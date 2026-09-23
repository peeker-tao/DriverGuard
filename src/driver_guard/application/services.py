"""应用服务层：编排（orchestration）与统计。

本层是"薄"的：它只负责 **编排** 领域对象与端口，几乎不含业务判断。
这样划分的好处是：

* 领域逻辑（阈值、融合、时序）可以在没有摄像头、没有 Qt 的环境下测试；
* 换一种驱动方式（Web 服务、批处理离线分析）只需重写本层的编排，
  领域与基础设施可原样复用。

包含三个服务：

======================  ====================================================
服务                    职责
======================  ====================================================
``DetectionService``    驱动"取帧 → 流水线 → 发布事件 → 落库"主循环
``AlertService``        订阅事件总线，把事件扇出到各告警通道（AlertSink）
``StatisticsService``   订阅事件总线，实时聚合统计指标
======================  ====================================================
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Iterable, Mapping, Sequence

from ..domain.models import BehaviorCategory, BehaviorEvent, BehaviorType, Severity
from .event_bus import EventBus, Subscription
from .pipeline import DetectionPipeline, FrameAnalysis
from .ports import AlertSink, EventRepository, FrameSource, FrameSourceError

logger = logging.getLogger(__name__)

__all__ = [
    "ServiceState",
    "SessionSummary",
    "DetectionService",
    "AlertService",
    "StatisticsService",
]


# ===========================================================================
class ServiceState(enum.Enum):
    """采集服务状态机。"""

    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    FINISHED = "finished"
    ERROR = "error"

    @property
    def label(self) -> str:
        return {
            ServiceState.IDLE: "未启动",
            ServiceState.RUNNING: "运行中",
            ServiceState.STOPPING: "停止中",
            ServiceState.FINISHED: "已结束",
            ServiceState.ERROR: "异常",
        }[self]


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """一次检测会话的汇总，用于结束报告。"""

    source_name: str
    frames: int
    duration: float
    avg_fps: float
    triggered_frames: int
    event_counts: Mapping[BehaviorType, int]
    stage_avg_ms: Mapping[str, float]
    started_at: datetime
    ended_at: datetime
    interrupted: bool = False

    @property
    def risk_index(self) -> float:
        """简单风险指数：单位时间加权告警数（0~100）。

        权重直接复用 :class:`Severity` 的数值，因此"危急"事件的贡献是
        "提示"的 4 倍。该指数用于会话间的横向对比，不是物理量。
        """
        if self.duration <= 0:
            return 0.0
        weighted = sum(
            count * (Severity.parse(behavior_severity(behavior)).value + 1)
            for behavior, count in self.event_counts.items()
        )
        per_minute = weighted / (self.duration / 60.0)
        return round(min(100.0, per_minute * 5.0), 2)


#: 行为 → 默认严重级别的兜底表（仅用于汇总报告；规则实例持有权威值）
_DEFAULT_SEVERITY: dict[BehaviorType, str] = {
    BehaviorType.EYES_CLOSED: "high",
    BehaviorType.YAWNING: "medium",
    BehaviorType.HEAD_DOWN: "high",
    BehaviorType.GAZE_OFF_ROAD: "high",
    BehaviorType.PHONE_CALL: "critical",
    BehaviorType.SMOKING: "high",
    BehaviorType.DRIVER_ABSENT: "critical",
}


def behavior_severity(behavior: BehaviorType) -> str:
    return _DEFAULT_SEVERITY.get(behavior, "medium")


# ===========================================================================
class DetectionService:
    """驱动检测主循环。

    Args:
        source: 帧来源端口。
        pipeline: 检测流水线。
        event_bus: 事件总线（发布领域事件）。
        repository: 事件仓储；``None`` 表示不持久化。
        max_frames: 最多处理帧数（``0`` 表示不限制），用于离线批处理与测试。
        target_fps: 目标帧率上限；``<=0`` 表示不限制（由数据源自身节奏决定）。
        frame_interval_tolerance: 允许的节奏偏差比例，避免过度 sleep。

    Notes:
        :meth:`run` 是阻塞式的，设计为在独立线程中调用；
        :meth:`stop` 可从任意线程调用，通过 :class:`threading.Event` 通知退出。
    """

    def __init__(
        self,
        *,
        source: FrameSource,
        pipeline: DetectionPipeline,
        event_bus: EventBus,
        repository: EventRepository | None = None,
        max_frames: int = 0,
        target_fps: float = 0.0,
        frame_interval_tolerance: float = 0.15,
    ) -> None:
        self._source = source
        self._pipeline = pipeline
        self._bus = event_bus
        self._repository = repository
        self._max_frames = max_frames
        self._target_fps = target_fps
        self._tolerance = frame_interval_tolerance

        self._stop_event = threading.Event()
        self._state = ServiceState.IDLE
        self._state_lock = threading.Lock()
        self._thread: threading.Thread | None = None

        self._frame_count = 0
        self._last_error: str | None = None
        self._last_analysis: FrameAnalysis | None = None

    # ------------------------------------------------------------------
    #  属性
    # ------------------------------------------------------------------
    @property
    def state(self) -> ServiceState:
        with self._state_lock:
            return self._state

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def last_analysis(self) -> FrameAnalysis | None:
        return self._last_analysis

    @property
    def source(self) -> FrameSource:
        return self._source

    @property
    def pipeline(self) -> DetectionPipeline:
        return self._pipeline

    def _set_state(self, state: ServiceState) -> None:
        with self._state_lock:
            self._state = state
        logger.debug("服务状态 → %s", state.label)

    # ------------------------------------------------------------------
    #  生命周期
    # ------------------------------------------------------------------
    def stop(self) -> None:
        """请求停止（线程安全，可从界面线程调用）。"""
        if self.state is ServiceState.RUNNING:
            self._set_state(ServiceState.STOPPING)
        self._stop_event.set()

    def limit_frames(self, count: int) -> None:
        """设置最多处理的帧数（``0`` 表示不限制）。

        用途：离线批处理、自动化回归测试，以及"演示时只跑固定时长"。
        可在运行中调用，下一个循环周期即生效。
        """
        self._max_frames = max(0, int(count))

    def set_target_fps(self, fps: float) -> None:
        """设置目标帧率上限（``<= 0`` 表示不节流）。运行中调用下一周期生效。"""
        self._target_fps = max(0.0, float(fps))

    def start_async(
        self,
        on_frame: Callable[[FrameAnalysis], None] | None = None,
        on_finished: Callable[[SessionSummary], None] | None = None,
    ) -> threading.Thread:
        """在后台线程启动 :meth:`run`，返回该线程对象。"""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("检测服务已在运行")

        def target() -> None:
            summary = self.run(on_frame=on_frame)
            if on_finished is not None:
                on_finished(summary)

        self._thread = threading.Thread(target=target, name="detection-loop", daemon=True)
        self._thread.start()
        return self._thread

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    #  主循环
    # ------------------------------------------------------------------
    def run(
        self,
        on_frame: Callable[[FrameAnalysis], None] | None = None,
    ) -> SessionSummary:
        """阻塞式执行采集与检测，直到数据结束或 :meth:`stop` 被调用。"""
        self._stop_event.clear()
        self._frame_count = 0
        self._last_error = None
        self._set_state(ServiceState.RUNNING)

        started_at = datetime.now()
        started = time.perf_counter()
        triggered_frames = 0
        stage_totals: dict[str, float] = {}
        event_counts: Counter[BehaviorType] = Counter()
        interrupted = False

        try:
            self._source.open()
            self._pipeline.warmup()
            interval = 1.0 / self._target_fps if self._target_fps > 0 else 0.0
            next_deadline = time.perf_counter() + interval

            while not self._stop_event.is_set():
                frame = self._source.read()
                if frame is None:
                    logger.info("数据源读取结束，共处理 %d 帧", self._frame_count)
                    break

                analysis = self._pipeline.process(frame)
                self._last_analysis = analysis
                self._frame_count += 1
                if analysis.triggered:
                    triggered_frames += 1
                for stage, elapsed in analysis.timings_ms.items():
                    stage_totals[stage] = stage_totals.get(stage, 0.0) + float(elapsed)

                for event in analysis.events:
                    event_counts[event.behavior] += 1
                    if self._repository is not None:
                        self._repository.save(event)
                    self._bus.publish(event, analysis)

                if on_frame is not None:
                    on_frame(analysis)

                if self._max_frames and self._frame_count >= self._max_frames:
                    logger.info("达到 max_frames=%d，正常结束", self._max_frames)
                    break

                if interval > 0.0:
                    next_deadline, sleep_for = _pace(next_deadline, interval, self._tolerance)
                    if sleep_for > 0:
                        # 用 wait 而非 sleep，使 stop() 能立即打断节流等待
                        self._stop_event.wait(sleep_for)

            else:
                interrupted = True
                logger.info("收到停止信号，共处理 %d 帧", self._frame_count)

            self._set_state(ServiceState.FINISHED if not interrupted else ServiceState.IDLE)

        except FrameSourceError as exc:
            self._last_error = str(exc)
            self._set_state(ServiceState.ERROR)
            logger.error("数据源错误: %s", exc)
        except Exception as exc:  # noqa: BLE001 - 服务级兜底，保证 UI 能收到结束通知
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._set_state(ServiceState.ERROR)
            logger.exception("检测主循环异常终止")
        finally:
            self._source.close()
            self._pipeline.close()

        duration = max(1e-6, time.perf_counter() - started)
        frames = max(1, self._frame_count)
        return SessionSummary(
            source_name=self._source.name,
            frames=self._frame_count,
            duration=duration,
            avg_fps=self._frame_count / duration,
            triggered_frames=triggered_frames,
            event_counts=dict(event_counts),
            stage_avg_ms={k: v / frames for k, v in stage_totals.items()},
            started_at=started_at,
            ended_at=datetime.now(),
            interrupted=interrupted,
        )


def _pace(deadline: float, interval: float, tolerance: float) -> tuple[float, float]:
    """计算到下一个周期的剩余等待时间，并返回更新后的截止时刻。

    若处理速度已经落后于目标帧率，则重置节奏而不补偿积压，
    避免"追赶"导致突发占用 CPU（这在实时视频场景中比帧率精度更重要）。
    """
    now = time.perf_counter()
    sleep_for = deadline - now
    if sleep_for < -interval * tolerance:
        return now + interval, 0.0
    return deadline + interval, max(0.0, sleep_for)


# ===========================================================================
class AlertService:
    """订阅事件总线，把领域事件扇出到各 :class:`AlertSink` 通道。

    支持按 **最低严重级别** 过滤，使声音/快照等"高打扰"通道只对严重事件生效。
    """

    def __init__(
        self,
        event_bus: EventBus,
        sinks: Iterable[AlertSink] = (),
        *,
        min_severity: Severity = Severity.LOW,
        per_sink_min_severity: Mapping[str, Severity] | None = None,
    ) -> None:
        self._bus = event_bus
        self._sinks = list(sinks)
        self._min_severity = min_severity
        self._per_sink = dict(per_sink_min_severity or {})
        self._subscription: Subscription | None = None
        self._routed = 0

    @property
    def sinks(self) -> tuple[AlertSink, ...]:
        return tuple(self._sinks)

    @property
    def routed_count(self) -> int:
        return self._routed

    def add_sink(self, sink: AlertSink) -> None:
        self._sinks.append(sink)

    def start(self) -> None:
        if self._subscription is None:
            self._subscription = self._bus.subscribe(
                self._on_event, label="AlertService", behaviors=None
            )

    def stop(self) -> None:
        if self._subscription is not None:
            self._subscription.close()
            self._subscription = None

    # ------------------------------------------------------------------
    def _on_event(self, event: BehaviorEvent, payload: object) -> None:
        if event.severity < self._min_severity:
            return
        frame = getattr(payload, "frame", None)
        for sink in self._sinks:
            threshold = self._per_sink.get(sink.name, self._min_severity)
            if event.severity < threshold:
                continue
            try:
                sink.emit(event, frame)
            except Exception:  # noqa: BLE001 - 单通道故障不影响其它通道
                logger.exception("告警通道 %s 输出失败", sink.name)
        self._routed += 1

    def close(self) -> None:
        self.stop()
        for sink in self._sinks:
            try:
                sink.close()
            except Exception:  # noqa: BLE001
                logger.exception("关闭告警通道 %s 失败", sink.name)


# ===========================================================================
class StatisticsService:
    """订阅事件总线并实时聚合统计（供界面仪表盘与会话报告使用）。"""

    def __init__(self, event_bus: EventBus) -> None:
        self._bus = event_bus
        self._lock = threading.RLock()
        self._counts: Counter[BehaviorType] = Counter()
        self._severity_counts: Counter[Severity] = Counter()
        self._first_at: datetime | None = None
        self._last_at: datetime | None = None
        self._subscription = event_bus.subscribe(self._on_event, label="StatisticsService")

    # ------------------------------------------------------------------
    @property
    def counts(self) -> Mapping[BehaviorType, int]:
        with self._lock:
            return dict(self._counts)

    @property
    def total(self) -> int:
        with self._lock:
            return sum(self._counts.values())

    @property
    def severity_counts(self) -> Mapping[Severity, int]:
        with self._lock:
            return dict(self._severity_counts)

    @property
    def window(self) -> tuple[datetime | None, datetime | None]:
        with self._lock:
            return self._first_at, self._last_at

    def category_counts(self) -> Mapping[BehaviorCategory, int]:
        """按"疲劳/分心/违规"三大类聚合，用于报告饼图。"""
        with self._lock:
            counter: Counter[BehaviorCategory] = Counter()
            for behavior, count in self._counts.items():
                counter[behavior.category] += count
            return dict(counter)

    def top(self, limit: int = 3) -> Sequence[tuple[BehaviorType, int]]:
        with self._lock:
            return self._counts.most_common(limit)

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()
            self._severity_counts.clear()
            self._first_at = None
            self._last_at = None

    def close(self) -> None:
        self._subscription.close()

    # ------------------------------------------------------------------
    def _on_event(self, event: BehaviorEvent, payload: object) -> None:
        del payload
        with self._lock:
            # 同一次持续行为的重复提醒不计入次数，避免拉高统计
            if not event.ongoing:
                self._counts[event.behavior] += 1
                self._severity_counts[event.severity] += 1
            self._last_at = event.wall_clock
            if self._first_at is None:
                self._first_at = event.wall_clock
