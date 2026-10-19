"""检测工作线程。

职责
----
把阻塞式的 :meth:`DetectionService.run` 放到独立线程，并通过 **Qt 信号**
把结果安全地送回界面线程。

线程模型
--------

.. code-block:: text

   界面线程 (GUI)                       工作线程 (QThread)
   ─────────────────                    ────────────────────
   ViewModel.start()  ──queued──▶       run()
                                        ├─ 取帧
   ◀──frameReady(QImage 数据)──queued──  ├─ 流水线处理
   ◀──eventRaised(事件)───────queued──   ├─ 发布事件总线
   ◀──finished(汇总)──────────queued──   └─ 循环
   ViewModel.stop()   ──直接调用──▶      service.stop()（内部用 Event 通知，线程安全）

关键设计
--------
1. **只传纯数据**：信号携带 ``numpy`` 数组与领域对象，不携带 Qt 图形类型。
   ``QPixmap`` 只能在 GUI 线程创建，因此图像转换放在界面线程完成。
   这是 Qt 多线程编程最常见的踩坑点，也是必须遵守的约束。
2. **停止信号不走事件循环**：:meth:`DetectionService.run` 阻塞了工作线程的
   事件循环，因此 :meth:`stop` 由界面线程 **直接调用**（其内部通过
   :class:`threading.Event` 通知，线程安全）。
3. **UI 刷新节流**：回放模式可能跑到上千 FPS，按墙钟时间限制信号频率，
   避免把界面线程压垮（而采集线程仍全速运行，不丢数据）。
4. **单帧异常不影响会话**：任何绘图/信号异常都被隔离，循环继续。
"""

from __future__ import annotations

import logging
import threading
import time

from PySide6.QtCore import QObject, Signal, Slot

from ..application.pipeline import FrameAnalysis
from ..application.services import SessionSummary
from ..domain.models import BehaviorEvent, BehaviorType
from ..bootstrap import ApplicationContext
from .overlay import OverlayOptions, draw_overlay

logger = logging.getLogger(__name__)

__all__ = ["DetectionWorker"]


class DetectionWorker(QObject):
    """在独立线程中驱动 :class:`~driver_guard.bootstrap.ApplicationContext`。"""

    #: ``(叠加后的 BGR numpy 数组, FrameAnalysis)``
    frameReady = Signal(object, object)
    #: 一条新的领域事件
    eventRaised = Signal(object)
    #: 服务状态变化（中文状态名）
    stateChanged = Signal(str)
    #: 会话结束（``SessionSummary``）
    finished = Signal(object)
    #: 发生错误（错误描述）
    failed = Signal(str)
    #: 供状态栏显示的运行信息 ``(帧号, 帧率, 感知耗时ms)``
    telemetry = Signal(int, float, float)

    def __init__(
        self,
        context: ApplicationContext,
        *,
        overlay_options: OverlayOptions | None = None,
        display_fps: float = 30.0,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._context = context
        self._options = overlay_options or OverlayOptions()
        self._min_interval = 1.0 / max(1.0, display_fps)
        self._last_emit = 0.0
        self._active: dict[BehaviorType, BehaviorEvent] = {}
        # 状态栏会在界面线程读取 _active，而写入发生在工作线程
        self._active_lock = threading.Lock()
        self._running = False
        self._pending_recalibration = False
        self._fps_window_start = time.perf_counter()
        self._fps_window_frames = 0
        self._current_fps = 0.0

    # ------------------------------------------------------------------
    #  属性
    # ------------------------------------------------------------------
    @property
    def context(self) -> ApplicationContext:
        return self._context

    @property
    def overlay_options(self) -> OverlayOptions:
        return self._options

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def current_fps(self) -> float:
        return self._current_fps

    @property
    def active_events(self) -> tuple[BehaviorEvent, ...]:
        with self._active_lock:
            return tuple(self._active.values())

    # ------------------------------------------------------------------
    #  控制（由界面线程调用）
    # ------------------------------------------------------------------
    def stop(self) -> None:
        """请求停止采集。**直接调用**，不走事件循环（见模块文档）。"""
        self._context.detection_service.stop()

    def recalibrate(self) -> None:
        """清除中立位标定，使其在后续帧重新学习。

        标定器只被采集线程访问，这里通过置位一个"待复位"标记避免竞争。
        """
        self._pending_recalibration = True

    # ------------------------------------------------------------------
    #  工作线程入口
    # ------------------------------------------------------------------
    @Slot()
    def run(self) -> None:
        """阻塞式主循环（在工作线程中执行）。"""
        if self._running:
            logger.warning("工作线程已在运行，忽略重复启动")
            return

        self._running = True
        self._pending_recalibration = False
        self._fps_window_start = time.perf_counter()
        self._fps_window_frames = 0
        self.stateChanged.emit("运行中")

        try:
            summary: SessionSummary = self._context.detection_service.run(
                on_frame=self._on_frame
            )
        except Exception as exc:  # noqa: BLE001 - 兜底，保证界面收到通知
            logger.exception("工作线程异常终止")
            self._running = False
            with self._active_lock:
                self._active.clear()
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            self.stateChanged.emit("异常")
            return

        self._running = False
        with self._active_lock:
            self._active.clear()
        if summary.interrupted:
            self.stateChanged.emit("已停止")
        elif self._context.detection_service.last_error:
            self.stateChanged.emit("异常")
            self.failed.emit(self._context.detection_service.last_error or "未知错误")
        else:
            self.stateChanged.emit("已完成")
        self.finished.emit(summary)

    # ------------------------------------------------------------------
    #  每帧回调（工作线程内）
    # ------------------------------------------------------------------
    def _on_frame(self, analysis: FrameAnalysis) -> None:
        self._sync_active_events()
        self._account_fps()

        if self._pending_recalibration:
            self._pending_recalibration = False
            self._context.pipeline.reset()
            logger.info("已触发中立位重新标定")

        now = time.perf_counter()
        if now - self._last_emit < self._min_interval:
            return
        self._last_emit = now

        try:
            canvas = draw_overlay(
                analysis.frame.image,
                analysis.observation,
                options=self._options,
                active_events=self.active_events,
                metrics=analysis.metrics.as_dict(),
                calibration_progress=analysis.calibration_progress,
                banner=self._banner_text(),
                latency_ms=analysis.observation.latency_ms,
            )
        except Exception:  # noqa: BLE001 - 绘图失败不应中断检测
            logger.exception("叠加层绘制失败，改用原始帧")
            canvas = analysis.frame.image

        self.frameReady.emit(canvas, analysis)
        self.telemetry.emit(
            analysis.frame.index,
            self._current_fps,
            float(analysis.timings_ms.get("perception", 0.0)),
        )

    def _sync_active_events(self) -> None:
        """移除已解除的行为；新增由 :meth:`on_event` 负责。"""
        active = set(self._context.pipeline.confirmer.active_behaviors)
        with self._active_lock:
            for behavior in list(self._active):
                if behavior not in active:
                    del self._active[behavior]

    def _account_fps(self) -> None:
        """按秒滚动窗口统计实际处理帧率。"""
        self._fps_window_frames += 1
        now = time.perf_counter()
        elapsed = now - self._fps_window_start
        if elapsed >= 0.5:
            self._current_fps = self._fps_window_frames / elapsed
            self._fps_window_start = now
            self._fps_window_frames = 0

    def _banner_text(self) -> str:
        provider = self._context.provider.name
        state = self._context.detection_service.state.label
        return f"{provider} · {state}"

    # ------------------------------------------------------------------
    #  告警通道回调（由 AlertService 在工作线程中调用）
    # ------------------------------------------------------------------
    def on_event(self, event: BehaviorEvent, frame: object) -> None:
        """记录当前激活行为并把领域事件转成 Qt 信号。

        作为 :class:`~driver_guard.infrastructure.alerts.CallbackAlertSink`
        的回调注册进告警服务，因此与"控制台/CSV/快照"等通道享受同一套
        过滤与错误隔离机制。

        持续中的重复提醒（``ongoing=True``）只刷新画面状态条，
        而所有事件都会送到事件列表，由界面自行决定去重策略。
        """
        del frame
        with self._active_lock:
            self._active[event.behavior] = event
        self.eventRaised.emit(event)
