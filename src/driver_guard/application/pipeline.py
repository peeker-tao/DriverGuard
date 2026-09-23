"""检测流水线（Pipeline）——管道-过滤器架构的骨架实现。

一帧图像的完整旅程
------------------

.. code-block:: text

   Frame
     │
     ├─[F1 感知]──► LandmarkProvider.infer        → Observation
     │
     ├─[F2 度量]──► MetricExtractor.extract       → DriverMetrics
     │
     ├─[F3 决策]──► RuleEngine.evaluate           → [RuleOutcome] × N
     │
     └─[F4 确认]──► BehaviorConfirmer.update      → [BehaviorEvent] × M

四个过滤器之间只通过值对象通信（``Observation`` / ``DriverMetrics`` /
``RuleOutcome`` / ``BehaviorEvent``），互不感知对方的实现，
因此每一级都可以独立替换、独立测试。

:class:`DetectionPipeline` 采用 **模板方法（Template Method）** 模式：
:meth:`process` 固定了过滤器的调用顺序与计时/埋点，具体每一步的
实现通过构造注入，子类也可以只覆盖某一个钩子（例如加入 GPU 批次推理）。
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ..domain.calibration import NeutralPoseCalibrator
from ..domain.confirmation import BehaviorConfirmer
from ..domain.metrics import MetricExtractor
from ..domain.models import (
    BehaviorEvent,
    BehaviorType,
    DetectionContext,
    DriverMetrics,
    Frame,
    NeutralProfile,
    Observation,
    RuleOutcome,
)
from ..domain.rules.base import RuleEngine
from .ports import LandmarkProvider

logger = logging.getLogger(__name__)

__all__ = ["FrameAnalysis", "DetectionPipeline"]


# ===========================================================================
@dataclass(frozen=True, slots=True)
class FrameAnalysis:
    """单帧处理的完整结果（DTO），供界面渲染与统计消费。"""

    frame: Frame
    observation: Observation
    metrics: DriverMetrics
    outcomes: tuple[RuleOutcome, ...] = ()
    events: tuple[BehaviorEvent, ...] = ()
    neutral: NeutralProfile | None = None
    calibration_progress: float = 0.0
    timings_ms: Mapping[str, float] = field(default_factory=dict)

    @property
    def triggered(self) -> tuple[RuleOutcome, ...]:
        """本帧成立（未经过时序确认）的规则结果。"""
        return tuple(o for o in self.outcomes if o.triggered)

    @property
    def triggered_behaviors(self) -> tuple[BehaviorType, ...]:
        return tuple(o.behavior for o in self.triggered)

    @property
    def total_ms(self) -> float:
        return float(sum(self.timings_ms.values()))


# ===========================================================================
class DetectionPipeline:
    """把四个过滤器串成一条流水线。

    Args:
        provider: 感知端口（第 1 级）。
        extractor: 度量提取器（第 2 级）。
        engine: 规则引擎（第 3 级）。
        confirmer: 时序确认状态机（第 4 级）。
        calibrator: 个人中立位标定器；``None`` 表示不标定，
            此时头部姿态规则自动跳过（其余规则不受影响）。
        smoothing_window: 指标滑动窗口长度（帧）。窗口内均值用于抑制抖动。
    """

    #: 各阶段使用的埋点名称，便于统一读取耗时
    STAGE_PERCEPTION = "perception"
    STAGE_METRICS = "metrics"
    STAGE_RULES = "rules"
    STAGE_CONFIRM = "confirm"

    def __init__(
        self,
        *,
        provider: LandmarkProvider,
        extractor: MetricExtractor,
        engine: RuleEngine,
        confirmer: BehaviorConfirmer,
        calibrator: NeutralPoseCalibrator | None = None,
        smoothing_window: int = 9,
    ) -> None:
        self._provider = provider
        self._extractor = extractor
        self._engine = engine
        self._confirmer = confirmer
        self._calibrator = calibrator
        self._history: deque[DriverMetrics] = deque(maxlen=max(1, smoothing_window))

    # ------------------------------------------------------------------
    #  只读属性（供界面展示与诊断）
    # ------------------------------------------------------------------
    @property
    def provider(self) -> LandmarkProvider:
        return self._provider

    @property
    def engine(self) -> RuleEngine:
        return self._engine

    @property
    def confirmer(self) -> BehaviorConfirmer:
        return self._confirmer

    @property
    def calibrator(self) -> NeutralPoseCalibrator | None:
        return self._calibrator

    @property
    def neutral(self) -> NeutralProfile | None:
        return self._calibrator.profile if self._calibrator else None

    @property
    def history(self) -> tuple[DriverMetrics, ...]:
        """当前滑动窗口内容（不含本帧），供界面绘制曲线。"""
        return tuple(self._history)

    # ------------------------------------------------------------------
    #  主流程（模板方法）
    # ------------------------------------------------------------------
    def process(self, frame: Frame) -> FrameAnalysis:
        """处理一帧并返回完整分析结果。

        每次调用都会依次经过 F1~F4 四个过滤器，并记录各阶段耗时（毫秒），
        供界面显示与性能分析使用。
        """
        timings: dict[str, float] = {}

        observation, timings[self.STAGE_PERCEPTION] = _timed(
            lambda: self._perceive(frame)
        )
        metrics, timings[self.STAGE_METRICS] = _timed(
            lambda: self._measure(observation, frame)
        )
        neutral = self._update_calibration(metrics)

        context = self._build_context(frame, observation, metrics, neutral)
        outcomes, timings[self.STAGE_RULES] = _timed(lambda: self._engine.evaluate(context))

        events, timings[self.STAGE_CONFIRM] = _timed(
            lambda: self._confirm(outcomes, frame)
        )
        self._history.append(metrics)

        return FrameAnalysis(
            frame=frame,
            observation=observation,
            metrics=metrics,
            outcomes=tuple(outcomes),
            events=tuple(events),
            neutral=neutral,
            calibration_progress=self._calibrator.progress if self._calibrator else 1.0,
            timings_ms=timings,
        )

    # ------------------------------------------------------------------
    #  可覆盖的钩子
    # ------------------------------------------------------------------
    def _perceive(self, frame: Frame) -> Observation:
        """F1：调用感知端口。子类可换为批次推理或多模型融合。"""
        return self._provider.infer(frame)

    def _measure(self, observation: Observation, frame: Frame) -> DriverMetrics:
        """F2：由感知结果计算指标。"""
        return self._extractor.extract(observation, frame.width, frame.height)

    def _build_context(
        self,
        frame: Frame,
        observation: Observation,
        metrics: DriverMetrics,
        neutral: NeutralProfile | None,
    ) -> DetectionContext:
        """组装规则上下文（含历史滑窗与中立位）。"""
        return DetectionContext(
            frame=frame,
            observation=observation,
            metrics=metrics,
            history=tuple(self._history),
            neutral=neutral,
        )

    def _confirm(self, outcomes: Sequence[RuleOutcome], frame: Frame) -> list[BehaviorEvent]:
        """F4：时序确认，并把真实帧号回填到事件上。"""
        confirmed: list[BehaviorEvent] = []
        for outcome in outcomes:
            event = self._confirmer.update(outcome, frame.timestamp)
            if event is None:
                continue
            # dataclasses.replace 需要导入；这里直接构造新事件以保持 frozen 语义
            confirmed.append(
                BehaviorEvent(
                    behavior=event.behavior,
                    severity=event.severity,
                    confidence=event.confidence,
                    frame_index=frame.index,
                    timestamp=event.timestamp,
                    message=event.message,
                    metrics=event.metrics,
                    duration=event.duration,
                    wall_clock=event.wall_clock,
                    event_id=event.event_id,
                    ongoing=event.ongoing,
                )
            )
        return confirmed

    # ------------------------------------------------------------------
    def _update_calibration(self, metrics: DriverMetrics) -> NeutralProfile | None:
        if self._calibrator is None:
            return None
        return self._calibrator.feed(metrics)

    def reset(self) -> None:
        """复位历史窗口与状态机（切换数据源/重新开始时调用）。"""
        self._history.clear()
        self._confirmer.reset()
        if self._calibrator is not None:
            self._calibrator.reset()

    # ------------------------------------------------------------------
    def warmup(self) -> None:
        """预热感知模型，避免前几帧耗时异常导致界面卡顿。"""
        try:
            self._provider.warmup()
        except Exception:  # noqa: BLE001 - 预热失败不应阻断启动
            logger.exception("感知模型预热失败（不影响后续运行）")

    def close(self) -> None:
        self._provider.close()


# ===========================================================================
def _timed(action) -> tuple[object, float]:
    """执行 ``action`` 并返回 ``(结果, 毫秒耗时)``。"""
    start = time.perf_counter()
    result = action()
    return result, (time.perf_counter() - start) * 1000.0
