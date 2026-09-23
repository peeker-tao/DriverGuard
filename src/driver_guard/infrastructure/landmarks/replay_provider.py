"""回放感知适配器：按脚本产出指标，用于无摄像头演示与自动化测试。

定位
----
这不是"造假数据"的临时手段，而是一个**受控的测试替身（Test Double）**，
用途明确且有价值：

* **可复现性**：同一脚本必然产生同一序列事件，因此可以写成回归测试；
* **无硬件演示**：答辩现场若摄像头不可用，仍能完整展示"检测—告警—统计—报告"链路；
* **规则调参**：把指标直接推到阈值附近，快速验证去抖与迟滞逻辑是否符合预期；
* **性能基线**：剥离感知耗时，单独测量流水线其余三级与界面的开销。

与真实感知的关系
----------------
它实现同一个 :class:`~driver_guard.application.ports.LandmarkProvider` 端口，
因此对上层完全透明；它通过 ``Observation.precomputed`` 直接提供指标，
从而跳过几何计算（:class:`~driver_guard.domain.metrics.MetricExtractor`
会原样采用该值）。这在架构上属于"可选通道"而非特例分支。

.. warning::
   本适配器 **不产生真实人脸关键点**，因此不能用于评测检测准确率。
   界面上会明确标注当前处于"回放模式"。
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np

from ...application.ports import LandmarkProvider
from ...domain.models import DriverMetrics, Frame, Landmark, Observation

logger = logging.getLogger(__name__)

__all__ = ["ReplayLandmarkProvider", "MetricSegment", "SCENARIOS"]


# ===========================================================================
@dataclass(frozen=True, slots=True)
class MetricSegment:
    """脚本中的一个时间片：``[start, end)`` 秒内持续输出给定指标。"""

    start: float
    end: float
    metrics: DriverMetrics
    label: str = ""

    def covers(self, t: float) -> bool:
        return self.start <= t < self.end


def _neutral(**overrides: object) -> DriverMetrics:
    """构造一份"正常驾驶"基准指标，便于脚本只覆盖关心的字段。"""
    base = DriverMetrics(
        face_detected=True,
        ear=0.30,
        ear_left=0.30,
        ear_right=0.30,
        mar=0.16,
        pitch=5.0,
        yaw=-2.0,
        roll=1.0,
        face_pitch_ratio=0.58,
        eye_blink=0.06,
        eye_blink_left=0.06,
        eye_blink_right=0.06,
        jaw_open=0.06,
        hand_face_ratio=1.60,
        hand_mouth_ratio=1.40,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


#: 预置演示脚本：``名称 -> 时间片序列``。
#: 每条脚本都是 **有限周期**，配合 ``loop=True`` 可无限循环演示。
SCENARIOS: dict[str, tuple[MetricSegment, ...]] = {
    # 全程正常驾驶，用于观察误报率
    "normal": (MetricSegment(0.0, 30.0, _neutral(), "正常驾驶"),),
    # 疲劳：闭眼 → 打哈欠 → 低头，再恢复
    "fatigue": (
        MetricSegment(0.0, 4.0, _neutral(), "正常驾驶"),
        MetricSegment(4.0, 9.0, _neutral(ear=0.12, eye_blink=0.88), "闭眼疲劳"),
        MetricSegment(9.0, 13.0, _neutral(mar=0.72, jaw_open=0.66), "打哈欠"),
        MetricSegment(
            13.0, 19.0, _neutral(pitch=-24.0, face_pitch_ratio=0.44), "低头"
        ),
        MetricSegment(19.0, 26.0, _neutral(), "恢复正常"),
    ),
    # 分心：左右偏头、低头看手机
    "distraction": (
        MetricSegment(0.0, 4.0, _neutral(), "正常驾驶"),
        MetricSegment(4.0, 9.0, _neutral(yaw=34.0, roll=6.0), "侧头分心"),
        MetricSegment(9.0, 14.0, _neutral(roll=29.0), "头部倾斜"),
        MetricSegment(
            14.0, 20.0, _neutral(pitch=-22.0, face_pitch_ratio=0.45), "低头看手机"
        ),
        MetricSegment(20.0, 27.0, _neutral(), "恢复正常"),
    ),
    # 违规：打电话 → 抽烟
    "violation": (
        MetricSegment(0.0, 4.0, _neutral(), "正常驾驶"),
        MetricSegment(
            4.0, 12.0, _neutral(hand_face_ratio=0.28, hand_mouth_ratio=0.55), "打电话"
        ),
        MetricSegment(
            12.0, 18.0, _neutral(hand_mouth_ratio=0.16, hand_face_ratio=0.80), "抽烟"
        ),
        MetricSegment(18.0, 25.0, _neutral(), "恢复正常"),
    ),
    # 驾驶员离位
    "absent": (
        MetricSegment(0.0, 3.0, _neutral(), "正常驾驶"),
        MetricSegment(3.0, 13.0, DriverMetrics(face_detected=False), "驾驶员离位"),
        MetricSegment(13.0, 20.0, _neutral(), "返回座位"),
    ),
    # 混合：用于演示"多行为并发 + 统计报表"
    "mixed": (
        MetricSegment(0.0, 3.0, _neutral(), "正常驾驶"),
        MetricSegment(3.0, 7.0, _neutral(ear=0.13, eye_blink=0.85), "闭眼疲劳"),
        MetricSegment(7.0, 11.0, _neutral(yaw=36.0), "侧头分心"),
        MetricSegment(11.0, 16.0, _neutral(hand_face_ratio=0.26), "打电话"),
        MetricSegment(16.0, 21.0, _neutral(mar=0.70, jaw_open=0.64), "打哈欠"),
        MetricSegment(21.0, 30.0, _neutral(), "恢复正常"),
    ),
}


# ===========================================================================
class ReplayLandmarkProvider(LandmarkProvider):
    """按脚本产出指标的感知适配器。

    Args:
        scenario: :data:`SCENARIOS` 中的名称，或直接传入自定义时间片序列。
        loop: 时间超出脚本末尾后是否从头循环（默认 ``True``，便于长时间演示）。
        with_hands: 是否附带合成手部关键点。打电话/抽烟规则要求手部存在，
            因此默认开启；置 ``False`` 可用于验证"手部通道缺失时的降级行为"。
        fps_hint: 用于把帧号换算为时间戳的名义帧率（仅当帧自带时间戳异常时使用）。
    """

    def __init__(
        self,
        scenario: str | Iterable[MetricSegment] = "mixed",
        *,
        loop: bool = True,
        with_hands: bool = True,
        fps_hint: float = 30.0,
    ) -> None:
        if isinstance(scenario, str):
            if scenario not in SCENARIOS:
                raise KeyError(
                    f"未知脚本 {scenario!r}，可选：{', '.join(sorted(SCENARIOS))}"
                )
            segments: Sequence[MetricSegment] = SCENARIOS[scenario]
            self._scenario_name = scenario
        else:
            segments = list(scenario)
            self._scenario_name = "custom"
            if not segments:
                raise ValueError("脚本不能为空")

        self._segments = sorted(segments, key=lambda s: s.start)
        self._starts = [s.start for s in self._segments]
        self._loop = loop
        self._with_hands = with_hands
        self._fps = max(1.0, fps_hint)
        self._span = max(s.end for s in self._segments)
        self._inferred_frames = 0

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return f"回放脚本[{self._scenario_name}]"

    @property
    def scenario_name(self) -> str:
        return self._scenario_name

    @property
    def duration(self) -> float:
        return self._span

    def describe(self) -> str:
        """人类可读的脚本说明，用于界面提示。"""
        parts = [f"{s.label or '片段'}({s.start:g}~{s.end:g}s)" for s in self._segments]
        return " → ".join(parts)

    # ------------------------------------------------------------------
    def infer(self, frame: Frame) -> Observation:
        t = frame.timestamp
        if not np.isfinite(t) or t < 0:
            t = self._inferred_frames / self._fps
        self._inferred_frames += 1

        if self._loop and self._span > 0 and t >= self._span:
            t = t % self._span

        metrics = self._metrics_at(t)
        return Observation(
            face=self._synthetic_face() if metrics.face_detected else (),
            hands=self._synthetic_hands() if (self._with_hands and metrics.face_detected) else (),
            precomputed=metrics,
            provider=self.name,
            latency_ms=0.0,
        )

    # ------------------------------------------------------------------
    def _metrics_at(self, t: float) -> DriverMetrics:
        index = bisect.bisect_right(self._starts, t) - 1
        if index < 0:
            index = 0
        segment = self._segments[min(index, len(self._segments) - 1)]
        return segment.metrics if segment.covers(t) else self._segments[-1].metrics

    # ------------------------------------------------------------------
    #  合成关键点（仅为让界面叠加层与"手部存在性"判定正常工作）
    # ------------------------------------------------------------------
    @staticmethod
    def _synthetic_face() -> tuple[Landmark, ...]:
        """构造 478 个占位关键点，围成一个居中的椭圆脸。

        仅用于界面绘制与"人脸存在"判定；不含真实语义，
        因此指标一律走 ``precomputed`` 通道，不参与几何计算。
        """
        count = 478
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        xs = 0.5 + 0.16 * np.cos(angles)
        ys = 0.46 + 0.21 * np.sin(angles)
        return tuple(
            Landmark(x=float(x), y=float(y), z=0.0)
            for x, y in zip(xs, ys, strict=True)
        )

    @staticmethod
    def _synthetic_hands() -> tuple[tuple[Landmark, ...], ...]:
        """构造 1 只手的 21 个占位关键点（贴近头面部）。"""
        base_x, base_y = 0.62, 0.42
        offsets = [
            (0.00, 0.00), (0.01, -0.02), (0.02, -0.04), (0.03, -0.05), (0.04, -0.06),
            (0.02, 0.01), (0.04, 0.00), (0.05, -0.01), (0.06, -0.02),
            (0.02, 0.03), (0.05, 0.03), (0.06, 0.02), (0.07, 0.01),
            (0.02, 0.05), (0.05, 0.06), (0.06, 0.05), (0.07, 0.04),
            (0.01, 0.06), (0.04, 0.08), (0.05, 0.07), (0.05, 0.06),
        ]
        hand = tuple(
            Landmark(x=base_x + dx, y=base_y + dy, z=0.0) for dx, dy in offsets
        )
        return (hand,)


# ---------------------------------------------------------------------------
#  脚本时长查询
# ---------------------------------------------------------------------------
def scenario_duration(name: str | None) -> float:
    """返回内置脚本的总时长（秒）；名字未知时返回 ``0.0``。

    组合根用它把"回放数据源"的长度对齐到脚本长度，使脚本跑完即自动收尾，
    而不是无限期地重复最后一段。
    """
    segments = SCENARIOS.get((name or "").strip())
    if not segments:
        return 0.0
    return max(segment.end for segment in segments)


# ---------------------------------------------------------------------------
#  一致性检查：脚本中出现的指标字段应存在于 DriverMetrics
# ---------------------------------------------------------------------------
def _validate_scenarios() -> list[str]:
    """校验内置脚本的区间合法性与无重叠性，返回问题列表（空表示合法）。"""
    problems: list[str] = []
    for name, segments in SCENARIOS.items():
        if not segments:
            problems.append(f"{name}: 脚本为空")
            continue
        previous_end = -1.0
        for segment in segments:
            if segment.end <= segment.start:
                problems.append(f"{name}: 片段区间非法 [{segment.start}, {segment.end})")
            if segment.start < previous_end:
                problems.append(f"{name}: 片段与上一段重叠（起点 {segment.start}）")
            previous_end = segment.end
    return problems


SCENARIO_PROBLEMS: tuple[str, ...] = tuple(_validate_scenarios())
if SCENARIO_PROBLEMS:  # pragma: no cover - 仅防止脚本编写错误
    logger.error("内置演示脚本存在 %d 处问题：%s", len(SCENARIO_PROBLEMS), SCENARIO_PROBLEMS)
