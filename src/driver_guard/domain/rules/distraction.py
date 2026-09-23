"""分心驾驶类规则：视线偏离、驾驶员离位。"""

from __future__ import annotations

from ..models import BehaviorType, DetectionContext, RuleOutcome
from .base import BehaviorRule, RuleConfig

__all__ = ["GazeOffRoadRule", "DriverAbsentRule"]


class GazeOffRoadRule(BehaviorRule):
    """视线/头部持续偏离行驶方向。

    判定依据是 **相对个人中立位的水平与倾斜偏离**：取 ``yaw`` 与 ``roll``
    偏离中立位后的绝对值，任一超过阈值即认为注意力离开前方。

    由于本规则天然使用绝对偏离量，不需要区分左右方向，
    因此同样规避了 MediaPipe 轴系语义不确定的问题。
    """

    behavior = BehaviorType.GAZE_OFF_ROAD
    requires_face = True

    def _evaluate(self, ctx: DetectionContext) -> RuleOutcome:
        yaw_threshold = self._magnitude_threshold(
            self.config.get_float("yaw_threshold_deg", 25.0), 25.0
        )
        roll_threshold = self._magnitude_threshold(
            self.config.get_float("roll_threshold_deg", 22.0), 22.0
        )

        deviation = ctx.pose_deviation
        if deviation is None:
            return self._skip()

        yaw = abs(deviation.yaw)
        roll = abs(deviation.roll)
        yaw_hit = yaw > yaw_threshold
        roll_hit = roll > roll_threshold
        if not (yaw_hit or roll_hit):
            return self._skip()

        confidence = max(
            self._score(yaw, yaw_threshold) if yaw_hit else 0.0,
            self._score(roll, roll_threshold) if roll_hit else 0.0,
        )
        return self._trigger(
            ctx, confidence,
            f"头部偏离行驶方向（水平偏转 {yaw:.1f}°，倾斜 {roll:.1f}°）",
            hit_yaw=1.0 if yaw_hit else 0.0,
            hit_roll=1.0 if roll_hit else 0.0,
            yaw_deviation=round(deviation.yaw, 3),
            roll_deviation=round(deviation.roll, 3),
            pitch_deviation=round(deviation.pitch, 3),
        )


class DriverAbsentRule(BehaviorRule):
    """驾驶员离位 / 长时间未被检测到。

    这条规则与其余规则相反：它的触发条件是 **没有检测到人脸**，
    因此 ``requires_face`` 必须为 ``False``，且需要自己累计"无人"的持续时间。

    之所以在规则内自行计时（而非交给时序确认层），是因为本规则的
    事件"开始时刻"就是无人状态的开始时刻，两者语义重合。
    """

    behavior = BehaviorType.DRIVER_ABSENT
    requires_face = False

    def __init__(self, config: RuleConfig | None = None) -> None:
        super().__init__(config)
        self._absent_since: float | None = None

    # -- 与其它规则不同：状态由帧时间驱动，需显式复位 -------------------
    def reset(self) -> None:
        self._absent_since = None

    def _evaluate(self, ctx: DetectionContext) -> RuleOutcome:
        if ctx.observation.has_face:
            self._absent_since = None
            return self._skip()

        if self._absent_since is None:
            self._absent_since = ctx.frame.timestamp

        absent_seconds = self._magnitude_threshold(
            self.config.get_float("absent_seconds", 3.0), 3.0
        )
        elapsed = ctx.frame.timestamp - self._absent_since
        if elapsed < absent_seconds:
            return self._skip()

        confidence = self._score(elapsed, absent_seconds)
        return self._trigger(
            ctx, confidence,
            f"已连续 {elapsed:.1f} 秒未检测到驾驶员",
            absent_seconds=round(elapsed, 2),
        )

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<DriverAbsentRule severity={self.severity.name}>"
