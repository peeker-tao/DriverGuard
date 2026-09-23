"""违规驾驶类规则：打电话、抽烟。

两条规则都依赖 **手部关键点与头面部的相对几何**。
距离统一用 **脸宽** 归一化，使阈值与摄像头距离、驾驶员体型无关。

索引选取
--------
* 手部：``0``(腕) ``4``(拇指尖) ``8``(食指尖) ``12``(中指尖)
* 面部：``168``(脸中心) ``61``/``291``(嘴角，取中点作为嘴中心)

媒体管道（MediaPipe HandLandmarker）最多给出两只手，取所有手的指尖
到脸/嘴的**最小**距离作为判据。
"""

from __future__ import annotations

from ..models import BehaviorType, DetectionContext, RuleOutcome
from .base import BehaviorRule

__all__ = ["PhoneCallRule", "SmokingRule"]


class PhoneCallRule(BehaviorRule):
    """打电话：手（或手持物）长时间贴近脸部。

    判据：``hand_face_ratio``（手部关键点到脸中心的最小距离 ÷ 脸宽）
    小于 ``hand_face_distance_ratio``。

    当手举到耳边通话时该比值显著下降；正常握方向盘时手远离脸，
    比值较大。因此叠加了相对时长门槛（配置项 ``min_duration``，
    由时序确认层消费）以避免抬手挠头等瞬时动作误报。
    """

    behavior = BehaviorType.PHONE_CALL
    requires_face = True
    requires_hands = True

    def _evaluate(self, ctx: DetectionContext) -> RuleOutcome:
        threshold = self._magnitude_threshold(
            self.config.get_float("hand_face_distance_ratio", 0.55), 0.55
        )
        ratio = ctx.mean("hand_face_ratio")
        if ratio is None:
            return self._skip()
        if ratio >= threshold:
            return self._skip()

        confidence = self._score(ratio, threshold, higher_means_more=False)
        return self._trigger(
            ctx, confidence,
            f"检测到手部贴近头面部（归一化距离 {ratio:.2f} < {threshold:.2f}）",
            hand_face_ratio=round(ratio, 4),
            threshold=threshold,
        )


class SmokingRule(BehaviorRule):
    """抽烟：手（或手持物）长时间贴近嘴部。

    与打电话的区别在于**贴近位置**：电话贴近耳/侧脸，抽烟贴近嘴。
    因此本规则使用 ``hand_mouth_ratio``（到嘴中心的最小距离 ÷ 脸宽），
    阈值比打电话更小，且要求更长的持续时间
    （配置项 ``hold_seconds``，由时序确认层消费）。
    """

    behavior = BehaviorType.SMOKING
    requires_face = True
    requires_hands = True

    def _evaluate(self, ctx: DetectionContext) -> RuleOutcome:
        threshold = self._magnitude_threshold(
            self.config.get_float("hand_mouth_distance_ratio", 0.35), 0.35
        )
        ratio = ctx.mean("hand_mouth_ratio")
        if ratio is None:
            return self._skip()
        if ratio >= threshold:
            return self._skip()

        confidence = self._score(ratio, threshold, higher_means_more=False)
        return self._trigger(
            ctx, confidence,
            f"检测到手部持续贴近嘴部（归一化距离 {ratio:.2f} < {threshold:.2f}）",
            hand_mouth_ratio=round(ratio, 4),
            threshold=threshold,
        )
