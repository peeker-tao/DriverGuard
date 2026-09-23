"""疲劳驾驶类规则：闭眼、打哈欠、低头。

每条规则都采用 **多证据融合（multi-cue fusion）**：把几何指标（EAR/MAR）与
MediaPipe 表情系数（blendshape）两条通道的结论合并。这样做的好处：

* 任一通道因遮挡、侧脸、降级模式失效时，另一通道仍可工作（可靠性）；
* 两通道都可用时，用"是否互相印证"调节置信度，降低误报。

融合规则：``triggered = 任一通道成立``；
置信度取各通道最大值，但若通道间结论**冲突**则乘以 0.6 惩罚。
"""

from __future__ import annotations

from ..models import BehaviorType, DetectionContext, RuleOutcome
from .base import BehaviorRule

__all__ = ["EyesClosedRule", "YawningRule", "HeadDownRule"]


class _FusableRule(BehaviorRule):
    """提供通用融合工具的基类（内部使用）。"""

    @staticmethod
    def _fuse(evidence: list[tuple[str, bool, float]]) -> tuple[bool, float, dict[str, float]]:
        """融合多条证据。

        Args:
            evidence: ``[(通道名, 是否成立, 置信度), ...]``

        Returns:
            ``(是否触发, 融合置信度, 通道明细)``

        .. note::
           通道明细的键统一加 ``hit_`` 前缀（如 ``hit_ear``），以免与
           指标键（``ear``、``mar``）同名而覆盖真实数值——这类命名冲突
           会让告警记录里的指标静默失真，属于必须避免的隐患。
        """
        if not evidence:
            return False, 0.0, {}
        triggered = any(item[1] for item in evidence)
        positives = [item[2] for item in evidence if item[1]]
        detail = {f"hit_{name}": 1.0 if flag else 0.0 for name, flag, _ in evidence}

        if not triggered:
            return False, 0.0, detail

        confidence = max(positives)
        flags = {flag for _, flag, _ in evidence}
        if len(flags) > 1:
            # 通道结论冲突：降低置信度，交由时序确认层用"持续时间"再筛一次
            confidence *= 0.6
        return True, round(confidence, 4), detail


class EyesClosedRule(_FusableRule):
    """闭眼疲劳。

    .. math::

        \\text{EAR} = \\frac{\\|p_2-p_6\\| + \\|p_3-p_5\\|}{2\\,\\|p_1-p_4\\|}

    EAR 对距离不敏感（是比值），睁眼约 0.25~0.35，闭眼降到 0.10~0.15。
    对滑窗均值做判定以抑制单帧关键点抖动。

    次要通道：MediaPipe ``eyeBlinkLeft``/``eyeBlinkRight`` 表情系数
    （0~1，越大越闭合），在侧脸导致 EAR 失真时更稳。
    """

    behavior = BehaviorType.EYES_CLOSED
    requires_face = True

    def _evaluate(self, ctx: DetectionContext) -> RuleOutcome:
        ear_threshold = self._magnitude_threshold(
            self.config.get_float("ear_threshold", 0.21), 0.21
        )
        blink_threshold = self._magnitude_threshold(
            self.config.get_float("blendshape_threshold", 0.55), 0.55
        )

        evidence: list[tuple[str, bool, float]] = []
        detail: dict[str, float] = {}

        ear = ctx.mean("ear")
        if ear is not None:
            hit = ear < ear_threshold
            evidence.append(("ear", hit, self._score(ear, ear_threshold, higher_means_more=False)))
            detail["ear"] = round(ear, 4)

        blink = ctx.mean("eye_blink")
        if blink is not None:
            hit = blink > blink_threshold
            evidence.append(("blink", hit, self._score(blink, blink_threshold)))
            detail["eye_blink"] = round(blink, 4)

        if not evidence:
            return self._skip()

        triggered, confidence, flags = self._fuse(evidence)
        detail.update(flags)
        if not triggered:
            return self._skip()
        return self._trigger(
            ctx,
            confidence,
            f"检测到闭眼（EAR={detail.get('ear')}，眨眼系数={detail.get('eye_blink')}）",
            **detail,
        )


class YawningRule(_FusableRule):
    """打哈欠。

    .. math::

        \\text{MAR} = \\frac{\\|p_{13}-p_{14}\\|}{\\|p_{78}-p_{308}\\|}

    即"上下唇内缘距离 / 嘴角宽度"。说话时 MAR < 0.3，打哈欠可超过 0.5。
    次要通道为 ``jawOpen`` 表情系数。
    """

    behavior = BehaviorType.YAWNING
    requires_face = True

    def _evaluate(self, ctx: DetectionContext) -> RuleOutcome:
        mar_threshold = self._magnitude_threshold(
            self.config.get_float("mar_threshold", 0.55), 0.55
        )
        jaw_threshold = self._magnitude_threshold(
            self.config.get_float("jaw_open_threshold", 0.45), 0.45
        )

        evidence: list[tuple[str, bool, float]] = []
        detail: dict[str, float] = {}

        mar = ctx.mean("mar")
        if mar is not None:
            evidence.append(("mar", mar > mar_threshold, self._score(mar, mar_threshold)))
            detail["mar"] = round(mar, 4)

        jaw = ctx.mean("jaw_open")
        if jaw is not None:
            evidence.append(("jaw_open", jaw > jaw_threshold, self._score(jaw, jaw_threshold)))
            detail["jaw_open"] = round(jaw, 4)

        if not evidence:
            return self._skip()

        triggered, confidence, flags = self._fuse(evidence)
        detail.update(flags)
        if not triggered:
            return self._skip()
        return self._trigger(
            ctx, confidence,
            f"检测到打哈欠（MAR={detail.get('mar')}，张口系数={detail.get('jaw_open')}）",
            **detail,
        )


class HeadDownRule(_FusableRule):
    """低头（看手机/看仪表盘导致的长时低头）。

    两条证据通道，均以 **相对个人中立位的偏离** 判定：

    1. ``pitch`` 偏离中立位超过 ``pitch_threshold_deg``。
    2. ``face_pitch_ratio``（眼-嘴纵向距离 ÷ 脸宽）相对中立位**下降**
       超过 ``pitch_ratio_drop``。低头时人脸纵向被透视压缩，该比值下降。
       该通道完全不依赖旋转矩阵的轴系约定，是抵抗轴系不确定性的关键，
       且不需要判断俯仰角的正负号方向。

    ``pitch_threshold_deg`` 取正数，表示"偏离幅度"；由于轴系语义不可靠，
    这里用绝对值判定，方向由 ``pitch_ratio`` 通道负责区分低头与抬头。
    """

    behavior = BehaviorType.HEAD_DOWN
    requires_face = True

    def _evaluate(self, ctx: DetectionContext) -> RuleOutcome:
        angle_threshold = self._magnitude_threshold(
            self.config.get_float("pitch_threshold_deg", 18.0), 18.0
        )
        ratio_drop = self._magnitude_threshold(
            self.config.get_float("pitch_ratio_drop", 0.06), 0.06
        )

        evidence: list[tuple[str, bool, float]] = []
        detail: dict[str, float] = {}

        deviation = ctx.pose_deviation
        if deviation is not None:
            magnitude = abs(deviation.pitch)
            evidence.append(("pitch", magnitude > angle_threshold,
                             self._score(magnitude, angle_threshold)))
            detail["pitch_deviation"] = round(deviation.pitch, 3)

        ratio_dev = ctx.pitch_ratio_deviation
        if ratio_dev is not None:
            evidence.append(("pitch_ratio", ratio_dev > ratio_drop,
                             self._score(ratio_dev, ratio_drop)))
            detail["pitch_ratio_drop"] = round(ratio_dev, 4)

        if not evidence:
            return self._skip()

        triggered, confidence, flags = self._fuse(evidence)
        detail.update(flags)
        if not triggered:
            return self._skip()
        return self._trigger(
            ctx, confidence,
            f"检测到持续低头（俯仰偏离={detail.get('pitch_deviation')}°，"
            f"纵向压缩={detail.get('pitch_ratio_drop')}）",
            **detail,
        )
