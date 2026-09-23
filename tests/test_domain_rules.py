"""领域规则、时序确认与中立位标定测试（管道第 3、4 级）。

这三块决定了"误报率"与"漏报率"，是系统可用性的关键，因此测试重点是：

* **边界**：恰好等于阈值、略高于/略低于阈值时的行为必须明确；
* **多通道融合**：单通道失效 / 两通道冲突时的结论与置信度；
* **时序语义**：连续帧门槛、最短时长、迟滞释放、冷却；
* **阈值健壮性**：配置写成负数等错误输入不应让规则变成"恒真"。
"""

from __future__ import annotations

import dataclasses

import pytest

from driver_guard.domain.calibration import NeutralPoseCalibrator
from driver_guard.domain.confirmation import BehaviorConfirmer
from driver_guard.domain.models import (
    BehaviorType,
    DriverMetrics,
    HeadPose,
    NeutralProfile,
    RuleOutcome,
    Severity,
)
from driver_guard.domain.rules import RULE_FACTORIES, build_engine, build_rules
from driver_guard.domain.rules.base import BehaviorRule, RuleConfig, RuleEngine
from driver_guard.domain.rules.distraction import DriverAbsentRule, GazeOffRoadRule
from driver_guard.domain.rules.fatigue import EyesClosedRule, HeadDownRule, YawningRule
from driver_guard.domain.rules.violation import PhoneCallRule, SmokingRule


# ===========================================================================
#  配置读取
# ===========================================================================
class TestRuleConfig:
    def test_defaults(self) -> None:
        config = RuleConfig()
        assert config.enabled is True
        assert config.severity is Severity.MEDIUM

    def test_reads_thresholds_into_options(self) -> None:
        config = RuleConfig.from_mapping(
            {"enabled": True, "severity": "critical", "ear_threshold": 0.25,
             "trigger_frames": 4, "min_duration": 1.5}
        )
        assert config.severity is Severity.CRITICAL
        assert config.get_float("ear_threshold", 0.0) == pytest.approx(0.25)
        assert config.trigger_frames == 4
        assert config.min_duration == pytest.approx(1.5)

    def test_invalid_number_falls_back_to_default(self) -> None:
        config = RuleConfig.from_mapping({"ear_threshold": "not-a-number"})
        assert config.get_float("ear_threshold", 0.21) == pytest.approx(0.21)

    def test_boolean_coercion(self) -> None:
        config = RuleConfig.from_mapping({"flag": "yes"})
        assert config.get_bool("flag", False) is True
        assert config.get_bool("missing", True) is True

    def test_severity_parse_rejects_garbage(self) -> None:
        with pytest.raises(ValueError):
            Severity.parse("super-critical")


# ===========================================================================
#  疲劳类规则
# ===========================================================================
class TestEyesClosedRule:
    def _outcome(self, context_factory, metrics: DriverMetrics, **config):
        rule = EyesClosedRule(RuleConfig.from_mapping(config))
        return rule.evaluate(context_factory(metrics))

    def test_triggers_on_low_ear(self, context_factory, normal_metrics: DriverMetrics) -> None:
        outcome = self._outcome(context_factory, normal_metrics.merged(ear=0.12))
        assert outcome.triggered
        assert outcome.behavior is BehaviorType.EYES_CLOSED
        assert 0.0 < outcome.confidence <= 1.0

    def test_severity_comes_from_delivered_config(
        self, default_config, context_factory, normal_metrics: DriverMetrics
    ) -> None:
        """严重级别必须取自交付配置文件，而不是代码里的默认值。"""
        from driver_guard.domain.rules import build_rules

        rule = next(
            r for r in build_rules(default_config.rules)
            if r.behavior is BehaviorType.EYES_CLOSED
        )
        outcome = rule.evaluate(context_factory(normal_metrics.merged(ear=0.12)))
        assert outcome.triggered
        assert outcome.severity is rule.config.severity == Severity.HIGH

    def test_does_not_trigger_when_ear_normal(
        self, context_factory, normal_metrics: DriverMetrics
    ) -> None:
        assert not self._outcome(context_factory, normal_metrics).triggered

    def test_boundary_is_exclusive(self, context_factory, normal_metrics: DriverMetrics) -> None:
        """恰好等于阈值不算触发（阈值语义为"严格越过"），保证可预测。"""
        at_threshold = normal_metrics.merged(ear=0.21)
        assert not self._outcome(context_factory, at_threshold).triggered
        just_below = normal_metrics.merged(ear=0.2099)
        assert self._outcome(context_factory, just_below).triggered

    def test_blendshape_channel_alone_can_trigger(
        self, context_factory, normal_metrics: DriverMetrics
    ) -> None:
        """EAR 不可用（如降级感知）时，仅凭眨眼系数也应能判定闭眼。"""
        metrics = normal_metrics.merged(ear=None, eye_blink=0.9)
        outcome = self._outcome(context_factory, metrics)
        assert outcome.triggered
        assert outcome.detail.get("hit_blink") == 1.0
        assert "ear" not in outcome.detail

    def test_conflicting_channels_lower_confidence(
        self, context_factory, normal_metrics: DriverMetrics
    ) -> None:
        """EAR 说闭眼、眨眼系数说睁眼：仍触发，但置信度应被惩罚。"""
        agreed = self._outcome(
            context_factory, normal_metrics.merged(ear=0.12, eye_blink=0.9)
        )
        conflicting = self._outcome(
            context_factory, normal_metrics.merged(ear=0.12, eye_blink=0.0)
        )
        assert conflicting.triggered
        assert conflicting.confidence < agreed.confidence

    def test_metrics_are_preserved_not_overwritten(
        self, context_factory, normal_metrics: DriverMetrics
    ) -> None:
        """指标值必须如实记录；标志位用 hit_ 前缀，不得覆盖真实数值。"""
        outcome = self._outcome(context_factory, normal_metrics.merged(ear=0.115))
        assert outcome.detail["ear"] == pytest.approx(0.115)
        assert outcome.detail["hit_ear"] == 1.0

    def test_without_face_skips(self, context_factory, normal_metrics: DriverMetrics) -> None:
        """未在座（感知层明确报告无人）时，行为规则不得凭空触发。"""
        metrics = normal_metrics.merged(face_detected=False, ear=0.05)
        context = context_factory(metrics, with_face=False)
        assert not EyesClosedRule(RuleConfig()).evaluate(context).triggered

    def test_no_evidence_skips(self, context_factory, normal_metrics: DriverMetrics) -> None:
        metrics = normal_metrics.merged(ear=None, eye_blink=None)
        assert not self._outcome(context_factory, metrics).triggered

    def test_negative_threshold_does_not_cause_false_positive(
        self, context_factory, normal_metrics: DriverMetrics
    ) -> None:
        """回归：配置误写成负数时，不能变成"每帧都触发"。"""
        outcome = self._outcome(
            context_factory, normal_metrics, ear_threshold=-0.21
        )
        assert not outcome.triggered


class TestYawningRule:
    def test_triggers_on_high_mar(self, context_factory, normal_metrics: DriverMetrics) -> None:
        outcome = YawningRule(RuleConfig.from_mapping({"mar_threshold": 0.55})).evaluate(
            context_factory(normal_metrics.merged(mar=0.70))
        )
        assert outcome.triggered
        assert outcome.behavior is BehaviorType.YAWNING

    def test_talking_does_not_trigger(self, context_factory, normal_metrics: DriverMetrics) -> None:
        outcome = YawningRule(RuleConfig.from_mapping({"mar_threshold": 0.55})).evaluate(
            context_factory(normal_metrics.merged(mar=0.30, jaw_open=0.20))
        )
        assert not outcome.triggered

    def test_jaw_open_channel(self, context_factory, normal_metrics: DriverMetrics) -> None:
        outcome = YawningRule(RuleConfig.from_mapping({"jaw_open_threshold": 0.45})).evaluate(
            context_factory(normal_metrics.merged(mar=None, jaw_open=0.8))
        )
        assert outcome.triggered
        assert outcome.detail["hit_jaw_open"] == 1.0


class TestHeadDownRule:
    def test_triggers_on_pitch_deviation(
        self, context_factory, normal_metrics: DriverMetrics, neutral_profile: NeutralProfile
    ) -> None:
        """相对中立位偏离超过阈值即触发（不关心俯仰角的正负号方向）。"""
        metrics = normal_metrics.merged(pitch=normal_metrics.pitch - 25.0)  # type: ignore[operator]
        outcome = HeadDownRule(RuleConfig.from_mapping({"pitch_threshold_deg": 18.0})).evaluate(
            context_factory(metrics, neutral=neutral_profile)
        )
        assert outcome.triggered
        assert outcome.detail["pitch_deviation"] == pytest.approx(-25.0, abs=1e-3)

    def test_pitch_ratio_channel_is_axis_free(
        self, context_factory, normal_metrics: DriverMetrics, neutral_profile: NeutralProfile
    ) -> None:
        """仅凭"纵向压缩"通道也能判定低头——这条通道完全不依赖旋转矩阵轴系。"""
        metrics = normal_metrics.merged(pitch=None, yaw=None, roll=None,
                                        face_pitch_ratio=0.45)
        outcome = HeadDownRule(RuleConfig.from_mapping({"pitch_ratio_drop": 0.06})).evaluate(
            context_factory(metrics, neutral=neutral_profile)
        )
        assert outcome.triggered
        assert outcome.detail["hit_pitch_ratio"] == 1.0

    def test_without_calibration_skips_gracefully(
        self, context_factory, normal_metrics: DriverMetrics
    ) -> None:
        """未完成标定时必须跳过——否则固定阈值会随摄像头角度造成误报。"""
        metrics = normal_metrics.merged(pitch=-40.0, face_pitch_ratio=0.30)
        outcome = HeadDownRule(RuleConfig()).evaluate(
            context_factory(metrics, with_neutral=False)
        )
        assert not outcome.triggered

    def test_negative_config_threshold_is_normalised(
        self, context_factory, normal_metrics: DriverMetrics, neutral_profile: NeutralProfile
    ) -> None:
        """回归：历史缺陷——负阈值使 ``0 > -18`` 恒真，导致每帧误报低头。"""
        outcome = HeadDownRule(
            RuleConfig.from_mapping({"pitch_threshold_deg": -18.0})
        ).evaluate(context_factory(normal_metrics, neutral=neutral_profile))
        assert not outcome.triggered


# ===========================================================================
#  分心类规则
# ===========================================================================
class TestGazeOffRoadRule:
    def test_triggers_on_yaw_deviation(
        self, context_factory, normal_metrics: DriverMetrics, neutral_profile: NeutralProfile
    ) -> None:
        metrics = normal_metrics.merged(yaw=normal_metrics.yaw + 35.0)  # type: ignore[operator]
        outcome = GazeOffRoadRule(RuleConfig.from_mapping({"yaw_threshold_deg": 25.0})).evaluate(
            context_factory(metrics, neutral=neutral_profile)
        )
        assert outcome.triggered

    def test_triggers_on_roll_only(
        self, context_factory, normal_metrics: DriverMetrics, neutral_profile: NeutralProfile
    ) -> None:
        metrics = normal_metrics.merged(roll=normal_metrics.roll + 30.0)  # type: ignore[operator]
        outcome = GazeOffRoadRule(RuleConfig.from_mapping({"roll_threshold_deg": 22.0})).evaluate(
            context_factory(metrics, neutral=neutral_profile)
        )
        assert outcome.triggered
        assert outcome.detail["hit_yaw"] == 0.0

    def test_symmetric_for_left_and_right(
        self, context_factory, normal_metrics: DriverMetrics, neutral_profile: NeutralProfile
    ) -> None:
        """判定基于绝对偏离，向左与向右应完全等价。"""
        rule = GazeOffRoadRule(RuleConfig.from_mapping({"yaw_threshold_deg": 25.0}))
        base_yaw = normal_metrics.yaw or 0.0
        left = rule.evaluate(
            context_factory(normal_metrics.merged(yaw=base_yaw + 40.0), neutral=neutral_profile)
        )
        right = rule.evaluate(
            context_factory(normal_metrics.merged(yaw=base_yaw - 40.0), neutral=neutral_profile)
        )
        assert left.triggered and right.triggered
        assert left.confidence == pytest.approx(right.confidence)

    def test_normal_head_position_does_not_trigger(
        self, context_factory, normal_metrics: DriverMetrics, neutral_profile: NeutralProfile
    ) -> None:
        assert not GazeOffRoadRule(RuleConfig()).evaluate(
            context_factory(normal_metrics, neutral=neutral_profile)
        ).triggered


class TestDriverAbsentRule:
    def test_respects_absence_duration(self, context_factory) -> None:
        rule = DriverAbsentRule(RuleConfig.from_mapping({"absent_seconds": 3.0}))
        blank = DriverMetrics(face_detected=False)

        assert not rule.evaluate(context_factory(blank, timestamp=0.0, with_face=False)).triggered
        assert not rule.evaluate(context_factory(blank, timestamp=2.9, with_face=False)).triggered
        assert rule.evaluate(context_factory(blank, timestamp=3.1, with_face=False)).triggered

    def test_resets_when_driver_returns(self, context_factory, normal_metrics: DriverMetrics) -> None:
        rule = DriverAbsentRule(RuleConfig.from_mapping({"absent_seconds": 1.0}))
        blank = DriverMetrics(face_detected=False)

        rule.evaluate(context_factory(blank, timestamp=0.0, with_face=False))
        rule.evaluate(context_factory(normal_metrics, timestamp=2.0))       # 驾驶员回来了
        outcome = rule.evaluate(context_factory(blank, timestamp=2.5, with_face=False))
        assert not outcome.triggered, "计时应在驾驶员出现时重置"

    def test_requires_face_is_false(self) -> None:
        assert DriverAbsentRule.requires_face is False


# ===========================================================================
#  违规类规则
# ===========================================================================
class TestPhoneCallAndSmoking:
    def test_phone_call_triggers_near_face(
        self, context_factory, normal_metrics: DriverMetrics
    ) -> None:
        outcome = PhoneCallRule(
            RuleConfig.from_mapping({"hand_face_distance_ratio": 0.55})
        ).evaluate(context_factory(normal_metrics.merged(hand_face_ratio=0.30)))
        assert outcome.triggered
        assert outcome.behavior is BehaviorType.PHONE_CALL

    def test_phone_call_skips_without_hands(
        self, context_factory, normal_metrics: DriverMetrics
    ) -> None:
        rule = PhoneCallRule(RuleConfig())
        assert rule.requires_hands is True
        assert not rule.evaluate(
            context_factory(normal_metrics.merged(hand_face_ratio=0.10), with_hands=False)
        ).triggered

    def test_smoking_triggers_near_mouth(
        self, context_factory, normal_metrics: DriverMetrics
    ) -> None:
        outcome = SmokingRule(
            RuleConfig.from_mapping({"hand_mouth_distance_ratio": 0.35})
        ).evaluate(context_factory(normal_metrics.merged(hand_mouth_ratio=0.15)))
        assert outcome.triggered
        assert outcome.behavior is BehaviorType.SMOKING

    def test_phone_and_smoking_use_different_signals(
        self, context_factory, normal_metrics: DriverMetrics
    ) -> None:
        """手在脸侧（电话）但不贴嘴时应只触发打电话。"""
        metrics = normal_metrics.merged(hand_face_ratio=0.30, hand_mouth_ratio=1.0)
        assert PhoneCallRule(RuleConfig()).evaluate(context_factory(metrics)).triggered
        assert not SmokingRule(RuleConfig()).evaluate(context_factory(metrics)).triggered


# ===========================================================================
#  规则引擎（微内核）
# ===========================================================================
class TestRuleEngine:
    def test_builds_all_registered_rules(self) -> None:
        rules = build_rules({})
        assert {rule.behavior for rule in rules} == {
            BehaviorType(value) for value in RULE_FACTORIES
        }

    def test_engine_survives_failing_plugin(self, context_factory, normal_metrics: DriverMetrics):
        """单个插件抛异常必须被隔离，其余规则照常工作。"""

        class ExplodingRule(BehaviorRule):
            behavior = BehaviorType.HEAD_DOWN

            def _evaluate(self, ctx):  # noqa: ANN001
                raise RuntimeError("插件内部错误")

        good = EyesClosedRule(RuleConfig())
        engine = RuleEngine([ExplodingRule(RuleConfig()), good])
        outcomes = engine.evaluate(context_factory(normal_metrics.merged(ear=0.1)))

        assert len(outcomes) == 1, "坏插件不应产生结果，也不应吞掉好插件的结果"
        assert outcomes[0].behavior is BehaviorType.EYES_CLOSED
        assert engine.failure_counts[BehaviorType.HEAD_DOWN] == 1

    def test_disabled_rule_is_not_evaluated(self, context_factory, normal_metrics: DriverMetrics):
        rule = EyesClosedRule(RuleConfig.from_mapping({"enabled": False}))
        engine = RuleEngine([rule])
        assert engine.evaluate(context_factory(normal_metrics.merged(ear=0.01))) == []

    def test_engine_from_config_uses_configured_severity(self, default_config) -> None:
        engine = build_engine(default_config.rules)
        by_behavior = {rule.behavior: rule for rule in engine.rules}
        assert by_behavior[BehaviorType.PHONE_CALL].severity is Severity.CRITICAL
        assert by_behavior[BehaviorType.YAWNING].severity is Severity.MEDIUM


# ===========================================================================
#  时序确认（去抖 / 迟滞 / 冷却）
# ===========================================================================
def _outcome(behavior: BehaviorType, triggered: bool = True,
             confidence: float = 0.8) -> RuleOutcome:
    return RuleOutcome(
        behavior=behavior,
        triggered=triggered,
        confidence=confidence if triggered else 0.0,
        severity=Severity.HIGH,
        message="test",
    )


class TestBehaviorConfirmer:
    BEHAVIOR = BehaviorType.EYES_CLOSED

    def _confirmer(self, **config) -> BehaviorConfirmer:
        payload = {"trigger_frames": 3, "release_frames": 2, "min_duration": 0.0,
                   "cooldown_seconds": 0.0}
        payload.update(config)
        return BehaviorConfirmer({self.BEHAVIOR: RuleConfig.from_mapping(payload)})

    def test_requires_consecutive_frames(self) -> None:
        confirmer = self._confirmer()
        for timestamp in (0.1, 0.2):
            assert confirmer.update(_outcome(self.BEHAVIOR), timestamp) is None
        event = confirmer.update(_outcome(self.BEHAVIOR), 0.3)
        assert event is not None and not event.ongoing

    def test_single_dropout_is_tolerated(self) -> None:
        """单帧丢失不应让一个真实行为漏报：未达 release_frames 前计数不清零。"""
        confirmer = self._confirmer(trigger_frames=3, release_frames=2)
        confirmer.update(_outcome(self.BEHAVIOR), 0.1)
        confirmer.update(_outcome(self.BEHAVIOR), 0.2)
        confirmer.update(_outcome(self.BEHAVIOR, triggered=False), 0.3)
        assert confirmer.update(_outcome(self.BEHAVIOR), 0.4) is not None

    def test_strict_consecutive_mode(self) -> None:
        """把 release_frames 设为 1 即可要求严格连续。"""
        confirmer = self._confirmer(trigger_frames=3, release_frames=1)
        confirmer.update(_outcome(self.BEHAVIOR), 0.1)
        confirmer.update(_outcome(self.BEHAVIOR), 0.2)
        confirmer.update(_outcome(self.BEHAVIOR, triggered=False), 0.3)
        assert confirmer.update(_outcome(self.BEHAVIOR), 0.4) is None

    def test_sustained_absence_resets_streak(self) -> None:
        confirmer = self._confirmer(trigger_frames=3, release_frames=2)
        confirmer.update(_outcome(self.BEHAVIOR), 0.1)
        confirmer.update(_outcome(self.BEHAVIOR), 0.2)
        for timestamp in (0.3, 0.4, 0.5):
            confirmer.update(_outcome(self.BEHAVIOR, triggered=False), timestamp)
        assert confirmer.update(_outcome(self.BEHAVIOR), 0.6) is None, "计数应已清零"

    def test_min_duration_gate(self) -> None:
        """帧数够但持续时间不足时仍不上报（用于区分打哈欠与说话）。"""
        confirmer = self._confirmer(trigger_frames=1, min_duration=1.5)
        assert confirmer.update(_outcome(self.BEHAVIOR), 0.0) is None
        assert confirmer.update(_outcome(self.BEHAVIOR), 1.0) is None
        event = confirmer.update(_outcome(self.BEHAVIOR), 1.6)
        assert event is not None
        assert event.duration == pytest.approx(1.6)

    def test_hysteresis_release(self) -> None:
        confirmer = self._confirmer(trigger_frames=1, release_frames=3)
        assert confirmer.update(_outcome(self.BEHAVIOR), 0.0) is not None
        assert self.BEHAVIOR in confirmer.active_behaviors

        confirmer.update(_outcome(self.BEHAVIOR, triggered=False), 0.1)
        confirmer.update(_outcome(self.BEHAVIOR, triggered=False), 0.2)
        assert self.BEHAVIOR in confirmer.active_behaviors, "未达释放帧数时不应解除"

        confirmer.update(_outcome(self.BEHAVIOR, triggered=False), 0.3)
        assert self.BEHAVIOR not in confirmer.active_behaviors

    def test_cooldown_controls_repeats(self) -> None:
        confirmer = self._confirmer(trigger_frames=1, cooldown_seconds=5.0)
        first = confirmer.update(_outcome(self.BEHAVIOR), 0.0)
        assert first is not None and not first.ongoing

        assert confirmer.update(_outcome(self.BEHAVIOR), 2.0) is None, "冷却期内不重复提醒"
        repeat = confirmer.update(_outcome(self.BEHAVIOR), 5.1)
        assert repeat is not None and repeat.ongoing

    def test_zero_cooldown_reports_once(self) -> None:
        confirmer = self._confirmer(trigger_frames=1, cooldown_seconds=0.0)
        assert confirmer.update(_outcome(self.BEHAVIOR), 0.0) is not None
        assert confirmer.update(_outcome(self.BEHAVIOR), 10.0) is None

    def test_reactivates_after_release(self) -> None:
        confirmer = self._confirmer(trigger_frames=1, release_frames=1, cooldown_seconds=0.0)
        assert confirmer.update(_outcome(self.BEHAVIOR), 0.0) is not None
        confirmer.update(_outcome(self.BEHAVIOR, triggered=False), 1.0)
        assert confirmer.update(_outcome(self.BEHAVIOR), 2.0) is not None, "应可再次触发"

    def test_reset_clears_state(self) -> None:
        confirmer = self._confirmer(trigger_frames=1)
        confirmer.update(_outcome(self.BEHAVIOR), 0.0)
        confirmer.reset()
        assert confirmer.active_behaviors == ()
        assert confirmer.update(_outcome(self.BEHAVIOR), 1.0) is not None

    def test_independent_state_per_behavior(self) -> None:
        confirmer = BehaviorConfirmer({
            BehaviorType.EYES_CLOSED: RuleConfig.from_mapping({"trigger_frames": 1}),
            BehaviorType.YAWNING: RuleConfig.from_mapping({"trigger_frames": 5}),
        })
        assert confirmer.update(_outcome(BehaviorType.EYES_CLOSED), 0.0) is not None
        assert confirmer.update(_outcome(BehaviorType.YAWNING), 0.0) is None

    def test_aggressive_defaults_do_not_lose_events(self) -> None:
        """未配置时序参数时退化为"瞬时直通"，便于调试与测试。"""
        confirmer = BehaviorConfirmer()
        assert confirmer.update(_outcome(self.BEHAVIOR), 0.0) is not None


# ===========================================================================
#  中立位标定
# ===========================================================================
def _pose_metrics(pitch: float, yaw: float, roll: float,
                  ratio: float | None = 0.58) -> DriverMetrics:
    return DriverMetrics(
        face_detected=True, pitch=pitch, yaw=yaw, roll=roll, face_pitch_ratio=ratio
    )


class TestNeutralPoseCalibrator:
    def test_not_ready_until_enough_samples(self) -> None:
        calibrator = NeutralPoseCalibrator(required_samples=10, warmup_frames=0)
        for _ in range(9):
            assert calibrator.feed(_pose_metrics(5.0, -2.0, 1.0)) is None
        assert not calibrator.ready
        assert calibrator.feed(_pose_metrics(5.0, -2.0, 1.0)) is not None
        assert calibrator.ready

    def test_uses_median_to_resist_outliers(self) -> None:
        """中位数估计：混入少量异常帧不应污染基准。

        构造的离群帧必须仍在"姿态合理"范围内（否则会被极值门限过滤掉，
        那就测不到中位数的稳健性了）：均值会被拉偏，中位数不会。
        """
        calibrator = NeutralPoseCalibrator(required_samples=10, warmup_frames=0)
        for _ in range(9):
            calibrator.feed(_pose_metrics(5.0, 0.0, 0.0))
        calibrator.feed(_pose_metrics(-20.0, 15.0, 10.0))    # 离群帧（仍在合理范围内）
        profile = calibrator.profile
        assert profile is not None and profile.head_pose is not None
        # 均值会是 (9*5 - 20)/10 = 2.5，中位数仍是 5.0
        assert profile.head_pose.pitch == pytest.approx(5.0)

    def test_warmup_frames_are_discarded(self) -> None:
        """预热帧不参与标定，避免模型尚未稳定时拉偏基准。"""
        calibrator = NeutralPoseCalibrator(
            required_samples=3, warmup_frames=2
        )
        assert calibrator.feed(_pose_metrics(99.0, 99.0, 99.0)) is None
        assert calibrator.feed(_pose_metrics(99.0, 99.0, 99.0)) is None
        for _ in range(3):
            calibrator.feed(_pose_metrics(5.0, 0.0, 0.0))
        assert calibrator.profile is not None
        assert calibrator.profile.head_pose is not None
        assert calibrator.profile.head_pose.pitch == pytest.approx(5.0)

    def test_ignores_frames_without_face_or_pose(self) -> None:
        calibrator = NeutralPoseCalibrator(required_samples=2, warmup_frames=0)
        assert calibrator.feed(DriverMetrics(face_detected=False)) is None
        assert calibrator.feed(DriverMetrics(face_detected=True, pitch=None)) is None
        assert not calibrator.ready

    def test_ignores_frames_with_extreme_pose(self) -> None:
        """驾驶员正在大幅偏头时不应把该姿态当作中立位。"""
        calibrator = NeutralPoseCalibrator(required_samples=2, warmup_frames=0)
        assert calibrator.feed(_pose_metrics(5.0, 80.0, 0.0)) is None
        assert not calibrator.ready

    def test_progress_reports_fraction(self) -> None:
        calibrator = NeutralPoseCalibrator(required_samples=10, warmup_frames=0)
        for _ in range(5):
            calibrator.feed(_pose_metrics(5.0, 0.0, 0.0))
        assert calibrator.progress == pytest.approx(0.5)
        assert calibrator.ready is False

    def test_reset_allows_recalibration(self) -> None:
        calibrator = NeutralPoseCalibrator(required_samples=2, warmup_frames=0)
        calibrator.feed(_pose_metrics(5.0, 0.0, 0.0))
        calibrator.feed(_pose_metrics(5.0, 0.0, 0.0))
        assert calibrator.ready

        calibrator.reset()
        assert not calibrator.ready
        assert calibrator.progress == 0.0

    def test_rejects_non_positive_required_samples(self) -> None:
        with pytest.raises(ValueError):
            NeutralPoseCalibrator(required_samples=0)

    def test_captures_pitch_ratio_baseline(self) -> None:
        calibrator = NeutralPoseCalibrator(required_samples=2, warmup_frames=0)
        calibrator.feed(_pose_metrics(5.0, 0.0, 0.0, ratio=0.60))
        calibrator.feed(_pose_metrics(5.0, 0.0, 0.0, ratio=0.58))
        assert calibrator.profile is not None
        assert calibrator.profile.pitch_ratio == pytest.approx(0.59)

    def test_profile_deviations_helper(self, normal_metrics: DriverMetrics) -> None:
        """NeutralProfile + 当前姿态 → 偏离量（供规则直接消费）。"""
        profile = NeutralProfile(head_pose=HeadPose(5.0, -2.0, 1.0), pitch_ratio=0.58)
        current = normal_metrics.merged(pitch=20.0, yaw=-2.0, roll=1.0)
        deviation = HeadPose(current.pitch or 0, current.yaw or 0, current.roll or 0) \
            .deviation_from(profile.head_pose)
        assert deviation.pitch == pytest.approx(15.0)
        assert deviation.yaw == pytest.approx(0.0)

    def test_metrics_are_immutable(self, normal_metrics: DriverMetrics) -> None:
        """指标对象是不可变值对象：任何"修改"都应返回新实例。"""
        with pytest.raises(dataclasses.FrozenInstanceError):
            normal_metrics.ear = 0.1  # type: ignore[misc]
