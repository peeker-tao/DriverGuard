"""集成测试：装配、数据源、叠加层与"演示脚本 → 期望告警"的端到端门禁。

与其它测试文件的区别
--------------------
这里 **跑的是真实装配过程**（:func:`build_application`），使用真实的
配置阈值、真实的规则集合、真实的事件总线与存储，只是把
"摄像头 + MediaPipe"换成了"合成画面 + 回放脚本"。

它回答的问题是：
    **"按交付配置跑起来，这套系统在无摄像头环境下能否复现预期行为？"**

因此它同时充当 **演示前自检** 与 **参数改动后的回归门禁**——
阈值一旦被误改，这里会立刻变红。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from driver_guard.bootstrap import build_application, create_alert_sinks, create_frame_source
from driver_guard.config import AppConfig
from driver_guard.domain.models import BehaviorType, Observation, Severity
from driver_guard.infrastructure.landmarks import (
    SCENARIOS,
    HaarLandmarkProvider,
    MediaPipeLandmarkProvider,
    MetricSegment,
    ReplayLandmarkProvider,
    available_providers,
    create_landmark_provider,
)
from driver_guard.infrastructure.sources import (
    ImageSequenceFrameSource,
    SyntheticFrameSource,
    VideoFileFrameSource,
    WebcamFrameSource,
)
from driver_guard.presentation.overlay import OverlayOptions, draw_overlay

import cv2


# ===========================================================================
#  演示脚本自检
# ===========================================================================
def _replay_config(scenario: str) -> AppConfig:
    config = AppConfig()
    return config.with_overrides(
        source=replace(config.source, type="replay", width=320, height=180, fps=30.0),
        perception=replace(config.perception, provider="replay", scenario=scenario),
        alert=replace(config.alert, sound_enabled=False, snapshot_enabled=False,
                      csv_enabled=False, log_console=False),
    )


def _run_scenario(scenario: str) -> dict[str, int]:
    """跑完一条脚本，返回 ``{行为: 确认次数}``。

    关闭软件节流（``set_target_fps(0)``）让脚本尽快跑完，但**帧时间戳仍由
    帧号 ÷ 帧率推导**，因此时序判定（时长门槛、冷却）的语义与真实播放完全一致。
    这样测试既快又不失真——若去掉节流的同时又依赖墙钟时间，测试就会变成
    "跑得越快、告警越难触发"，那是错的。
    """
    app = build_application(_replay_config(scenario), persist_events=False)
    try:
        app.detection_service.set_target_fps(0.0)
        summary = app.detection_service.run()
    finally:
        app.shutdown()
    return {behavior.value: count for behavior, count in summary.event_counts.items()}


class TestScenarioGate:
    """每条内置演示脚本必须产生预期告警——不多不少。"""

    EXPECTED: dict[str, set[str]] = {
        "normal": set(),
        "fatigue": {"eyes_closed", "yawning", "head_down"},
        "distraction": {"gaze_off_road", "head_down"},
        "violation": {"phone_call", "smoking"},
        "absent": {"driver_absent"},
        "mixed": {"eyes_closed", "gaze_off_road", "phone_call", "yawning"},
    }

    @pytest.mark.parametrize("scenario", sorted(SCENARIOS))
    def test_scenario_triggers_expected_behaviors(self, scenario: str) -> None:
        triggered = set(_run_scenario(scenario))
        expected = self.EXPECTED[scenario]
        assert triggered == expected, (
            f"脚本 {scenario} 的告警集合不符："
            f"缺少 {sorted(expected - triggered)}，多出 {sorted(triggered - expected)}"
        )

    def test_normal_scenario_has_no_false_positive(self) -> None:
        """零误报是交付底线：正常驾驶脚本不得产生任何告警。"""
        assert _run_scenario("normal") == {}

    def test_scenario_definitions_are_valid(self) -> None:
        """脚本本身必须合法：区间有序、不重叠、从 0 开始。"""
        for name, segments in SCENARIOS.items():
            assert segments, f"{name} 为空"
            assert segments[0].start == 0.0, f"{name} 未从 0 秒开始"
            for previous, current in zip(segments, segments[1:], strict=False):
                assert current.start >= previous.end, f"{name} 存在重叠片段"
            assert all(segment.end > segment.start for segment in segments), \
                f"{name} 存在非法区间"


# ===========================================================================
#  装配（组合根）
# ===========================================================================
class TestBootstrap:
    def test_builds_complete_context(self) -> None:
        # 回放配置默认关闭了全部告警通道；这里开启控制台通道，
        # 以便同时验证"告警通道确实被装配进去"。
        config = _replay_config("mixed")
        config = config.with_overrides(
            alert=replace(config.alert, log_console=True)
        )
        app = build_application(config, persist_events=False)
        try:
            assert app.source is not None
            assert app.provider.ready
            assert app.pipeline.engine.rules
            assert app.alert_service.sinks
            assert app.statistics is not None
            assert app.detection_service is not None
        finally:
            app.shutdown()

    def test_all_alert_channels_disabled_is_allowed(self) -> None:
        """全部告警通道关闭时装配仍应成功（无输出通道是合法配置）。"""
        app = build_application(_replay_config("mixed"), persist_events=False)
        try:
            assert app.alert_service.sinks == ()
            assert app.snapshot_sink is None
        finally:
            app.shutdown()

    def test_warns_when_mediapipe_paired_with_synthetic_source(self) -> None:
        """把 MediaPipe 和合成画面配在一起是常见误用，必须给出提示。"""
        config = _replay_config("mixed")
        config = config.with_overrides(
            source=replace(config.source, type="synthetic"),
            perception=replace(config.perception, provider="mediapipe",
                               model_dir="assets/models"),
        )
        app = build_application(config, persist_events=False)
        try:
            assert any("合成画面" in warning for warning in app.warnings)
        finally:
            app.shutdown()

    def test_shutdown_is_idempotent(self) -> None:
        app = build_application(_replay_config("mixed"), persist_events=False)
        app.shutdown()
        app.shutdown()          # 第二次调用不应抛异常

    def test_session_summary_is_written(self, tmp_path) -> None:
        config = _replay_config("mixed")
        config = config.with_overrides(
            output=replace(config.output, dir=str(tmp_path))
        )
        app = build_application(config, persist_events=True)
        try:
            summary = app.detection_service.run()
            path = app.write_session_summary(summary)
        finally:
            app.shutdown()
        assert path is not None and path.exists()

    def test_disabled_rules_are_not_instantiated_as_enabled(self) -> None:
        config = _replay_config("mixed")
        rules = dict(config.rules)
        rules["yawning"] = {**rules.get("yawning", {}), "enabled": False}
        config = config.with_overrides(rules=rules)

        app = build_application(config, persist_events=False)
        try:
            enabled = {rule.behavior for rule in app.pipeline.engine.rules if rule.enabled}
            assert BehaviorType.YAWNING not in enabled
        finally:
            app.shutdown()


# ===========================================================================
#  数据源与感知工厂
# ===========================================================================
class TestFactories:
    def test_available_providers(self) -> None:
        assert set(available_providers()) == {"mediapipe", "haar", "replay"}

    def test_create_known_provider(self) -> None:
        provider = create_landmark_provider("replay", {"scenario": "fatigue"})
        assert isinstance(provider, ReplayLandmarkProvider)
        assert provider.scenario_name == "fatigue"

    def test_unknown_provider_raises_without_fallback(self) -> None:
        with pytest.raises(KeyError, match="未知感知实现"):
            create_landmark_provider("does-not-exist", {})

    def test_unknown_provider_falls_back(self) -> None:
        provider = create_landmark_provider("does-not-exist", {}, fallback="replay")
        assert isinstance(provider, ReplayLandmarkProvider)

    def test_create_camera_source(self) -> None:
        config = AppConfig()
        source = create_frame_source(config)
        assert isinstance(source, WebcamFrameSource)
        assert source.fps > 0

    def test_create_synthetic_source(self) -> None:
        config = AppConfig().with_overrides(
            source=replace(AppConfig().source, type="synthetic")
        )
        assert isinstance(create_frame_source(config), SyntheticFrameSource)

    def test_create_video_source(self, tmp_path) -> None:
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"not-a-real-video")   # 仅验证工厂与存在性检查
        config = AppConfig().with_overrides(
            source=replace(AppConfig().source, type="video", video_path=str(video))
        )
        assert isinstance(create_frame_source(config), VideoFileFrameSource)

    def test_missing_video_raises(self, tmp_path) -> None:
        config = AppConfig().with_overrides(
            source=replace(AppConfig().source, type="video",
                           video_path=str(tmp_path / "nope.mp4"))
        )
        with pytest.raises(Exception):
            create_frame_source(config)

    def test_unknown_source_type_raises(self) -> None:
        config = AppConfig().with_overrides(
            source=replace(AppConfig().source, type="telepathy")
        )
        with pytest.raises(ValueError, match="未知数据源类型"):
            create_frame_source(config)

    def test_alert_sinks_follow_config(self) -> None:
        config = AppConfig().with_overrides(
            alert=replace(AppConfig().alert, log_console=True, csv_enabled=False,
                          snapshot_enabled=False, sound_enabled=False)
        )
        sinks, snapshot = create_alert_sinks(config, repository=None)
        assert [sink.name for sink in sinks] == ["控制台"]
        assert snapshot is None

    def test_missing_models_are_reported_not_raised(self) -> None:
        provider = MediaPipeLandmarkProvider(model_dir="definitely/not/here")
        assert provider.ready is False
        assert provider.missing_models(), "应能列出缺失的模型文件供界面提示"

    def test_haar_reports_missing_cascades(self) -> None:
        provider = HaarLandmarkProvider(cascade_dir="definitely/not/here")
        assert provider.ready is False
        assert provider.missing_files()


# ===========================================================================
#  数据源行为
# ===========================================================================
class TestSyntheticSource:
    def test_frames_have_expected_shape_and_advance(self) -> None:
        source = SyntheticFrameSource(width=200, height=120, fps=25.0)
        with source:
            first = source.read()
            second = source.read()
        assert first is not None and second is not None
        assert (first.height, first.width) == (120, 200)
        assert second.index == first.index + 1
        assert second.timestamp > first.timestamp

    def test_respects_duration(self) -> None:
        source = SyntheticFrameSource(width=64, height=64, fps=10.0, duration=0.5)
        frames = list(source)
        assert len(frames) == 5

    def test_read_before_open_returns_none(self) -> None:
        assert SyntheticFrameSource().read() is None

    def test_is_iterable_and_closes(self) -> None:
        source = SyntheticFrameSource(width=64, height=64, fps=10.0, duration=0.3)
        assert len(list(source)) == 3
        assert source.read() is None, "迭代结束后应处于关闭状态"


class TestImageSequenceSource:
    def test_iterates_existing_images(self, tmp_path) -> None:
        paths = []
        for index in range(3):
            path = tmp_path / f"frame_{index}.png"
            cv2.imwrite(str(path), np.full((20, 30, 3), index * 60, dtype=np.uint8))
            paths.append(path)

        source = ImageSequenceFrameSource(paths, fps=10.0)
        frames = list(source)
        assert len(frames) == 3
        assert frames[1].timestamp == pytest.approx(0.1)

    def test_skips_missing_files(self, tmp_path) -> None:
        good = tmp_path / "ok.png"
        cv2.imwrite(str(good), np.zeros((10, 10, 3), dtype=np.uint8))
        source = ImageSequenceFrameSource([good, tmp_path / "missing.png"], fps=10.0)
        assert len(list(source)) == 1

    def test_all_missing_raises(self, tmp_path) -> None:
        source = ImageSequenceFrameSource([tmp_path / "a.png"], fps=10.0)
        with pytest.raises(Exception):
            source.open()

    def test_empty_list_raises(self) -> None:
        with pytest.raises(Exception):
            ImageSequenceFrameSource([])


# ===========================================================================
#  回放感知适配器
# ===========================================================================
class TestReplayProvider:
    def test_unknown_scenario_raises(self) -> None:
        with pytest.raises(KeyError, match="未知脚本"):
            ReplayLandmarkProvider("no-such-scenario")

    def test_loops_after_span(self) -> None:
        provider = ReplayLandmarkProvider("absent", loop=True)
        span = provider.duration
        frame = _dummy_frame(timestamp=span + 1.0)
        observation = provider.infer(frame)
        assert observation.precomputed is not None
        # 循环回 t≈1.0，此时驾驶员应在座
        assert observation.precomputed.face_detected is True

    def test_without_loop_holds_last_segment(self) -> None:
        provider = ReplayLandmarkProvider("absent", loop=False)
        frame = _dummy_frame(timestamp=provider.duration + 5.0)
        observation = provider.infer(frame)
        assert observation.precomputed is not None
        assert observation.precomputed.face_detected is True

    def test_absent_segment_reports_no_face(self) -> None:
        provider = ReplayLandmarkProvider("absent", loop=False)
        observation = provider.infer(_dummy_frame(timestamp=6.0))
        assert observation.precomputed is not None
        assert observation.precomputed.face_detected is False
        assert not observation.has_face

    def test_hands_can_be_disabled(self) -> None:
        provider = ReplayLandmarkProvider("violation", with_hands=False)
        assert not provider.infer(_dummy_frame(timestamp=6.0)).has_hands

    def test_describe_is_human_readable(self) -> None:
        text = ReplayLandmarkProvider("fatigue").describe()
        assert "闭眼疲劳" in text and "低头" in text

    def test_custom_segments(self) -> None:
        from driver_guard.domain.models import DriverMetrics

        segments = [
            MetricSegment(0.0, 1.0, DriverMetrics(face_detected=True, ear=0.30)),
            MetricSegment(1.0, 2.0, DriverMetrics(face_detected=True, ear=0.10,
                                                  eye_blink=0.9)),
        ]
        provider = ReplayLandmarkProvider(segments, loop=False)
        assert provider.duration == pytest.approx(2.0)
        assert provider.infer(_dummy_frame(timestamp=1.5)).precomputed.ear == pytest.approx(0.10)


def _dummy_frame(timestamp: float):
    from driver_guard.domain.models import Frame

    return Frame(index=int(timestamp * 30), timestamp=timestamp,
                 image=np.zeros((32, 32, 3), dtype=np.uint8))


# ===========================================================================
#  叠加层渲染
# ===========================================================================
class TestOverlay:
    def _face(self):
        from driver_guard.domain.models import Landmark

        return tuple(Landmark(x=0.5 + 0.1 * (i % 5) / 5, y=0.5 + 0.1 * (i % 7) / 7)
                     for i in range(478))

    def test_returns_new_array_and_does_not_mutate_input(self) -> None:
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        original = image.copy()
        observation = Observation(face=self._face(), provider="test")

        result = draw_overlay(image, observation, options=OverlayOptions())

        assert result is not image
        assert np.array_equal(image, original), "原始帧不应被就地修改"

    def test_runs_with_empty_observation(self) -> None:
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        result = draw_overlay(image, Observation(), options=OverlayOptions())
        assert result.shape == image.shape

    def test_runs_with_hand_landmarks(self) -> None:
        from driver_guard.domain.models import Landmark

        hand = tuple(Landmark(x=0.5, y=0.5) for _ in range(21))
        observation = Observation(face=self._face(), hands=(hand,))
        result = draw_overlay(np.zeros((120, 160, 3), dtype=np.uint8), observation,
                              options=OverlayOptions())
        assert result.shape == (120, 160, 3)

    def test_hud_with_metrics_dict(self) -> None:
        metrics = {"ear": 0.22, "mar": 0.65, "pitch": -12.0, "yaw": 4.0,
                   "roll": 2.0, "eye_blink": 0.7, "jaw_open": 0.5,
                   "hand_face_ratio": 0.9}
        result = draw_overlay(np.zeros((300, 400, 3), dtype=np.uint8),
                              Observation(face=self._face()),
                              options=OverlayOptions(),
                              metrics=metrics, latency_ms=21.5)
        assert result.shape == (300, 400, 3)
        assert result.sum() > 0, "应至少绘制出 HUD 面板"

    def test_draws_active_event_banner(self, event_factory) -> None:
        from driver_guard.domain.models import BehaviorType

        events = (event_factory(BehaviorType.PHONE_CALL, severity=Severity.CRITICAL),)
        result = draw_overlay(np.zeros((200, 300, 3), dtype=np.uint8),
                              Observation(face=self._face()),
                              options=OverlayOptions(), active_events=events)
        assert result.sum() > 0

    def test_calibration_progress_is_rendered(self) -> None:
        result = draw_overlay(np.zeros((200, 300, 3), dtype=np.uint8),
                              Observation(face=self._face()),
                              options=OverlayOptions(), calibration_progress=0.4)
        assert result.sum() > 0

    def test_options_can_disable_all_optional_layers(self) -> None:
        """除底部状态条外，其余叠加图层都可以被完全关闭。

        状态条是刻意不提供开关的：它给出的是"当前是否正常"的最终结论，
        属于安全性/易用性需求（S-29 结论直给），不应被用户关掉。
        """
        options = OverlayOptions(show_landmarks=False, show_face_box=False,
                                 show_hud=False, show_hand_points=False)
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        result = draw_overlay(image, Observation(face=self._face()), options=options)

        changed = np.any(result != image, axis=2)
        rows = np.where(changed.any(axis=1))[0]
        assert rows.size > 0, "状态条应当始终绘制"
        assert rows.min() > image.shape[0] * 0.4, "上半部分不应有任何绘制"

    def test_status_banner_is_not_switchable_off(self) -> None:
        """即使所有开关都关掉，状态条仍会给出"正常驾驶"结论。"""
        options = OverlayOptions(show_landmarks=False, show_face_box=False,
                                 show_hud=False, show_hand_points=False)
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        result = draw_overlay(image, Observation(face=self._face()), options=options)
        assert not np.array_equal(result, image)

    def test_chinese_text_does_not_crash_without_font(self, monkeypatch) -> None:
        """没有中文字体时应退化为英文，而不是抛异常或画出方块。"""
        from driver_guard.presentation import overlay as overlay_module

        monkeypatch.setattr(overlay_module._RENDERER, "_font_path", None)
        monkeypatch.setattr(overlay_module._RENDERER, "_cache", {})
        result = draw_overlay(np.zeros((120, 200, 3), dtype=np.uint8),
                              Observation(face=self._face(),
                                          provider="混合模式·中文测试"),
                              options=OverlayOptions(), banner="回放脚本[混合]")
        assert result.shape == (120, 200, 3)
