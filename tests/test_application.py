"""应用层测试：事件总线、流水线、服务编排、存储与配置。

应用层是"薄"的，但它是**故障最容易扩散**的地方——一个订阅者抛异常、
一次管道阶段顺序错误、一个 CSV 字段错位，都会以"界面卡死"或
"事件丢失"的形式暴露。因此这里的测试重点是 **容错与契约**，
而不是业务正确性（业务正确性由 domain 层测试保证）。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from driver_guard.application.event_bus import EventBus, Topic
from driver_guard.application.pipeline import DetectionPipeline
from driver_guard.application.services import (
    AlertService,
    DetectionService,
    ServiceState,
    StatisticsService,
)
from driver_guard.application.ports import AlertSink, FrameSource, LandmarkProvider
from driver_guard.config import AppConfig, dump_config, load_config
from driver_guard.domain.calibration import NeutralPoseCalibrator
from driver_guard.domain.confirmation import BehaviorConfirmer
from driver_guard.domain.metrics import MetricExtractor
from driver_guard.domain.models import (
    BehaviorEvent,
    BehaviorType,
    DriverMetrics,
    Frame,
    Observation,
    Severity,
)
from driver_guard.domain.rules import build_engine
from driver_guard.infrastructure.alerts import (
    CallbackAlertSink,
    CompositeAlertSink,
    ConsoleAlertSink,
    NullAlertSink,
)
from driver_guard.infrastructure.sources import SyntheticFrameSource
from driver_guard.infrastructure.storage import (
    CsvEventRepository,
    CsvSessionWriter,
    InMemoryEventRepository,
    SnapshotArchiver,
)

import numpy as np


# ===========================================================================
#  事件总线（发布-订阅）
# ===========================================================================
class TestEventBus:
    def test_delivers_to_all_subscribers(self, event_factory) -> None:
        bus = EventBus()
        seen: list[str] = []
        bus.subscribe(lambda e, p: seen.append("A"), label="A")
        bus.subscribe(lambda e, p: seen.append("B"), label="B")
        assert bus.publish(event_factory()) == 2
        assert sorted(seen) == ["A", "B"]

    def test_filters_by_behavior(self, event_factory) -> None:
        bus = EventBus()
        received: list[BehaviorType] = []
        bus.subscribe(lambda e, p: received.append(e.behavior),
                      behaviors=[BehaviorType.PHONE_CALL], label="only-phone")
        bus.publish(event_factory(BehaviorType.YAWNING))
        bus.publish(event_factory(BehaviorType.PHONE_CALL))
        assert received == [BehaviorType.PHONE_CALL]

    def test_failing_subscriber_does_not_break_others(self, event_factory) -> None:
        """核心可靠性要求：一个坏订阅者不能阻断其它订阅者。"""
        bus = EventBus()
        good: list[int] = []

        def bad(event, payload):
            raise RuntimeError("订阅者内部错误")

        bus.subscribe(bad, label="bad")
        bus.subscribe(lambda e, p: good.append(1), label="good")

        assert bus.publish(event_factory()) == 1
        assert good == [1]
        assert bus.error_counts["bad"] == 1

    def test_publish_error_does_not_propagate(self, event_factory) -> None:
        bus = EventBus()

        def bad(event, payload):
            raise ValueError("boom")

        bus.subscribe(bad, label="bad")
        bus.publish(event_factory())      # 不应抛出

    def test_unsubscribe_stops_delivery(self, event_factory) -> None:
        bus = EventBus()
        received: list[int] = []
        sub = bus.subscribe(lambda e, p: received.append(1))
        bus.publish(event_factory())
        sub.close()
        bus.publish(event_factory())
        assert received == [1]
        assert bus.subscriber_count == 0

    def test_subscription_supports_context_manager(self, event_factory) -> None:
        bus = EventBus()
        with bus.subscribe(lambda e, p: None):
            assert bus.subscriber_count == 1
        assert bus.subscriber_count == 0

    def test_topic_matching(self, event_factory) -> None:
        topic = Topic(behaviors=frozenset({BehaviorType.YAWNING}))
        assert topic.matches(event_factory(BehaviorType.YAWNING))
        assert not topic.matches(event_factory(BehaviorType.SMOKING))

    def test_payload_is_passed_through(self, event_factory) -> None:
        bus = EventBus()
        captured: list[object] = []
        bus.subscribe(lambda e, p: captured.append(p))
        payload = {"frame": 42}
        bus.publish(event_factory(), payload)
        assert captured == [payload]

    def test_thread_safe_publish(self, event_factory) -> None:
        """多线程发布不应丢事件或抛异常（采集线程 + 可能的多个输出通道）。"""
        bus = EventBus()
        counter = {"n": 0}
        lock = threading.Lock()

        def handler(event, payload):
            with lock:
                counter["n"] += 1

        bus.subscribe(handler)

        def worker():
            for _ in range(50):
                bus.publish(event_factory())

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert counter["n"] == 200


# ===========================================================================
#  流水线（模板方法）
# ===========================================================================
class _StubProvider(LandmarkProvider):
    """固定输出的感知桩，用于隔离流水线本身的行为。"""

    def __init__(self, observation: Observation) -> None:
        self._observation = observation
        self.calls = 0

    @property
    def name(self) -> str:
        return "stub"

    def infer(self, frame: Frame) -> Observation:
        self.calls += 1
        return self._observation


def _build_pipeline(observation: Observation, *, calibrator=None, window: int = 5):
    engine = build_engine({
        "eyes_closed": {"enabled": True, "trigger_frames": 1, "min_duration": 0.0,
                        "cooldown_seconds": 0.0},
        "yawning": {"enabled": True, "trigger_frames": 1, "min_duration": 0.0,
                    "cooldown_seconds": 0.0},
    })
    confirmer = BehaviorConfirmer({rule.behavior: rule.config for rule in engine.rules})
    provider = _StubProvider(observation)
    pipeline = DetectionPipeline(
        provider=provider,
        extractor=MetricExtractor(),
        engine=engine,
        confirmer=confirmer,
        calibrator=calibrator,
        smoothing_window=window,
    )
    return pipeline, provider


def _frame(index: int) -> Frame:
    return Frame(index=index, timestamp=index / 30.0,
                 image=np.zeros((48, 64, 3), dtype=np.uint8))


class TestDetectionPipeline:
    def test_produces_full_analysis(self) -> None:
        observation = Observation(
            face_box=None,
            precomputed=DriverMetrics(face_detected=True, ear=0.10, eye_blink=0.9),
        )
        pipeline, provider = _build_pipeline(observation)
        analysis = pipeline.process(_frame(0))

        assert provider.calls == 1
        assert analysis.metrics.ear == pytest.approx(0.10)
        assert BehaviorType.EYES_CLOSED in analysis.triggered_behaviors
        assert len(analysis.events) == 1
        assert analysis.events[0].frame_index == 0, "服务层应回填真实帧号"

    def test_records_stage_timings(self) -> None:
        pipeline, _ = _build_pipeline(Observation(
            precomputed=DriverMetrics(face_detected=True, ear=0.3)
        ))
        analysis = pipeline.process(_frame(0))
        assert {
            DetectionPipeline.STAGE_PERCEPTION,
            DetectionPipeline.STAGE_METRICS,
            DetectionPipeline.STAGE_RULES,
            DetectionPipeline.STAGE_CONFIRM,
        } <= set(analysis.timings_ms)
        assert analysis.total_ms >= 0.0

    def test_history_window_is_bounded(self) -> None:
        pipeline, _ = _build_pipeline(
            Observation(precomputed=DriverMetrics(face_detected=True, ear=0.3)),
            window=3,
        )
        for index in range(10):
            pipeline.process(_frame(index))
        assert len(pipeline.history) == 3

    def test_no_events_for_normal_metrics(self) -> None:
        pipeline, _ = _build_pipeline(Observation(
            precomputed=DriverMetrics(face_detected=True, ear=0.30, mar=0.15,
                                      eye_blink=0.05, jaw_open=0.05)
        ))
        analysis = pipeline.process(_frame(0))
        assert analysis.triggered == ()
        assert analysis.events == ()

    def test_calibrator_receives_metrics(self) -> None:
        calibrator = NeutralPoseCalibrator(required_samples=3, warmup_frames=0)
        pipeline, _ = _build_pipeline(
            Observation(precomputed=DriverMetrics(face_detected=True, pitch=5.0,
                                                  yaw=0.0, roll=0.0)),
            calibrator=calibrator,
        )
        for index in range(3):
            analysis = pipeline.process(_frame(index))
        assert analysis.neutral is not None
        assert analysis.calibration_progress == pytest.approx(1.0)

    def test_reset_clears_history_and_state(self) -> None:
        pipeline, _ = _build_pipeline(Observation(
            precomputed=DriverMetrics(face_detected=True, ear=0.10, eye_blink=0.9)
        ))
        pipeline.process(_frame(0))
        pipeline.reset()
        assert pipeline.history == ()
        assert pipeline.confirmer.active_behaviors == ()

    def test_provider_exception_propagates_for_service_to_handle(self) -> None:
        """流水线不吞感知异常，由服务层统一兜底并上报状态（职责清晰）。"""

        class BrokenProvider(LandmarkProvider):
            @property
            def name(self) -> str:
                return "broken"

            def infer(self, frame: Frame) -> Observation:
                raise RuntimeError("模型崩溃")

        pipeline = DetectionPipeline(
            provider=BrokenProvider(),
            extractor=MetricExtractor(),
            engine=build_engine({}),
            confirmer=BehaviorConfirmer(),
        )
        with pytest.raises(RuntimeError):
            pipeline.process(_frame(0))


# ===========================================================================
#  检测服务
# ===========================================================================
class _FiniteSource(FrameSource):
    """产出固定帧数的数据源（测试用，无需文件与摄像头）。"""

    def __init__(self, frames: int, *, fps: float = 1000.0) -> None:
        self._remaining = frames
        self._fps = fps
        self._index = 0
        self.opened = False
        self.closed = False

    @property
    def name(self) -> str:
        return "固定帧源"

    @property
    def fps(self) -> float:
        return self._fps

    def open(self) -> None:
        self.opened = True

    def read(self) -> Frame | None:
        if self._remaining <= 0:
            return None
        self._remaining -= 1
        frame = _frame(self._index)
        self._index += 1
        return frame

    def close(self) -> None:
        self.closed = True


class _ExplodingSource(FrameSource):
    @property
    def name(self) -> str:
        return "会抛异常的数据源"

    def open(self) -> None:
        raise RuntimeError("摄像头被占用")

    def read(self) -> Frame | None:
        return None


def _service(frames: int = 5, **kwargs) -> tuple[DetectionService, EventBus]:
    observation = Observation(precomputed=DriverMetrics(face_detected=True, ear=0.10,
                                                        eye_blink=0.9))
    pipeline, _ = _build_pipeline(observation)
    bus = EventBus()
    service = DetectionService(
        source=_FiniteSource(frames),
        pipeline=pipeline,
        event_bus=bus,
        **kwargs,
    )
    return service, bus


class TestDetectionService:
    def test_runs_to_completion_and_summarises(self) -> None:
        service, _ = _service(frames=4)
        summary = service.run()
        assert summary.frames == 4
        assert not summary.interrupted
        assert service.state is ServiceState.FINISHED
        assert summary.event_counts[BehaviorType.EYES_CLOSED] == 1

    def test_publishes_events_to_bus(self) -> None:
        service, bus = _service(frames=3)
        received: list[BehaviorEvent] = []
        bus.subscribe(lambda e, p: received.append(e))
        service.run()
        assert len(received) == 1

    def test_persists_events_to_repository(self) -> None:
        repository = InMemoryEventRepository()
        service, _ = _service(frames=3, repository=repository)
        service.run()
        assert len(repository) == 1

    def test_frame_callback_receives_analyses(self) -> None:
        service, _ = _service(frames=4)
        seen: list[int] = []
        service.run(on_frame=lambda analysis: seen.append(analysis.frame.index))
        assert seen == [0, 1, 2, 3]

    def test_source_error_is_captured_not_raised(self) -> None:
        observation = Observation(precomputed=DriverMetrics(face_detected=True))
        pipeline, _ = _build_pipeline(observation)
        service = DetectionService(
            source=_ExplodingSource(), pipeline=pipeline, event_bus=EventBus()
        )
        summary = service.run()          # 不应抛出
        assert service.state is ServiceState.ERROR
        assert service.last_error is not None
        assert summary.frames == 0

    def test_stop_interrupts_loop(self) -> None:
        """stop() 必须在合理时间内让阻塞主循环退出（响应性要求）。"""

        class EndlessSource(FrameSource):
            @property
            def name(self) -> str:
                return "无限源"

            def open(self) -> None:
                pass

            def read(self) -> Frame | None:
                time.sleep(0.001)
                return _frame(0)

        observation = Observation(precomputed=DriverMetrics(face_detected=True))
        pipeline, _ = _build_pipeline(observation)
        service = DetectionService(
            source=EndlessSource(), pipeline=pipeline, event_bus=EventBus()
        )
        thread = service.start_async()
        time.sleep(0.15)
        service.stop()
        thread.join(timeout=3.0)
        assert not thread.is_alive(), "stop() 后主循环应及时退出"

    def test_max_frames_limit(self) -> None:
        service, _ = _service(frames=100, max_frames=7)
        assert service.run().frames == 7

    def test_limit_frames_can_be_changed(self) -> None:
        service, _ = _service(frames=100)
        service.limit_frames(3)
        assert service.run().frames == 3

    def test_source_is_closed_after_run(self) -> None:
        source = _FiniteSource(2)
        observation = Observation(precomputed=DriverMetrics(face_detected=True))
        pipeline, _ = _build_pipeline(observation)
        service = DetectionService(source=source, pipeline=pipeline, event_bus=EventBus())
        service.run()
        assert source.closed, "数据源必须在 finally 中被释放"

    def test_start_async_rejects_double_start(self) -> None:
        """防止两个采集循环同时驱动同一数据源（会导致画面与事件错乱）。"""

        class EndlessSource(FrameSource):
            @property
            def name(self) -> str:
                return "无限源"

            def open(self) -> None:
                pass

            def read(self) -> Frame | None:
                time.sleep(0.001)
                return _frame(0)

        observation = Observation(precomputed=DriverMetrics(face_detected=True))
        pipeline, _ = _build_pipeline(observation)
        service = DetectionService(
            source=EndlessSource(), pipeline=pipeline, event_bus=EventBus()
        )
        service.start_async()
        try:
            with pytest.raises(RuntimeError):
                service.start_async()
        finally:
            service.stop()
            service.join(timeout=3.0)


# ===========================================================================
#  告警服务
# ===========================================================================
class _RecordingSink(AlertSink):
    def __init__(self, name: str = "rec", *, fail: bool = False) -> None:
        self._name = name
        self._fail = fail
        self.events: list[BehaviorEvent] = []

    @property
    def name(self) -> str:
        return self._name

    def emit(self, event: BehaviorEvent, frame: Frame | None = None) -> None:
        if self._fail:
            raise RuntimeError("通道故障")
        self.events.append(event)


class TestAlertService:
    def test_fans_out_to_all_sinks(self, event_factory) -> None:
        bus = EventBus()
        sink_a, sink_b = _RecordingSink("A"), _RecordingSink("B")
        service = AlertService(bus, [sink_a, sink_b])
        service.start()
        bus.publish(event_factory())
        assert len(sink_a.events) == 1 and len(sink_b.events) == 1

    def test_min_severity_filter(self, event_factory) -> None:
        bus = EventBus()
        sink = _RecordingSink()
        service = AlertService(bus, [sink], min_severity=Severity.HIGH)
        service.start()
        bus.publish(event_factory(severity=Severity.LOW))
        bus.publish(event_factory(severity=Severity.CRITICAL))
        assert len(sink.events) == 1

    def test_per_sink_threshold(self, event_factory) -> None:
        bus = EventBus()
        quiet, loud = _RecordingSink("quiet"), _RecordingSink("loud")
        service = AlertService(
            bus, [quiet, loud],
            min_severity=Severity.LOW,
            per_sink_min_severity={"quiet": Severity.CRITICAL, "loud": Severity.LOW},
        )
        service.start()
        bus.publish(event_factory(severity=Severity.HIGH))
        assert not quiet.events
        assert len(loud.events) == 1

    def test_failing_sink_does_not_block_others(self, event_factory) -> None:
        bus = EventBus()
        bad, good = _RecordingSink("bad", fail=True), _RecordingSink("good")
        service = AlertService(bus, [bad, good])
        service.start()
        bus.publish(event_factory())          # 不应抛出
        assert len(good.events) == 1

    def test_stop_prevents_further_delivery(self, event_factory) -> None:
        bus = EventBus()
        sink = _RecordingSink()
        service = AlertService(bus, [sink])
        service.start()
        service.stop()
        bus.publish(event_factory())
        assert not sink.events

    def test_frame_payload_is_forwarded(self, event_factory) -> None:
        bus = EventBus()
        frame = _frame(0)

        class FrameAwareSink(AlertSink):
            got_frame: Frame | None = None

            @property
            def name(self) -> str:
                return "frame-aware"

            def emit(self, event: BehaviorEvent, frame: Frame | None = None) -> None:
                self.got_frame = frame

        sink = FrameAwareSink()
        service = AlertService(bus, [sink])
        service.start()
        bus.publish(event_factory(), SimpleNamespace(frame=frame))
        assert sink.got_frame is frame


from types import SimpleNamespace  # noqa: E402  （放在使用处附近便于阅读）


def test_callback_sink_bridges_to_arbitrary_target(event_factory) -> None:
    received: list[BehaviorEvent] = []
    sink = CallbackAlertSink(lambda e, f: received.append(e), label="bridge")
    sink.emit(event_factory())
    assert len(received) == 1


def test_composite_and_null_sinks(event_factory) -> None:
    inner = _RecordingSink("inner")
    composite = CompositeAlertSink([inner])
    composite.emit(event_factory())
    assert len(inner.events) == 1

    NullAlertSink().emit(event_factory())      # 不应抛异常，也不产生副作用


def test_console_sink_never_raises(event_factory) -> None:
    ConsoleAlertSink().emit(event_factory(), _frame(0))


# ===========================================================================
#  统计服务
# ===========================================================================
class TestStatisticsService:
    def test_counts_only_new_events(self, event_factory) -> None:
        """同一次持续行为的重复提醒不应拉高统计次数。"""
        bus = EventBus()
        stats = StatisticsService(bus)
        bus.publish(event_factory(BehaviorType.YAWNING))
        bus.publish(event_factory(BehaviorType.YAWNING, ongoing=True))
        bus.publish(event_factory(BehaviorType.YAWNING, ongoing=True))
        assert stats.total == 1

    def test_groups_by_category(self, event_factory) -> None:
        bus = EventBus()
        stats = StatisticsService(bus)
        bus.publish(event_factory(BehaviorType.EYES_CLOSED))
        bus.publish(event_factory(BehaviorType.GAZE_OFF_ROAD))
        bus.publish(event_factory(BehaviorType.PHONE_CALL))
        categories = stats.category_counts()
        assert sum(categories.values()) == 3

    def test_severity_breakdown(self, event_factory) -> None:
        bus = EventBus()
        stats = StatisticsService(bus)
        bus.publish(event_factory(severity=Severity.HIGH))
        bus.publish(event_factory(severity=Severity.HIGH))
        bus.publish(event_factory(severity=Severity.LOW))
        assert stats.severity_counts[Severity.HIGH] == 2

    def test_top_returns_descending(self, event_factory) -> None:
        bus = EventBus()
        stats = StatisticsService(bus)
        for _ in range(3):
            bus.publish(event_factory(BehaviorType.YAWNING))
        bus.publish(event_factory(BehaviorType.SMOKING))
        assert stats.top(1)[0] == (BehaviorType.YAWNING, 3)

    def test_reset_clears(self, event_factory) -> None:
        bus = EventBus()
        stats = StatisticsService(bus)
        bus.publish(event_factory())
        stats.reset()
        assert stats.total == 0

    def test_listener_sees_updated_counts(self, event_factory) -> None:
        """监听者必须在计数更新之后被通知，否则界面统计会慢一条事件。"""
        bus = EventBus()
        stats = StatisticsService(bus)
        snapshots: list[int] = []
        stats.add_listener(lambda: snapshots.append(stats.total))
        bus.publish(event_factory(BehaviorType.YAWNING))
        assert snapshots == [1]

    def test_listener_notified_on_reset(self, event_factory) -> None:
        bus = EventBus()
        stats = StatisticsService(bus)
        bus.publish(event_factory())
        snapshots: list[int] = []
        stats.add_listener(lambda: snapshots.append(stats.total))
        stats.reset()
        assert snapshots == [0]

    def test_listener_ignores_ongoing_duplicates(self, event_factory) -> None:
        bus = EventBus()
        stats = StatisticsService(bus)
        bus.publish(event_factory())
        snapshots: list[int] = []
        stats.add_listener(lambda: snapshots.append(stats.total))
        bus.publish(event_factory(ongoing=True))
        assert snapshots == []

    def test_close_unsubscribes(self, event_factory) -> None:
        bus = EventBus()
        stats = StatisticsService(bus)
        stats.close()
        bus.publish(event_factory())
        assert stats.total == 0


# ===========================================================================
#  存储
# ===========================================================================
class TestCsvEventRepository:
    def test_round_trip(self, tmp_path: Path, event_factory) -> None:
        path = tmp_path / "events.csv"
        event = event_factory(
            BehaviorType.PHONE_CALL, severity=Severity.CRITICAL,
            metrics={"hand_face_ratio": 0.28, "confidence": 0.91},
            duration=2.5, message="检测到手部贴近头面部",
        )
        with CsvEventRepository(path, flush_every=1) as repo:
            repo.save(event)

        with CsvEventRepository(path) as reopened:
            loaded = reopened.all()

        assert len(loaded) == 1
        restored = loaded[0]
        assert restored.event_id == event.event_id
        assert restored.behavior is event.behavior
        assert restored.severity is event.severity
        assert restored.confidence == pytest.approx(event.confidence)
        assert restored.duration == pytest.approx(2.5)
        assert restored.metrics["hand_face_ratio"] == pytest.approx(0.28)
        # CSV 以毫秒精度记录墙钟时间，比较时也要对齐到毫秒
        assert restored.wall_clock == event.wall_clock.replace(microsecond=(
            event.wall_clock.microsecond // 1000) * 1000)

    def test_header_written_once_and_appends(self, tmp_path: Path, event_factory) -> None:
        path = tmp_path / "events.csv"
        repo = CsvEventRepository(path, flush_every=1)
        repo.save(event_factory())
        repo.save(event_factory())
        repo.close()

        text = path.read_text(encoding="utf-8-sig")
        assert text.count("event_id") == 1, "表头只能出现一次"
        assert len([line for line in text.splitlines() if line.strip()]) == 3

    def test_in_memory_cache_stays_consistent(self, tmp_path: Path, event_factory) -> None:
        repo = CsvEventRepository(tmp_path / "e.csv", flush_every=1)
        repo.save(event_factory())
        assert len(repo.all()) == 1        # 触发一次全量读取并建立缓存
        repo.save(event_factory())
        assert len(repo.all()) == 2        # 缓存应同步增长
        repo.close()

    def test_skips_corrupt_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "events.csv"
        path.write_text(
            "event_id,wall_clock,timestamp,frame_index,behavior,behavior_label,"
            "category,severity,severity_value,confidence,duration,ongoing,message,metrics\n"
            "good,2024-01-01T00:00:00.000,1.0,5,yawning,打哈欠,fatigue,HIGH,3,0.8,1.0,0,ok,mar=0.7\n"
            "broken,not-a-date,x,y,unknown,x,x,x,x,x,x,x,x,x\n",
            encoding="utf-8-sig",
        )
        with CsvEventRepository(path) as repo:
            events = repo.all()
        assert len(events) == 1, "坏行应被跳过而不是让整个加载失败"

    def test_clear_removes_file(self, tmp_path: Path, event_factory) -> None:
        path = tmp_path / "events.csv"
        repo = CsvEventRepository(path, flush_every=1)
        repo.save(event_factory())
        repo.clear()
        assert repo.all() == ()
        repo.close()

    def test_creates_parent_directory(self, tmp_path: Path, event_factory) -> None:
        path = tmp_path / "deep" / "nested" / "events.csv"
        repo = CsvEventRepository(path, flush_every=1)
        repo.save(event_factory())
        repo.close()
        assert path.exists()


class TestSnapshotArchiver:
    def test_saves_annotated_image(self, tmp_path: Path, event_factory, frame_factory) -> None:
        archiver = SnapshotArchiver(tmp_path, annotate=True)
        path = archiver.save(event_factory(severity=Severity.CRITICAL), frame_factory())
        assert path is not None and path.exists()
        assert path.suffix == ".jpg"

    def test_returns_none_without_frame(self, tmp_path: Path, event_factory) -> None:
        archiver = SnapshotArchiver(tmp_path)
        assert archiver.save(event_factory(), None) is None

    def test_handles_empty_image(self, tmp_path: Path, event_factory) -> None:
        archiver = SnapshotArchiver(tmp_path)
        empty = Frame(index=0, timestamp=0.0, image=np.zeros((0, 0, 3), dtype=np.uint8))
        assert archiver.save(event_factory(), empty) is None

    def test_enforces_quota(self, tmp_path: Path, event_factory, frame_factory) -> None:
        archiver = SnapshotArchiver(tmp_path, max_files=10)
        for _ in range(25):
            archiver.save(event_factory(), frame_factory())
        assert len(list(tmp_path.glob("*.jpg"))) <= 10


class TestCsvSessionWriter:
    def test_appends_header_once(self, tmp_path: Path) -> None:
        from driver_guard.application.services import SessionSummary
        from datetime import datetime

        writer = CsvSessionWriter(tmp_path / "sessions.csv")
        summary = SessionSummary(
            source_name="测试源", frames=100, duration=10.0, avg_fps=10.0,
            triggered_frames=20, event_counts={BehaviorType.YAWNING: 2},
            stage_avg_ms={"perception": 12.5}, started_at=datetime.now(),
            ended_at=datetime.now(),
        )
        writer.append(summary, risk_index=7.5)
        writer.append(summary, risk_index=8.0)
        text = writer.path.read_text(encoding="utf-8-sig")
        assert text.count("started_at") == 1
        assert "yawning=2" in text


def test_session_summary_risk_index_is_bounded() -> None:
    from datetime import datetime

    from driver_guard.application.services import SessionSummary

    summary = SessionSummary(
        source_name="x", frames=1, duration=0.001, avg_fps=1.0, triggered_frames=1,
        event_counts={BehaviorType.PHONE_CALL: 100},
        stage_avg_ms={}, started_at=datetime.now(), ended_at=datetime.now(),
    )
    assert 0.0 <= summary.risk_index <= 100.0

    zero = replace(summary, duration=0.0)
    assert zero.risk_index == 0.0


# ===========================================================================
#  配置
# ===========================================================================
class TestConfig:
    def test_default_config_is_usable(self) -> None:
        config = AppConfig()
        assert config.source.type == "camera"
        assert config.perception.provider == "mediapipe"

    def test_load_and_merge_partial_file(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"source": {"type": "replay", "fps": 15}}),
                        encoding="utf-8")
        config = load_config(path, validate_paths=False)
        assert config.source.type == "replay"
        assert config.source.fps == pytest.approx(15.0)
        assert config.source.width == 1280, "未覆盖的字段应保留默认值"

    def test_unknown_key_is_reported_not_silently_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"source": {"camra_index": 2, "camera_index": 1}}),
                        encoding="utf-8")
        config = load_config(path, validate_paths=False)
        assert config.source.camera_index == 1
        assert any("camra_index" in w for w in config.warnings)

    def test_out_of_range_value_is_clamped(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"source": {"fps": 100000}}), encoding="utf-8")
        config = load_config(path, validate_paths=False)
        assert config.source.fps <= 240.0
        assert any("fps" in w for w in config.warnings)

    def test_bad_scalar_type_falls_back(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"ui": {"max_events_in_table": "many"}}),
                        encoding="utf-8")
        config = load_config(path, validate_paths=False)
        assert config.ui.max_events_in_table == 500

    def test_severity_string_is_parsed(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"alert": {"min_severity": "high"}}), encoding="utf-8")
        config = load_config(path, validate_paths=False)
        assert config.alert.min_severity is Severity.HIGH

    def test_missing_file_uses_defaults(self, tmp_path: Path) -> None:
        config = load_config(tmp_path / "nope.json")
        assert config.source.type == "camera"

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_config(path)

    def test_dump_round_trips(self, default_config) -> None:
        payload = json.loads(dump_config(default_config))
        assert payload["source"]["type"] == default_config.source.type
        assert payload["alert"]["min_severity"] in {"low", "medium", "high", "critical", "info"}

    def test_paths_resolve_under_project_root(self, default_config) -> None:
        from driver_guard.config import PROJECT_ROOT

        assert default_config.resolved_model_dir().is_absolute()
        assert str(default_config.resolved_model_dir()).startswith(str(PROJECT_ROOT))
        assert default_config.output.resolve("x.csv").is_absolute()

    def test_provider_options_include_scenario(self, default_config) -> None:
        options = default_config.perception.provider_options()
        assert "model_dir" in options
        assert options["scenario"] == default_config.perception.scenario


def test_placeholder_fixtures_are_wired(normal_metrics, neutral_profile, clock) -> None:
    """保证 conftest 里的夹具本身可用（否则其它测试的失败会被误读）。"""
    assert normal_metrics.face_detected
    assert neutral_profile.ready
    assert clock.tick() > 0.0
