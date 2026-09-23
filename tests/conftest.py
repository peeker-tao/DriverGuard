"""pytest 全局配置与共享夹具。

设计原则
--------
测试 **不依赖摄像头、不依赖模型文件、不依赖图形界面**：

* 用 :class:`ReplayLandmarkProvider` 与构造的 ``DriverMetrics`` 代替真实感知；
* 用 :class:`InMemoryEventRepository` 代替 CSV；
* 用固定时间戳代替墙钟时间，使时序判定 **完全可复现**。

因此整套测试可以在任何环境（含 CI 容器）中秒级跑完。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from driver_guard.config import AppConfig, load_config  # noqa: E402
from driver_guard.domain.models import (  # noqa: E402
    BehaviorEvent,
    BehaviorType,
    BoundingBox,
    DriverMetrics,
    Frame,
    HeadPose,
    Landmark,
    NeutralProfile,
    Observation,
    Severity,
)


# ===========================================================================
#  时间与帧
# ===========================================================================
class FakeClock:
    """可手动推进的假时钟，取代 ``time.perf_counter``。

    用固定步长推进使时序测试（去抖、迟滞、冷却）具有确定性——
    这是"可测试性"质量属性的直接体现。
    """

    def __init__(self, step: float = 1.0 / 30.0) -> None:
        self.step = step
        self.now = 0.0

    def tick(self) -> float:
        self.now += self.step
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def frame_factory():
    """构造一帧纯色图像，避免测试依赖真实视频。"""

    counter = {"index": 0}

    def _make(timestamp: float | None = None, size: tuple[int, int] = (120, 160)) -> Frame:
        index = counter["index"]
        counter["index"] += 1
        image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        return Frame(
            index=index,
            timestamp=timestamp if timestamp is not None else index / 30.0,
            image=image,
        )

    return _make


# ===========================================================================
#  指标构造
# ===========================================================================
@pytest.fixture
def normal_metrics() -> DriverMetrics:
    """一份"正常驾驶"的基准指标，测试中按需覆盖字段。"""
    return DriverMetrics(
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


@pytest.fixture
def neutral_profile(normal_metrics: DriverMetrics) -> NeutralProfile:
    """与 :func:`normal_metrics` 一致的中立位，使偏离量为 0。"""
    return NeutralProfile(
        head_pose=HeadPose(
            pitch=normal_metrics.pitch or 0.0,
            yaw=normal_metrics.yaw or 0.0,
            roll=normal_metrics.roll or 0.0,
        ),
        pitch_ratio=normal_metrics.face_pitch_ratio,
        samples=30,
    )


@pytest.fixture
def context_factory(frame_factory, neutral_profile):
    """构造 :class:`DetectionContext`，默认带中立位与完整人脸/手部。

    ``with_neutral=False`` 用于验证"尚未完成标定时规则应跳过而非误判"。
    """
    from driver_guard.domain.models import DetectionContext

    def _make(
        metrics: DriverMetrics,
        *,
        timestamp: float = 0.0,
        neutral: NeutralProfile | None = None,
        with_neutral: bool = True,
        with_face: bool = True,
        with_hands: bool = True,
        history: tuple[DriverMetrics, ...] = (),
    ) -> DetectionContext:
        face = tuple(Landmark(x=0.5, y=0.5) for _ in range(478)) if with_face else ()
        hands = (
            tuple(tuple(Landmark(x=0.6, y=0.4) for _ in range(21)) for _ in range(1))
            if with_hands
            else ()
        )
        observation = Observation(
            face=face,
            face_box=BoundingBox(0.3, 0.2, 0.7, 0.8) if with_face else None,
            hands=hands,
            blendshapes={
                key: value
                for key, value in (
                    ("eyeBlinkLeft", metrics.eye_blink_left),
                    ("eyeBlinkRight", metrics.eye_blink_right),
                    ("jawOpen", metrics.jaw_open),
                )
                if value is not None
            },
            precomputed=metrics,
        )
        return DetectionContext(
            frame=frame_factory(timestamp),
            observation=observation,
            metrics=metrics,
            history=history,
            neutral=(neutral_profile if neutral is None else neutral) if with_neutral else neutral,
        )

    return _make


# ===========================================================================
#  事件构造
# ===========================================================================
@pytest.fixture
def event_factory():
    def _make(
        behavior: BehaviorType = BehaviorType.YAWNING,
        severity: Severity = Severity.MEDIUM,
        **kwargs,
    ) -> BehaviorEvent:
        payload = {
            "behavior": behavior,
            "severity": severity,
            "confidence": 0.8,
            "frame_index": 10,
            "timestamp": 1.0,
            "message": "测试事件",
        }
        payload.update(kwargs)
        return BehaviorEvent(**payload)

    return _make


# ===========================================================================
#  配置
# ===========================================================================
@pytest.fixture(scope="session")
def default_config() -> AppConfig:
    """加载项目真实配置，保证测试使用的阈值与交付配置一致。"""
    path = PROJECT_ROOT / "config" / "default.json"
    if path.exists():
        return load_config(path, validate_paths=False)
    return AppConfig()
