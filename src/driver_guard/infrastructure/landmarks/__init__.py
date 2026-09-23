"""感知适配器集合与简单工厂。

.. code-block:: text

   provider 名        实现                            适用场景
   ─────────────────────────────────────────────────────────────────
   mediapipe        MediaPipeLandmarkProvider        生产默认，精度最高
   haar             HaarLandmarkProvider             模型不可用时的降级通道
   replay           回放脚本                          无摄像头演示 / 回归测试

新增一种感知实现只需：实现 ``LandmarkProvider`` 端口 + 在
:data:`PROVIDER_FACTORIES` 登记一行。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

from ...application.ports import LandmarkProvider
from .haar_provider import HaarLandmarkProvider
from .mediapipe_provider import MediaPipeLandmarkProvider
from .replay_provider import (
    SCENARIOS,
    MetricSegment,
    ReplayLandmarkProvider,
    scenario_duration,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MediaPipeLandmarkProvider",
    "HaarLandmarkProvider",
    "ReplayLandmarkProvider",
    "MetricSegment",
    "SCENARIOS",
    "scenario_duration",
    "PROVIDER_FACTORIES",
    "create_landmark_provider",
    "available_providers",
]


def _build_mediapipe(options: Mapping[str, Any]) -> LandmarkProvider:
    return MediaPipeLandmarkProvider(
        model_dir=options.get("model_dir", "assets/models"),
        face_model=options.get("face_model", "face_landmarker.task"),
        hand_model=options.get("hand_model", "hand_landmarker.task"),
        enable_hands=bool(options.get("enable_hands", True)),
        max_faces=int(options.get("max_faces", 1)),
        max_hands=int(options.get("max_hands", 2)),
        min_face_detection_confidence=float(options.get("min_face_detection_confidence", 0.5)),
        min_face_presence_confidence=float(options.get("min_face_presence_confidence", 0.5)),
        min_tracking_confidence=float(options.get("min_tracking_confidence", 0.5)),
        min_hand_detection_confidence=float(options.get("min_hand_detection_confidence", 0.5)),
        min_hand_presence_confidence=float(options.get("min_hand_presence_confidence", 0.5)),
        min_hand_tracking_confidence=float(options.get("min_hand_tracking_confidence", 0.5)),
        fps_hint=float(options.get("fps_hint", 30.0)),
    )


def _build_haar(options: Mapping[str, Any]) -> LandmarkProvider:
    return HaarLandmarkProvider(
        cascade_dir=options.get("cascade_dir", "assets/cascades"),
        scale_factor=float(options.get("scale_factor", 1.1)),
        min_neighbors=int(options.get("min_neighbors", 6)),
        min_size_ratio=float(options.get("min_size_ratio", 0.12)),
    )


def _build_replay(options: Mapping[str, Any]) -> LandmarkProvider:
    return ReplayLandmarkProvider(
        scenario=options.get("scenario", "mixed"),
        loop=bool(options.get("loop", True)),
        with_hands=bool(options.get("with_hands", True)),
        fps_hint=float(options.get("fps_hint", 30.0)),
    )


#: ``provider 名 -> 构造函数``
PROVIDER_FACTORIES: dict[str, Callable[[Mapping[str, Any]], LandmarkProvider]] = {
    "mediapipe": _build_mediapipe,
    "haar": _build_haar,
    "replay": _build_replay,
}


def available_providers() -> tuple[str, ...]:
    return tuple(PROVIDER_FACTORIES)


def create_landmark_provider(
    name: str,
    options: Mapping[str, Any] | None = None,
    *,
    fallback: str | None = None,
) -> LandmarkProvider:
    """按名字创建感知适配器。

    Args:
        name: ``mediapipe`` / ``haar`` / ``replay``。
        options: 传给具体实现的构造参数。
        fallback: 当 ``name`` 未知或构造失败时退回的实现名。
            设为 ``"replay"`` 可以在模型缺失时仍能启动界面（演示友好）。

    Raises:
        KeyError: ``name`` 未知且未提供有效的 ``fallback``。
        Exception: 构造失败且未提供 ``fallback`` 时原样抛出。
    """
    options = options or {}
    factory = PROVIDER_FACTORIES.get(name)
    if factory is None:
        if fallback and fallback in PROVIDER_FACTORIES:
            logger.warning("未知感知实现 %r，退回 %r", name, fallback)
            return PROVIDER_FACTORIES[fallback](options)
        raise KeyError(
            f"未知感知实现 {name!r}，可选：{', '.join(available_providers())}"
        )

    try:
        return factory(options)
    except Exception as exc:  # noqa: BLE001 - 允许降级
        if not fallback or fallback == name:
            raise
        logger.warning("感知实现 %r 构造失败（%s），退回 %r", name, exc, fallback)
        return PROVIDER_FACTORIES[fallback](options)
