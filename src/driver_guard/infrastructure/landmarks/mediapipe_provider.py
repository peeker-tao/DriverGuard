"""MediaPipe Tasks 感知适配器（生产环境默认实现）。

为什么用 Tasks API 而不是 ``mp.solutions``
------------------------------------------
MediaPipe 1.x 已移除旧的 ``mp.solutions.face_mesh`` / ``pose`` 接口，
只保留 ``mediapipe.tasks``；本研究环境实测 ``mediapipe 1.0.1`` 中
``hasattr(mp, "solutions") is False``。因此本适配器直接使用
``FaceLandmarker`` / ``HandLandmarker`` 两个 Tasks 任务。

本适配器一次推理同时产出三类证据，供上层融合：

====================================  ==========================================
证据                                    用途
====================================  ==========================================
478 点人脸关键点                        EAR / MAR / 低头比例（可解释、可复核）
``facial_transformation_matrixes``      头部姿态（来自模型，比 solvePnP 更稳）
``face_blendshapes``                    ``eyeBlinkL/R``、``jawOpen`` 语义系数
21 点手部关键点                         打电话 / 抽烟判定的相对几何
====================================  ==========================================

降级策略
--------
* 手部模型缺失或加载失败 → 自动关闭手部通道，其余功能不受影响（``degraded`` 为真）；
* 模型文件缺失 → :attr:`ready` 为 ``False``，界面据此提示运行模型下载脚本。

时间戳处理
----------
Tasks 的 ``VIDEO`` 模式要求时间戳 **严格单调递增**。视频循环播放时
时间戳会回绕，因此这里维护独立计数器并强制递增（实测若不处理会抛
``ValueError: Input timestamp must be monotonically increasing``）。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from ...application.ports import LandmarkProvider
from ...domain.models import BoundingBox, Frame, HeadPose, Landmark, Observation
from ...domain.geometry import FaceMeshIndex, head_pose_from_matrix

logger = logging.getLogger(__name__)

__all__ = ["MediaPipeLandmarkProvider"]


class MediaPipeLandmarkProvider(LandmarkProvider):
    """基于 MediaPipe Tasks 的人脸 + 手部关键点提供者。

    Args:
        model_dir: 存放 ``.task`` 模型的目录。
        face_model: 人脸模型文件名。
        hand_model: 手部模型文件名。
        enable_hands: 是否加载手部模型（关闭可省约 7 MB 内存与部分耗时）。
        max_faces: 最大人脸数；驾驶员场景固定为 1。
        max_hands: 最大手部数；方向盘场景 2 足够。
        min_*_confidence: 各检测/跟踪置信度门槛。
        fps_hint: 用于把帧号换算为毫秒时间戳的名义帧率。
    """

    def __init__(
        self,
        *,
        model_dir: str | Path = "assets/models",
        face_model: str = "face_landmarker.task",
        hand_model: str = "hand_landmarker.task",
        enable_hands: bool = True,
        max_faces: int = 1,
        max_hands: int = 2,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_hand_tracking_confidence: float = 0.5,
        fps_hint: float = 30.0,
    ) -> None:
        self._model_dir = Path(model_dir)
        self._face_path = self._model_dir / face_model
        self._hand_path = self._model_dir / hand_model
        self._enable_hands = enable_hands
        self._max_faces = max(1, max_faces)
        self._max_hands = max(1, max_hands)
        self._face_conf = (
            min_face_detection_confidence,
            min_face_presence_confidence,
            min_tracking_confidence,
        )
        self._hand_conf = (
            min_hand_detection_confidence,
            min_hand_presence_confidence,
            min_hand_tracking_confidence,
        )
        self._fps_hint = max(1.0, fps_hint)

        self._face_landmarker: Any = None
        self._hand_landmarker: Any = None
        self._mp: Any = None
        self._last_timestamp_ms = -1
        self._tick = 0
        self._degraded = False
        self._degraded_reason = ""

    # ------------------------------------------------------------------
    #  元信息
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "mediapipe-tasks" + ("（手部通道已降级关闭）" if self._degraded and self._hand_landmarker is None else "")

    @property
    def ready(self) -> bool:
        return self._face_landmarker is not None

    @property
    def degraded(self) -> bool:
        """是否处于降级状态（部分通道不可用）。"""
        return self._degraded

    @property
    def degraded_reason(self) -> str:
        return self._degraded_reason

    @property
    def hands_enabled(self) -> bool:
        return self._hand_landmarker is not None

    @property
    def model_paths(self) -> tuple[Path, Path]:
        return self._face_path, self._hand_path

    def missing_models(self) -> tuple[Path, ...]:
        """列出缺失且被启用的模型文件，供界面给出可操作提示。"""
        required = [self._face_path]
        if self._enable_hands:
            required.append(self._hand_path)
        return tuple(path for path in required if not path.exists())

    # ------------------------------------------------------------------
    #  生命周期
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._face_landmarker is not None:
            return

        missing = self.missing_models()
        if missing:
            raise FileNotFoundError(
                "缺少 MediaPipe 模型文件："
                + ", ".join(str(p) for p in missing)
                + "\n请先运行： python tools/download_models.py"
            )

        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        detection, presence, tracking = self._face_conf
        face_options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self._face_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=self._max_faces,
            min_face_detection_confidence=detection,
            min_face_presence_confidence=presence,
            min_tracking_confidence=tracking,
            # 头姿矩阵 + 表情系数：两者都是本系统的关键证据通道
            output_facial_transformation_matrixes=True,
            output_face_blendshapes=True,
        )
        self._face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
        logger.info("已加载人脸模型：%s", self._face_path.name)

        if self._enable_hands:
            self._try_load_hands(vision, mp_python)

    def _try_load_hands(self, vision: Any, mp_python: Any) -> None:
        """手部模型加载失败时降级，不阻断整体功能。"""
        if not self._hand_path.exists():
            self._mark_degraded(f"未找到手部模型 {self._hand_path.name}，打电话/抽烟检测已关闭")
            return
        detection, presence, tracking = self._hand_conf
        try:
            hand_options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(self._hand_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=self._max_hands,
                min_hand_detection_confidence=detection,
                min_hand_presence_confidence=presence,
                min_tracking_confidence=tracking,
            )
            self._hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
            logger.info("已加载手部模型：%s", self._hand_path.name)
        except Exception as exc:  # noqa: BLE001 - 降级而非崩溃
            self._mark_degraded(f"手部模型加载失败（{type(exc).__name__}: {exc}），打电话/抽烟检测已关闭")

    def _mark_degraded(self, reason: str) -> None:
        self._degraded = True
        self._degraded_reason = reason
        logger.warning("感知通道降级：%s", reason)

    def warmup(self) -> None:
        """用一张小尺寸空白图预热，避免首帧耗时抖动影响界面。"""
        self._ensure_loaded()
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        dummy = Frame(index=0, timestamp=0.0, image=blank)
        try:
            self.infer(dummy)
        except Exception:  # noqa: BLE001 - 预热失败不影响后续
            logger.exception("预热推理失败（已忽略）")
        finally:
            # 复位计数，使真实会话的时间戳从 0 开始
            self._last_timestamp_ms = -1
            self._tick = 0

    def close(self) -> None:
        for landmarker in (self._face_landmarker, self._hand_landmarker):
            if landmarker is not None:
                try:
                    landmarker.close()
                except Exception:  # noqa: BLE001
                    logger.exception("关闭 MediaPipe 任务失败")
        self._face_landmarker = None
        self._hand_landmarker = None

    # ------------------------------------------------------------------
    #  推理
    # ------------------------------------------------------------------
    def _next_timestamp_ms(self, frame: Frame) -> int:
        """产生严格单调递增的毫秒时间戳。

        以真实帧号换算为主，同时参考墙钟时间以防帧号回绕（视频循环）：
        取三者的最大值再加 1，确保单调性成立。
        """
        self._tick += 1
        from_frame = int(frame.timestamp * 1000.0)
        from_index = int(frame.index * 1000.0 / self._fps_hint)
        from_clock = int((time.perf_counter() % 1e6) * 1000.0) % 1_000_000
        candidate = max(from_frame, from_index)
        if candidate <= self._last_timestamp_ms:
            candidate = self._last_timestamp_ms + 1
        del from_clock  # 仅用于说明为何不采用墙钟时间：跨会话会回绕
        self._last_timestamp_ms = candidate
        return candidate

    def infer(self, frame: Frame) -> Observation:
        self._ensure_loaded()
        assert self._mp is not None

        started = time.perf_counter()
        timestamp_ms = self._next_timestamp_ms(frame)
        rgb = np.ascontiguousarray(frame.rgb())
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        face_result = self._face_landmarker.detect_for_video(mp_image, timestamp_ms)
        observation = self._to_observation(face_result)

        hand_result = None
        if self._hand_landmarker is not None:
            try:
                hand_result = self._hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            except Exception:  # noqa: BLE001 - 手部通道故障不影响人脸通道
                logger.exception("手部推理失败，本帧跳过手部证据")
        hands, hand_boxes = self._to_hands(hand_result)

        return Observation(
            face=observation.face,
            face_box=observation.face_box,
            head_pose=observation.head_pose,
            blendshapes=observation.blendshapes,
            hands=hands,
            hand_boxes=hand_boxes,
            provider=self.name,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    # ------------------------------------------------------------------
    #  结果转换
    # ------------------------------------------------------------------
    def _to_observation(self, result: Any) -> Observation:
        if not result.face_landmarks:
            return Observation(provider=self.name)

        raw = result.face_landmarks[0]
        face = tuple(Landmark(x=float(p.x), y=float(p.y), z=float(p.z)) for p in raw)
        face_box = BoundingBox.from_landmarks(face, padding=0.06)

        head_pose: HeadPose | None = None
        matrices = getattr(result, "facial_transformation_matrixes", None)
        if matrices:
            head_pose = head_pose_from_matrix(np.asarray(matrices[0]))

        blendshapes: dict[str, float] = {}
        shapes = getattr(result, "face_blendshapes", None)
        if shapes:
            for category in shapes[0]:
                blendshapes[str(category.category_name)] = float(category.score)

        return Observation(
            face=face,
            face_box=face_box,
            head_pose=head_pose,
            blendshapes=blendshapes,
            provider=self.name,
        )

    def _to_hands(self, result: Any) -> tuple[tuple[tuple[Landmark, ...], ...], tuple[BoundingBox, ...]]:
        if result is None or not getattr(result, "hand_landmarks", None):
            return (), ()
        hands: list[tuple[Landmark, ...]] = []
        boxes: list[BoundingBox] = []
        for raw_hand in result.hand_landmarks:
            points = tuple(Landmark(x=float(p.x), y=float(p.y), z=float(p.z)) for p in raw_hand)
            hands.append(points)
            box = BoundingBox.from_landmarks(points, padding=0.05)
            if box is not None:
                boxes.append(box)
        return tuple(hands), tuple(boxes)


# ---------------------------------------------------------------------------
#  便于诊断的辅助信息
# ---------------------------------------------------------------------------
FACE_LANDMARK_COUNT = 478
HAND_LANDMARK_COUNT = 21
NOSE_TIP_INDEX = FaceMeshIndex.NOSE_TIP
