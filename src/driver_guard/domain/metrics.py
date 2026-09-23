"""指标提取：把感知结果 :class:`Observation` 转换为可度量的 :class:`DriverMetrics`。

这是管道-过滤器架构中的**第 2 个过滤器**。它只做纯计算，不持有状态、
不产生副作用，因此可被单元测试穷举覆盖。

设计要点
--------
1. **可空即融合**：任一指标算不出来时置 ``None``，规则层据此自动切换到
   其他证据通道（例如 EAR 不可用时改用 MediaPipe blendshape），实现优雅降级。
2. **优先采用上游预计算值**：回放/自检模式下的感知实现可直接给出指标
   （``Observation.precomputed``），此时跳过几何计算。
3. **不依赖绝对姿态语义**：同时输出旋转矩阵角与与轴系无关的
   ``face_pitch_ratio``，让规则可以选择更稳健的判据。
"""

from __future__ import annotations

import math
from typing import Sequence

from .geometry import (
    FaceMeshIndex,
    eye_aspect_ratio,
    head_pose_from_landmarks,
    head_pose_from_matrix,
    mouth_aspect_ratio,
)
from .models import DriverMetrics, Landmark, Observation

__all__ = ["MetricExtractor"]


class MetricExtractor:
    """由感知结果计算驾驶员状态指标。

    Args:
        enable_hand_metrics: 是否计算手部相对几何（需要感知层提供手部关键点）。
        enable_solvepnp_fallback: 当感知层未提供头姿矩阵时，
            是否退回用 ``solvePnP`` 由 2D 关键点反解头姿。
    """

    def __init__(
        self,
        *,
        enable_hand_metrics: bool = True,
        enable_solvepnp_fallback: bool = True,
    ) -> None:
        self._enable_hands = enable_hand_metrics
        self._enable_pnp = enable_solvepnp_fallback

    # ------------------------------------------------------------------
    def extract(self, observation: Observation, width: int, height: int) -> DriverMetrics:
        if observation.precomputed is not None:
            # 上游已给出指标（回放/自检模式）：保留其通道，仅补齐人脸存在标志。
            return observation.precomputed

        face = observation.face
        if not face and observation.face_box is None:
            return DriverMetrics(face_detected=False)

        metrics = DriverMetrics(face_detected=True)

        if face:
            metrics = self._with_eye_metrics(metrics, face)
            metrics = self._with_mouth_metrics(metrics, face)
            metrics = self._with_pitch_ratio(metrics, face)
            metrics = self._with_head_pose(metrics, observation, face, width, height)

        metrics = self._with_blendshapes(metrics, observation)
        if self._enable_hands:
            metrics = self._with_hand_metrics(metrics, observation)
        return metrics

    # ------------------------------------------------------------------
    #  眼部
    # ------------------------------------------------------------------
    @staticmethod
    def _with_eye_metrics(metrics: DriverMetrics, face: Sequence[Landmark]) -> DriverMetrics:
        left = eye_aspect_ratio(face, FaceMeshIndex.LEFT_EYE)
        right = eye_aspect_ratio(face, FaceMeshIndex.RIGHT_EYE)
        if left is None or right is None:
            single = left if left is not None else right
            return metrics.merged(ear=single, ear_left=left, ear_right=right)
        return metrics.merged(ear=(left + right) / 2.0, ear_left=left, ear_right=right)

    # ------------------------------------------------------------------
    #  嘴部
    # ------------------------------------------------------------------
    @staticmethod
    def _with_mouth_metrics(metrics: DriverMetrics, face: Sequence[Landmark]) -> DriverMetrics:
        return metrics.merged(mar=mouth_aspect_ratio(face))

    # ------------------------------------------------------------------
    #  与轴系无关的低头比例
    # ------------------------------------------------------------------
    @staticmethod
    def _with_pitch_ratio(metrics: DriverMetrics, face: Sequence[Landmark]) -> DriverMetrics:
        """``face_pitch_ratio`` = 眼-嘴纵向距离 ÷ 脸宽。

        低头时人脸的纵向投影被压缩，该比值下降；抬头时上升。
        该量与旋转矩阵无关，是低头判定的一条独立证据通道。
        """
        needed = (
            FaceMeshIndex.EYE_LEFT_CORNER,
            FaceMeshIndex.EYE_RIGHT_CORNER,
            FaceMeshIndex.MOUTH_CORNER_LEFT,
            FaceMeshIndex.MOUTH_CORNER_RIGHT,
            FaceMeshIndex.CHEEK_LEFT,
            FaceMeshIndex.CHEEK_RIGHT,
        )
        if len(face) <= max(needed):
            return metrics

        eye_mid_y = (face[FaceMeshIndex.EYE_LEFT_CORNER].y + face[FaceMeshIndex.EYE_RIGHT_CORNER].y) / 2.0
        mouth_mid_y = (
            face[FaceMeshIndex.MOUTH_CORNER_LEFT].y + face[FaceMeshIndex.MOUTH_CORNER_RIGHT].y
        ) / 2.0
        face_width = math.hypot(
            face[FaceMeshIndex.CHEEK_LEFT].x - face[FaceMeshIndex.CHEEK_RIGHT].x,
            face[FaceMeshIndex.CHEEK_LEFT].y - face[FaceMeshIndex.CHEEK_RIGHT].y,
        )
        if face_width < 1e-6:
            return metrics
        return metrics.merged(face_pitch_ratio=abs(mouth_mid_y - eye_mid_y) / face_width)

    # ------------------------------------------------------------------
    #  头部姿态
    # ------------------------------------------------------------------
    def _with_head_pose(
        self,
        metrics: DriverMetrics,
        observation: Observation,
        face: Sequence[Landmark],
        width: int,
        height: int,
    ) -> DriverMetrics:
        pose = observation.head_pose
        if pose is None and self._enable_pnp:
            pose = head_pose_from_landmarks(face, width, height)
        if pose is None:
            return metrics
        return metrics.merged(pitch=pose.pitch, yaw=pose.yaw, roll=pose.roll)

    # ------------------------------------------------------------------
    #  表情/动作系数
    # ------------------------------------------------------------------
    @staticmethod
    def _with_blendshapes(metrics: DriverMetrics, observation: Observation) -> DriverMetrics:
        if not observation.blendshapes:
            return metrics
        blink_left = observation.blendshape("eyeBlinkLeft")
        blink_right = observation.blendshape("eyeBlinkRight")
        blink = None
        if blink_left is not None and blink_right is not None:
            blink = (blink_left + blink_right) / 2.0
        elif blink_left is not None or blink_right is not None:
            blink = blink_left if blink_left is not None else blink_right
        return metrics.merged(
            eye_blink=blink,
            eye_blink_left=blink_left,
            eye_blink_right=blink_right,
            jaw_open=observation.blendshape("jawOpen"),
        )

    # ------------------------------------------------------------------
    #  手部相对几何
    # ------------------------------------------------------------------
    @staticmethod
    def _with_hand_metrics(metrics: DriverMetrics, observation: Observation) -> DriverMetrics:
        """计算手部到「脸中心」「嘴中心」的最小归一化距离。

        距离以 **脸宽** 归一化，从而对摄像头距离与驾驶员体型不敏感。

        关键点选择：手部取 ``0``（腕）、``4``（拇指尖）、``8``（食指尖）、
        ``12``（中指尖）—— 打电话、抽烟时这几个点最能代表"手/物体贴近头面部"。
        """
        if not observation.hands or not observation.face:
            return metrics
        face = observation.face
        if len(face) <= FaceMeshIndex.FACE_CENTER:
            return metrics

        face_width = 0.0
        if len(face) > FaceMeshIndex.CHEEK_LEFT:
            face_width = math.hypot(
                face[FaceMeshIndex.CHEEK_LEFT].x - face[FaceMeshIndex.CHEEK_RIGHT].x,
                face[FaceMeshIndex.CHEEK_LEFT].y - face[FaceMeshIndex.CHEEK_RIGHT].y,
            )
        if face_width < 1e-6:
            return metrics

        face_center = face[FaceMeshIndex.FACE_CENTER]
        mouth_center = _midpoint(
            face[FaceMeshIndex.MOUTH_CORNER_LEFT], face[FaceMeshIndex.MOUTH_CORNER_RIGHT]
        )

        hand_face = math.inf
        hand_mouth = math.inf
        for hand in observation.hands:
            for point in _hand_tips(hand):
                hand_face = min(hand_face, point.distance_to(face_center))
                hand_mouth = min(hand_mouth, point.distance_to(mouth_center))

        return metrics.merged(
            hand_face_ratio=None if math.isinf(hand_face) else hand_face / face_width,
            hand_mouth_ratio=None if math.isinf(hand_mouth) else hand_mouth / face_width,
        )


# ---------------------------------------------------------------------------
#  模块级小工具
# ---------------------------------------------------------------------------
#: 手部关键点中具有代表性的指尖/腕部索引（MediaPipe HandLandmarker 为 21 点）
_HAND_TIP_INDICES = (0, 4, 8, 12)


def _hand_tips(hand: Sequence[Landmark]) -> list[Landmark]:
    return [hand[i] for i in _HAND_TIP_INDICES if i < len(hand)]


def _midpoint(a: Landmark, b: Landmark) -> Landmark:
    return Landmark(x=(a.x + b.x) / 2.0, y=(a.y + b.y) / 2.0)
