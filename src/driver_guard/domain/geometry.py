"""几何度量算法：EAR / MAR / 头部姿态。

这些函数是纯粹的"数学工具"，不持有状态、不依赖任何推理库，
因此可以被高密度单元测试覆盖，也便于在文档中给出可复核的公式。

参考
----
* Soukupová & Čech (2016), *Real-Time Eye Blink Detection using Facial Landmarks*
  —— EAR 的经典定义与阈值。
* OpenCV ``solvePnP`` + ``RQDecomp3x3`` —— 由 2D/3D 对应点求头部欧拉角。
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .models import HeadPose, Landmark

__all__ = [
    "FaceMeshIndex",
    "eye_aspect_ratio",
    "mouth_aspect_ratio",
    "euler_to_matrix",
    "rotation_matrix_to_euler",
    "head_pose_from_matrix",
    "head_pose_from_landmarks",
]


class FaceMeshIndex:
    """MediaPipe FaceMesh（478 点）关键点索引常量。

    索引取自 MediaPipe 官方 FaceMesh 拓扑图；集中定义避免魔法数字散落各处。
    """

    #: 右眼 6 点环：外眼角、上眼睑外、上眼睑内、内眼角、下眼睑内、下眼睑外
    RIGHT_EYE = (33, 160, 158, 133, 153, 144)
    #: 左眼 6 点环（顺序同上）
    LEFT_EYE = (362, 385, 387, 263, 373, 380)

    #: 嘴部：内唇角 / 上唇内缘 / 下唇内缘
    MOUTH_LEFT = 78
    MOUTH_RIGHT = 308
    LIP_TOP_INNER = 13
    LIP_BOTTOM_INNER = 14

    #: 头部姿态估计所用的 6 个稳定点（鼻尖、下巴、双外眼角、双嘴角）
    NOSE_TIP = 1
    CHIN = 152
    EYE_LEFT_OUTER = 263
    EYE_RIGHT_OUTER = 33
    MOUTH_LEFT_OUTER = 291
    MOUTH_RIGHT_OUTER = 61

    #: 脸部中心（鼻梁附近），用于"手贴近脸"的距离基准
    FACE_CENTER = 168

    #: 眼睛连线两侧点（用于求眼睛中心）
    EYE_LEFT_CORNER = 263
    EYE_RIGHT_CORNER = 33
    #: 嘴部两侧点（用于求嘴部中心）
    MOUTH_CORNER_LEFT = 291
    MOUTH_CORNER_RIGHT = 61
    #: 脸宽基准：两侧颧点
    CHEEK_LEFT = 454
    CHEEK_RIGHT = 234
    #: 额头中心与下巴（用于纵向尺度参考）
    FOREHEAD_CENTER = 10


# ---------------------------------------------------------------------------
#  眼 / 嘴 纵横比
# ---------------------------------------------------------------------------
def _dist(a: Landmark, b: Landmark) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def eye_aspect_ratio(landmarks: Sequence[Landmark], indices: Sequence[int]) -> float | None:
    """计算单眼 EAR。

    给定 6 个眼部关键点 ``p1..p6``（见 :attr:`FaceMeshIndex.RIGHT_EYE` 的顺序）::

        EAR = (‖p2-p6‖ + ‖p3-p5‖) / (2 · ‖p1-p4‖)

    其中 ``p1``、``p4`` 为眼角，分子是两组上下眼睑距离之和，分母是眼宽。
    眼睛睁开时角膜弧度大、EAR 约为 0.25~0.35；闭合时上下眼睑贴合，EAR 骤降至 0.1 附近。
    由于是"比值"，EAR 对人与摄像头的距离不敏感，是其被广泛采用的原因。

    Returns:
        归一化 EAR；索引缺失或退化（分母为 0）时返回 ``None``。
    """
    if len(landmarks) <= max(indices):
        return None
    try:
        p1, p2, p3, p4, p5, p6 = (landmarks[i] for i in indices)
    except IndexError:  # pragma: no cover - 由上面的长度检查保证
        return None

    horizontal = _dist(p1, p4)
    if horizontal < 1e-6:
        return None
    vertical = _dist(p2, p6) + _dist(p3, p5)
    return float(vertical / (2.0 * horizontal))


def mouth_aspect_ratio(landmarks: Sequence[Landmark]) -> float | None:
    """计算嘴部纵横比 MAR = 上下唇内缘距离 ÷ 嘴角宽度。

    打哈欠时口部竖向张开幅度显著大于说话，MAR 通常从 < 0.3 升到 > 0.5。
    """
    needed = (FaceMeshIndex.LIP_TOP_INNER, FaceMeshIndex.LIP_BOTTOM_INNER,
              FaceMeshIndex.MOUTH_LEFT, FaceMeshIndex.MOUTH_RIGHT)
    if len(landmarks) <= max(needed):
        return None
    top = landmarks[FaceMeshIndex.LIP_TOP_INNER]
    bottom = landmarks[FaceMeshIndex.LIP_BOTTOM_INNER]
    left = landmarks[FaceMeshIndex.MOUTH_LEFT]
    right = landmarks[FaceMeshIndex.MOUTH_RIGHT]
    width = _dist(left, right)
    if width < 1e-6:
        return None
    return float(_dist(top, bottom) / width)


# ---------------------------------------------------------------------------
#  头部姿态
# ---------------------------------------------------------------------------
def euler_to_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """把欧拉角 ``(pitch, yaw, roll)``（度）组合为旋转矩阵。

    与 :func:`rotation_matrix_to_euler` 互为逆运算，采用同一约定::

        R = Rz(roll) · Ry(yaw) · Rx(pitch)

    提供该函数的主要目的是让"往返一致性"成为可断言的性质：
    ``rotation_matrix_to_euler(euler_to_matrix(p, y, r)) == (p, y, r)``，
    这是单元测试中最有价值的检查，也是本模块数值正确性的锚点。
    """
    p = math.radians(pitch)
    y = math.radians(yaw)
    r = math.radians(roll)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    cr, sr = math.cos(r), math.sin(r)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def rotation_matrix_to_euler(rotation: np.ndarray) -> tuple[float, float, float]:
    """把旋转矩阵分解为欧拉角 ``(pitch, yaw, roll)``，单位为度。

    采用 OpenCV ``RQDecomp3x3`` 的分解约定 ``R = Rz(roll) · Ry(yaw) · Rx(pitch)``，
    展开后可得::

        pitch = atan2(R[2,1], R[2,2])
        yaw   = atan2(-R[2,0], √(R[0,0]² + R[1,0]²))
        roll  = atan2(R[1,0], R[0,0])

    已验证的数学性质
    ----------------

    1. **往返一致性（严格等式）**：与 :func:`euler_to_matrix` 互为逆运算::

           rotation_matrix_to_euler(euler_to_matrix(p, y, r)) == (p, y, r)

    2. **叠加近似（实用精度）**：在已有姿态上叠加一个相机系旋转 ``Δ`` 时::

           euler(Δ ⊗ R) ≈ euler(R) + Δ

       实测：``|角度| ≤ 30°`` 时误差在 **2° 以内**（角度越大，欧拉角之间的
       耦合越明显）。这说明本函数足以可靠地度量"姿态变化量"，但不能
       当成精确的绝对角度计使用。

    以上两条均由 ``tests/test_geometry.py`` 逐轴、逐角度校验。

    关于绝对角语义（重要）
    ----------------------
    MediaPipe 规范人脸模型的轴系与 OpenCV 相机轴系（X 右 / Y 下 / Z 前）
    **并不一致**，且官方未公开说明该差异。实测（见 ``.calib/``）表明
    直接套用相机轴语义会得到错误的方位判断。

    因此本系统 **不依赖绝对角的物理含义**：三轴角度仅作为"相对量"使用，
    规则判定一律基于 **相对驾驶员个人中立位的偏离量**（由
    :class:`~driver_guard.domain.calibration.NeutralPoseCalibrator`
    在会话开头自动标定）。该设计对轴系差异、镜头安装角度、驾驶员坐姿差异
    全部免疫，且比硬编码阈值更贴合个体。

    Returns:
        ``(pitch, yaw, roll)`` 三元组，单位为度。
    """
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    sy = math.hypot(float(r[0, 0]), float(r[1, 0]))
    if sy > 1e-6:
        pitch = math.atan2(float(r[2, 1]), float(r[2, 2]))
        yaw = math.atan2(-float(r[2, 0]), sy)
        roll = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:  # 万向节死锁：yaw = ±90°，roll 与 pitch 退化，约定 roll = 0
        pitch = math.atan2(-float(r[1, 2]), float(r[1, 1]))
        yaw = math.atan2(-float(r[2, 0]), sy)
        roll = 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def head_pose_from_matrix(matrix: np.ndarray | None) -> HeadPose | None:
    """由 MediaPipe ``facial_transformation_matrixes`` 输出直接构造头部姿态。

    MediaPipe 在 ``output_facial_transformation_matrixes=True`` 时会返回 4×4
    齐次变换矩阵，把规范人脸模型映射到相机坐标系；其左上 3×3 子块即旋转矩阵。
    相比 ``solvePnP`` 需要手工指定 3D 模型点，该路径更稳定，故列为**首选**。

    实测（``.calib/which_side.py``）：关键点索引 ``234`` 位于画面左侧
    （``x=0.397``），``454`` 位于画面右侧（``x=0.655``）；偏航角与该两点的
    相对深度差 ``z(234) - z(454)`` 符号一致，说明该矩阵的偏航分量确实编码了
    真实的左右转脸信息，而非镜像或逆变换。
    """
    if matrix is None:
        return None
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape not in {(4, 4), (3, 3)}:
        return None
    rotation = arr[:3, :3]
    pitch, yaw, roll = rotation_matrix_to_euler(rotation)
    return HeadPose(
        pitch=pitch,
        yaw=yaw,
        roll=roll,
        matrix=tuple(tuple(float(v) for v in row) for row in rotation),
    )


#: 规范 3D 人脸模型（单位：毫米），顺序与 :meth:`head_pose_from_landmarks` 的 2D 点一一对应。
#: 数值取自通用人脸尺寸统计（唇间距约 57.3mm、眼角间距约 86.6mm 等）。
_CANONICAL_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),          # 鼻尖
        (0.0, -63.6, -12.5),      # 下巴
        (-43.3, 32.7, -26.0),     # 左外眼角（被检者左侧）
        (43.3, 32.7, -26.0),      # 右外眼角
        (-28.9, -28.9, -24.1),    # 左嘴角
        (28.9, -28.9, -24.1),     # 右嘴角
    ],
    dtype=np.float64,
)


def head_pose_from_landmarks(
    landmarks: Sequence[Landmark],
    width: int,
    height: int,
) -> HeadPose | None:
    """**降级路径**：仅有 2D 关键点时，用 ``solvePnP`` 反解头部姿态。

    当感知实现不提供变换矩阵（例如自定义模型、录像回放）时使用。
    需要 ``opencv-contrib-python``；导入失败时返回 ``None`` 而非抛异常，
    以保证感知链路在部分依赖缺失时仍能降级运行。
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - 环境缺 opencv 时
        return None

    if not landmarks:
        return None
    try:
        image_points = np.array(
            [
                _to_pixel(landmarks[FaceMeshIndex.NOSE_TIP], width, height),
                _to_pixel(landmarks[FaceMeshIndex.CHIN], width, height),
                _to_pixel(landmarks[FaceMeshIndex.EYE_LEFT_OUTER], width, height),
                _to_pixel(landmarks[FaceMeshIndex.EYE_RIGHT_OUTER], width, height),
                _to_pixel(landmarks[FaceMeshIndex.MOUTH_LEFT_OUTER], width, height),
                _to_pixel(landmarks[FaceMeshIndex.MOUTH_RIGHT_OUTER], width, height),
            ],
            dtype=np.float64,
        )
    except IndexError:
        return None

    focal = float(width)
    center = (width / 2.0, height / 2.0)
    camera_matrix = np.array(
        [[focal, 0.0, center[0]], [0.0, focal, center[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, _ = cv2.solvePnP(
        _CANONICAL_MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    rotation, _ = cv2.Rodrigues(rotation_vector)
    pose = head_pose_from_matrix(rotation)
    return pose


def _to_pixel(landmark: Landmark, width: int, height: int) -> tuple[float, float]:
    return landmark.x * width, landmark.y * height
