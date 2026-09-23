"""几何算法测试。

这是整个测试套件中最重要的一组文件：EAR / MAR / 欧拉角分解是所有
上层判定的数值基础，一旦出错会以"误报"或"漏报"的形式表现出来，
极难通过端到端观察定位。

因此这里采用 **性质测试（property-based thinking）** 而不是简单的
"输入 A 得到 B"：断言公式应当满足的数学关系（往返一致、叠加近似、
尺度不变、退化安全），这类断言更难被"改代码后顺手改期望值"破坏。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from driver_guard.domain.geometry import (
    FaceMeshIndex,
    euler_to_matrix,
    eye_aspect_ratio,
    head_pose_from_landmarks,
    head_pose_from_matrix,
    mouth_aspect_ratio,
    rotation_matrix_to_euler,
)
from driver_guard.domain.models import Landmark, Observation


# ===========================================================================
#  欧拉角分解
# ===========================================================================
class TestEulerRoundTrip:
    """往返一致性：euler → matrix → euler 必须还原。"""

    @pytest.mark.parametrize(
        "angles",
        [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (0.0, 15.0, 0.0),
            (0.0, 0.0, 20.0),
            (12.0, -8.0, 6.0),
            (-25.0, 30.0, -18.0),
            (45.0, 45.0, 0.0),
            (-33.3, 12.7, 27.9),
        ],
    )
    def test_round_trip(self, angles: tuple[float, float, float]) -> None:
        matrix = euler_to_matrix(*angles)
        recovered = rotation_matrix_to_euler(matrix)
        assert recovered == pytest.approx(angles, abs=1e-9)

    def test_round_trip_over_grid(self) -> None:
        """在角网格上扫一遍，确认没有任何盲区。"""
        worst = 0.0
        for pitch in range(-60, 61, 15):
            for yaw in range(-60, 61, 15):
                for roll in range(-60, 61, 20):
                    angles = (float(pitch), float(yaw), float(roll))
                    recovered = rotation_matrix_to_euler(euler_to_matrix(*angles))
                    error = max(abs(a - b) for a, b in zip(angles, recovered, strict=True))
                    worst = max(worst, error)
        assert worst < 1e-9, f"最大往返误差 {worst}"

    def test_matrix_is_orthonormal(self) -> None:
        """生成的必须是合法旋转矩阵（正交且行列式为 +1）。"""
        matrix = euler_to_matrix(17.0, -23.0, 41.0)
        assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
        assert float(np.linalg.det(matrix)) == pytest.approx(1.0, abs=1e-12)


class TestEulerSuperposition:
    """"叠加一个相机系旋转"这一性质的精度上界（本系统的关键假设）。"""

    @pytest.mark.parametrize("axis", ["pitch", "yaw", "roll"])
    @pytest.mark.parametrize("delta", [-30.0, -15.0, -5.0, 5.0, 15.0, 30.0])
    def test_single_axis_superposition(self, axis: str, delta: float) -> None:
        """在任意基准姿态上叠加绕单轴旋转，分解结果应增加同样的量。"""
        base = euler_to_matrix(11.0, -20.0, 3.5)
        angle = math.radians(delta)
        c, s = math.cos(angle), math.sin(angle)
        delta_matrix = {
            "pitch": np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float),
            "yaw": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float),
            "roll": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float),
        }[axis]

        before = rotation_matrix_to_euler(base)
        after = rotation_matrix_to_euler(delta_matrix @ base)
        index = {"pitch": 0, "yaw": 1, "roll": 2}[axis]

        assert (after[index] - before[index]) == pytest.approx(delta, abs=2.0)

    def test_identity_delta_recovers_base(self) -> None:
        base = euler_to_matrix(-7.0, 19.0, -4.0)
        assert rotation_matrix_to_euler(np.eye(3) @ base) == pytest.approx(
            (-7.0, 19.0, -4.0), abs=1e-9
        )


class TestEulerDegenerate:
    """退化输入的健壮性：绝不能抛异常，否则会中断采集循环。"""

    @pytest.mark.parametrize("bad", [None, np.zeros((2, 2)), np.zeros((4, 5)), np.zeros(3)])
    def test_head_pose_from_invalid_matrix(self, bad) -> None:
        assert head_pose_from_matrix(bad) is None

    def test_gimbal_lock_branch(self) -> None:
        """yaw = ±90° 时 roll 与 pitch 退化，函数应返回有限值而非 NaN。"""
        matrix = euler_to_matrix(10.0, 90.0, 0.0)
        pitch, yaw, roll = rotation_matrix_to_euler(matrix)
        assert all(math.isfinite(v) for v in (pitch, yaw, roll))
        assert abs(yaw) == pytest.approx(90.0, abs=1e-6)

    def test_non_square_but_flattenable_rejected(self) -> None:
        assert head_pose_from_matrix(np.arange(9, dtype=float)) is None


# ===========================================================================
#  EAR
# ===========================================================================
def _eye_points(
    indices: tuple[int, ...],
    *,
    horizontal: float = 0.10,
    vertical: float = 0.03,
) -> list[Landmark]:
    """按 6 点顺序构造一只眼睛：p1 外眼角、p4 内眼角、其余为上下眼睑。"""
    landmarks = [Landmark(x=0.0, y=0.0) for _ in range(max(indices) + 1)]
    p1, p2, p3, p4, p5, p6 = indices
    landmarks[p1] = Landmark(x=0.0, y=0.5)
    landmarks[p4] = Landmark(x=horizontal, y=0.5)
    landmarks[p2] = Landmark(x=horizontal * 0.3, y=0.5 - vertical)
    landmarks[p3] = Landmark(x=horizontal * 0.7, y=0.5 - vertical)
    landmarks[p5] = Landmark(x=horizontal * 0.7, y=0.5 + vertical)
    landmarks[p6] = Landmark(x=horizontal * 0.3, y=0.5 + vertical)
    return landmarks


class TestEyeAspectRatio:
    def test_formula_matches_definition(self) -> None:
        landmarks = _eye_points(FaceMeshIndex.RIGHT_EYE, horizontal=0.10, vertical=0.03)
        # EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|) = (0.06 + 0.06) / 0.20
        assert eye_aspect_ratio(landmarks, FaceMeshIndex.RIGHT_EYE) == pytest.approx(0.6)

    def test_open_eye_above_closed_eye(self) -> None:
        opened = _eye_points(FaceMeshIndex.RIGHT_EYE, vertical=0.035)
        closed = _eye_points(FaceMeshIndex.RIGHT_EYE, vertical=0.004)
        assert eye_aspect_ratio(opened, FaceMeshIndex.RIGHT_EYE) > (
            eye_aspect_ratio(closed, FaceMeshIndex.RIGHT_EYE)
        )

    def test_scale_invariance(self) -> None:
        """EAR 是比值：整体放大 10 倍不应改变结果（对距离不敏感的关键性质）。"""
        small = _eye_points(FaceMeshIndex.RIGHT_EYE)
        large = [
            Landmark(x=p.x * 10.0, y=p.y * 10.0) for p in small
        ]
        assert eye_aspect_ratio(small, FaceMeshIndex.RIGHT_EYE) == pytest.approx(
            eye_aspect_ratio(large, FaceMeshIndex.RIGHT_EYE)
        )

    def test_left_and_right_use_different_indices(self) -> None:
        landmarks = _eye_points(FaceMeshIndex.RIGHT_EYE, horizontal=0.10, vertical=0.03)
        # 左侧索引处的点尚未定义（仍是 (0,0)），因此计算会退化
        assert eye_aspect_ratio(landmarks, FaceMeshIndex.LEFT_EYE) is None

    def test_insufficient_landmarks_returns_none(self) -> None:
        assert eye_aspect_ratio([Landmark(x=0, y=0)] * 5, FaceMeshIndex.RIGHT_EYE) is None
        assert eye_aspect_ratio([], FaceMeshIndex.RIGHT_EYE) is None

    def test_zero_width_returns_none_instead_of_dividing_by_zero(self) -> None:
        indices = FaceMeshIndex.RIGHT_EYE
        landmarks = [Landmark(x=0.5, y=0.5) for _ in range(max(indices) + 1)]
        assert eye_aspect_ratio(landmarks, indices) is None


# ===========================================================================
#  MAR
# ===========================================================================
class TestMouthAspectRatio:
    @staticmethod
    def _mouth(opening: float, width: float = 0.10) -> list[Landmark]:
        landmarks = [Landmark(x=0.0, y=0.0) for _ in range(311)]
        landmarks[FaceMeshIndex.MOUTH_LEFT] = Landmark(x=0.4, y=0.6)
        landmarks[FaceMeshIndex.MOUTH_RIGHT] = Landmark(x=0.4 + width, y=0.6)
        landmarks[FaceMeshIndex.LIP_TOP_INNER] = Landmark(x=0.4 + width / 2, y=0.6 - opening / 2)
        landmarks[FaceMeshIndex.LIP_BOTTOM_INNER] = Landmark(x=0.4 + width / 2, y=0.6 + opening / 2)
        return landmarks

    def test_matches_definition(self) -> None:
        landmarks = self._mouth(opening=0.05, width=0.10)
        assert mouth_aspect_ratio(landmarks) == pytest.approx(0.5)

    def test_yawn_exceeds_talk(self) -> None:
        assert mouth_aspect_ratio(self._mouth(0.08)) > mouth_aspect_ratio(self._mouth(0.02))

    def test_insufficient_landmarks(self) -> None:
        assert mouth_aspect_ratio([Landmark(x=0, y=0)] * 10) is None

    def test_zero_width_returns_none(self) -> None:
        landmarks = self._mouth(opening=0.05, width=0.0)
        assert mouth_aspect_ratio(landmarks) is None


# ===========================================================================
#  头姿的降级路径
# ===========================================================================
class TestHeadPoseFromLandmarks:
    def test_returns_none_when_opencv_missing(self, monkeypatch) -> None:
        """opencv 不可用时应降级为 None，而不是抛 ImportError 中断采集。"""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert head_pose_from_landmarks([Landmark(0.5, 0.5)] * 478, 640, 480) is None

    def test_insufficient_landmarks(self) -> None:
        assert head_pose_from_landmarks([Landmark(0.5, 0.5)] * 10, 640, 480) is None
        assert head_pose_from_landmarks([], 640, 480) is None

    def test_returns_finite_pose_for_plausible_landmarks(self) -> None:
        """用一组近似正面的人脸点跑通 solvePnP 分支，断言结果有限。"""
        face = [Landmark(x=0.5, y=0.5, z=0.0) for _ in range(478)]
        face[FaceMeshIndex.NOSE_TIP] = Landmark(0.50, 0.55)
        face[FaceMeshIndex.CHIN] = Landmark(0.50, 0.78)
        face[FaceMeshIndex.EYE_LEFT_OUTER] = Landmark(0.62, 0.44)
        face[FaceMeshIndex.EYE_RIGHT_OUTER] = Landmark(0.38, 0.44)
        face[FaceMeshIndex.MOUTH_LEFT_OUTER] = Landmark(0.58, 0.66)
        face[FaceMeshIndex.MOUTH_RIGHT_OUTER] = Landmark(0.42, 0.66)

        pose = head_pose_from_landmarks(face, 640, 480)
        if pose is None:
            pytest.skip("当前 OpenCV 未提供 solvePnP")
        assert all(math.isfinite(v) for v in (pose.pitch, pose.yaw, pose.roll))


# ===========================================================================
#  Observation 辅助
# ===========================================================================
def test_observation_helpers() -> None:
    empty = Observation()
    assert not empty.has_face
    assert not empty.has_hands
    assert empty.blendshape("eyeBlinkLeft") is None

    populated = Observation(
        face=(Landmark(0.5, 0.5),),
        blendshapes={"eyeBlinkLeft": 0.9},
        hands=((Landmark(0.6, 0.4),),),
    )
    assert populated.has_face
    assert populated.has_hands
    assert populated.blendshape("eyeBlinkLeft") == 0.9
