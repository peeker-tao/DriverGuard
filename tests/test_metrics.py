"""指标提取测试（管道第 2 级）。

用 **构造的关键点** 生成可控的人脸，从而精确验证：
"关键点 → 指标"这一步是否忠实，以及各条指标缺失时是否正确置空
（置空而非填 0 是"优雅降级"的前提）。
"""

from __future__ import annotations

import math

import pytest

from driver_guard.domain.geometry import FaceMeshIndex
from driver_guard.domain.metrics import MetricExtractor
from driver_guard.domain.models import DriverMetrics, HeadPose, Landmark, Observation

WIDTH, HEIGHT = 640, 480


# ===========================================================================
#  合成人脸
# ===========================================================================
def make_face(
    *,
    horizontal: float = 0.10,
    eye_opening: float = 0.03,
    mouth_opening: float = 0.03,
    mouth_width: float = 0.10,
    eye_to_mouth: float = 0.20,
    cheek_span: float = 0.24,
) -> tuple[Landmark, ...]:
    """构造一副具有指定几何的 478 点人脸。

    参数直接对应各条指标的分子/分母，因此测试可以"从几何推指标"，
    而不是把实现里的数字抄一遍。
    """
    face = [Landmark(x=0.5, y=0.5) for _ in range(478)]

    # --- 眼睛 ---------------------------------------------------------
    for indices, base_x in ((FaceMeshIndex.RIGHT_EYE, 0.35), (FaceMeshIndex.LEFT_EYE, 0.55)):
        p1, p2, p3, p4, p5, p6 = indices
        face[p1] = Landmark(x=base_x, y=0.42)
        face[p4] = Landmark(x=base_x + horizontal, y=0.42)
        face[p2] = Landmark(x=base_x + horizontal * 0.3, y=0.42 - eye_opening)
        face[p3] = Landmark(x=base_x + horizontal * 0.7, y=0.42 - eye_opening)
        face[p5] = Landmark(x=base_x + horizontal * 0.7, y=0.42 + eye_opening)
        face[p6] = Landmark(x=base_x + horizontal * 0.3, y=0.42 + eye_opening)

    # --- 眼中心 / 眼角 / 颧点 -----------------------------------------
    face[FaceMeshIndex.EYE_RIGHT_CORNER] = Landmark(x=0.35, y=0.42)
    face[FaceMeshIndex.EYE_LEFT_CORNER] = Landmark(x=0.55 + horizontal, y=0.42)
    face[FaceMeshIndex.CHEEK_RIGHT] = Landmark(x=0.5 - cheek_span / 2, y=0.50)
    face[FaceMeshIndex.CHEEK_LEFT] = Landmark(x=0.5 + cheek_span / 2, y=0.50)

    # --- 嘴 -----------------------------------------------------------
    mouth_y = 0.42 + eye_to_mouth
    face[FaceMeshIndex.MOUTH_CORNER_RIGHT] = Landmark(x=0.5 - mouth_width / 2, y=mouth_y)
    face[FaceMeshIndex.MOUTH_CORNER_LEFT] = Landmark(x=0.5 + mouth_width / 2, y=mouth_y)
    face[FaceMeshIndex.MOUTH_RIGHT] = Landmark(x=0.5 - mouth_width / 2, y=mouth_y)
    face[FaceMeshIndex.MOUTH_LEFT] = Landmark(x=0.5 + mouth_width / 2, y=mouth_y)
    face[FaceMeshIndex.LIP_TOP_INNER] = Landmark(x=0.5, y=mouth_y - mouth_opening / 2)
    face[FaceMeshIndex.LIP_BOTTOM_INNER] = Landmark(x=0.5, y=mouth_y + mouth_opening / 2)

    # --- 脸中心 / 鼻 --------------------------------------------------
    face[FaceMeshIndex.FACE_CENTER] = Landmark(x=0.5, y=0.46)
    face[FaceMeshIndex.NOSE_TIP] = Landmark(x=0.5, y=0.50)
    face[FaceMeshIndex.CHIN] = Landmark(x=0.5, y=0.80)
    return tuple(face)


def make_hand(cx: float = 0.60, cy: float = 0.40) -> tuple[Landmark, ...]:
    """构造一只手；``0/4/8/12`` 是提取器实际使用的指尖索引。"""
    hand = [Landmark(x=cx + 0.05, y=cy + 0.05) for _ in range(21)]
    hand[0] = Landmark(x=cx, y=cy)
    hand[4] = Landmark(x=cx + 0.01, y=cy - 0.01)
    hand[8] = Landmark(x=cx + 0.02, y=cy - 0.02)
    hand[12] = Landmark(x=cx + 0.03, y=cy - 0.03)
    return tuple(hand)


@pytest.fixture
def extractor() -> MetricExtractor:
    return MetricExtractor()


# ===========================================================================
#  眼部
# ===========================================================================
class TestEyeMetrics:
    def test_ear_matches_geometry(self, extractor: MetricExtractor) -> None:
        face = make_face(horizontal=0.10, eye_opening=0.03)
        metrics = extractor.extract(Observation(face=face), WIDTH, HEIGHT)
        # (2 * 0.06) / (2 * 0.10) = 0.6
        assert metrics.ear == pytest.approx(0.6)
        assert metrics.ear_left == pytest.approx(0.6)
        assert metrics.ear_right == pytest.approx(0.6)

    def test_closed_eye_yields_low_ear(self, extractor: MetricExtractor) -> None:
        face = make_face(horizontal=0.10, eye_opening=0.005)
        metrics = extractor.extract(Observation(face=face), WIDTH, HEIGHT)
        assert metrics.ear is not None and metrics.ear < 0.21

    def test_ear_is_mean_of_both_eyes(self, extractor: MetricExtractor) -> None:
        face = list(make_face(horizontal=0.10, eye_opening=0.03))
        # 单独把左眼压扁：右眼 EAR = 0.6，左眼 EAR = 0.1
        for index in FaceMeshIndex.LEFT_EYE[1:]:
            face[index] = Landmark(x=face[index].x, y=0.42)
        metrics = extractor.extract(Observation(face=tuple(face)), WIDTH, HEIGHT)
        assert metrics.ear_left == pytest.approx(0.0, abs=1e-6)
        assert metrics.ear_right == pytest.approx(0.6)
        assert metrics.ear == pytest.approx(0.3)


# ===========================================================================
#  嘴部与低头比例
# ===========================================================================
class TestMouthAndPitchRatio:
    def test_mar_matches_geometry(self, extractor: MetricExtractor) -> None:
        face = make_face(mouth_opening=0.05, mouth_width=0.10)
        metrics = extractor.extract(Observation(face=face), WIDTH, HEIGHT)
        assert metrics.mar == pytest.approx(0.5)

    def test_face_pitch_ratio_matches_geometry(self, extractor: MetricExtractor) -> None:
        face = make_face(eye_to_mouth=0.20, cheek_span=0.25)
        metrics = extractor.extract(Observation(face=face), WIDTH, HEIGHT)
        # 眼中心 y = 0.42（两眼角平均）到嘴中心 y = 0.62，脸宽 = 0.25
        assert metrics.face_pitch_ratio == pytest.approx(0.20 / 0.25)

    def test_head_down_shrinks_pitch_ratio(self, extractor: MetricExtractor) -> None:
        """低头 → 眼-嘴纵向距离被压缩 → 比值下降（与轴系无关的判据）。"""
        normal = extractor.extract(
            Observation(face=make_face(eye_to_mouth=0.20)), WIDTH, HEIGHT
        )
        head_down = extractor.extract(
            Observation(face=make_face(eye_to_mouth=0.12)), WIDTH, HEIGHT
        )
        assert (
            normal.face_pitch_ratio is not None
            and head_down.face_pitch_ratio is not None
            and head_down.face_pitch_ratio < normal.face_pitch_ratio
        )


# ===========================================================================
#  头姿
# ===========================================================================
class TestHeadPoseMetrics:
    def test_uses_provided_head_pose(self, extractor: MetricExtractor) -> None:
        pose = HeadPose(pitch=12.5, yaw=-8.0, roll=3.0)
        metrics = extractor.extract(Observation(face=make_face(), head_pose=pose),
                                    WIDTH, HEIGHT)
        assert (metrics.pitch, metrics.yaw, metrics.roll) == (12.5, -8.0, 3.0)

    def test_pose_absent_leaves_fields_none(self, extractor: MetricExtractor) -> None:
        """没有姿态矩阵且禁用 solvePnP 时，三个角度必须为 None 而非 0。"""
        extractor_no_pnp = MetricExtractor(enable_solvepnp_fallback=False)
        metrics = extractor_no_pnp.extract(Observation(face=make_face()), WIDTH, HEIGHT)
        assert metrics.pitch is None
        assert metrics.yaw is None
        assert metrics.roll is None


# ===========================================================================
#  表情系数
# ===========================================================================
class TestBlendshapeMetrics:
    def test_averages_left_and_right_blink(self, extractor: MetricExtractor) -> None:
        observation = Observation(
            face=make_face(),
            blendshapes={"eyeBlinkLeft": 0.8, "eyeBlinkRight": 0.4, "jawOpen": 0.25},
        )
        metrics = extractor.extract(observation, WIDTH, HEIGHT)
        assert metrics.eye_blink == pytest.approx(0.6)
        assert metrics.eye_blink_left == pytest.approx(0.8)
        assert metrics.eye_blink_right == pytest.approx(0.4)
        assert metrics.jaw_open == pytest.approx(0.25)

    def test_single_side_falls_back_to_available_value(self, extractor: MetricExtractor) -> None:
        observation = Observation(face=make_face(), blendshapes={"eyeBlinkLeft": 0.7})
        metrics = extractor.extract(observation, WIDTH, HEIGHT)
        assert metrics.eye_blink == pytest.approx(0.7)
        assert metrics.eye_blink_right is None

    def test_absent_blendshapes_leave_none(self, extractor: MetricExtractor) -> None:
        metrics = extractor.extract(Observation(face=make_face()), WIDTH, HEIGHT)
        assert metrics.eye_blink is None
        assert metrics.jaw_open is None


# ===========================================================================
#  手部相对几何
# ===========================================================================
class TestHandMetrics:
    def test_hand_near_face_yields_small_ratio(self, extractor: MetricExtractor) -> None:
        face = make_face(cheek_span=0.25)
        near = extractor.extract(
            Observation(face=face, hands=(make_hand(0.50, 0.46),)), WIDTH, HEIGHT
        )
        far = extractor.extract(
            Observation(face=face, hands=(make_hand(0.95, 0.10),)), WIDTH, HEIGHT
        )
        assert near.hand_face_ratio is not None and far.hand_face_ratio is not None
        assert near.hand_face_ratio < far.hand_face_ratio
        assert near.hand_face_ratio < 0.55   # 应能触发打电话阈值

    def test_no_hands_leaves_ratios_none(self, extractor: MetricExtractor) -> None:
        metrics = extractor.extract(Observation(face=make_face()), WIDTH, HEIGHT)
        assert metrics.hand_face_ratio is None
        assert metrics.hand_mouth_ratio is None

    def test_hand_metrics_can_be_disabled(self) -> None:
        extractor = MetricExtractor(enable_hand_metrics=False)
        metrics = extractor.extract(
            Observation(face=make_face(), hands=(make_hand(),)), WIDTH, HEIGHT
        )
        assert metrics.hand_face_ratio is None

    def test_ratio_normalised_by_face_width(self, extractor: MetricExtractor) -> None:
        """同一相对位置、不同脸宽（即不同距离）应得到相近的归一化距离。"""
        small = extractor.extract(
            Observation(face=make_face(cheek_span=0.12), hands=(make_hand(0.56, 0.46),)),
            WIDTH, HEIGHT,
        )
        large = extractor.extract(
            Observation(face=make_face(cheek_span=0.24), hands=(make_hand(0.62, 0.46),)),
            WIDTH, HEIGHT,
        )
        assert small.hand_face_ratio is not None and large.hand_face_ratio is not None
        assert small.hand_face_ratio == pytest.approx(large.hand_face_ratio, abs=0.05)


# ===========================================================================
#  降级与透传
# ===========================================================================
class TestDegradation:
    def test_no_face_marks_not_detected(self, extractor: MetricExtractor) -> None:
        metrics = extractor.extract(Observation(), WIDTH, HEIGHT)
        assert metrics.face_detected is False
        assert metrics.ear is None

    def test_face_box_only_still_counts_as_detected(self, extractor: MetricExtractor) -> None:
        """降级感知只给包围盒时，应判定"有人在座"，只是没有细节指标。"""
        from driver_guard.domain.models import BoundingBox

        observation = Observation(face_box=BoundingBox(0.3, 0.2, 0.7, 0.8),
                                  provider="haar-cascade")
        metrics = extractor.extract(observation, WIDTH, HEIGHT)
        assert metrics.face_detected is True
        assert metrics.ear is None
        assert metrics.pitch is None

    def test_precomputed_metrics_pass_through(self, extractor: MetricExtractor) -> None:
        """回放/自检模式给出的指标应原样采用，不做二次计算。"""
        given = DriverMetrics(face_detected=True, ear=0.123, mar=0.456)
        metrics = extractor.extract(Observation(precomputed=given), WIDTH, HEIGHT)
        assert metrics is given

    def test_missing_landmarks_do_not_raise(self, extractor: MetricExtractor) -> None:
        """关键点数量不足时必须安全返回，不能抛异常打断采集循环。"""
        short = tuple(Landmark(x=0.5, y=0.5) for _ in range(20))
        metrics = extractor.extract(Observation(face=short), WIDTH, HEIGHT)
        assert metrics.face_detected is True
        assert metrics.ear is None
        assert metrics.mar is None
        assert metrics.face_pitch_ratio is None


# ===========================================================================
#  指标容器本身
# ===========================================================================
class TestDriverMetricsContainer:
    def test_as_dict_only_exports_numbers(self) -> None:
        metrics = DriverMetrics(face_detected=True, ear=0.3, mar=None, pitch=5.0)
        payload = metrics.as_dict()
        assert payload["ear"] == pytest.approx(0.3)
        assert payload["pitch"] == pytest.approx(5.0)
        assert "mar" not in payload          # None 不导出
        assert payload["face_detected"] == 1.0

    def test_merged_returns_new_instance(self, normal_metrics: DriverMetrics) -> None:
        updated = normal_metrics.merged(ear=0.1)
        assert updated.ear == pytest.approx(0.1)
        assert normal_metrics.ear == pytest.approx(0.30)   # 原对象不可变


def test_geometry_constants_are_consistent() -> None:
    """眼部索引必须是 6 个互不相同的点（公式依赖 p1..p6 的顺序）。"""
    assert len(FaceMeshIndex.RIGHT_EYE) == 6
    assert len(FaceMeshIndex.LEFT_EYE) == 6
    assert len(set(FaceMeshIndex.RIGHT_EYE)) == 6
    assert len(set(FaceMeshIndex.LEFT_EYE)) == 6
    assert not set(FaceMeshIndex.RIGHT_EYE) & set(FaceMeshIndex.LEFT_EYE)
    assert all(0 <= index < 478 for index in FaceMeshIndex.RIGHT_EYE + FaceMeshIndex.LEFT_EYE)


def test_eye_index_order_matches_formula() -> None:
    """EAR 公式假定 p1/p4 是眼角、p2/p3 在上、p5/p6 在下。"""
    face = make_face(horizontal=0.10, eye_opening=0.03)
    p1, p2, p3, p4, p5, p6 = (face[i] for i in FaceMeshIndex.RIGHT_EYE)
    # 眼角间距应最大
    assert math.hypot(p1.x - p4.x, p1.y - p4.y) > math.hypot(p2.x - p2.x, p2.y - p6.y)
    # 上睑 y 应小于下睑 y（图像 y 轴向下）
    assert p2.y < p6.y and p3.y < p5.y
