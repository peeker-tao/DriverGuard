"""领域模型层。

本模块定义与 **框架、GUI、推理库完全无关** 的纯业务概念：

* :class:`Severity` / :class:`BehaviorType` —— 业务枚举
* :class:`Landmark` / :class:`BoundingBox` / :class:`HeadPose` —— 感知结果值对象
* :class:`Frame` —— 一帧视频
* :class:`DriverMetrics` —— 驾驶员状态的可度量指标
* :class:`RuleOutcome` / :class:`BehaviorEvent` —— 规则判定结果与领域事件

整个模块只依赖 ``numpy``，因此可以被单元测试高速覆盖，
也可以被替换掉表现层（例如改成 Web 服务）而不需修改。
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "Severity",
    "BehaviorCategory",
    "BehaviorType",
    "Landmark",
    "BoundingBox",
    "HeadPose",
    "Observation",
    "Frame",
    "DriverMetrics",
    "NeutralProfile",
    "RuleOutcome",
    "BehaviorEvent",
    "DetectionContext",
]


# ===========================================================================
#  枚举
# ===========================================================================
class Severity(enum.IntEnum):
    """告警严重程度。

    继承 :class:`enum.IntEnum` 以便直接比较大小（``>=``）与排序，
    使 ``alert.sound_min_severity`` 之类的阈值判断只需一次比较。
    """

    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @property
    def label(self) -> str:
        return _SEVERITY_LABELS[self]

    @classmethod
    def parse(cls, value: object) -> "Severity":
        """从配置字符串/整数宽松地解析出严重级别。

        统一抛出 :class:`ValueError`（而不是让字典的 ``KeyError`` 泄漏），
        使调用方只需捕获一种异常类型。
        """
        if isinstance(value, Severity):
            return value
        if isinstance(value, str):
            try:
                return cls[value.strip().upper()]
            except KeyError as exc:
                raise ValueError(
                    f"未知严重级别 {value!r}，可选："
                    + ", ".join(member.name for member in cls)
                ) from exc
        if isinstance(value, (int, float)):
            try:
                return cls(int(value))
            except ValueError as exc:
                raise ValueError(f"严重级别数值 {value!r} 超出 [0, {max(cls)}]") from exc
        raise ValueError(f"无法解析严重级别: {value!r}")


_SEVERITY_LABELS: dict[Severity, str] = {
    Severity.INFO: "提示",
    Severity.LOW: "轻微",
    Severity.MEDIUM: "中等",
    Severity.HIGH: "严重",
    Severity.CRITICAL: "危急",
}


class BehaviorCategory(enum.Enum):
    """危险行为所属大类，用于统计报表分组。"""

    FATIGUE = "fatigue"
    DISTRACTION = "distraction"
    VIOLATION = "violation"

    @property
    def label(self) -> str:
        return {
            BehaviorCategory.FATIGUE: "疲劳驾驶",
            BehaviorCategory.DISTRACTION: "分心驾驶",
            BehaviorCategory.VIOLATION: "违规驾驶",
        }[self]


class BehaviorType(enum.Enum):
    """系统可识别的危险行为类型。

    枚举值即配置键名，保证 ``config/default.json`` 与代码一一对应。
    """

    EYES_CLOSED = "eyes_closed"
    YAWNING = "yawning"
    HEAD_DOWN = "head_down"
    GAZE_OFF_ROAD = "gaze_off_road"
    PHONE_CALL = "phone_call"
    SMOKING = "smoking"
    DRIVER_ABSENT = "driver_absent"

    @property
    def label(self) -> str:
        return _BEHAVIOR_LABELS[self]

    @property
    def category(self) -> BehaviorCategory:
        return _BEHAVIOR_CATEGORIES[self]


_BEHAVIOR_LABELS: dict[BehaviorType, str] = {
    BehaviorType.EYES_CLOSED: "闭眼疲劳",
    BehaviorType.YAWNING: "打哈欠",
    BehaviorType.HEAD_DOWN: "低头",
    BehaviorType.GAZE_OFF_ROAD: "视线偏离",
    BehaviorType.PHONE_CALL: "打电话",
    BehaviorType.SMOKING: "抽烟",
    BehaviorType.DRIVER_ABSENT: "驾驶员离位",
}

_BEHAVIOR_CATEGORIES: dict[BehaviorType, BehaviorCategory] = {
    BehaviorType.EYES_CLOSED: BehaviorCategory.FATIGUE,
    BehaviorType.YAWNING: BehaviorCategory.FATIGUE,
    BehaviorType.HEAD_DOWN: BehaviorCategory.FATIGUE,
    BehaviorType.GAZE_OFF_ROAD: BehaviorCategory.DISTRACTION,
    BehaviorType.PHONE_CALL: BehaviorCategory.VIOLATION,
    BehaviorType.SMOKING: BehaviorCategory.VIOLATION,
    BehaviorType.DRIVER_ABSENT: BehaviorCategory.DISTRACTION,
}


# ===========================================================================
#  感知结果值对象
# ===========================================================================
@dataclass(frozen=True, slots=True)
class Landmark:
    """归一化坐标的关键点（x、y ∈ [0, 1]，相对图像宽高）。"""

    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0

    def to_pixel(self, width: int, height: int) -> tuple[int, int]:
        return int(round(self.x * width)), int(round(self.y * height))

    def distance_to(self, other: "Landmark") -> float:
        """归一化欧氏距离（忽略 z，避免不同 provider 的 z 尺度差异影响判定）。"""
        return float(np.hypot(self.x - other.x, self.y - other.y))


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """归一化坐标轴对齐包围盒。"""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> tuple[float, float]:
        return (self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)

    def as_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """转换为 ``(x1, y1, x2, y2)`` 像素坐标，并夹紧到图像范围内。"""
        x1, y1 = int(round(self.x_min * width)), int(round(self.y_min * height))
        x2, y2 = int(round(self.x_max * width)), int(round(self.y_max * height))
        x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
        y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
        return x1, y1, x2, y2

    @classmethod
    def from_landmarks(
        cls, landmarks: Sequence[Landmark], padding: float = 0.0
    ) -> "BoundingBox | None":
        """由关键点集合求包围盒，``padding`` 为相对膨胀比例。"""
        if not landmarks:
            return None
        xs = [p.x for p in landmarks]
        ys = [p.y for p in landmarks]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        pad_x = (x_max - x_min) * padding
        pad_y = (y_max - y_min) * padding
        return cls(x_min - pad_x, y_min - pad_y, x_max + pad_x, y_max + pad_y)


@dataclass(frozen=True, slots=True)
class HeadPose:
    """头部姿态角（单位：度）。

    三个角度分别对应绕传感器 X / Y / Z 轴的旋转量，由
    :func:`~driver_guard.domain.geometry.rotation_matrix_to_euler` 从旋转矩阵分解而来。

    .. note::
       由于 MediaPipe 规范模型的轴系与 OpenCV 相机轴系不一致（详见
       :func:`~driver_guard.domain.geometry.rotation_matrix_to_euler` 的说明），
       这里 **不约定** pitch/yaw/roll 的绝对方位含义。三者只在"相对变化量"
       意义下使用：所有头部姿态规则都基于 **相对驾驶员个人中立位（标定值）的
       偏离量** 做判定。
    """

    pitch: float
    yaw: float
    roll: float
    matrix: tuple[tuple[float, ...], ...] | None = None

    @property
    def max_deviation(self) -> float:
        """三个角度中绝对值最大者，用于快速判断"偏离中立位"的整体幅度。"""
        return max(abs(self.pitch), abs(self.yaw), abs(self.roll))

    def deviation_from(self, neutral: "HeadPose") -> "HeadPose":
        """相对中立位的偏离（分量相减）。"""
        return HeadPose(
            pitch=self.pitch - neutral.pitch,
            yaw=self.yaw - neutral.yaw,
            roll=self.roll - neutral.roll,
        )


@dataclass(frozen=True, slots=True)
class Observation:
    """**感知层** 单帧输出：由 :class:`~driver_guard.application.ports.LandmarkProvider` 产生。

    该对象是"感知"与"决策"之间的契约，使决策规则无需了解
    MediaPipe / Haar / 录像回放等具体实现（依赖倒置）。
    """

    face: tuple[Landmark, ...] = ()
    """人脸关键点（MediaPipe FaceMesh 为 478 点；降级模式可能为空）。"""

    face_box: BoundingBox | None = None
    head_pose: HeadPose | None = None
    blendshapes: Mapping[str, float] = field(default_factory=dict)
    """表情/动作系数，如 ``eyeBlinkLeft``、``jawOpen``。"""

    hands: tuple[tuple[Landmark, ...], ...] = ()
    hand_boxes: tuple[BoundingBox, ...] = ()
    precomputed: "DriverMetrics | None" = None
    """感知实现若已直接给出指标（如回放/自检模式），决策层可直接采用。"""

    provider: str = "unknown"
    latency_ms: float = 0.0

    @property
    def has_face(self) -> bool:
        """感知层是否认为画面中存在驾驶员。

        判定顺序：有人脸关键点 / 有人脸包围盒 / 上游在预计算指标中
        明确声明 ``face_detected``。最后一条使 **只提供指标而不提供关键点**
        的感知实现（如回放通道、第三方 DMS 接入）也能正常驱动规则，
        无需为了满足接口而伪造关键点。
        """
        if self.face or self.face_box is not None:
            return True
        return self.precomputed is not None and self.precomputed.face_detected

    @property
    def has_hands(self) -> bool:
        return bool(self.hands)

    def blendshape(self, name: str) -> float | None:
        return self.blendshapes.get(name)


# ===========================================================================
#  帧
# ===========================================================================
@dataclass(frozen=True, slots=True)
class Frame:
    """一帧视频图像（BGR 通道，OpenCV 原生布局）。"""

    index: int
    timestamp: float
    """相对本次会话开始的秒数（单调递增，用于时间域判定）。"""
    image: np.ndarray

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    def rgb(self) -> np.ndarray:
        """转换为 RGB 视图（供 Qt 显示 / MediaPipe 输入）。"""
        import cv2

        return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)


# ===========================================================================
#  可度量指标
# ===========================================================================
@dataclass(frozen=True, slots=True)
class DriverMetrics:
    """从 :class:`Observation` 计算出的驾驶员状态指标。

    字段全部可空：当某项因遮挡、降级模式或模型缺失而无法计算时保持 ``None``，
    规则据此自动选择"融合路径"，从而优雅降级而非误判。
    """

    face_detected: bool = False

    # --- 眼部：EAR (Eye Aspect Ratio)，眨眼/闭眼判据 ---
    ear: float | None = None
    ear_left: float | None = None
    ear_right: float | None = None

    # --- 嘴部：MAR (Mouth Aspect Ratio)，打哈欠判据 ---
    mar: float | None = None

    # --- 头部姿态（度；仅作相对量使用，见 HeadPose 说明）---
    pitch: float | None = None
    yaw: float | None = None
    roll: float | None = None

    # --- 与轴系无关的低头辅助指标 ---
    face_pitch_ratio: float | None = None
    """眼睛连线中心到嘴部中心的图像纵向距离 ÷ 脸宽。

    低头时人脸发生纵向透视压缩，该比值相对中立位变小。
    因为它只用到图像 2D 距离，**不依赖任何旋转矩阵的轴系约定**，
    因此可作为低头判定的一条独立、稳健的辅助依据。
    """

    # --- 表情系数（MediaPipe blendshapes，取值范围 0~1）---
    eye_blink: float | None = None
    eye_blink_left: float | None = None
    eye_blink_right: float | None = None
    jaw_open: float | None = None

    # --- 手部与头部的相对几何（电话/抽烟判据）---
    hand_face_ratio: float | None = None
    """手部关键点到脸部中心的最小距离 ÷ 脸宽，越小表示手越贴近脸。"""
    hand_mouth_ratio: float | None = None
    """手部关键点到嘴部中心的最小距离 ÷ 脸宽，越小表示手越贴近嘴。"""

    def as_dict(self) -> dict[str, float]:
        """仅导出已计算的指标，便于写入 CSV 与界面表格。"""
        out: dict[str, float] = {}
        for name in self.__slots__:
            value = getattr(self, name)
            if isinstance(value, bool):
                out[name] = float(value)
            elif isinstance(value, (int, float)):
                out[name] = float(value)
        return out

    def merged(self, **overrides: float | None) -> "DriverMetrics":
        return replace(self, **overrides)


# ===========================================================================
#  决策结果
# ===========================================================================
@dataclass(frozen=True, slots=True)
class RuleOutcome:
    """单条规则的瞬时判定输出（尚未经过时序确认）。"""

    behavior: BehaviorType
    triggered: bool
    confidence: float = 0.0
    severity: Severity = Severity.MEDIUM
    message: str = ""
    detail: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def negative(cls, behavior: BehaviorType) -> "RuleOutcome":
        """构造"未触发"结果，便于规则提前返回。"""
        return cls(behavior=behavior, triggered=False)


@dataclass(frozen=True, slots=True)
class BehaviorEvent:
    """**领域事件**：一条已被时序确认、可对外告警的危险行为记录。"""

    behavior: BehaviorType
    severity: Severity
    confidence: float
    frame_index: int
    timestamp: float
    message: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    duration: float = 0.0
    """该行为从首次触发到本次确认已持续的秒数（去抖确认后 > 0）。"""

    wall_clock: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    #: 是否为该行为"持续段"中的重复上报（True 表示仍在持续，非新事件）
    ongoing: bool = False

    @property
    def summary(self) -> str:
        return f"[{self.severity.label}] {self.behavior.label} — {self.message}"


# ===========================================================================
#  规则上下文
# ===========================================================================
@dataclass(frozen=True, slots=True)
class NeutralProfile:
    """驾驶员个人中立位档案（会话开始时自动标定得到）。

    之所以要标定个体中立位，而不是使用全局固定阈值：

    * 摄像头安装角度、座椅高度、驾驶员身高体型都会整体平移姿态角；
    * MediaPipe 规范模型的轴系与相机轴系不完全一致（详见
      :func:`~driver_guard.domain.geometry.rotation_matrix_to_euler` 的说明），
      绝对角语义不可依赖。

    用「相对中立位的偏离量」判定，可一次性规避以上全部干扰。
    """

    head_pose: HeadPose | None = None
    pitch_ratio: float | None = None
    """中立位下的 ``face_pitch_ratio``，用于低头判定的相对比较。"""
    samples: int = 0
    """参与标定的有效帧数，用于评估标定可信度。"""

    @property
    def ready(self) -> bool:
        return self.head_pose is not None


@dataclass(frozen=True, slots=True)
class DetectionContext:
    """传入规则的只读上下文：当前帧 + 感知结果 + 指标 + 历史滑窗 + 中立位。"""

    frame: Frame
    observation: Observation
    metrics: DriverMetrics
    history: tuple[DriverMetrics, ...] = ()
    neutral: NeutralProfile | None = None

    @property
    def pose_deviation(self) -> HeadPose | None:
        """相对中立位的头部姿态偏离量；未标定或无姿态数据时返回 ``None``。

        头部姿态规则 **只使用本属性**，从而不依赖任何绝对轴语义。
        """
        if self.neutral is None or self.neutral.head_pose is None:
            return None
        pitch, yaw, roll = self.metrics.pitch, self.metrics.yaw, self.metrics.roll
        if pitch is None or yaw is None or roll is None:
            return None
        return HeadPose(pitch=pitch, yaw=yaw, roll=roll).deviation_from(self.neutral.head_pose)

    @property
    def pitch_ratio_deviation(self) -> float | None:
        """当前 ``face_pitch_ratio`` 与中立位之差；抬头为负、低头为正。

        ``face_pitch_ratio`` 是"眼睛中线到嘴中线"的纵向距离除以脸宽。
        低头时人脸纵向被压缩，该比值 **减小**，故偏离量取
        ``中立位 - 当前`` 时低头为正。
        """
        if self.neutral is None or self.neutral.pitch_ratio is None:
            return None
        current = self.metrics.face_pitch_ratio
        if current is None:
            return None
        return self.neutral.pitch_ratio - current

    def mean(self, attr: str) -> float | None:
        """返回历史滑窗中某指标的平均值（忽略 ``None``），无数据则返回 ``None``。"""
        values = [
            value
            for item in (*self.history, self.metrics)
            if (value := getattr(item, attr, None)) is not None
        ]
        if not values:
            return None
        return float(np.mean(values))

    def streak(self, predicate) -> int:
        """从最新一帧往前数，满足 ``predicate(metrics)`` 的连续帧数。

        用于"连续 N 帧才确认"的去抖逻辑，是抑制误报的关键。
        """
        count = 0
        for item in reversed((*self.history, self.metrics)):
            if predicate(item):
                count += 1
            else:
                break
        return count
