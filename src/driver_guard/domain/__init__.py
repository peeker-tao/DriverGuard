"""领域层导出入口。

领域层是整个系统的核心，**不依赖任何外部框架**（无 Qt、无 OpenCV 推理、
无网络、无数据库）。所有外部依赖都通过
:mod:`driver_guard.application.ports` 中定义的端口（接口）注入，
这就是依赖倒置原则（DIP）在本项目中的落地方式。
"""

from .calibration import NeutralPoseCalibrator
from .confirmation import BehaviorConfirmer
from .geometry import (
    FaceMeshIndex,
    eye_aspect_ratio,
    head_pose_from_landmarks,
    head_pose_from_matrix,
    mouth_aspect_ratio,
    rotation_matrix_to_euler,
)
from .metrics import MetricExtractor
from .models import (
    BehaviorCategory,
    BehaviorEvent,
    BehaviorType,
    BoundingBox,
    DetectionContext,
    DriverMetrics,
    Frame,
    HeadPose,
    Landmark,
    NeutralProfile,
    Observation,
    RuleOutcome,
    Severity,
)
from .rules import RULE_FACTORIES, BehaviorRule, RuleConfig, RuleEngine, build_engine, build_rules

__all__ = [
    # 模型
    "Severity", "BehaviorCategory", "BehaviorType",
    "Landmark", "BoundingBox", "HeadPose", "Observation", "Frame",
    "DriverMetrics", "NeutralProfile", "RuleOutcome", "BehaviorEvent", "DetectionContext",
    # 几何
    "FaceMeshIndex", "eye_aspect_ratio", "mouth_aspect_ratio",
    "rotation_matrix_to_euler", "head_pose_from_matrix", "head_pose_from_landmarks",
    # 服务
    "MetricExtractor", "NeutralPoseCalibrator", "BehaviorConfirmer",
    # 规则
    "RULE_FACTORIES", "BehaviorRule", "RuleConfig", "RuleEngine",
    "build_rules", "build_engine",
]
