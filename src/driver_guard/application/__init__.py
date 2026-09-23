"""应用层：编排领域对象与端口适配器。"""

from .event_bus import EventBus, Subscription, Topic
from .pipeline import DetectionPipeline, FrameAnalysis
from .ports import AlertSink, EventRepository, FrameSource, FrameSourceError, LandmarkProvider
from .services import (
    AlertService,
    DetectionService,
    ServiceState,
    SessionSummary,
    StatisticsService,
)

__all__ = [
    "EventBus", "Subscription", "Topic",
    "DetectionPipeline", "FrameAnalysis",
    "FrameSource", "FrameSourceError", "LandmarkProvider", "AlertSink", "EventRepository",
    "DetectionService", "AlertService", "StatisticsService", "ServiceState", "SessionSummary",
]
