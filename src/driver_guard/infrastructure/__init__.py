"""基础设施层导出入口。"""

from .alerts import (
    CallbackAlertSink,
    CompositeAlertSink,
    ConsoleAlertSink,
    CsvAlertSink,
    NullAlertSink,
    SnapshotAlertSink,
    SoundAlertSink,
)
from .landmarks import (
    PROVIDER_FACTORIES,
    SCENARIOS,
    HaarLandmarkProvider,
    MediaPipeLandmarkProvider,
    MetricSegment,
    ReplayLandmarkProvider,
    available_providers,
    create_landmark_provider,
)
from .sources import (
    ImageSequenceFrameSource,
    SyntheticFrameSource,
    VideoFileFrameSource,
    WebcamFrameSource,
)
from .storage import (
    EVENT_COLUMNS,
    CsvEventRepository,
    CsvSessionWriter,
    InMemoryEventRepository,
    SnapshotArchiver,
)

__all__ = [
    # 数据源
    "WebcamFrameSource", "VideoFileFrameSource", "ImageSequenceFrameSource", "SyntheticFrameSource",
    # 感知
    "MediaPipeLandmarkProvider", "HaarLandmarkProvider", "ReplayLandmarkProvider",
    "MetricSegment", "SCENARIOS", "PROVIDER_FACTORIES",
    "create_landmark_provider", "available_providers",
    # 告警
    "ConsoleAlertSink", "SoundAlertSink", "SnapshotAlertSink", "CsvAlertSink",
    "CallbackAlertSink", "CompositeAlertSink", "NullAlertSink",
    # 存储
    "CsvEventRepository", "InMemoryEventRepository", "SnapshotArchiver", "CsvSessionWriter",
    "EVENT_COLUMNS",
]
