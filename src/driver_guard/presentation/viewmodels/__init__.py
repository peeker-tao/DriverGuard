"""表现层视图模型。"""

from .monitor_viewmodel import (
    BEHAVIOR_LABELS,
    CATEGORY_LABELS,
    SEVERITY_LABELS,
    MonitorViewModel,
)

__all__ = [
    "MonitorViewModel",
    "BEHAVIOR_LABELS",
    "CATEGORY_LABELS",
    "SEVERITY_LABELS",
]
