"""告警输出适配器。

.. code-block:: text

   BehaviorEvent ──► AlertService ──┬──► ConsoleAlertSink   控制台/日志
                                    ├──► SoundAlertSink     蜂鸣提示
                                    ├──► SnapshotAlertSink  保存现场画面
                                    ├──► CsvAlertSink       事件流水
                                    └──► CallbackAlertSink  桥接到界面(Qt Signal)

每个适配器都实现 :class:`~driver_guard.application.ports.AlertSink`，
因此 :class:`~driver_guard.application.services.AlertService` 可以在
**不知道输出介质** 的前提下把同一个事件扇出到全部通道。

设计约束
--------
* 每个 :meth:`emit` **必须自行吞掉异常**：告警输出失败不能中断视频采集；
* 声音通道支持按最低严重级别过滤，避免轻微事件频繁鸣响；
* ``CallbackAlertSink`` 是界面层的接入点，把领域事件转换为 Qt 信号，
  从而保证跨线程更新 UI 时走 Qt 的队列机制（线程安全）。
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Callable, Iterable, Sequence

from ..application.ports import AlertSink
from ..domain.models import BehaviorEvent, Frame, Severity
from .storage import CsvEventRepository, SnapshotArchiver

logger = logging.getLogger(__name__)

__all__ = [
    "ConsoleAlertSink",
    "SoundAlertSink",
    "SnapshotAlertSink",
    "CsvAlertSink",
    "CallbackAlertSink",
    "CompositeAlertSink",
    "NullAlertSink",
]


class _BaseSink(AlertSink):
    """统一实现"异常不外抛"的约束。"""

    def emit(self, event: BehaviorEvent, frame: Frame | None = None) -> None:
        try:
            self._emit(event, frame)
        except Exception:  # noqa: BLE001 - 告警通道故障隔离
            logger.exception("告警通道 %s 输出失败（已忽略）", self.name)

    def _emit(self, event: BehaviorEvent, frame: Frame | None) -> None:  # pragma: no cover
        raise NotImplementedError


# ===========================================================================
class ConsoleAlertSink(_BaseSink):
    """把告警写入日志。

    Args:
        with_metrics: 是否附带指标明细，便于调参时观察阈值余量。
    """

    def __init__(self, *, with_metrics: bool = True) -> None:
        self._with_metrics = with_metrics

    @property
    def name(self) -> str:
        return "控制台"

    def _emit(self, event: BehaviorEvent, frame: Frame | None) -> None:
        del frame
        suffix = ""
        if self._with_metrics and event.metrics:
            suffix = "  " + " ".join(f"{k}={v:g}" for k, v in sorted(event.metrics.items()))
        marker = "持续" if event.ongoing else "新增"
        logger.warning(
            "[%s] %s %s | 置信度=%.2f | 时长=%.2fs | 帧#%d%s",
            marker, event.severity.label, event.behavior.label,
            event.confidence, event.duration, event.frame_index, suffix,
        )


# ===========================================================================
class SoundAlertSink(_BaseSink):
    """蜂鸣提示。

    按严重级别给出不同的鸣响模式（次数 × 时长），使驾驶员无需看屏幕
    即可分辨告警紧迫程度——这是"注意资源有限"的人因工程考虑。

    * Windows 使用 :mod:`winsound`；
    * 其它平台回退为终端响铃 ``\\a``（无阻塞）。
    """

    #: ``严重级别 -> (蜂鸣次数, 单次时长毫秒, 频率 Hz)``
    PATTERNS: dict[Severity, tuple[int, int, int]] = {
        Severity.MEDIUM: (1, 120, 880),
        Severity.HIGH: (2, 150, 1046),
        Severity.CRITICAL: (3, 200, 1318),
    }

    def __init__(
        self,
        *,
        min_severity: Severity = Severity.HIGH,
        repeat_gap_ms: int = 90,
        enabled: bool = True,
    ) -> None:
        self._min_severity = min_severity
        self._gap = repeat_gap_ms
        self._enabled = enabled
        self._lock = threading.Lock()
        self._warned = False

    @property
    def name(self) -> str:
        return "声音"

    def _emit(self, event: BehaviorEvent, frame: Frame | None) -> None:
        del frame
        if not self._enabled or event.severity < self._min_severity:
            return
        # 同一次持续行为的重复提醒不再鸣响，避免长时间嘈杂
        if event.ongoing:
            return

        count, duration, frequency = self.PATTERNS.get(
            event.severity, self.PATTERNS[Severity.HIGH]
        )
        # 串行化播放，避免多个事件同时触发时蜂鸣器互相打断
        with self._lock:
            self._play(count, duration, frequency)

    def _play(self, count: int, duration_ms: int, frequency: int) -> None:
        if sys.platform.startswith("win"):
            try:
                import winsound

                for index in range(count):
                    winsound.Beep(frequency, duration_ms)
                    if index + 1 < count:
                        threading.Event().wait(self._gap / 1000.0)
                return
            except Exception:  # noqa: BLE001 - 无音频设备时静默降级
                if not self._warned:
                    self._warned = True
                    logger.warning("系统蜂鸣不可用，改为终端响铃")
        sys.stdout.write("\a")
        sys.stdout.flush()


# ===========================================================================
class SnapshotAlertSink(_BaseSink):
    """把告警现场画面存档。

    Args:
        directory: 快照输出目录。
        min_severity: 最低严重级别。
        skip_ongoing: 是否跳过"同一次持续行为"的重复提醒。
            默认 ``True``：一次长时间疲劳只保留 **首次触发** 的一帧，
            因为首帧已足以人工复核，而重复存图会让目录迅速膨胀
            （实测一次 30 秒回放可产生数百张近乎相同的图片）。
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        min_severity: Severity = Severity.HIGH,
        skip_ongoing: bool = True,
    ) -> None:
        self._archiver = SnapshotArchiver(directory)
        self._min_severity = min_severity
        self._skip_ongoing = skip_ongoing

    @property
    def name(self) -> str:
        return "现场快照"

    @property
    def archiver(self) -> SnapshotArchiver:
        return self._archiver

    def _emit(self, event: BehaviorEvent, frame: Frame | None) -> None:
        if event.severity < self._min_severity:
            return
        if self._skip_ongoing and event.ongoing:
            return
        path = self._archiver.save(event, frame)
        if path is not None:
            logger.info("已保存快照：%s", path.name)


# ===========================================================================
class CsvAlertSink(_BaseSink):
    """把事件追加到 CSV 流水（复用仓储适配器，避免重复实现）。"""

    def __init__(self, path: str | Path, repo: CsvEventRepository | None = None) -> None:
        self._repo = repo or CsvEventRepository(path)

    @property
    def name(self) -> str:
        return "CSV 流水"

    @property
    def repository(self) -> CsvEventRepository:
        return self._repo

    def _emit(self, event: BehaviorEvent, frame: Frame | None) -> None:
        del frame
        self._repo.save(event)

    def close(self) -> None:
        self._repo.close()


# ===========================================================================
class CallbackAlertSink(_BaseSink):
    """把事件转交给任意回调（表现层用它桥接到 Qt Signal）。

    Args:
        callback: ``(event, frame) -> None``。**注意**：它会在采集线程中被调用，
            因此界面实现必须使用信号槽（而非直接操作控件）来保证线程安全。
        label: 通道名称。
        min_severity: 最低严重级别过滤。
    """

    def __init__(
        self,
        callback: Callable[[BehaviorEvent, Frame | None], None],
        *,
        label: str = "回调",
        min_severity: Severity = Severity.LOW,
    ) -> None:
        self._callback = callback
        self._label = label
        self._min_severity = min_severity

    @property
    def name(self) -> str:
        return self._label

    def _emit(self, event: BehaviorEvent, frame: Frame | None) -> None:
        if event.severity < self._min_severity:
            return
        self._callback(event, frame)


# ===========================================================================
class CompositeAlertSink(_BaseSink):
    """组合（Composite）模式：把多个通道当成一个通道，便于分组管理。"""

    def __init__(self, sinks: Iterable[AlertSink] = (), *, label: str = "组合通道") -> None:
        self._sinks: list[AlertSink] = list(sinks)
        self._label = label

    @property
    def name(self) -> str:
        return self._label

    @property
    def sinks(self) -> Sequence[AlertSink]:
        return tuple(self._sinks)

    def add(self, sink: AlertSink) -> None:
        self._sinks.append(sink)

    def _emit(self, event: BehaviorEvent, frame: Frame | None) -> None:
        for sink in self._sinks:
            sink.emit(event, frame)

    def close(self) -> None:
        for sink in self._sinks:
            try:
                sink.close()
            except Exception:  # noqa: BLE001
                logger.exception("关闭告警通道 %s 失败", sink.name)


# ===========================================================================
class NullAlertSink(_BaseSink):
    """空对象（Null Object）模式：占位而不引入 ``if sink is not None`` 分支。"""

    @property
    def name(self) -> str:
        return "（无）"

    def _emit(self, event: BehaviorEvent, frame: Frame | None) -> None:
        del event, frame
