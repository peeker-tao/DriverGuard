"""存储适配器：事件仓储与快照归档。

实现了 :class:`~driver_guard.application.ports.EventRepository` 端口，
采用**仓储（Repository）模式**，把"事件的集合"与"CSV 文件格式"解耦。

* :class:`CsvEventRepository` —— 追加写 CSV，进程重启后可继续累加；
  写入使用行缓冲 + 定期 flush，兼顾实时可见性与 IO 开销。
* :class:`InMemoryEventRepository` —— 测试与"不落盘"模式使用。
* :class:`SnapshotArchiver` —— 保存告警瞬间的画面（含时间戳水印），
  供事后人工复核，是降低误报成本的重要手段。
"""

from __future__ import annotations

import csv
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np

from ..application.ports import EventRepository
from ..domain.models import BehaviorEvent, Frame

logger = logging.getLogger(__name__)

__all__ = [
    "CsvEventRepository",
    "InMemoryEventRepository",
    "SnapshotArchiver",
    "CsvSessionWriter",
]

#: CSV 列顺序。新增字段时请追加到末尾，保证向后兼容。
EVENT_COLUMNS: tuple[str, ...] = (
    "event_id",
    "wall_clock",
    "timestamp",
    "frame_index",
    "behavior",
    "behavior_label",
    "category",
    "severity",
    "severity_value",
    "confidence",
    "duration",
    "ongoing",
    "message",
    "metrics",
)


def _flatten_metrics(metrics: Mapping[str, float]) -> str:
    """把指标字典压成 ``k=v; k=v`` 形式，避免列数随规则变化而漂移。"""
    return "; ".join(f"{k}={v:g}" for k, v in sorted(metrics.items()))


# ===========================================================================
class CsvEventRepository(EventRepository):
    """把事件追加写入 CSV 的仓储实现。

    Args:
        path: CSV 文件路径；父目录会自动创建。
        flush_every: 每写入多少条后强制 flush（``1`` 表示每条都落盘，
            实时可见性最好但 IO 开销最大）。
        write_header: 文件不存在时是否写表头。
    """

    def __init__(
        self,
        path: str | Path,
        *,
        flush_every: int = 5,
        write_header: bool = True,
        encoding: str = "utf-8-sig",
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._flush_every = max(1, flush_every)
        self._encoding = encoding
        self._lock = threading.RLock()
        self._handle = None
        self._writer: csv.DictWriter | None = None
        self._pending = 0
        self._cached: list[BehaviorEvent] | None = None
        self._write_header = write_header
        self._new_file = not self._path.exists() or self._path.stat().st_size == 0

    # ------------------------------------------------------------------
    @property
    def path(self) -> Path:
        return self._path

    def _ensure_open(self) -> None:
        if self._handle is not None:
            return
        self._handle = self._path.open("a", newline="", encoding=self._encoding)
        self._writer = csv.DictWriter(self._handle, fieldnames=list(EVENT_COLUMNS))
        if self._write_header and self._new_file:
            self._writer.writeheader()
            self._handle.flush()
            self._new_file = False

    # ------------------------------------------------------------------
    def save(self, event: BehaviorEvent) -> None:
        with self._lock:
            self._ensure_open()
            assert self._writer is not None and self._handle is not None
            self._writer.writerow(self._to_row(event))
            self._pending += 1
            if self._pending >= self._flush_every:
                self._handle.flush()
                self._pending = 0
            if self._cached is not None:
                self._cached.append(event)

    def extend(self, events: Iterable[BehaviorEvent]) -> None:
        with self._lock:
            for event in events:
                self.save(event)
            if self._handle is not None:
                self._handle.flush()
                self._pending = 0

    def all(self) -> Sequence[BehaviorEvent]:
        """读取全部事件。

        首次调用会从磁盘读入并缓存；此后若通过 :meth:`save` 写入新事件，
        会同步追加到缓存，保持"读写一致"的集合语义。
        """
        with self._lock:
            if self._cached is None:
                self._cached = self._read_from_disk()
            return tuple(self._cached)

    def clear(self) -> None:
        with self._lock:
            self.close()
            if self._path.exists():
                self._path.unlink()
            self._cached = []
            self._new_file = True

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                try:
                    self._handle.flush()
                    self._handle.close()
                finally:
                    self._handle = None
                    self._writer = None
                    self._pending = 0

    def __enter__(self) -> "CsvEventRepository":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    @staticmethod
    def _to_row(event: BehaviorEvent) -> dict[str, object]:
        return {
            "event_id": event.event_id,
            "wall_clock": event.wall_clock.isoformat(timespec="milliseconds"),
            "timestamp": f"{event.timestamp:.3f}",
            "frame_index": event.frame_index,
            "behavior": event.behavior.value,
            "behavior_label": event.behavior.label,
            "category": event.behavior.category.value,
            "severity": event.severity.name,
            "severity_value": int(event.severity),
            "confidence": f"{event.confidence:.4f}",
            "duration": f"{event.duration:.3f}",
            "ongoing": int(event.ongoing),
            "message": event.message.replace("\n", " "),
            "metrics": _flatten_metrics(event.metrics),
        }

    def _read_from_disk(self) -> list[BehaviorEvent]:
        """尽力把 CSV 还原为领域对象；无法解析的行会被跳过并记录。"""
        if not self._path.exists():
            return []
        from ..domain.models import BehaviorType, Severity

        events: list[BehaviorEvent] = []
        with self._path.open("r", newline="", encoding=self._encoding) as handle:
            for line_no, row in enumerate(csv.DictReader(handle), start=2):
                try:
                    events.append(
                        BehaviorEvent(
                            behavior=BehaviorType(row["behavior"]),
                            severity=Severity.parse(row["severity"]),
                            confidence=float(row["confidence"]),
                            frame_index=int(row["frame_index"]),
                            timestamp=float(row["timestamp"]),
                            message=row["message"],
                            metrics=_parse_metrics(row.get("metrics", "")),
                            duration=float(row["duration"]),
                            wall_clock=datetime.fromisoformat(row["wall_clock"]),
                            event_id=row["event_id"],
                            ongoing=bool(int(row.get("ongoing", "0") or 0)),
                        )
                    )
                except (KeyError, ValueError, TypeError) as exc:
                    logger.warning("跳过无法解析的事件行（第 %d 行）：%s", line_no, exc)
        logger.info("从 %s 载入 %d 条历史事件", self._path.name, len(events))
        return events


def _parse_metrics(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        key, _, value = chunk.partition("=")
        try:
            result[key.strip()] = float(value)
        except ValueError:
            continue
    return result


# ===========================================================================
class InMemoryEventRepository(EventRepository):
    """内存仓储：测试与"不落盘"演示模式使用。"""

    def __init__(self, capacity: int = 10_000) -> None:
        self._events: list[BehaviorEvent] = []
        self._capacity = max(1, capacity)

    def save(self, event: BehaviorEvent) -> None:
        self._events.append(event)
        if len(self._events) > self._capacity:
            del self._events[: len(self._events) - self._capacity]

    def all(self) -> Sequence[BehaviorEvent]:
        return tuple(self._events)

    def clear(self) -> None:
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)


# ===========================================================================
class SnapshotArchiver:
    """把告警瞬间的画面保存为图片。

    Args:
        directory: 输出目录。
        annotate: 是否叠加时间戳与行为文字水印（便于人工复核）。
        jpeg_quality: JPEG 质量（1~100）。
        max_files: 目录内最多保留的图片数，超出后删除最旧的，防止磁盘写满。
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        annotate: bool = True,
        jpeg_quality: int = 88,
        max_files: int = 2000,
    ) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._annotate = annotate
        self._quality = int(np.clip(jpeg_quality, 1, 100))
        self._max_files = max(10, max_files)
        self._saved = 0

    @property
    def directory(self) -> Path:
        return self._dir

    @property
    def saved_count(self) -> int:
        return self._saved

    def save(self, event: BehaviorEvent, frame: Frame | None) -> Path | None:
        """保存一帧快照，返回写入路径；无图像时返回 ``None``。"""
        if frame is None or frame.image is None or frame.image.size == 0:
            return None

        image = frame.image.copy()
        if self._annotate:
            _draw_watermark(image, event)

        name = (
            f"{event.wall_clock:%Y%m%d_%H%M%S_%f}"[:22]
            + f"_{event.behavior.value}.jpg"
        )
        target = self._dir / name
        try:
            ok = cv2.imwrite(
                str(target), image, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality]
            )
        except Exception:  # noqa: BLE001 - 归档失败不能影响检测
            logger.exception("写入快照失败：%s", target)
            return None

        if not ok:
            logger.warning("cv2.imwrite 返回失败：%s", target)
            return None

        self._saved += 1
        self._enforce_quota()
        return target

    def _enforce_quota(self) -> None:
        try:
            files = sorted(
                self._dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime
            )
        except OSError:  # pragma: no cover
            return
        excess = len(files) - self._max_files
        for path in files[:excess]:
            try:
                path.unlink()
            except OSError:  # pragma: no cover
                logger.warning("删除旧快照失败：%s", path)


def _draw_watermark(image: np.ndarray, event: BehaviorEvent) -> None:
    """在图像底部绘制半透明信息条，标明时间、行为与置信度。"""
    height, width = image.shape[:2]
    bar_height = max(28, int(height * 0.075))
    overlay = image.copy()
    cv2.rectangle(overlay, (0, height - bar_height), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

    lines = [
        f"{event.wall_clock:%Y-%m-%d %H:%M:%S.%f}"[:-3],
        f"{event.behavior.label} | {event.severity.label} | "
        f"conf={event.confidence:.2f} | t={event.timestamp:.2f}s | frame={event.frame_index}",
    ]
    font_scale = max(0.45, min(0.8, width / 1400.0))
    thickness = max(1, int(font_scale * 1.6))
    color = (255, 255, 255)
    for index, text in enumerate(lines):
        y = height - bar_height + int(bar_height * (0.36 + 0.46 * index)) + 2
        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)


# ===========================================================================
class CsvSessionWriter:
    """把每次会话的汇总追加写入 CSV，便于横向对比不同会话的风险。"""

    COLUMNS: tuple[str, ...] = (
        "started_at", "ended_at", "source", "frames", "duration_s", "avg_fps",
        "triggered_frames", "total_events", "risk_index", "interrupted",
        "event_counts", "stage_avg_ms",
    )

    def __init__(self, path: str | Path, *, encoding: str = "utf-8-sig") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._encoding = encoding

    @property
    def path(self) -> Path:
        return self._path

    def append(self, summary, *, risk_index: float = 0.0) -> None:
        """写入一条会话汇总。

        Args:
            summary: :class:`~driver_guard.application.services.SessionSummary`。
            risk_index: 由调用方计算的风险指数（避免本类反向依赖应用层类型）。
        """
        is_new = not self._path.exists() or self._path.stat().st_size == 0
        with self._path.open("a", newline="", encoding=self._encoding) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.COLUMNS))
            if is_new:
                writer.writeheader()
            writer.writerow(self._to_row(summary, risk_index))

    @staticmethod
    def _to_row(summary, risk_index: float) -> dict[str, object]:
        counts = "; ".join(
            f"{behavior.value}={count}"
            for behavior, count in sorted(
                summary.event_counts.items(), key=lambda item: item[0].value
            )
        )
        stages = "; ".join(
            f"{name}={value:.2f}" for name, value in sorted(summary.stage_avg_ms.items())
        )
        return {
            "started_at": summary.started_at.isoformat(timespec="seconds"),
            "ended_at": summary.ended_at.isoformat(timespec="seconds"),
            "source": summary.source_name,
            "frames": summary.frames,
            "duration_s": f"{summary.duration:.2f}",
            "avg_fps": f"{summary.avg_fps:.2f}",
            "triggered_frames": summary.triggered_frames,
            "total_events": sum(summary.event_counts.values()),
            "risk_index": f"{risk_index:.2f}",
            "interrupted": int(summary.interrupted),
            "event_counts": counts,
            "stage_avg_ms": stages,
        }
