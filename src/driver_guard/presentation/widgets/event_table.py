"""事件列表与统计图表控件。"""

from __future__ import annotations

from typing import Mapping, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from ...domain.models import BehaviorEvent, Severity
from ..theme import BEHAVIOR_COLORS, CATEGORY_COLORS, Palette, SEVERITY_COLORS

__all__ = ["EventTable", "StatisticsChart"]


# ===========================================================================
class EventTable(QTableWidget):
    """告警事件流水表（新事件插入顶部，便于值守人员第一时间看到）。"""

    HEADERS: tuple[tuple[str, int], ...] = (
        ("时间", 108),
        ("行为", 100),
        ("级别", 66),
        ("置信度", 72),
        ("时长", 72),
        ("帧号", 76),
        ("说明", 320),
    )

    def __init__(self, max_rows: int = 500, parent: QWidget | None = None) -> None:
        super().__init__(0, len(self.HEADERS), parent)
        self._max_rows = max(10, max_rows)

        self.setHorizontalHeaderLabels([title for title, _ in self.HEADERS])
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setWordWrap(False)
        self.setShowGrid(False)
        self.verticalHeader().setDefaultSectionSize(26)

        header = self.horizontalHeader()
        for index, (_, width) in enumerate(self.HEADERS):
            if index == len(self.HEADERS) - 1:
                header.setSectionResizeMode(index, QHeaderView.ResizeMode.Stretch)
            else:
                header.setSectionResizeMode(index, QHeaderView.ResizeMode.Interactive)
                self.setColumnWidth(index, width)
        header.setHighlightSections(False)

        self._ongoing_rows: dict[str, int] = {}

    # ------------------------------------------------------------------
    def append_event(self, event: BehaviorEvent) -> None:
        """插入一条事件。

        同一次持续行为的重复提醒（``ongoing``）不会新增行，而是 **更新
        已有行** 的时长与置信度——否则一次长时间疲劳会刷满整个表格，
        把真正的新事件挤下去。
        """
        if event.ongoing and event.event_id not in self._ongoing_rows:
            key = self._find_ongoing_row(event)
            if key is not None:
                self._ongoing_rows[event.event_id] = key
        if event.ongoing:
            row = self._ongoing_rows.get(event.event_id)
            if row is not None and row < self.rowCount():
                self._fill_row(row, event, ongoing=True)
                return

        self.insertRow(0)
        self._fill_row(0, event)
        self._rebind_ongoing_rows()
        self._trim()

    def clear_events(self) -> None:
        self.setRowCount(0)
        self._ongoing_rows.clear()

    # ------------------------------------------------------------------
    def _fill_row(self, row: int, event: BehaviorEvent, *, ongoing: bool = False) -> None:
        values = (
            event.wall_clock.strftime("%H:%M:%S"),
            event.behavior.label,
            event.severity.label,
            f"{event.confidence:.0%}",
            f"{event.duration:.1f}s",
            str(event.frame_index),
            ("↻ " if ongoing else "") + event.message,
        )
        for column, text in enumerate(values):
            item = self.item(row, column)
            if item is None:
                item = QTableWidgetItem()
                self.setItem(row, column, item)
            item.setText(text)
            item.setToolTip(text)
            if column == 1:
                item.setForeground(QColor(BEHAVIOR_COLORS.get(event.behavior, Palette.TEXT)))
            elif column == 2:
                item.setForeground(QColor(SEVERITY_COLORS[event.severity]))
                font = QFont(item.font())
                font.setBold(True)
                item.setFont(font)
            elif column == 3:
                item.setForeground(QColor(_confidence_color(event.confidence)))
            else:
                item.setForeground(QColor(Palette.TEXT))

    def _rebind_ongoing_rows(self) -> None:
        """插入新行后，已存在的映射行号需要整体下移 1。"""
        self._ongoing_rows = {k: v + 1 for k, v in self._ongoing_rows.items()}

    def _find_ongoing_row(self, event: BehaviorEvent) -> int | None:
        """找到同一行为最近的一行（用于把"持续提醒"合并进原行）。"""
        for row in range(self.rowCount()):
            item = self.item(row, 1)
            if item is not None and item.text() == event.behavior.label:
                return row
        return None

    def _trim(self) -> None:
        excess = self.rowCount() - self._max_rows
        if excess > 0:
            self.setRowCount(self._max_rows)
            self._ongoing_rows = {
                k: v for k, v in self._ongoing_rows.items() if v < self._max_rows
            }


def _confidence_color(confidence: float) -> str:
    if confidence >= 0.85:
        return Palette.OK
    if confidence >= 0.65:
        return "#f59e0b"
    return Palette.TEXT_MUTED


# ===========================================================================
class StatisticsChart(QWidget):
    """极简水平条形图：展示各危险行为的累计次数。

    刻意不使用图表库（matplotlib/pyqtgraph）：

    * 少一个重依赖，部署更简单；
    * 只画条形图，``QPainter`` 几十行就够，且与主题完全一致。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(150)
        self._data: list[tuple[str, str, int]] = []
        self._total = 0

    # ------------------------------------------------------------------
    def set_data(self, data: Sequence[tuple[str, str, int]], total: int) -> None:
        """``data`` 为 ``[(行为键, 显示名, 次数), ...]``（已按次数降序）。"""
        self._data = list(data)
        self._total = total
        self.update()

    def clear_data(self) -> None:
        self._data = []
        self._total = 0
        self.update()

    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt 命名风格
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(Palette.BG_PANEL))

        if not self._data:
            painter.setPen(QColor(Palette.TEXT_DIM))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "暂无告警数据")
            painter.end()
            return

        margin = 10
        row_height = 22
        label_width = 84
        value_width = 44
        bar_left = margin + label_width
        bar_max = max(24, self.width() - bar_left - value_width - margin)
        maximum = max(count for _, _, count in self._data) or 1

        font = QFont(self.font())
        font.setPointSizeF(max(8.5, font.pointSizeF() - 0.5))
        painter.setFont(font)

        for index, (key, label, count) in enumerate(self._data):
            y = margin + index * row_height
            if y + row_height > self.height():
                break
            color = QColor(_behavior_color(key))

            painter.setPen(QColor(Palette.TEXT_MUTED))
            painter.drawText(margin, y + 15, label)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(Palette.BG_INPUT))
            painter.drawRoundedRect(bar_left, y + 4, bar_max, row_height - 12, 3, 3)

            width = int(bar_max * count / maximum)
            painter.setBrush(color)
            painter.drawRoundedRect(bar_left, y + 4, max(3, width), row_height - 12, 3, 3)

            painter.setPen(QColor(Palette.TEXT))
            painter.drawText(
                self.width() - value_width - margin + 4, y + 15, f"{count}"
            )

        painter.setPen(QPen(QColor(Palette.BORDER)))
        painter.drawLine(margin, self.height() - 6, self.width() - margin, self.height() - 6)
        painter.setPen(QColor(Palette.TEXT_DIM))
        painter.drawText(margin, self.height() - 10, f"合计 {self._total} 次")
        painter.end()


def _behavior_color(key: str) -> str:
    from ...domain.models import BehaviorType

    try:
        return BEHAVIOR_COLORS[BehaviorType(key)]
    except (ValueError, KeyError):
        return Palette.ACCENT


def severity_color(severity: Severity) -> str:
    """供其它模块复用的映射（避免各处重复取色）。"""
    return SEVERITY_COLORS[severity]


def category_color_map() -> Mapping[str, str]:
    return {category.value: color for category, color in CATEGORY_COLORS.items()}
