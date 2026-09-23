"""指标面板与状态面板。

两个控件的共同设计原则：**只接受"已经算好"的数值与文本**，
不含任何阈值判断——是否该变红、是否算超标，全部由 ViewModel / 领域规则决定。
这样界面逻辑可以被直接单元测试（给相同的输入必然得到相同的渲染状态）。
"""

from __future__ import annotations

from typing import Mapping

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...domain.models import BehaviorEvent
from ..theme import Palette, SEVERITY_COLORS

__all__ = ["MetricsPanel", "StatusPanel", "MetricRow"]


# ===========================================================================
class MetricRow(QFrame):
    """一行「指标名 + 数值 + 进度条」。进度条用于直观呈现"离阈值还有多远」。"""

    def __init__(self, title: str, unit: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._unit = unit
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)

        self._label = QLabel(title)
        self._label.setFixedWidth(60)
        self._label.setStyleSheet(f"color: {Palette.TEXT_MUTED};")

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(7)
        self._bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._value = QLabel("—")
        self._value.setFixedWidth(66)
        self._value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._value.setStyleSheet("font-family: Consolas, 'Cascadia Mono', monospace;")

        layout.addWidget(self._label)
        layout.addWidget(self._bar, 1)
        layout.addWidget(self._value)

    # ------------------------------------------------------------------
    def set_value(
        self,
        value: float | None,
        *,
        ratio: float = 0.0,
        color: str | None = None,
    ) -> None:
        """更新数值。

        Args:
            value: 显示数值；``None`` 表示该指标不可用（显示 ``—``）。
            ratio: 进度条填充比例（0~1），由调用方根据阈值换算。
            color: 进度条颜色；``None`` 使用主题强调色。
        """
        if value is None:
            self._value.setText("—")
            self._bar.setValue(0)
            return
        self._value.setText(f"{value:.3f}{self._unit}" if self._unit else f"{value:.3f}")
        self._bar.setValue(int(max(0.0, min(1.0, ratio)) * 100))
        chunk = color or Palette.ACCENT
        self._bar.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {chunk}; border-radius: 3px; }}"
        )

    def set_raw_text(self, text: str) -> None:
        self._value.setText(text)


# ===========================================================================
class MetricsPanel(QWidget):
    """驾驶员状态指标面板。"""

    #: ``(指标键, 显示名, 单位, 参考上限)``
    #: 参考上限仅用于把数值映射到进度条长度，**不是** 判定阈值。
    SPECS: tuple[tuple[str, str, str, float], ...] = (
        ("ear", "EAR", "", 0.45),
        ("mar", "MAR", "", 0.90),
        ("eye_blink", "眨眼", "", 1.0),
        ("jaw_open", "张口", "", 1.0),
        ("pitch", "俯仰", "°", 60.0),
        ("yaw", "偏航", "°", 60.0),
        ("roll", "翻滚", "°", 45.0),
        ("face_pitch_ratio", "低头比", "", 0.80),
        ("hand_face_ratio", "手-脸", "", 1.60),
        ("hand_mouth_ratio", "手-嘴", "", 1.60),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        self._rows: dict[str, MetricRow] = {}
        for key, title, unit, _ in self.SPECS:
            row = MetricRow(title, unit)
            self._rows[key] = row
            layout.addWidget(row)

        self._footer = QLabel("指标不可用时显示为 —（例如降级感知通道不提供头姿）")
        self._footer.setProperty("role", "hint")
        self._footer.setWordWrap(True)
        layout.addSpacing(4)
        layout.addWidget(self._footer)
        layout.addStretch(1)

    # ------------------------------------------------------------------
    def update_metrics(self, metrics: Mapping[str, float]) -> None:
        for key, _, _, ceiling in self.SPECS:
            raw = metrics.get(key)
            value = float(raw) if raw is not None else None
            ratio = 0.0 if value is None else abs(value) / ceiling
            self._rows[key].set_value(value, ratio=ratio)

    def clear(self) -> None:
        for row in self._rows.values():
            row.set_value(None)


# ===========================================================================
class StatusPanel(QWidget):
    """当前状态总览：一句话结论 + 各行为芯片 + 生效通道说明。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._headline = QLabel("未开始检测")
        self._headline.setProperty("role", "bigValue")
        self._headline.setWordWrap(True)
        layout.addWidget(self._headline)

        self._subline = QLabel("—")
        self._subline.setProperty("role", "hint")
        self._subline.setWordWrap(True)
        layout.addWidget(self._subline)

        self._chips_host = QWidget()
        self._chips = QGridLayout(self._chips_host)
        self._chips.setContentsMargins(0, 4, 0, 0)
        self._chips.setHorizontalSpacing(6)
        self._chips.setVerticalSpacing(6)
        layout.addWidget(self._chips_host)

        self._chips_labels: dict[str, QLabel] = {}
        layout.addStretch(1)

    # ------------------------------------------------------------------
    def configure_behaviors(self, behaviors: Mapping[str, str]) -> None:
        """按启用规则动态生成芯片（``{行为键: 显示名}``）。"""
        while self._chips.count():
            item = self._chips.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._chips_labels.clear()

        for index, (key, label) in enumerate(behaviors.items()):
            chip = QLabel(label)
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chip.setContentsMargins(8, 3, 8, 3)
            self._apply_chip_style(chip, Palette.BORDER, Palette.TEXT_DIM)
            self._chips.addWidget(chip, index // 3, index % 3)
            self._chips_labels[key] = chip

    def set_state(self, headline: str, subline: str = "", color: str | None = None) -> None:
        self._headline.setText(headline)
        self._headline.setStyleSheet(
            f"color: {color or Palette.TEXT}; font-size: 24px; font-weight: 700;"
        )
        self._subline.setText(subline)

    def set_active(self, events: tuple[BehaviorEvent, ...]) -> None:
        """高亮当前激活的行为芯片。"""
        active = {event.behavior.value: event for event in events}
        for key, chip in self._chips_labels.items():
            event = active.get(key)
            if event is None:
                self._apply_chip_style(chip, Palette.BORDER, Palette.TEXT_DIM)
                chip.setText(_strip_percent(chip.text()))
            else:
                color = SEVERITY_COLORS[event.severity]
                self._apply_chip_style(chip, color, color, bold=True)
                chip.setText(f"{_strip_percent(chip.text())} {event.confidence:.0%}")

    @staticmethod
    def _apply_chip_style(
        chip: QLabel, border: str, text: str, *, bold: bool = False
    ) -> None:
        chip.setStyleSheet(
            f"QLabel {{ border: 1px solid {border}; border-radius: 10px;"
            f" color: {text}; padding: 3px 9px;"
            f" font-weight: {'700' if bold else '400'}; }}"
        )


def _strip_percent(text: str) -> str:
    """把芯片文字尾部的百分比去掉，避免多次追加而越变越长。"""
    head, _, tail = text.rpartition(" ")
    if tail.endswith("%") and head:
        return head
    return text.strip()
