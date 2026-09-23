"""界面主题：配色表与 Qt 样式表。

把"颜色"集中到一处，使 **严重级别 → 视觉编码** 在全系统保持一致
（告警表格、状态卡、曲线、快照水印都用同一套色）。这类一致性能显著
降低驾驶员/监管员的识别负担，属于易用性质量属性的具体落地。

配色取暗色驾驶舱风格：深灰底 + 高对比强调色，直视强光下也可读。
"""

from __future__ import annotations

from ..domain.models import BehaviorCategory, BehaviorType, Severity

__all__ = ["Palette", "SEVERITY_COLORS", "BEHAVIOR_COLORS", "CATEGORY_COLORS", "STYLESHEET"]


class Palette:
    """基础调色板（集中定义，避免魔法色值散落）。"""

    BG_WINDOW = "#0f1419"
    BG_PANEL = "#161c24"
    BG_ELEVATED = "#1d252f"
    BG_INPUT = "#111820"

    BORDER = "#2a3441"
    BORDER_STRONG = "#3a4756"

    TEXT = "#e6edf3"
    TEXT_MUTED = "#8b98a5"
    TEXT_DIM = "#5f6b78"

    ACCENT = "#38bdf8"
    ACCENT_DIM = "#0e7490"
    OK = "#22c55e"


#: 严重级别 → 主色
SEVERITY_COLORS: dict[Severity, str] = {
    Severity.INFO: "#64748b",
    Severity.LOW: "#38bdf8",
    Severity.MEDIUM: "#f59e0b",
    Severity.HIGH: "#f97316",
    Severity.CRITICAL: "#ef4444",
}

#: 危险行为 → 图表强调色（用于统计柱状图区分）
BEHAVIOR_COLORS: dict[BehaviorType, str] = {
    BehaviorType.EYES_CLOSED: "#ef4444",
    BehaviorType.YAWNING: "#f59e0b",
    BehaviorType.HEAD_DOWN: "#f97316",
    BehaviorType.GAZE_OFF_ROAD: "#a78bfa",
    BehaviorType.PHONE_CALL: "#e11d48",
    BehaviorType.SMOKING: "#94a3b8",
    BehaviorType.DRIVER_ABSENT: "#0891b2",
}

#: 大类 → 颜色
CATEGORY_COLORS: dict[BehaviorCategory, str] = {
    BehaviorCategory.FATIGUE: "#f59e0b",
    BehaviorCategory.DISTRACTION: "#a78bfa",
    BehaviorCategory.VIOLATION: "#ef4444",
}


# ---------------------------------------------------------------------------
#  Qt 样式表
# ---------------------------------------------------------------------------
STYLESHEET = f"""
QWidget {{
    background-color: {Palette.BG_WINDOW};
    color: {Palette.TEXT};
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 13px;
}}

QMainWindow, QDialog {{ background-color: {Palette.BG_WINDOW}; }}

QGroupBox {{
    background-color: {Palette.BG_PANEL};
    border: 1px solid {Palette.BORDER};
    border-radius: 8px;
    margin-top: 14px;
    padding: 10px 10px 8px 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 2px 8px;
    color: {Palette.TEXT_MUTED};
    background-color: {Palette.BG_PANEL};
    border: 1px solid {Palette.BORDER};
    border-radius: 6px;
}}

QPushButton {{
    background-color: {Palette.BG_ELEVATED};
    border: 1px solid {Palette.BORDER_STRONG};
    border-radius: 6px;
    padding: 6px 14px;
    color: {Palette.TEXT};
    min-height: 20px;
}}
QPushButton:hover {{ background-color: #263241; border-color: {Palette.ACCENT_DIM}; }}
QPushButton:pressed {{ background-color: #0f1a24; }}
QPushButton:disabled {{ color: {Palette.TEXT_DIM}; border-color: {Palette.BORDER}; }}
QPushButton[accent="true"] {{
    background-color: {Palette.ACCENT_DIM};
    border-color: {Palette.ACCENT};
    font-weight: 600;
}}
QPushButton[accent="true"]:hover {{ background-color: #155e75; }}
QPushButton[danger="true"] {{
    background-color: #7f1d1d;
    border-color: #ef4444;
    font-weight: 600;
}}
QPushButton[danger="true"]:hover {{ background-color: #991b1b; }}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background-color: {Palette.BG_INPUT};
    border: 1px solid {Palette.BORDER};
    border-radius: 5px;
    padding: 4px 8px;
    min-height: 20px;
    selection-background-color: {Palette.ACCENT_DIM};
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{
    border-color: {Palette.BORDER_STRONG};
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background-color: {Palette.BG_ELEVATED};
    border: 1px solid {Palette.BORDER_STRONG};
    selection-background-color: {Palette.ACCENT_DIM};
    outline: none;
}}

QCheckBox, QRadioButton {{ spacing: 6px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 15px; height: 15px; }}
QCheckBox::indicator:unchecked {{
    background-color: {Palette.BG_INPUT};
    border: 1px solid {Palette.BORDER_STRONG};
    border-radius: 3px;
}}
QCheckBox::indicator:checked {{
    background-color: {Palette.ACCENT};
    border: 1px solid {Palette.ACCENT};
    border-radius: 3px;
}}

QTableWidget, QTableView {{
    background-color: {Palette.BG_PANEL};
    alternate-background-color: {Palette.BG_ELEVATED};
    border: 1px solid {Palette.BORDER};
    border-radius: 6px;
    gridline-color: {Palette.BORDER};
    selection-background-color: {Palette.ACCENT_DIM};
    selection-color: #ffffff;
}}
QHeaderView::section {{
    background-color: {Palette.BG_ELEVATED};
    color: {Palette.TEXT_MUTED};
    border: none;
    border-right: 1px solid {Palette.BORDER};
    border-bottom: 1px solid {Palette.BORDER};
    padding: 6px 8px;
    font-weight: 600;
}}
QTableCornerButton::section {{ background-color: {Palette.BG_ELEVATED}; border: none; }}

QProgressBar {{
    background-color: {Palette.BG_INPUT};
    border: 1px solid {Palette.BORDER};
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{ background-color: {Palette.ACCENT}; border-radius: 3px; }}

QTabWidget::pane {{
    border: 1px solid {Palette.BORDER};
    border-radius: 6px;
    background-color: {Palette.BG_PANEL};
    top: -1px;
}}
QTabBar::tab {{
    background-color: {Palette.BG_WINDOW};
    color: {Palette.TEXT_MUTED};
    border: 1px solid {Palette.BORDER};
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 6px 16px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background-color: {Palette.BG_PANEL};
    color: {Palette.TEXT};
    font-weight: 600;
}}
QTabBar::tab:hover:!selected {{ color: {Palette.TEXT}; }}

QScrollBar:vertical {{
    background: transparent; width: 10px; margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {Palette.BORDER_STRONG}; border-radius: 4px; min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: {Palette.TEXT_DIM}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{
    background: {Palette.BORDER_STRONG}; border-radius: 4px; min-width: 28px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0px; width: 0px; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QStatusBar {{
    background-color: {Palette.BG_PANEL};
    border-top: 1px solid {Palette.BORDER};
    color: {Palette.TEXT_MUTED};
}}
QStatusBar::item {{ border: none; }}

QSplitter::handle {{ background-color: {Palette.BORDER}; }}
QSplitter::handle:horizontal {{ width: 2px; }}
QSplitter::handle:vertical {{ height: 2px; }}

QToolTip {{
    background-color: {Palette.BG_ELEVATED};
    color: {Palette.TEXT};
    border: 1px solid {Palette.BORDER_STRONG};
    padding: 5px 8px;
    border-radius: 4px;
}}

QMenuBar {{ background-color: {Palette.BG_PANEL}; }}
QMenuBar::item:selected {{ background-color: {Palette.BG_ELEVATED}; }}
QMenu {{
    background-color: {Palette.BG_ELEVATED};
    border: 1px solid {Palette.BORDER_STRONG};
}}
QMenu::item:selected {{ background-color: {Palette.ACCENT_DIM}; }}
QLabel[role="title"] {{ font-size: 15px; font-weight: 600; }}
QLabel[role="hint"] {{ color: {Palette.TEXT_MUTED}; font-size: 12px; }}
QLabel[role="value"] {{ font-size: 15px; font-weight: 600; }}
QLabel[role="bigValue"] {{ font-size: 24px; font-weight: 700; }}
"""
