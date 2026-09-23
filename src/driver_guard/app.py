"""应用引导：日志、Qt 环境与界面启动。

把"进程级初始化"（日志、Qt 属性、样式表）与"业务装配"分开：
前者在 :mod:`driver_guard.main` 中被调用一次，后者由
:mod:`driver_guard.bootstrap` 负责。
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from .config import AppConfig, PROJECT_ROOT

__all__ = ["setup_logging", "create_qapplication", "run_gui"]

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-38s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


# ===========================================================================
def setup_logging(config: AppConfig, *, console_level: str | None = None) -> Path | None:
    """配置根日志器：控制台 + 滚动文件。

    Args:
        config: 应用配置（提供日志级别与文件路径）。
        console_level: 覆盖控制台日志级别（命令行 ``--log-level``）。

    Returns:
        日志文件路径；无法创建时返回 ``None``。
    """
    level = _parse_level(console_level or config.logging.level)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    root.addHandler(console)

    log_path: Path | None = None
    try:
        log_path = Path(config.logging.file)
        log_path = log_path if log_path.is_absolute() else PROJECT_ROOT / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # 滚动文件：单文件 2 MB，保留 3 个备份，避免长期运行写满磁盘
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(file_handler)
    except OSError:
        log_path = None
        logger.warning("无法创建日志文件，仅输出到控制台")

    # 第三方库日志过于聒噪，统一压到 WARNING
    for noisy in ("mediapipe", "absl", "PIL", "matplotlib", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.debug("日志初始化完成，级别=%s", logging.getLevelName(level))
    return log_path


def _parse_level(name: str) -> int:
    value = getattr(logging, (name or "INFO").strip().upper(), None)
    if isinstance(value, int):
        return value
    logger.warning("未知日志级别 %r，改用 INFO", name)
    return logging.INFO


# ===========================================================================
def create_qapplication(argv: list[str] | None = None):
    """创建 :class:`QApplication` 并应用暗色主题（同时适配高分屏）。"""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    from .presentation.theme import STYLESHEET

    # 高分屏缩放：Qt6 默认已开启，这里显式保留以便在 Qt5 上行为一致
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setApplicationName("Driver Guard")
    app.setApplicationDisplayName("驾驶员危险行为检测系统")
    app.setOrganizationName("软件体系结构课程设计")
    app.setApplicationVersion("1.0.0")
    app.setStyleSheet(STYLESHEET)
    return app


# ===========================================================================
def run_gui(config: AppConfig) -> int:
    """启动图形界面，返回进程退出码。"""
    from PySide6.QtWidgets import QMessageBox

    from .presentation.main_window import MainWindow, center_on_screen
    from .presentation.viewmodels.monitor_viewmodel import MonitorViewModel

    app = create_qapplication()

    view_model = MonitorViewModel(config)
    window = MainWindow(view_model)
    center_on_screen(window)
    window.show()

    # 提前装配：把模型缺失、摄像头不可用等问题在界面上即时暴露，
    # 而不是等用户点了"开始检测"才失败。
    if not view_model.initialize():
        QMessageBox.warning(
            window,
            "初始化未完成",
            "应用已启动，但检测组件初始化失败。\n"
            "界面可正常浏览，请根据提示处理后点击「开始检测」重试。",
        )

    for warning in config.warnings:
        logger.warning("配置提示：%s", warning)

    return app.exec()
