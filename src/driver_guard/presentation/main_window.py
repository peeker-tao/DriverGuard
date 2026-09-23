"""主窗口（View 层）。

View 的纪律
-----------
本文件 **不做任何业务判断**，只做三件事：

1. 把用户操作转发给 :class:`~driver_guard.presentation.viewmodels.monitor_viewmodel.MonitorViewModel`；
2. 把 ViewModel 的信号渲染到控件上；
3. 管理窗口自身的布局与生命周期。

布局
----

.. code-block:: text

   ┌────────────────────────────────────────────── 工具栏 ──────────────────────────────────────────────┐
   │ 数据源 ▾   开始检测  停止  │ 重新标定 │ ☑关键点 ☑人脸框 ☑指标 │ 导出报告 │ 系统日志              │
   ├────────────────────── 视频区 ───────────────────────┬────────────── 右侧信息栏 ──────────────────┤
   │                                                       │  运行状态（大号文字 + 行为芯片）            │
   │                 （叠加了关键点/框/HUD）                │  状态指标（EAR/MAR/头姿/手距）              │
   │                                                       │  本次会话统计（条形图）                     │
   ├───────────────────────────────────────────────────────┴────────────────────────────────────────────┤
   │  告警事件流水（按时间倒序，持续行为合并为一行）                                                        │
   └──────────────────────────────────────────────────────────────────────────────────────────────────────┘
   │ 状态栏：帧号 | FPS | 推理耗时 | 事件数 | 感知模式 | 相机                                              │
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QSize, Qt, Slot
from PySide6.QtGui import QAction, QCloseEvent, QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..application.services import SessionSummary
from ..domain.models import BehaviorEvent, BehaviorType, Severity
from .theme import SEVERITY_COLORS, Palette
from .viewmodels.monitor_viewmodel import MonitorViewModel
from .widgets.event_table import EventTable, StatisticsChart
from .widgets.panels import MetricsPanel, StatusPanel
from .widgets.video_panel import VideoPanel

logger = logging.getLogger(__name__)

__all__ = ["MainWindow"]


class MainWindow(QMainWindow):
    """系统主窗口。"""

    def __init__(self, view_model: MonitorViewModel) -> None:
        super().__init__()
        self._vm = view_model
        self._summary: SessionSummary | None = None

        self.setWindowTitle("驾驶员危险行为检测系统  ·  Driver Guard")
        self.resize(1520, 920)
        self.setMinimumSize(QSize(1180, 740))

        self._build_ui()
        self._bind()
        self._apply_source_hints()

    # ==================================================================
    #  构建界面
    # ==================================================================
    def _build_ui(self) -> None:
        self._build_toolbar()

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(10, 6, 10, 6)
        outer.setSpacing(8)

        vertical = QSplitter(Qt.Orientation.Vertical)
        vertical.addWidget(self._build_upper_area())
        vertical.addWidget(self._build_event_area())
        vertical.setStretchFactor(0, 5)
        vertical.setStretchFactor(1, 2)
        vertical.setSizes([620, 220])
        outer.addWidget(vertical)

        self.setCentralWidget(central)

        status = QStatusBar()
        self.setStatusBar(status)
        self._lbl_frame = QLabel("帧 —")
        self._lbl_fps = QLabel("FPS —")
        self._lbl_latency = QLabel("推理 —")
        self._lbl_events = QLabel("告警 0")
        self._lbl_mode = QLabel("感知 —")
        for label in (self._lbl_frame, self._lbl_fps, self._lbl_latency,
                      self._lbl_events, self._lbl_mode):
            label.setStyleSheet(f"color: {Palette.TEXT_MUTED}; padding: 0 10px;")
            status.addPermanentWidget(label)

    # ------------------------------------------------------------------
    def _build_toolbar(self) -> None:
        bar = QToolBar("主工具栏")
        bar.setMovable(False)
        bar.setIconSize(QSize(18, 18))
        self.addToolBar(bar)

        bar.addWidget(QLabel(" 数据源 "))
        self._cmb_source = QComboBox()
        self._cmb_source.addItem("摄像头", "camera")
        self._cmb_source.addItem("回放演示（无需摄像头）", "replay")
        self._cmb_source.addItem("合成画面", "synthetic")
        self._cmb_source.addItem("视频文件", "video")
        self._cmb_source.setMinimumWidth(210)
        self._cmb_source.setToolTip(
            "摄像头：实时检测\n"
            "回放演示：合成画面 + 内置行为脚本，答辩/无摄像头时使用\n"
            "合成画面：仅验证界面与流水线（不会有告警）\n"
            "视频文件：离线分析 assets/samples 下的录像"
        )
        bar.addWidget(self._cmb_source)
        bar.addSeparator()

        self._act_start = QAction("▶  开始检测", self)
        self._act_start.setShortcut("F5")
        self._act_stop = QAction("■  停止", self)
        self._act_stop.setShortcut("F6")
        self._act_stop.setEnabled(False)
        bar.addAction(self._act_start)
        bar.addAction(self._act_stop)
        bar.addSeparator()

        self._act_recalibrate = QAction("重新标定中立位", self)
        self._act_recalibrate.setToolTip(
            "清除个人中立位基准，随后约 1 秒内重新学习驾驶员自然姿态。\n"
            "更换驾驶员或调整摄像头后建议执行一次。"
        )
        bar.addAction(self._act_recalibrate)
        bar.addSeparator()

        self._chk_landmarks = QCheckBox("关键点")
        self._chk_facebox = QCheckBox("人脸框")
        self._chk_hud = QCheckBox("指标 HUD")
        for check, checked in (
            (self._chk_landmarks, self._vm.overlay_options.show_landmarks),
            (self._chk_facebox, self._vm.overlay_options.show_face_box),
            (self._chk_hud, self._vm.overlay_options.show_hud),
        ):
            check.setChecked(checked)
            check.setStyleSheet("padding: 0 8px;")
            bar.addWidget(check)
        bar.addSeparator()

        self._act_export = QAction("导出报告", self)
        self._act_export.setEnabled(False)
        bar.addAction(self._act_export)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)

    # ------------------------------------------------------------------
    def _build_upper_area(self) -> QWidget:
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._video = VideoPanel()
        video_box = QGroupBox("实时画面")
        video_layout = QVBoxLayout(video_box)
        video_layout.setContentsMargins(6, 4, 6, 6)
        video_layout.addWidget(self._video)
        splitter.addWidget(video_box)

        splitter.addWidget(self._build_side_panel())
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([1020, 440])
        return splitter

    def _build_side_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        status_box = QGroupBox("运行状态")
        status_layout = QVBoxLayout(status_box)
        status_layout.setContentsMargins(10, 6, 10, 10)
        self._status = StatusPanel()
        status_layout.addWidget(self._status)
        layout.addWidget(status_box)

        tabs = QTabWidget()
        metrics_box = QWidget()
        metrics_layout = QVBoxLayout(metrics_box)
        metrics_layout.setContentsMargins(10, 8, 10, 8)
        self._metrics = MetricsPanel()
        metrics_layout.addWidget(self._metrics)
        tabs.addTab(metrics_box, "状态指标")

        stats_page = QWidget()
        stats_layout = QVBoxLayout(stats_page)
        stats_layout.setContentsMargins(6, 6, 6, 6)
        self._chart = StatisticsChart()
        stats_layout.addWidget(self._chart)
        self._lbl_stats = QLabel("暂无数据")
        self._lbl_stats.setProperty("role", "hint")
        self._lbl_stats.setWordWrap(True)
        stats_layout.addWidget(self._lbl_stats)
        tabs.addTab(stats_page, "本次统计")

        self._tabs = tabs
        layout.addWidget(tabs, 1)
        return container

    def _build_event_area(self) -> QWidget:
        box = QGroupBox("告警事件流水")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 6)

        header = QHBoxLayout()
        self._lbl_alert_state = QLabel("暂无告警")
        self._lbl_alert_state.setProperty("role", "hint")
        header.addWidget(self._lbl_alert_state)
        header.addStretch(1)
        self._btn_clear = QPushButton("清空列表")
        self._btn_clear.setToolTip("只清空界面显示，不影响已写入 output/events/events.csv 的记录")
        header.addWidget(self._btn_clear)
        layout.addLayout(header)

        self._table = EventTable(max_rows=self._vm.config.ui.max_events_in_table)
        layout.addWidget(self._table)
        return box

    # ==================================================================
    #  信号绑定
    # ==================================================================
    def _bind(self) -> None:
        self._act_start.triggered.connect(self._on_start)
        self._act_stop.triggered.connect(self._vm.stop)
        self._act_recalibrate.triggered.connect(self._vm.recalibrate)
        self._act_export.triggered.connect(self._on_export_report)
        self._btn_clear.clicked.connect(self._on_clear_events)
        self._cmb_source.currentIndexChanged.connect(self._on_source_changed)

        self._chk_landmarks.toggled.connect(
            lambda on: self._vm.set_overlay("landmarks", on)
        )
        self._chk_facebox.toggled.connect(lambda on: self._vm.set_overlay("face_box", on))
        self._chk_hud.toggled.connect(lambda on: self._vm.set_overlay("hud", on))

        self._vm.frameReady.connect(self._on_frame)
        self._vm.eventRaised.connect(self._on_event)
        self._vm.stateChanged.connect(self._on_state)
        self._vm.telemetry.connect(self._on_telemetry)
        self._vm.sessionFinished.connect(self._on_session_finished)
        self._vm.errorOccurred.connect(self._on_error)
        self._vm.providerChanged.connect(self._on_provider_changed)
        self._vm.statisticsChanged.connect(self._on_statistics)

    # ==================================================================
    #  槽：用户操作
    # ==================================================================
    @Slot()
    def _on_start(self) -> None:
        if self._vm.is_running:
            return
        self._table.clear_events()
        self._metrics.clear()
        self._chart.clear_data()
        self._lbl_stats.setText("暂无数据")
        self._lbl_alert_state.setText("暂无告警")
        self._video.clear("正在启动…", "首次启动需加载感知模型，约 1~3 秒")
        if self._vm.start():
            self._act_start.setEnabled(False)
            self._act_stop.setEnabled(True)
            self._act_export.setEnabled(False)
            self._cmb_source.setEnabled(False)
        else:
            self._act_start.setEnabled(True)

    @Slot()
    def _on_clear_events(self) -> None:
        self._table.clear_events()
        self._lbl_alert_state.setText("界面列表已清空（CSV 记录保留）")

    @Slot(int)
    def _on_source_changed(self, index: int) -> None:
        """切换数据源：重建数据源与感知适配器（需要重启会话）。"""
        if self._vm.is_running:
            QMessageBox.information(self, "请先停止", "切换数据源前请先停止当前检测。")
            return
        kind = self._cmb_source.itemData(index)
        logger.info("用户选择数据源：%s", kind)

        config = self._vm.config
        provider = config.perception.provider
        warnings: list[str] = []
        if kind in {"replay", "synthetic"} and provider != "replay":
            warnings.append(
                "合成画面不含真实人脸，回放演示需要把感知实现切为 replay，"
                "否则界面会一直显示「未检测到驾驶员」。\n"
                "请用命令行参数启动：--provider replay --source replay"
            )
        if kind == "video" and not config.resolved_video_path().exists():
            warnings.append(f"视频文件不存在：{config.resolved_video_path()}")
        if warnings:
            QMessageBox.warning(self, "数据源提示", "\n\n".join(warnings))

        self._apply_source_hints(kind)

    @Slot()
    def _on_export_report(self) -> None:
        if self._summary is None:
            return
        default = str(self._vm.config.output.resolve("session_report.txt"))
        path, _ = QFileDialog.getSaveFileName(
            self, "导出本次会话报告", default, "文本文件 (*.txt);;所有文件 (*)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self._render_report(self._summary))
        except OSError as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        QMessageBox.information(self, "导出完成", f"报告已保存到：\n{path}")

    # ==================================================================
    #  槽：ViewModel 信号
    # ==================================================================
    @Slot(object, object)
    def _on_frame(self, canvas: Any, analysis: Any) -> None:
        self._video.set_frame(canvas)
        self._metrics.update_metrics(analysis.metrics.as_dict())
        self._status.set_active(self._vm.active_events)

        if analysis.calibration_progress < 1.0:
            self._status.set_state(
                "正在标定中立位",
                "请保持正常坐姿，约 1 秒后自动开始检测",
                Palette.ACCENT,
            )
        elif not analysis.observation.has_face:
            self._status.set_state("未检测到驾驶员", "请确认驾驶员入座且摄像头未被遮挡", "#f59e0b")
        else:
            active = self._vm.active_events
            if active:
                worst = max(active, key=lambda e: e.severity)
                self._status.set_state(
                    "检测到危险行为",
                    f"当前最严重：{worst.behavior.label}（{worst.severity.label}）",
                    SEVERITY_COLORS[worst.severity],
                )
            else:
                self._status.set_state("驾驶状态正常", "未发现危险行为", Palette.OK)

    @Slot(object)
    def _on_event(self, event: BehaviorEvent) -> None:
        self._table.append_event(event)
        if not event.ongoing:
            self._lbl_alert_state.setText(
                f"最近告警：{event.wall_clock:%H:%M:%S}  {event.summary}"
            )

    @Slot(str, str)
    def _on_state(self, state: str, detail: str) -> None:
        self._lbl_mode.setText(f"感知 {state}")
        if detail:
            self.statusBar().showMessage(detail, 4000)
        running = state in {"运行中", "停止中"}
        self._act_start.setEnabled(not running)
        self._act_stop.setEnabled(running)
        self._cmb_source.setEnabled(not running)
        if state == "已停止":
            self._video.clear("已停止", "点击「开始检测」重新开始")
        elif state == "已完成":
            self._video.clear("本次数据源已播放完毕", "可切换数据源或点击「开始检测」重放")
        elif state == "异常":
            self._video.clear("检测异常终止")

    @Slot(int, float, float, int)
    def _on_telemetry(self, frame_index: int, fps: float, latency_ms: float, events: int) -> None:
        self._lbl_frame.setText(f"帧 {frame_index}")
        self._lbl_fps.setText(f"FPS {fps:.1f}")
        self._lbl_latency.setText(f"推理 {latency_ms:.0f} ms")
        self._lbl_events.setText(f"告警 {events}")

    @Slot(object)
    def _on_session_finished(self, summary: SessionSummary) -> None:
        self._summary = summary
        self._act_start.setEnabled(True)
        self._act_stop.setEnabled(False)
        self._act_export.setEnabled(True)
        self._cmb_source.setEnabled(True)
        self._status.set_active(())
        self._on_statistics(self._vm.statistics_snapshot())

        stages = "  ".join(f"{k} {v:.1f}ms" for k, v in summary.stage_avg_ms.items())
        self.statusBar().showMessage(
            f"会话结束：{summary.frames} 帧 / {summary.duration:.1f}s / "
            f"平均 {summary.avg_fps:.1f} FPS / 风险指数 {summary.risk_index}　{stages}",
            15000,
        )
        QMessageBox.information(
            self,
            "本次检测结束",
            self._render_report(summary),
        )

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._act_start.setEnabled(True)
        self._act_stop.setEnabled(False)
        self._cmb_source.setEnabled(True)
        self._video.clear("启动失败")
        logger.error("界面收到错误：%s", message)
        QMessageBox.critical(self, "运行错误", message)

    @Slot(object)
    def _on_provider_changed(self, info: Any) -> None:
        if not isinstance(info, dict) or not info.get("initialized"):
            return
        self._lbl_mode.setText(f"感知 {info.get('provider', '—')}")
        self._status.configure_behaviors(
            {key: BehaviorType(key).label for key in info.get("rules", [])}
        )
        self._lbl_stats.setText(
            f"数据源：{info.get('source')}　告警通道：{'、'.join(info.get('sinks', [])) or '无'}"
        )
        if info.get("degraded"):
            self.statusBar().showMessage(f"注意：{info.get('degraded_reason', '部分通道降级')}", 12000)

    @Slot(object)
    def _on_statistics(self, snapshot: Any) -> None:
        if not isinstance(snapshot, dict):
            return
        self._chart.set_data(snapshot.get("top", []), snapshot.get("total", 0))
        by_severity: dict[str, int] = snapshot.get("by_severity", {})
        if by_severity:
            detail = "　".join(
                f"{Severity.parse(name).label} {count}"
                for name, count in sorted(by_severity.items(),
                                          key=lambda kv: -Severity.parse(kv[0]))
            )
            self._lbl_stats.setText(detail)

    # ==================================================================
    #  辅助
    # ==================================================================
    def _apply_source_hints(self, kind: str | None = None) -> None:
        kind = kind or self._cmb_source.currentData()
        hints = {
            "camera": "摄像头实时检测",
            "replay": "回放演示（内置行为脚本，无需摄像头）",
            "synthetic": "合成画面（无真实人脸，不会触发告警）",
            "video": "离线分析视频文件",
        }
        self._video.clear("未开始检测", hints.get(kind, ""))

    def _render_report(self, summary: SessionSummary) -> str:
        lines = [
            "驾驶员危险行为检测系统 — 会话报告",
            "=" * 46,
            f"数据源      ：{summary.source_name}",
            f"开始时间    ：{summary.started_at:%Y-%m-%d %H:%M:%S}",
            f"结束时间    ：{summary.ended_at:%Y-%m-%d %H:%M:%S}",
            f"处理帧数    ：{summary.frames}",
            f"会话时长    ：{summary.duration:.2f} 秒",
            f"平均帧率    ：{summary.avg_fps:.2f} FPS",
            f"含危险行为帧：{summary.triggered_frames}",
            f"风险指数    ：{summary.risk_index} / 100",
            "",
            "告警统计",
            "-" * 46,
        ]
        if summary.event_counts:
            for behavior, count in sorted(
                summary.event_counts.items(), key=lambda kv: -kv[1]
            ):
                lines.append(
                    f"  {behavior.label:<8} {behavior.category.label:<6} × {count}"
                )
        else:
            lines.append("  本次会话未产生任何告警")

        if summary.stage_avg_ms:
            lines += ["", "各阶段平均耗时", "-" * 46]
            for stage, value in sorted(summary.stage_avg_ms.items()):
                lines.append(f"  {stage:<12} {value:7.2f} ms")

        lines += [
            "",
            "说明：风险指数 = 单位时间（分钟）加权告警数 × 5，按严重级别加权后截断到 100。",
            "      该指标用于会话间横向对比，不是物理量。",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt 命名风格
        logger.info("窗口关闭，正在释放资源…")
        self._vm.shutdown()
        event.accept()


# ===========================================================================
def center_on_screen(window: QMainWindow) -> None:
    """把窗口居中到主屏幕（多显示器下体验更稳定）。"""
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        return
    geometry = window.frameGeometry()
    geometry.moveCenter(screen.availableGeometry().center())
    window.move(geometry.topLeft())


__all__ = ["MainWindow", "center_on_screen"]
