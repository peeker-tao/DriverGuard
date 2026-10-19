"""视图模型（ViewModel）—— MVVM 的核心。

职责边界
--------

============  ==========================================================
层次           负责
============  ==========================================================
Model         ``domain`` + ``application``：检测逻辑与领域状态
ViewModel     本模块：持有模型、暴露 **可直接绑定** 的属性与信号、管理线程
View          ``main_window`` 与 ``widgets``：只做渲染与用户输入转发
============  ==========================================================

View 里 **不出现任何业务判断**（例如"置信度是否够高""该不该显示红色"），
这类决策全部由 ViewModel 以"已加工好的数据"形式提供。收益：

* 界面可以被替换（改成 Web 前端）而业务判断零改动；
* ViewModel 的信号契约可被单元测试，无需真实 GUI。

信号契约（View 只需连接这些）
-----------------------------
===========================  ==================================================
信号                          载荷
===========================  ==================================================
``frameReady``                ``(QImage 数据源: numpy 数组, FrameAnalysis)``
``eventRaised``                ``BehaviorEvent``
``stateChanged``               ``(状态名: str, 附加说明: str)``
``telemetry``                  ``(帧号, FPS, 感知耗时ms, 事件累计数)``
``statisticsChanged``          ``dict``：行为/级别统计快照
``sessionFinished``            ``SessionSummary``
``errorOccurred``              ``str``
``providerChanged``            提供者/数据源/降级信息的字典
===========================  ==========================================================
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from PySide6.QtCore import QObject, QThread, Signal, Slot

from ...application.pipeline import FrameAnalysis
from ...application.services import SessionSummary
from ...bootstrap import ApplicationContext, build_application
from ...config import AppConfig
from ...domain.models import BehaviorCategory, BehaviorType, Severity
from ...infrastructure.alerts import CallbackAlertSink
from ..overlay import OverlayOptions
from ..worker import DetectionWorker

logger = logging.getLogger(__name__)

__all__ = ["MonitorViewModel"]


class MonitorViewModel(QObject):
    """检测会话的视图模型。"""

    # --- 数据流 --------------------------------------------------------
    frameReady = Signal(object, object)
    eventRaised = Signal(object)
    # --- 状态 ----------------------------------------------------------
    stateChanged = Signal(str, str)
    telemetry = Signal(int, float, float, int)
    sessionFinished = Signal(object)
    errorOccurred = Signal(str)
    providerChanged = Signal(object)
    # --- 统计 ----------------------------------------------------------
    statisticsChanged = Signal(object)

    def __init__(self, config: AppConfig, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._context: ApplicationContext | None = None
        self._worker: DetectionWorker | None = None
        self._thread: QThread | None = None
        self._options = OverlayOptions(
            show_landmarks=config.ui.show_landmarks,
            show_face_box=config.ui.show_face_box,
            show_hud=config.ui.show_hud,
        )

    # ==================================================================
    #  只读属性（View 绑定用）
    # ==================================================================
    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def context(self) -> ApplicationContext | None:
        return self._context

    @property
    def overlay_options(self) -> OverlayOptions:
        return self._options

    @property
    def event_total(self) -> int:
        """本会话已确认的告警总数（与统计服务同源）。"""
        if self._context is None:
            return 0
        return self._context.statistics.total

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.is_running

    @property
    def active_events(self) -> tuple[Any, ...]:
        return self._worker.active_events if self._worker else ()

    # ==================================================================
    #  生命周期
    # ==================================================================
    def initialize(self) -> bool:
        """装配应用上下文。失败时发出 :attr:`errorOccurred` 并返回 ``False``。"""
        if self._context is not None:
            return True

        # 界面通过告警通道之一接收事件，从而复用同一套过滤与错误隔离机制
        try:
            context = build_application(self._config)
        except Exception as exc:  # noqa: BLE001 - 把装配错误转成界面可读信息
            logger.exception("应用装配失败")
            self.errorOccurred.emit(_friendly_build_error(exc))
            return False

        worker = DetectionWorker(context, overlay_options=self._options,
                                 display_fps=self._config.source.fps or 30.0)
        thread = QThread()
        worker.moveToThread(thread)

        # worker 作为告警通道接收事件：维护"当前激活行为"并向界面转发信号。
        # 注意必须在启动线程前注册，否则运行中修改通道列表会与事件分发竞争。
        context.alert_service.add_sink(
            CallbackAlertSink(worker.on_event, label="界面")
        )

        # 信号转发（工作线程 → 界面线程）
        worker.frameReady.connect(self._on_frame_ready)
        worker.eventRaised.connect(self.eventRaised)
        worker.telemetry.connect(self._on_telemetry)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self.errorOccurred)
        worker.stateChanged.connect(self._on_state_changed)

        thread.started.connect(worker.run)
        thread.finished.connect(worker.deleteLater)

        self._context = context
        self._worker = worker
        self._thread = thread

        # 统计服务聚合完成后直接通知界面，不能再经由告警通道触发，
        # 否则快照会落后一条事件（告警服务先于统计服务订阅事件总线）。
        context.statistics.add_listener(self._on_statistics_changed)

        self.providerChanged.emit(self.describe_sources())
        for warning in context.warnings:
            logger.warning("装配提示：%s", warning)
        return True

    def start(self) -> bool:
        """启动检测（幂等）。"""
        if self._context is None and not self.initialize():
            return False
        assert self._thread is not None
        if self._thread.isRunning():
            return False
        # 每次启动都是新会话，"本次统计"必须从零开始
        if self._context is not None:
            self._context.statistics.reset()
        self._thread.start()
        return True

    def stop(self) -> None:
        """请求停止（非阻塞；结束时发出 :attr:`sessionFinished`）。"""
        if self._worker is not None:
            self._worker.stop()

    def shutdown(self) -> None:
        """停止并彻底释放资源（窗口关闭时调用）。"""
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None:
            self._thread.quit()
            if not self._thread.wait(4000):
                logger.warning("工作线程未能在 4 秒内退出")
            self._thread = None
        if self._context is not None:
            self._context.statistics.remove_listener(self._on_statistics_changed)
            self._context.shutdown()
            self._context = None
        self._worker = None

    # ==================================================================
    #  用户操作
    # ==================================================================
    @Slot()
    def recalibrate(self) -> None:
        """重新标定驾驶员个人中立位。"""
        if self._worker is not None:
            self._worker.recalibrate()
            self.stateChanged.emit("运行中", "已触发中立位重新标定")

    def set_overlay(self, key: str, enabled: bool) -> None:
        """开关画面叠加元素（``landmarks`` / ``face_box`` / ``hud``）。"""
        mapping = {
            "landmarks": "show_landmarks",
            "face_box": "show_face_box",
            "hud": "show_hud",
        }
        attribute = mapping.get(key)
        if attribute is None:
            logger.warning("未知的叠加开关：%s", key)
            return
        setattr(self._options, attribute, bool(enabled))

    def describe_sources(self) -> dict[str, Any]:
        """汇总数据源/感知/降级信息，供界面标题栏与提示区展示。"""
        context = self._context
        if context is None:
            return {"initialized": False}
        provider = context.provider
        return {
            "initialized": True,
            "source": context.source.name,
            "provider": provider.name,
            "provider_ready": provider.ready,
            "degraded": getattr(provider, "degraded", False),
            "degraded_reason": getattr(provider, "degraded_reason", ""),
            "model_dir": str(self._config.resolved_model_dir()),
            "warnings": list(context.warnings),
            "sinks": [sink.name for sink in context.alert_service.sinks],
            "rules": [rule.behavior.value for rule in context.pipeline.engine.rules
                      if rule.enabled],
        }

    def statistics_snapshot(self) -> dict[str, Any]:
        """当前统计快照（供界面仪表盘绘制）。"""
        context = self._context
        if context is None:
            return {"total": 0, "by_behavior": {}, "by_category": {}, "by_severity": {}}
        stats = context.statistics
        return {
            "total": stats.total,
            "by_behavior": {b.value: c for b, c in stats.counts.items()},
            "by_category": {c.value: n for c, n in stats.category_counts().items()},
            "by_severity": {s.name: n for s, n in stats.severity_counts.items()},
            "top": [(b.value, b.label, c) for b, c in stats.top(5)],
        }

    # ==================================================================
    #  内部槽
    # ==================================================================
    @Slot(object, object)
    def _on_frame_ready(self, canvas: object, analysis: object) -> None:
        self.frameReady.emit(canvas, analysis)

    @Slot(int, float, float)
    def _on_telemetry(self, frame_index: int, fps: float, perception_ms: float) -> None:
        self.telemetry.emit(frame_index, fps, perception_ms, self.event_total)

    @Slot(str)
    def _on_state_changed(self, state: str) -> None:
        self.stateChanged.emit(state, "")

    @Slot(object)
    def _on_finished(self, summary: object) -> None:
        assert isinstance(summary, SessionSummary) or summary is not None
        if self._context is not None:
            self._context.write_session_summary(summary)  # type: ignore[arg-type]
        self.sessionFinished.emit(summary)

    def _on_statistics_changed(self) -> None:
        """统计服务聚合完成后的回调（工作线程）；只发信号，由界面线程渲染。"""
        self.statisticsChanged.emit(self.statistics_snapshot())


# ===========================================================================
def _friendly_build_error(exc: Exception) -> str:
    """把装配期异常翻译成"用户能照做"的提示。"""
    text = str(exc)
    if isinstance(exc, FileNotFoundError) and "模型" in text:
        return f"{text}"
    if isinstance(exc, FileNotFoundError) and "级联" in text:
        return f"{text}"
    return (
        f"启动失败：{type(exc).__name__}: {text}\n\n"
        "可尝试的排查方向：\n"
        "1) 运行 python tools/download_models.py 下载感知模型；\n"
        "2) 若摄像头不可用，改用 --source replay --provider mediapipe 之外的回放模式；\n"
        "3) 查看 output/driver_guard.log 获取详细堆栈。"
    )


#: 供界面直接使用的枚举→中文映射（避免 View 里出现业务映射表）
BEHAVIOR_LABELS: Mapping[BehaviorType, str] = {b: b.label for b in BehaviorType}
CATEGORY_LABELS: Mapping[BehaviorCategory, str] = {c: c.label for c in BehaviorCategory}
SEVERITY_LABELS: Mapping[Severity, str] = {s: s.label for s in Severity}
