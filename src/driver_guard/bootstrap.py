"""组合根（Composition Root）与应用装配。

**组合根** 是依赖注入的唯一落点：只有这里知道"哪个接口用哪个实现"。
其余模块一律通过构造函数接收依赖，因此：

* 领域层零外部依赖，可脱机单元测试；
* 表现层可以在测试中被替换为"无界面"的假实现；
* 想换成降级感知/回放数据源，只改这一处装配代码。

装配顺序（自内向外）::

    Config
      └─ MetricExtractor / RuleEngine / BehaviorConfirmer / NeutralPoseCalibrator
           └─ DetectionPipeline（注入 LandmarkProvider）
                └─ DetectionService（注入 FrameSource + EventBus + Repository）
                     └─ AlertService（订阅 EventBus，扇出到 AlertSink）
                     └─ StatisticsService（订阅 EventBus）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .application.event_bus import EventBus
from .application.pipeline import DetectionPipeline
from .application.ports import AlertSink, EventRepository, FrameSource, LandmarkProvider
from .application.services import (
    AlertService,
    DetectionService,
    ServiceState,
    SessionSummary,
    StatisticsService,
)
from .config import AppConfig, PROJECT_ROOT
from .domain.calibration import NeutralPoseCalibrator
from .domain.confirmation import BehaviorConfirmer
from .domain.metrics import MetricExtractor
from .domain.models import BehaviorType
from .domain.rules import RuleConfig, build_engine
from .infrastructure import (
    ConsoleAlertSink,
    CsvAlertSink,
    CsvEventRepository,
    CsvSessionWriter,
    InMemoryEventRepository,
    SnapshotAlertSink,
    SoundAlertSink,
    SyntheticFrameSource,
    VideoFileFrameSource,
    WebcamFrameSource,
    create_landmark_provider,
)
from .infrastructure.landmarks import scenario_duration
from .infrastructure.sources import ImageSequenceFrameSource

logger = logging.getLogger(__name__)

__all__ = ["ApplicationContext", "build_application", "create_frame_source", "create_alert_sinks"]


# ===========================================================================
@dataclass
class ApplicationContext:
    """一次应用会话的全部装配结果。"""

    config: AppConfig
    source: FrameSource
    provider: LandmarkProvider
    pipeline: DetectionPipeline
    event_bus: EventBus
    repository: EventRepository
    alert_service: AlertService
    statistics: StatisticsService
    detection_service: DetectionService
    session_writer: CsvSessionWriter | None = None
    snapshot_sink: SnapshotAlertSink | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------
    @property
    def state(self) -> ServiceState:
        return self.detection_service.state

    @property
    def event_count(self) -> int:
        return self.statistics.total

    def shutdown(self) -> None:
        """停止服务并释放全部资源（幂等）。"""
        try:
            self.detection_service.stop()
            self.detection_service.join(timeout=3.0)
        except Exception:  # noqa: BLE001
            logger.exception("停止检测服务失败")
        self.alert_service.close()
        self.statistics.close()
        self.event_bus.clear()
        if hasattr(self.repository, "close"):
            try:
                self.repository.close()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                logger.exception("关闭事件仓储失败")
        logger.info("应用资源已释放")

    def write_session_summary(self, summary: SessionSummary) -> Path | None:
        """把会话汇总写入 CSV，返回文件路径。"""
        if self.session_writer is None:
            return None
        try:
            self.session_writer.append(summary, risk_index=summary.risk_index)
        except Exception:  # noqa: BLE001 - 报告写入失败不应影响退出
            logger.exception("写入会话汇总失败")
            return None
        return self.session_writer.path


# ===========================================================================
#  数据源工厂
# ===========================================================================
def create_frame_source(config: AppConfig) -> FrameSource:
    """按 ``source.type`` 构造数据源适配器。

    支持 ``camera`` / ``video`` / ``synthetic`` / ``images`` / ``replay``。
    ``replay`` 等价于"合成画面 + 回放感知"，用于无摄像头演示。
    """
    kind = (config.source.type or "camera").strip().lower()

    if kind in {"replay", "demo"}:
        # 回放模式的数据源长度**默认对齐脚本长度**，使脚本跑完即自动收尾。
        # 若不这样做，`synthetic_duration = 0`（= 无限）会让"回放"永不结束，
        # 既拿不到会话汇总，也无法在无人工干预的场景下结束演示。
        duration = config.source.synthetic_duration
        if duration <= 0:
            duration = scenario_duration(config.perception.scenario)
        return SyntheticFrameSource(
            width=config.source.width,
            height=config.source.height,
            fps=config.source.fps,
            duration=duration,
        )

    if kind == "synthetic":
        return SyntheticFrameSource(
            width=config.source.width,
            height=config.source.height,
            fps=config.source.fps,
            duration=config.source.synthetic_duration,
        )

    if kind == "video":
        path = config.resolved_video_path()
        return VideoFileFrameSource(
            path,
            loop=config.source.loop_video,
            speed=config.source.speed,
        )

    if kind == "images":
        directory = config.resolved_image_dir()
        if not directory.is_dir():
            raise FileNotFoundError(f"图片目录不存在：{directory}")
        patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        files: list[Path] = []
        for pattern in patterns:
            files.extend(sorted(directory.glob(pattern)))
        if not files:
            raise FileNotFoundError(f"图片目录中没有可用图片：{directory}")
        return ImageSequenceFrameSource(files, fps=config.source.fps or 25.0)

    if kind == "camera":
        backend = _resolve_camera_backend(config.source.camera_backend)
        return WebcamFrameSource(
            index=config.source.camera_index,
            width=config.source.width,
            height=config.source.height,
            fps=config.source.fps,
            backend=backend,
        )

    raise ValueError(
        f"未知数据源类型 {config.source.type!r}，"
        "可选：camera / video / synthetic / images / replay"
    )


def _resolve_camera_backend(name: str) -> int | None:
    """把配置里的后端名映射为 OpenCV 标志。"""
    import cv2

    mapping = {
        "any": None,
        "dshow": getattr(cv2, "CAP_DSHOW", None),
        "msmf": getattr(cv2, "CAP_MSMF", None),
    }
    key = (name or "any").strip().lower()
    if key not in mapping:
        logger.warning("未知摄像头后端 %r，改为自动选择", name)
        return None
    return mapping[key]


# ===========================================================================
#  告警通道工厂
# ===========================================================================
def create_alert_sinks(
    config: AppConfig,
    *,
    extra: Sequence[AlertSink] = (),
    repository: EventRepository | None = None,
) -> tuple[list[AlertSink], SnapshotAlertSink | None]:
    """按配置组装告警通道。

    Returns:
        ``(全部通道, 快照通道或 None)``
    """
    sinks: list[AlertSink] = []

    if config.alert.log_console:
        sinks.append(ConsoleAlertSink())

    if config.alert.csv_enabled:
        if isinstance(repository, CsvEventRepository):
            # 复用同一仓储，避免两个句柄同时写同一文件
            sinks.append(CsvAlertSink(repository.path, repo=repository))
        else:
            sinks.append(CsvAlertSink(config.output.resolve(config.output.events_csv)))

    if config.alert.snapshot_enabled:
        snapshot = SnapshotAlertSink(
            config.output.resolve(config.output.snapshot_dir),
            min_severity=config.alert.snapshot_min_severity,
        )
        sinks.append(snapshot)
    else:
        snapshot = None

    if config.alert.sound_enabled:
        sinks.append(
            SoundAlertSink(min_severity=config.alert.sound_min_severity)
        )

    sinks.extend(extra)
    return sinks, snapshot


# ===========================================================================
#  总装配
# ===========================================================================
def build_application(
    config: AppConfig,
    *,
    extra_sinks: Sequence[AlertSink] = (),
    source: FrameSource | None = None,
    persist_events: bool = True,
    max_frames: int = 0,
) -> ApplicationContext:
    """把配置与各层实现装配为一个可运行的 :class:`ApplicationContext`。

    Args:
        config: 应用配置。
        extra_sinks: 额外的告警通道（例如表现层的 Qt 桥接通道）。
        source: 覆盖数据源（测试或界面切换数据源时使用）。
        persist_events: 是否把事件写入 CSV 仓储。
        max_frames: 最多处理帧数（``0`` 表示不限制），用于离线批处理与测试。

    Raises:
        FileNotFoundError: 感知模型缺失且未提供回放降级。
        ValueError: 数据源类型或感知实现名非法。
    """
    warnings: list[str] = list(config.warnings)

    # --- 数据源 --------------------------------------------------------
    frame_source = source if source is not None else create_frame_source(config)

    # --- 感知 ----------------------------------------------------------
    provider_name = config.perception.provider
    if isinstance(frame_source, SyntheticFrameSource) and provider_name == "mediapipe":
        # 合成画面没有人脸，用 MediaPipe 会一直"未检测到驾驶员"。
        # 这不是错误，而是验收时容易踩的坑，因此显式提示。
        warnings.append(
            "数据源为合成画面，但感知实现是 mediapipe：合成画面不含真实人脸，"
            "请改用 --provider replay 才能看到完整告警链路"
        )
        logger.warning(warnings[-1])

    provider = create_landmark_provider(
        provider_name,
        _provider_options(config),
    )
    warnings.extend(_provider_readiness_warnings(provider))

    # --- 流水线 --------------------------------------------------------
    engine = build_engine(config.rules)
    confirmer = BehaviorConfirmer(_confirmer_configs(engine))
    calibrator = (
        NeutralPoseCalibrator(
            required_samples=config.calibration.required_samples,
            warmup_frames=config.calibration.warmup_frames,
        )
        if config.calibration.enabled
        else None
    )
    pipeline = DetectionPipeline(
        provider=provider,
        extractor=MetricExtractor(),
        engine=engine,
        confirmer=confirmer,
        calibrator=calibrator,
        smoothing_window=config.smoothing.window,
    )

    # --- 事件总线与仓储 ------------------------------------------------
    event_bus = EventBus()
    repository: EventRepository
    if persist_events:
        repository = CsvEventRepository(config.output.resolve(config.output.events_csv))
    else:
        repository = InMemoryEventRepository()

    # --- 告警 ----------------------------------------------------------
    sinks, snapshot_sink = create_alert_sinks(config, extra=extra_sinks, repository=repository)
    alert_service = AlertService(
        event_bus,
        sinks,
        min_severity=config.alert.min_severity,
    )
    alert_service.start()

    # --- 统计 ----------------------------------------------------------
    statistics = StatisticsService(event_bus)

    # --- 检测服务 ------------------------------------------------------
    detection_service = DetectionService(
        source=frame_source,
        pipeline=pipeline,
        event_bus=event_bus,
        repository=repository,
        max_frames=max_frames,
        target_fps=config.source.fps if _needs_pacing(frame_source) else 0.0,
    )

    session_writer = (
        CsvSessionWriter(config.output.resolve(config.output.session_csv))
        if persist_events
        else None
    )

    return ApplicationContext(
        config=config,
        source=frame_source,
        provider=provider,
        pipeline=pipeline,
        event_bus=event_bus,
        repository=repository,
        alert_service=alert_service,
        statistics=statistics,
        detection_service=detection_service,
        session_writer=session_writer,
        snapshot_sink=snapshot_sink,
        warnings=tuple(warnings),
    )


def _needs_pacing(source: FrameSource) -> bool:
    """摄像头自身有实时节奏，其它源需要软件节流。"""
    return not isinstance(source, WebcamFrameSource)


def _provider_options(config: AppConfig) -> dict:
    options = config.perception.provider_options()
    options["fps_hint"] = config.source.fps or 30.0
    return options


def _provider_readiness_warnings(provider: LandmarkProvider) -> list[str]:
    """对"未就绪/降级"的感知实现给出可操作的提示。"""
    messages: list[str] = []
    missing = getattr(provider, "missing_models", None)
    if callable(missing):
        absent = list(missing())
        if absent:
            names = ", ".join(str(p) for p in absent)
            messages.append(
                f"感知实现 {provider.name} 缺少模型文件：{names}；"
                "请运行 python tools/download_models.py"
            )
    missing_cascades = getattr(provider, "missing_files", None)
    if callable(missing_cascades):
        absent = list(missing_cascades())
        if absent:
            names = ", ".join(str(p) for p in absent)
            messages.append(
                f"感知实现 {provider.name} 缺少级联文件：{names}；"
                "请运行 python tools/fetch_cascades.py"
            )
    if getattr(provider, "degraded", False):
        reason = getattr(provider, "degraded_reason", "")
        if reason:
            messages.append(f"感知通道已降级：{reason}")
    for message in messages:
        logger.warning(message)
    return messages


def _confirmer_configs(engine) -> dict[BehaviorType, RuleConfig]:
    """从规则实例抽出时序确认参数，供 :class:`BehaviorConfirmer` 使用。

    规则只关心"瞬时是否成立"，时间域的门槛（连续帧、持续时间、冷却）
    统一由确认器消费——这是第 3、4 个过滤器之间的职责边界。
    """
    return {rule.behavior: rule.config for rule in engine.rules}
