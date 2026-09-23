"""配置加载与校验。

设计
----
配置文件是 JSON（而非 YAML），以 **零额外依赖** 为前提：课程交付环境
可能无法联网安装 ``pyyaml``。配置被映射为**不可变 dataclass**，
从而在程序入口处一次性完成校验，运行期不再出现"配置键拼错但静默生效"
这类难以定位的问题。

关键行为
--------
* **深度合并**：用户配置只需写出与默认值不同的部分；
* **未知键告警**：拼错的键会被记录到 :attr:`AppConfig.warnings`，
  由界面或日志提示，而不是被静默忽略；
* **路径归一化**：所有相对路径按 **项目根目录** 解析，
  因此从任意工作目录启动都能找到模型与输出目录。
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from .domain.models import Severity

logger = logging.getLogger(__name__)

__all__ = ["AppConfig", "load_config", "default_config", "PROJECT_ROOT", "DEFAULT_CONFIG_PATH"]

#: 项目根目录（``src/driver_guard/config.py`` 上溯三级）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.json"


# ===========================================================================
@dataclass(frozen=True, slots=True)
class AppMeta:
    name: str = "驾驶员危险行为检测系统"
    code: str = "DriverGuard"
    version: str = "1.0.0"


@dataclass(frozen=True, slots=True)
class SourceConfig:
    type: str = "camera"
    """``camera`` / ``video`` / ``synthetic`` / ``images`` / ``replay``。"""
    camera_index: int = 0
    video_path: str = "assets/samples/driver_demo.mp4"
    image_dir: str = "assets/samples/frames"
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    loop_video: bool = True
    speed: float = 1.0
    synthetic_duration: float = 0.0
    """合成画面的时长（秒）；``0`` 表示无限。

    对 ``replay`` 数据源，``0`` 的含义是"自动对齐回放脚本长度"，
    使脚本跑完即自动收尾（见 :func:`bootstrap.create_frame_source`）。
    """
    camera_backend: str = "any"
    """``any`` / ``dshow`` / ``msmf``；Windows 上 ``dshow`` 通常打开更快。"""


@dataclass(frozen=True, slots=True)
class PerceptionConfig:
    provider: str = "mediapipe"
    model_dir: str = "assets/models"
    face_model: str = "face_landmarker.task"
    hand_model: str = "hand_landmarker.task"
    cascade_dir: str = "assets/cascades"
    enable_hands: bool = True
    max_faces: int = 1
    max_hands: int = 2
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_hand_tracking_confidence: float = 0.5
    frame_stride: int = 1
    """隔帧推理步长；``2`` 表示每两帧推理一次（用上一帧结果填充），
    用于低算力设备的性能降级手段。"""

    scenario: str = "mixed"
    """回放脚本名（仅 ``provider == "replay"`` 时生效）。

    可选值见 :data:`driver_guard.infrastructure.landmarks.SCENARIOS`。
    放在感知配置里而不是数据源配置里，是因为该参数直接决定
    “感知适配器会产出什么指标”，属于感知语义。
    """

    def provider_options(self) -> dict[str, Any]:
        """转成基础设施层可消费的普通字典。"""
        return {
            "model_dir": self.model_dir,
            "face_model": self.face_model,
            "hand_model": self.hand_model,
            "cascade_dir": self.cascade_dir,
            "enable_hands": self.enable_hands,
            "max_faces": self.max_faces,
            "max_hands": self.max_hands,
            "min_face_detection_confidence": self.min_face_detection_confidence,
            "min_face_presence_confidence": self.min_face_presence_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
            "min_hand_detection_confidence": self.min_hand_detection_confidence,
            "min_hand_presence_confidence": self.min_hand_presence_confidence,
            "min_hand_tracking_confidence": self.min_hand_tracking_confidence,
            "scenario": self.scenario,
            "fps_hint": 30.0,
        }


@dataclass(frozen=True, slots=True)
class SmoothingConfig:
    window: int = 9
    min_samples: int = 3


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    enabled: bool = True
    required_samples: int = 30
    warmup_frames: int = 5


@dataclass(frozen=True, slots=True)
class AlertConfig:
    log_console: bool = True
    csv_enabled: bool = True
    sound_enabled: bool = True
    sound_min_severity: Severity = Severity.HIGH
    snapshot_enabled: bool = True
    snapshot_min_severity: Severity = Severity.HIGH
    snapshot_scale: float = 1.0
    min_severity: Severity = Severity.LOW


@dataclass(frozen=True, slots=True)
class OutputConfig:
    dir: str = "output"
    events_csv: str = "events/events.csv"
    snapshot_dir: str = "snapshots"
    session_csv: str = "events/sessions.csv"
    report_html: str = "report.html"

    def resolve(self, relative: str) -> Path:
        """把输出相对路径解析为项目内的绝对路径。"""
        base = Path(self.dir)
        base = base if base.is_absolute() else PROJECT_ROOT / base
        return (base / relative).resolve()


@dataclass(frozen=True, slots=True)
class UiConfig:
    show_landmarks: bool = True
    show_face_box: bool = True
    show_hud: bool = True
    theme: str = "dark"
    max_events_in_table: int = 500
    display_scale: float = 1.0


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    level: str = "INFO"
    file: str = "output/driver_guard.log"


@dataclass(frozen=True, slots=True)
class AppConfig:
    """完整应用配置。"""

    app: AppMeta = field(default_factory=AppMeta)
    source: SourceConfig = field(default_factory=SourceConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    rules: Mapping[str, Any] = field(default_factory=dict)

    #: 加载过程中发现的未知键（供界面提示"配置可能写错"）
    warnings: tuple[str, ...] = ()
    #: 本次配置来源文件
    source_path: Path | None = None

    # ------------------------------------------------------------------
    def resolved_model_dir(self) -> Path:
        return _resolve(self.perception.model_dir)

    def resolved_cascade_dir(self) -> Path:
        return _resolve(self.perception.cascade_dir)

    def resolved_video_path(self) -> Path:
        return _resolve(self.source.video_path)

    def resolved_image_dir(self) -> Path:
        return _resolve(self.source.image_dir)

    def with_overrides(self, **kwargs: Any) -> "AppConfig":
        """返回替换了顶层字段的新配置（不可变语义）。"""
        return replace(self, **kwargs)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


# ===========================================================================
#  默认配置
# ===========================================================================
def default_config() -> AppConfig:
    """返回内置默认配置。

    与 :data:`DEFAULT_CONFIG_PATH` 的区别：本函数在配置文件缺失时兜底，
    保证程序永远能启动（可用性优先）。
    """
    config = AppConfig()
    if DEFAULT_CONFIG_PATH.exists():
        try:
            config = load_config(DEFAULT_CONFIG_PATH, validate_paths=False)
        except Exception:  # noqa: BLE001 - 默认配置损坏不应阻止启动
            logger.exception("读取默认配置文件失败，改用内置默认值")
    return config


# ===========================================================================
#  加载
# ===========================================================================
_SECTIONS: tuple[tuple[str, type], ...] = (
    ("app", AppMeta),
    ("source", SourceConfig),
    ("perception", PerceptionConfig),
    ("smoothing", SmoothingConfig),
    ("calibration", CalibrationConfig),
    ("alert", AlertConfig),
    ("output", OutputConfig),
    ("ui", UiConfig),
    ("logging", LoggingConfig),
)

#: 需要对特定字段做类型转换的映射：``(段名, 字段名) -> 转换函数``
_COERCERS: dict[tuple[str, str], Any] = {
    ("alert", "min_severity"): Severity.parse,
    ("alert", "sound_min_severity"): Severity.parse,
    ("alert", "snapshot_min_severity"): Severity.parse,
}

#: 配置项的合法取值范围（用于早期校验，避免运行期出现诡异行为）
_RANGES: dict[tuple[str, str], tuple[float, float]] = {
    ("source", "width"): (160, 7680),
    ("source", "height"): (120, 4320),
    ("source", "fps"): (1.0, 240.0),
    ("source", "speed"): (0.05, 8.0),
    ("perception", "max_faces"): (1, 4),
    ("perception", "max_hands"): (1, 4),
    ("perception", "frame_stride"): (1, 5),
    ("smoothing", "window"): (1, 121),
    ("calibration", "required_samples"): (3, 3600),
    ("ui", "max_events_in_table"): (10, 100_000),
}


def load_config(path: str | Path | None = None, *, validate_paths: bool = True) -> AppConfig:
    """加载配置文件并与默认值深度合并。

    Args:
        path: 配置文件路径；``None`` 或文件不存在时返回默认配置。
        validate_paths: 是否校验数据源相关路径的存在性（缺失只告警不报错）。

    Raises:
        ValueError: 配置值超出合法范围或类型无法转换。
        json.JSONDecodeError: 文件不是合法 JSON。
    """
    if path is None:
        return default_config()

    config_path = Path(path)
    if not config_path.exists():
        logger.warning("配置文件不存在，使用默认配置：%s", config_path)
        return default_config()

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("配置文件顶层必须是 JSON 对象")

    warnings: list[str] = []
    sections: dict[str, Any] = {}
    for section_name, section_type in _SECTIONS:
        user_section = raw.get(section_name, {})
        if not isinstance(user_section, dict):
            warnings.append(f"配置段 {section_name!r} 不是对象，已忽略")
            user_section = {}
        sections[section_name] = _build_section(
            section_name, section_type, user_section, warnings
        )

    unknown_top = set(raw) - {name for name, _ in _SECTIONS} - {"rules", "perception", "calibration"}
    for key in sorted(unknown_top):
        warnings.append(f"未知的顶层配置段：{key!r}")

    config = AppConfig(
        **sections,
        rules=dict(raw.get("rules") or {}),
        warnings=tuple(warnings),
        source_path=config_path.resolve(),
    )

    for message in warnings:
        logger.warning("配置提示：%s", message)
    if validate_paths:
        _validate_source_paths(config)
    logger.info("已加载配置：%s（%d 条提示）", config_path.name, len(warnings))
    return config


def _build_section(
    section_name: str,
    section_type: type,
    user_section: Mapping[str, Any],
    warnings: list[str],
) -> Any:
    """构造单个配置段，处理类型转换、范围校验与未知键告警。"""
    fields = {f.name: f for f in section_type.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    values: dict[str, Any] = {}

    for key, value in user_section.items():
        if key not in fields:
            warnings.append(f"{section_name}.{key} 不是已知配置项，已忽略")
            continue
        coerce = _COERCERS.get((section_name, key))
        try:
            if coerce is not None:
                values[key] = coerce(value)
            elif fields[key].type in {"int", int}:
                values[key] = int(value)
            elif fields[key].type in {"float", float}:
                values[key] = float(value)
            else:
                values[key] = value
        except (TypeError, ValueError, KeyError) as exc:
            warnings.append(f"{section_name}.{key}={value!r} 无法解析（{exc}），使用默认值")
            continue

        bounds = _RANGES.get((section_name, key))
        if bounds and isinstance(values[key], (int, float)):
            low, high = bounds
            if not low <= values[key] <= high:
                warnings.append(
                    f"{section_name}.{key}={values[key]} 超出合理范围 [{low}, {high}]，已夹紧"
                )
                values[key] = type(values[key])(min(max(values[key], low), high))

    return section_type(**values)


def _validate_source_paths(config: AppConfig) -> None:
    """对数据源相关路径做存在性检查（仅告警，不阻断启动）。"""
    kind = config.source.type.lower()
    if kind == "video":
        path = config.resolved_video_path()
        if not path.exists():
            logger.warning(
                "视频文件不存在：%s（可改用 --source synthetic --provider replay 演示）", path
            )
    elif kind == "images":
        directory = config.resolved_image_dir()
        if not directory.is_dir():
            logger.warning("图片目录不存在：%s", directory)
    elif kind == "camera":
        logger.info("数据源为摄像头 #%d", config.source.camera_index)


# ===========================================================================
#  保存（供界面"导出当前配置"使用）
# ===========================================================================
def dump_config(config: AppConfig) -> str:
    """把配置序列化为 JSON 文本（不包含运行时字段）。"""
    payload: dict[str, Any] = {}
    for section_name, section_type in _SECTIONS:
        section = getattr(config, section_name)
        payload[section_name] = {
            key: _jsonable(getattr(section, key)) for key in section_type.__dataclass_fields__
        }
    payload["rules"] = copy.deepcopy(dict(config.rules))
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Severity):
        return value.name.lower()
    if isinstance(value, Path):
        return str(value)
    return value
