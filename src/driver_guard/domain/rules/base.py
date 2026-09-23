"""规则抽象与规则引擎。

这是管道-过滤器架构中的**第 3 个过滤器**（决策级），也是本系统
「微内核 + 插件」风格的落点：

* :class:`BehaviorRule` 是**策略（Strategy）**抽象，每个危险行为对应一个具体策略；
* :class:`RuleEngine` 是**微内核**，只负责调度与容错，不含任何业务阈值；
* 新增一种危险行为 = 新增一个 ``BehaviorRule`` 子类 + 一行注册，
  **无需修改任何已有代码**（开闭原则）。

规则只依赖 :class:`DetectionContext`（只读），不产生副作用、不访问 IO，
因此每条规则都可用构造的指标对象做纯函数式单元测试。
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterable, Mapping

from ..models import BehaviorType, DetectionContext, RuleOutcome, Severity

logger = logging.getLogger(__name__)

__all__ = ["RuleConfig", "BehaviorRule", "RuleEngine"]


# ===========================================================================
#  配置
# ===========================================================================
@dataclass(frozen=True, slots=True)
class RuleConfig:
    """单条规则的配置（来自 ``config/default.json`` 的 ``rules.<name>`` 段）。

    采用「通用容器 + 类型化读取器」而非为每条规则建一个 dataclass：
    新增规则时只需读取自己需要的键，不必改动本类，代价是失去静态检查，
    收益是可扩展性与配置文件的自由度。读取器带默认值，因此配置文件
    可以只覆盖关心的键。
    """

    enabled: bool = True
    severity: Severity = Severity.MEDIUM
    min_duration: float = 0.0
    """行为需持续的最短秒数（时间域门槛，与帧数门槛取"或"）。"""
    trigger_frames: int = 1
    """连续满足条件的帧数达到该值才确认触发（去抖）。"""
    release_frames: int = 1
    """连续不满足条件的帧数达到该值才解除（迟滞，避免状态抖动）。"""
    cooldown_seconds: float = 0.0
    """同一行为两次告警之间的最小间隔秒数，防止告警轰炸。"""
    options: Mapping[str, Any] = field(default_factory=dict)
    """规则自定义阈值。"""

    # -- 通用读取器 -----------------------------------------------------
    def get_float(self, key: str, default: float) -> float:
        value = self.options.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            logger.warning("规则选项 %s=%r 不是数值，回退到默认值 %s", key, value, default)
            return default

    def get_int(self, key: str, default: int) -> int:
        return int(self.get_float(key, default))

    def get_bool(self, key: str, default: bool) -> bool:
        value = self.options.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "RuleConfig":
        """从配置字典构造；未知键进入 :attr:`options`。"""
        data = dict(data or {})
        known = {"enabled", "severity", "min_duration", "trigger_frames", "release_frames", "cooldown_seconds", "options"}
        options = dict(data.get("options") or {})
        options.update({k: v for k, v in data.items() if k not in known})
        return cls(
            enabled=bool(data.get("enabled", True)),
            severity=Severity.parse(data.get("severity", "medium")),
            min_duration=float(data.get("min_duration", 0.0)),
            trigger_frames=int(data.get("trigger_frames", 1)),
            release_frames=int(data.get("release_frames", 1)),
            cooldown_seconds=float(data.get("cooldown_seconds", 0.0)),
            options=options,
        )


# ===========================================================================
#  规则策略
# ===========================================================================
class BehaviorRule(abc.ABC):
    """危险行为判定策略的抽象基类。

    子类需实现 :meth:`_evaluate`；基类统一处理：
    ``关闭开关`` / ``缺少必要证据`` 两种短路情形，避免每个子类重复判断。
    """

    #: 该规则负责的危险行为类型（子类必须覆盖）
    behavior: ClassVar[BehaviorType]
    #: 判定是否需要检测到人脸；否则跳过（避免"无人"时误报）
    requires_face: ClassVar[bool] = True
    #: 判定是否需要检测到手部
    requires_hands: ClassVar[bool] = False

    def __init__(self, config: RuleConfig | None = None) -> None:
        self.config = config or RuleConfig()

    # -- 对外统一入口 ---------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def severity(self) -> Severity:
        return self.config.severity

    def evaluate(self, ctx: DetectionContext) -> RuleOutcome:
        """统一入口：短路条件 → 子类判定 → 统一封装严重级别。"""
        if not self.enabled:
            return RuleOutcome.negative(self.behavior)
        if self.requires_face and not ctx.observation.has_face:
            return RuleOutcome.negative(self.behavior)
        if self.requires_hands and not ctx.observation.has_hands:
            return RuleOutcome.negative(self.behavior)

        outcome = self._evaluate(ctx)
        if not outcome.triggered:
            return outcome
        return RuleOutcome(
            behavior=outcome.behavior,
            triggered=True,
            confidence=outcome.confidence,
            severity=self.severity,
            message=outcome.message,
            detail=outcome.detail,
        )

    @abc.abstractmethod
    def _evaluate(self, ctx: DetectionContext) -> RuleOutcome:
        """子类实现的实际判定逻辑。"""

    # -- 供子类使用的工具 -----------------------------------------------
    def _trigger(self, ctx: DetectionContext, confidence: float, message: str, **detail: float) -> RuleOutcome:
        return RuleOutcome(
            behavior=self.behavior,
            triggered=True,
            confidence=confidence,
            severity=self.severity,
            message=message,
            detail={k: v for k, v in detail.items() if v is not None},
        )

    def _skip(self) -> RuleOutcome:
        return RuleOutcome.negative(self.behavior)

    @staticmethod
    def _score(value: float, threshold: float, *, higher_means_more: bool = True) -> float:
        """把"指标越过阈值的程度"映射为 0~1 的置信度。

        以阈值为原点，超出 50% 即认为完全确信，避免置信度长期贴边。
        阈值统一取绝对值，因为本系统所有阈值都表达"偏离幅度"，
        负号无意义；若配置写成负数，此处自动纠正而不是静默地把
        判定条件变成恒真（历史上该错误曾导致低头规则每帧误报）。
        """
        threshold = abs(threshold)
        if threshold <= 0:
            return 1.0
        if higher_means_more:
            margin = (value - threshold) / threshold
        else:
            margin = (threshold - value) / threshold
        return max(0.0, min(1.0, 0.5 + margin))

    @staticmethod
    def _magnitude_threshold(value: float, default: float) -> float:
        """读取"偏离幅度"型阈值并取绝对值，防止负号把比较条件变成恒真。"""
        return abs(value) if abs(value) > 1e-9 else default


# ===========================================================================
#  规则引擎（微内核）
# ===========================================================================
class RuleEngine:
    """按注册顺序执行全部规则，并做**单条失败隔离**。

    容错策略：任何一条规则抛出异常都被捕获并记日志，不影响其余规则，
    也不会中断采集循环——这是"单个插件故障不导致系统失效"这一
    可靠性质量属性的实现手段。
    """

    def __init__(self, rules: Iterable[BehaviorRule] = ()) -> None:
        self._rules: list[BehaviorRule] = list(rules)
        self._failure_counts: dict[BehaviorType, int] = {}

    def register(self, rule: BehaviorRule) -> None:
        self._rules.append(rule)

    def clear(self) -> None:
        self._rules.clear()

    @property
    def rules(self) -> tuple[BehaviorRule, ...]:
        return tuple(self._rules)

    @property
    def failure_counts(self) -> Mapping[BehaviorType, int]:
        """各规则的累计异常次数，供界面/日志观测（可观测性）。"""
        return dict(self._failure_counts)

    def evaluate(self, ctx: DetectionContext) -> list[RuleOutcome]:
        outcomes: list[RuleOutcome] = []
        for rule in self._rules:
            if not rule.enabled:
                continue
            try:
                outcomes.append(rule.evaluate(ctx))
            except Exception:  # noqa: BLE001 - 故意兜底：插件级故障隔离
                self._failure_counts[rule.behavior] = self._failure_counts.get(rule.behavior, 0) + 1
                logger.exception("规则 %s 执行失败，已跳过该规则", rule.behavior.value)
        return outcomes
