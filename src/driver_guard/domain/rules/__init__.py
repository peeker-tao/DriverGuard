"""规则插件注册表（简单工厂 + 微内核插件表）。

新增一种危险行为的完整步骤
--------------------------
1. 在 :class:`~driver_guard.domain.models.BehaviorType` 中增加枚举值；
2. 新建 ``BehaviorRule`` 子类实现 ``_evaluate``；
3. 在本模块 :data:`RULE_FACTORIES` 中登记一行；
4. （可选）在 ``config/default.json`` 的 ``rules`` 段下增加同名配置节。

第 1~3 步之外 **不需要修改任何既有文件**，符合开闭原则。
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..models import BehaviorType
from .base import BehaviorRule, RuleConfig, RuleEngine
from .distraction import DriverAbsentRule, GazeOffRoadRule
from .fatigue import EyesClosedRule, HeadDownRule, YawningRule
from .violation import PhoneCallRule, SmokingRule

__all__ = [
    "RULE_FACTORIES",
    "RuleEngine",
    "RuleConfig",
    "BehaviorRule",
    "EyesClosedRule",
    "YawningRule",
    "HeadDownRule",
    "GazeOffRoadRule",
    "DriverAbsentRule",
    "PhoneCallRule",
    "SmokingRule",
    "build_rules",
    "build_engine",
]

#: ``配置键 -> 规则构造函数``。配置键与 :class:`BehaviorType` 的取值一一对应。
RULE_FACTORIES: dict[str, Callable[[RuleConfig], BehaviorRule]] = {
    BehaviorType.EYES_CLOSED.value: EyesClosedRule,
    BehaviorType.YAWNING.value: YawningRule,
    BehaviorType.HEAD_DOWN.value: HeadDownRule,
    BehaviorType.GAZE_OFF_ROAD.value: GazeOffRoadRule,
    BehaviorType.PHONE_CALL.value: PhoneCallRule,
    BehaviorType.SMOKING.value: SmokingRule,
    BehaviorType.DRIVER_ABSENT.value: DriverAbsentRule,
}


def build_rules(rules_config: Mapping[str, Any] | None = None) -> list[BehaviorRule]:
    """根据配置字典构造全部规则实例。

    Args:
        rules_config: ``config["rules"]`` 段；``None`` 表示使用各规则内置默认值。

    Returns:
        规则列表，顺序与 :data:`RULE_FACTORIES` 的登记顺序一致
        （即：疲劳 → 分心 → 违规），保证日志与界面展示顺序稳定。
    """
    rules_config = rules_config or {}
    rules: list[BehaviorRule] = []
    for key, factory in RULE_FACTORIES.items():
        config = RuleConfig.from_mapping(rules_config.get(key))
        rules.append(factory(config))
    return rules


def build_engine(rules_config: Mapping[str, Any] | None = None) -> RuleEngine:
    """组合根友好的便捷函数：直接得到装配好的 :class:`RuleEngine`。"""
    return RuleEngine(build_rules(rules_config))
