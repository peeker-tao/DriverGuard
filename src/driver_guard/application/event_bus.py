"""进程内事件总线（发布-订阅架构样式）。

作用
----
采集线程（生产者）与界面线程（消费者）之间通过总线解耦：

.. code-block:: text

    采集线程 ──publish──► EventBus ──dispatch──► 订阅者A（界面，转成 Qt Signal）
                                        └────────► 订阅者B（CSV 仓储）
                                        └────────► 订阅者C（统计服务）

设计取舍
--------
* **同步分发**：``publish`` 在调用线程内直接回调所有订阅者。实现简单、
  事件顺序确定，适合本系统"单生产者 + 少量轻量订阅者"的场景。
  订阅者若要做重活（写盘、编解码）应自行异步化。
* **订阅者异常隔离**：任一订阅者抛异常只记日志，不影响其它订阅者，
  也不向生产者传播——告警系统自身不能成为故障源（可靠性属性）。
* **主题过滤**：支持按 :class:`~driver_guard.domain.models.BehaviorType` 订阅，
  避免订阅者收到不关心的事件。

线程安全：``publish`` 与 ``subscribe`` 用 :class:`threading.RLock` 保护，
允许在采集线程发布的同时由界面线程增删订阅。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, TypeVar

from ..domain.models import BehaviorEvent, BehaviorType

logger = logging.getLogger(__name__)

__all__ = ["EventBus", "Subscription", "Topic"]

T = TypeVar("T")

#: 订阅回调签名：``(事件, 附加载荷) -> None``
EventHandler = Callable[[BehaviorEvent, Any], None]


@dataclass(frozen=True, slots=True)
class Topic:
    """订阅主题：把"事件类型"与"行为过滤"组合成一个可比较的键。

    ``behaviors=None`` 表示不按行为过滤（接收全部）。
    """

    event_type: type = BehaviorEvent
    behaviors: frozenset[BehaviorType] | None = None

    def matches(self, event: BehaviorEvent) -> bool:
        if not isinstance(event, self.event_type):
            return False
        return self.behaviors is None or event.behavior in self.behaviors


@dataclass(slots=True)
class Subscription:
    """一个订阅记录，``close()`` 取消订阅（支持 ``with`` 语法）。"""

    topic: Topic
    handler: EventHandler
    label: str = ""
    _bus: "EventBus | None" = field(default=None, repr=False)
    _closed: bool = field(default=False, repr=False)

    def close(self) -> None:
        if self._closed or self._bus is None:
            return
        self._bus.unsubscribe(self)
        self._closed = True

    def __enter__(self) -> "Subscription":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class EventBus:
    """轻量级同步发布-订阅总线。"""

    def __init__(self, *, error_threshold: int = 3) -> None:
        self._lock = threading.RLock()
        self._subs: list[Subscription] = []
        self._error_threshold = error_threshold
        self._publish_count = 0
        self._error_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    #  订阅管理
    # ------------------------------------------------------------------
    def subscribe(
        self,
        handler: EventHandler,
        *,
        event_type: type = BehaviorEvent,
        behaviors: Iterable[BehaviorType] | None = None,
        label: str = "",
    ) -> Subscription:
        """注册订阅者并返回可取消的句柄。"""
        topic = Topic(
            event_type=event_type,
            behaviors=frozenset(behaviors) if behaviors is not None else None,
        )
        sub = Subscription(
            topic=topic,
            handler=handler,
            label=label or _describe(handler),
            _bus=self,  # 必须回填，否则 Subscription.close() 永远无效
        )
        with self._lock:
            self._subs.append(sub)
        logger.debug("已订阅: %s", sub.label)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            try:
                self._subs.remove(sub)
            except ValueError:
                return
        logger.debug("已取消订阅: %s", sub.label)

    def clear(self) -> None:
        """移除全部订阅（主要供测试使用）。"""
        with self._lock:
            self._subs.clear()

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)

    @property
    def publish_count(self) -> int:
        return self._publish_count

    @property
    def error_counts(self) -> dict[str, int]:
        """各订阅者的累计异常次数（可观测性）。"""
        with self._lock:
            return dict(self._error_counts)

    # ------------------------------------------------------------------
    #  发布
    # ------------------------------------------------------------------
    def publish(self, event: BehaviorEvent, payload: Any = None) -> int:
        """向所有匹配的订阅者分发事件，返回成功分发的订阅者数量。"""
        with self._lock:
            targets = [sub for sub in self._subs if sub.topic.matches(event)]
            self._publish_count += 1

        delivered = 0
        for sub in targets:
            try:
                sub.handler(event, payload)
                delivered += 1
            except Exception:  # noqa: BLE001 - 订阅者故障隔离
                with self._lock:
                    count = self._error_counts.get(sub.label, 0) + 1
                    self._error_counts[sub.label] = count
                logger.exception("订阅者 %s 处理事件时异常（第 %d 次）", sub.label, count)
                if self._error_threshold and count == self._error_threshold:
                    logger.error(
                        "订阅者 %s 已连续异常 %d 次，建议检查实现（总线仍会继续分发）",
                        sub.label, count,
                    )
        return delivered


def _describe(handler: Callable[..., Any]) -> str:
    """为回调生成可读标签，便于日志定位。"""
    return getattr(handler, "__qualname__", None) or repr(handler)
