"""时序确认：把规则的「瞬时判定」升级为「可信事件」。

这是管道-过滤器架构中的**第 4 个过滤器**，也是抑制误报的核心机制。

三级过滤
--------
1. **连续帧门槛** ``trigger_frames``
   连续 N 帧成立才确认。单帧关键点抖动（眨眼、说话瞬间）不足以触发。
2. **时长门槛** ``min_duration``（秒）
   行为持续时间不足则不上报，用于区分"打哈欠"与"张嘴说话"。
3. **迟滞释放** ``release_frames``
   连续 M 帧不成立才解除。否则在阈值附近会反复触发/解除（状态抖动）。

另加 **冷却时间** ``cooldown_seconds``：同一行为持续期间按固定间隔
重复提醒，避免告警轰炸；``cooldown_seconds == 0`` 表示只报一次。

关于抖动容忍（重要）
--------------------
``hit_frames`` 在遇到不成立的帧时 **不立即清零**，只有当连续不成立的
帧数达到 ``release_frames`` 时才清零。也就是说 ``release_frames`` 同时
承担了两个作用：解除已激活的状态，以及容忍短时抖动。

例如 ``trigger_frames=3, release_frames=2`` 时序列
``成立 成立 不成立 成立`` 仍会触发：单帧丢失（关键点瞬时失败）
不应该让一个真实行为漏报。反过来，若希望"必须严格连续"，
把 ``release_frames`` 设为 ``1`` 即可。

状态机
------
每个行为维护独立的 :class:`_Track`::

    空闲 ──(连续 trigger_frames 帧成立 且 时长 ≥ min_duration)──► 激活
      ▲                                                            │
      └──────────────(连续 release_frames 帧不成立)────────────────┘
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping

from .models import BehaviorEvent, BehaviorType, RuleOutcome
from .rules.base import RuleConfig

logger = logging.getLogger(__name__)

__all__ = ["BehaviorConfirmer"]


@dataclass
class _Track:
    """单个行为的状态机内部状态。"""

    config: RuleConfig
    first_hit_at: float | None = None
    """本轮"成立"段的起始时间戳（用于时长门槛与 duration 统计）。"""
    hit_frames: int = 0
    miss_frames: int = 0
    active: bool = False
    activated_at: float | None = None
    last_emit_at: float | None = None
    emitted_count: int = 0
    last_confidence: float = 0.0
    last_message: str = ""
    detail: Mapping[str, float] = field(default_factory=dict)

    def duration(self, now: float) -> float:
        if self.first_hit_at is None:
            return 0.0
        return max(0.0, now - self.first_hit_at)


class BehaviorConfirmer:
    """带迟滞与冷却的事件确认器。

    Args:
        configs: ``行为类型 -> 规则配置``。缺少某项时使用默认配置
            （``trigger_frames=1, release_frames=1, min_duration=0``），
            即退化为"瞬时直通"，便于测试与调试。

    线程安全：设计为 **只在采集线程内使用**，不做加锁。
    跨线程传递事件由 :mod:`driver_guard.application.event_bus` 负责。
    """

    def __init__(self, configs: Mapping[BehaviorType, RuleConfig] | None = None) -> None:
        self._tracks: dict[BehaviorType, _Track] = {}
        for behavior, config in (configs or {}).items():
            self._tracks[behavior] = _Track(config=config)

    # ------------------------------------------------------------------
    def _track(self, behavior: BehaviorType) -> _Track:
        track = self._tracks.get(behavior)
        if track is None:
            track = _Track(config=RuleConfig())
            self._tracks[behavior] = track
        return track

    # ------------------------------------------------------------------
    @property
    def active_behaviors(self) -> tuple[BehaviorType, ...]:
        """当前处于激活状态的行为，供界面实时展示"正在发生的危险行为"。"""
        return tuple(b for b, t in self._tracks.items() if t.active)

    def stats(self, behavior: BehaviorType) -> dict[str, float] | None:
        """返回某行为的统计信息（用于界面/报表）。"""
        track = self._tracks.get(behavior)
        if track is None:
            return None
        return {
            "active": float(track.active),
            "emitted": float(track.emitted_count),
            "hit_frames": float(track.hit_frames),
            "miss_frames": float(track.miss_frames),
            "confidence": track.last_confidence,
        }

    def reset(self) -> None:
        """复位全部状态机（例如切换数据源时）。"""
        for track in self._tracks.values():
            track.first_hit_at = None
            track.hit_frames = 0
            track.miss_frames = 0
            track.active = False
            track.activated_at = None
            track.last_emit_at = None

    # ------------------------------------------------------------------
    def update(self, outcome: RuleOutcome, timestamp: float) -> BehaviorEvent | None:
        """推入一条瞬时判定，返回应当对外发布的领域事件（或 ``None``）。

        Args:
            outcome: 规则引擎给出的瞬时结果。
            timestamp: 当前帧相对会话开始的秒数（单调递增）。

        Returns:
            需要告警时返回 :class:`BehaviorEvent`；``ongoing=True`` 表示是
            同一次持续行为的重复提醒（受冷却时间控制）。
        """
        track = self._track(outcome.behavior)
        config = track.config

        if not outcome.triggered:
            track.miss_frames += 1
            if track.miss_frames >= max(1, config.release_frames):
                self._deactivate(track, timestamp)
            return None

        # --- 成立 ------------------------------------------------------
        track.hit_frames += 1
        track.miss_frames = 0
        track.last_confidence = outcome.confidence
        track.last_message = outcome.message
        track.detail = outcome.detail
        if track.first_hit_at is None:
            track.first_hit_at = timestamp

        if not track.active:
            if track.hit_frames < max(1, config.trigger_frames):
                return None
            if track.duration(timestamp) < config.min_duration:
                return None
            track.active = True
            track.activated_at = timestamp
            track.last_emit_at = timestamp
            track.emitted_count += 1
            return self._build_event(outcome, timestamp, track, ongoing=False)

        # --- 已激活：按冷却时间决定是否重复提醒 -------------------------
        if config.cooldown_seconds <= 0:
            return None
        last = track.last_emit_at if track.last_emit_at is not None else timestamp
        if timestamp - last < config.cooldown_seconds:
            return None

        track.last_emit_at = timestamp
        track.emitted_count += 1
        return self._build_event(outcome, timestamp, track, ongoing=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _deactivate(track: _Track, timestamp: float) -> None:
        if not track.active:
            track.hit_frames = 0
            track.first_hit_at = None
            return
        logger.debug("行为 %s 解除激活", track.config)
        track.active = False
        track.activated_at = None
        track.hit_frames = 0
        track.first_hit_at = None

    @staticmethod
    def _build_event(
        outcome: RuleOutcome, timestamp: float, track: _Track, *, ongoing: bool
    ) -> BehaviorEvent:
        duration = track.duration(timestamp)
        return BehaviorEvent(
            behavior=outcome.behavior,
            severity=outcome.severity,
            confidence=outcome.confidence,
            frame_index=-1,  # 由调用方（服务层）填入真实帧号
            timestamp=timestamp,
            message=outcome.message,
            metrics=dict(outcome.detail),
            duration=duration,
            wall_clock=datetime.now(),
            ongoing=ongoing,
        )
