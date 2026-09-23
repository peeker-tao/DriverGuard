"""端口（Ports）：应用层对基础设施的抽象接口。

本模块是 **六边形架构 / 端口-适配器（Ports & Adapters）** 风格的核心：

.. code-block:: text

     ┌──────────────────────── 应用层（内核） ─────────────────────────┐
     │   DetectionService / DetectionPipeline / StatisticsService      │
     └───────┬──────────────┬──────────────┬──────────────┬───────────┘
             │ 端口          │ 端口          │ 端口          │ 端口
      FrameSource      LandmarkProvider  AlertSink    EventRepository
             │              │              │              │
     ┌───────┴──────────────┴──────────────┴──────────────┴───────────┐
     │  基础设施适配器：摄像头/视频文件/合成源、MediaPipe/Haar/回放、   │
     │  控制台/CSV/声音/快照/界面 告警、CSV 事件仓储                    │
     └────────────────────────────────────────────────────────────────┘

应用层只依赖这里定义的抽象，因此：

* 换成 Web 服务只需新增适配器，内核零改动；
* 单元测试可注入内存实现（``tests/fakes.py``），无需摄像头与模型。
"""

from __future__ import annotations

import abc
from typing import Iterable, Iterator, Sequence

from ..domain.models import BehaviorEvent, Frame, Observation

__all__ = ["FrameSource", "LandmarkProvider", "AlertSink", "EventRepository"]


# ===========================================================================
class FrameSource(abc.ABC):
    """视频帧来源端口。

    实现可以是摄像头、视频文件、图片序列、合成动画，或测试用的固定队列。
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """数据源的人类可读描述，用于界面展示与日志。"""

    @property
    def fps(self) -> float:
        """声明帧率，用于时序判定与显示；未知时返回 30.0。"""
        return 30.0

    @property
    def resolution(self) -> tuple[int, int] | None:
        """``(宽, 高)``；未知返回 ``None``。"""
        return None

    @abc.abstractmethod
    def open(self) -> None:
        """打开数据源。失败时应抛出 :class:`FrameSourceError`。"""

    @abc.abstractmethod
    def read(self) -> Frame | None:
        """读取下一帧；数据结束或读取失败返回 ``None``。"""

    def close(self) -> None:
        """释放资源。默认空实现，子类按需覆盖。"""

    # -- 上下文管理器语法糖 ---------------------------------------------
    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[Frame]:
        """支持 ``for frame in source`` 迭代（会自动 open/close）。"""
        self.open()
        try:
            while (frame := self.read()) is not None:
                yield frame
        finally:
            self.close()


class FrameSourceError(RuntimeError):
    """数据源无法打开或读取失败。"""


# ===========================================================================
class LandmarkProvider(abc.ABC):
    """关键点（感知）推理端口。

    实现负责把一帧图像转换为 :class:`Observation`。该接口刻意不暴露
    任何推理库类型，使上层完全不知道用的是 MediaPipe、Haar 还是回放。
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """实现名称（如 ``mediapipe``、``haar``、``replay``），用于诊断信息。"""

    @property
    def ready(self) -> bool:
        """是否已就绪（模型已加载）。未就绪时 UI 应给出可操作提示。"""
        return True

    @abc.abstractmethod
    def infer(self, frame: Frame) -> Observation:
        """对单帧推理，返回感知结果。"""

    def warmup(self) -> None:
        """可选：预热模型（首次推理通常明显更慢）。"""

    def close(self) -> None:
        """释放模型资源。"""


# ===========================================================================
class AlertSink(abc.ABC):
    """告警输出端口（单向，应用层不关心输出去向）。"""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """输出通道名称，用于界面展示已启用的告警通道。"""

    @abc.abstractmethod
    def emit(self, event: BehaviorEvent, frame: Frame | None = None) -> None:
        """发布一条告警。实现**必须自行吞掉异常**并记录日志，
        以免告警输出失败中断采集主循环。"""

    def close(self) -> None:
        """释放资源（如关闭文件句柄）。"""


# ===========================================================================
class EventRepository(abc.ABC):
    """事件仓储端口（持久化抽象）。

    命名沿用领域驱动设计的 Repository 模式：上层只看到"集合"语义，
    不关心背后是 CSV、SQLite 还是内存。
    """

    @abc.abstractmethod
    def save(self, event: BehaviorEvent) -> None:
        """追加保存一条事件。"""

    @abc.abstractmethod
    def all(self) -> Sequence[BehaviorEvent]:
        """返回全部事件（按写入顺序）。"""

    def extend(self, events: Iterable[BehaviorEvent]) -> None:
        """批量保存。默认逐条调用 :meth:`save`，子类可覆写为批量写。"""
        for event in events:
            self.save(event)

    def clear(self) -> None:
        """清空仓储。默认空实现。"""
