"""表现层：PySide6 界面（MVVM）。

.. code-block:: text

   View            main_window.MainWindow + widgets/*
        ▲  信号
        │
   ViewModel       viewmodels.MonitorViewModel
        ▲
        │
   Worker          worker.DetectionWorker（QThread 中驱动应用服务）
        ▲
        │
   Application     bootstrap.ApplicationContext

约定：View 里只有"渲染 + 转发"，不含业务判断；Worker 里只有"线程调度 + 数据搬运"。
"""

from .overlay import OverlayOptions, draw_overlay
from .theme import STYLESHEET, Palette
from .worker import DetectionWorker

__all__ = ["OverlayOptions", "draw_overlay", "STYLESHEET", "Palette", "DetectionWorker"]
