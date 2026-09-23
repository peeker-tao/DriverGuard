"""视频显示面板。

职责单一：把 ``numpy`` 帧显示出来，并保持宽高比。

Qt 线程约束（重要）
-------------------
``QPixmap`` 只能在 GUI 线程创建，因此：
* 工作线程只传 ``numpy`` 数组（纯数据，可安全跨线程）；
* ``QImage``/``QPixmap`` 的构造全部发生在 :meth:`set_frame`（界面线程）。

缩放策略
--------
保持宽高比做"内接缩放"，并在两侧留黑边。对监控场景来说，
**绝不拉伸** 是底线——人脸比例被拉伸会直接影响驾驶员的主观判读。
"""

from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..theme import Palette

logger = logging.getLogger(__name__)

__all__ = ["VideoPanel"]


class VideoPanel(QWidget):
    """按宽高比显示视频帧的控件。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(QSize(480, 270))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAutoFillBackground(False)
        self._pixmap: QPixmap | None = None
        self._placeholder = "未开始检测"
        self._hint = "点击「开始检测」启动摄像头"

    # ------------------------------------------------------------------
    #  对外接口
    # ------------------------------------------------------------------
    def set_frame(self, canvas: np.ndarray) -> None:
        """显示一帧 BGR 图像（必须在 GUI 线程调用）。"""
        if canvas is None or canvas.size == 0:
            return
        height, width = canvas.shape[:2]
        if canvas.ndim != 3 or canvas.shape[2] != 3:
            logger.warning("非法帧形状 %s，已忽略", canvas.shape)
            return
        # numpy 数组不保证连续内存，QImage 要求连续；ascontiguousarray 开销可忽略
        contiguous = np.ascontiguousarray(canvas)
        image = QImage(contiguous.data, width, height, 3 * width,
                       QImage.Format.Format_BGR888)
        # copy() 让 QPixmap 拥有独立缓冲，避免 numpy 数组被回收后出现花屏
        self._pixmap = QPixmap.fromImage(image.copy())
        self.update()

    def clear(self, placeholder: str = "未开始检测", hint: str = "") -> None:
        self._pixmap = None
        self._placeholder = placeholder
        self._hint = hint
        self.update()

    def clear_hint(self) -> None:
        """只清掉副标题（例如标定结束后）。"""
        self._hint = ""
        self.update()

    def grab_current(self) -> QPixmap | None:
        """返回当前画面副本，用于界面截图留存。"""
        return self._pixmap.copy() if self._pixmap is not None else None

    # ------------------------------------------------------------------
    #  绘制
    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt 命名风格
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor(Palette.BG_INPUT))

        if self._pixmap is None:
            self._draw_placeholder(painter)
            painter.end()
            return

        target = self._fit_rect(self._pixmap.size())
        painter.drawPixmap(target, self._pixmap)
        painter.end()

    def _fit_rect(self, source: QSize):
        """计算保持宽高比的内接矩形（居中，可留黑边）。"""
        from PySide6.QtCore import QRect

        area = self.rect()
        if source.width() <= 0 or source.height() <= 0:
            return area
        scale = min(area.width() / source.width(), area.height() / source.height())
        width = max(1, int(source.width() * scale))
        height = max(1, int(source.height() * scale))
        return QRect(
            area.x() + (area.width() - width) // 2,
            area.y() + (area.height() - height) // 2,
            width,
            height,
        )

    def _draw_placeholder(self, painter: QPainter) -> None:
        painter.setPen(QColor(Palette.TEXT_MUTED))
        font = QFont(self.font())
        font.setPointSizeF(max(11.0, font.pointSizeF() + 4))
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)

        if self._hint:
            hint_font = QFont(self.font())
            hint_font.setPointSizeF(max(9.0, hint_font.pointSizeF()))
            painter.setFont(hint_font)
            painter.setPen(QColor(Palette.TEXT_DIM))
            text_rect = self.rect().adjusted(0, 46, 0, 0)
            painter.drawText(
                text_rect,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                self._hint,
            )
