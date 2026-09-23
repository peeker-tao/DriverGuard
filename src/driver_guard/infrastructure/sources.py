"""帧来源适配器：把各种视频输入统一成 :class:`FrameSource` 端口。

实现了 **适配器（Adapter）模式**：每个具体输入方式都被包装成应用层
能理解的统一接口，因此更换数据源对上层完全透明。

提供的适配器
------------
===========================  ==================================================
类                            用途
===========================  ==================================================
``WebcamFrameSource``         摄像头实时采集
``VideoFileFrameSource``      视频文件回放（支持循环、变速）
``SyntheticFrameSource``      程序合成画面，**无需任何硬件即可演示与自检**
``ImageSequenceFrameSource``  图片序列（离线批量分析）
===========================  ==================================================

``SyntheticFrameSource`` 的价值：在没有摄像头、没有真实视频的场合
（例如答辩现场、CI 环境），仍能驱动完整流水线，验证除"感知精度"以外的
所有环节。它与 ``ReplayLandmarkProvider`` 配合使用可复现指定场景。
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from ..application.ports import FrameSource, FrameSourceError
from ..domain.models import Frame

logger = logging.getLogger(__name__)

__all__ = [
    "WebcamFrameSource",
    "VideoFileFrameSource",
    "ImageSequenceFrameSource",
    "SyntheticFrameSource",
]


# ===========================================================================
class _OpenCvSourceBase(FrameSource):
    """基于 ``cv2.VideoCapture`` 的适配器公共实现。"""

    def __init__(self, *, width: int = 0, height: int = 0, fps: float = 0.0) -> None:
        self._capture: cv2.VideoCapture | None = None
        self._index = 0
        self._started_at = 0.0
        self._requested_width = width
        self._requested_height = height
        self._requested_fps = fps
        self._actual_fps = fps if fps > 0 else 30.0

    # -- 子类实现 -------------------------------------------------------
    def _create_capture(self) -> cv2.VideoCapture:  # pragma: no cover - 需硬件
        raise NotImplementedError

    @property
    def name(self) -> str:  # pragma: no cover - 由子类覆盖
        return "opencv"

    # -- 公共生命周期 ---------------------------------------------------
    def open(self) -> None:
        if self._capture is not None:
            return
        capture = self._create_capture()
        if not capture.isOpened():
            capture.release()
            raise FrameSourceError(f"无法打开数据源：{self.name}")
        if self._requested_width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._requested_width)
        if self._requested_height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested_height)
        if self._requested_fps > 0:
            capture.set(cv2.CAP_PROP_FPS, self._requested_fps)

        reported = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        # 某些后端对摄像头谎报 fps（返回 1000 或 0），做合理性夹紧
        self._actual_fps = reported if 1.0 <= reported <= 240.0 else (self._requested_fps or 30.0)
        self._capture = capture
        self._index = 0
        self._started_at = time.perf_counter()
        logger.info("已打开数据源：%s（%.1f fps，%dx%d）", self.name, self._actual_fps, *self._resolution_or_default())

    def _resolution_or_default(self) -> tuple[int, int]:
        return self.resolution or (0, 0)

    @property
    def fps(self) -> float:
        return self._actual_fps

    @property
    def resolution(self) -> tuple[int, int] | None:
        if self._capture is None:
            return None
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return (width, height) if width and height else None

    def read(self) -> Frame | None:
        if self._capture is None:
            return None
        ok, image = self._capture.read()
        if not ok or image is None:
            return None
        frame = Frame(
            index=self._index,
            timestamp=time.perf_counter() - self._started_at,
            image=image,
        )
        self._index += 1
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("已关闭数据源：%s", self.name)


# ===========================================================================
class WebcamFrameSource(_OpenCvSourceBase):
    """摄像头实时采集适配器。

    Args:
        index: 摄像头序号（``0`` 为默认摄像头）。
        backend: OpenCV 后端标志，``None`` 表示自动选择。
            在 Windows 上显式使用 ``cv2.CAP_DSHOW`` 通常能显著缩短打开耗时。
    """

    def __init__(
        self,
        *,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        backend: int | None = None,
    ) -> None:
        super().__init__(width=width, height=height, fps=fps)
        self._index_number = index
        self._backend = backend

    @property
    def name(self) -> str:
        return f"摄像头 #{self._index_number}"

    def _create_capture(self) -> cv2.VideoCapture:
        if self._backend is None:
            return cv2.VideoCapture(self._index_number)
        return cv2.VideoCapture(self._index_number, self._backend)


# ===========================================================================
class VideoFileFrameSource(_OpenCvSourceBase):
    """视频文件回放适配器。

    Args:
        path: 视频文件路径。
        loop: 播放到结尾后是否从头循环（演示时很有用）。
        speed: 播放速率，``1.0`` 为原速。
    """

    def __init__(
        self,
        path: str | Path,
        *,
        loop: bool = False,
        speed: float = 1.0,
        width: int = 0,
        height: int = 0,
    ) -> None:
        super().__init__(width=width, height=height)
        self._path = Path(path)
        self._loop = loop
        self._speed = max(0.05, float(speed))
        self._loops = 0
        if not self._path.exists():
            raise FrameSourceError(f"视频文件不存在：{self._path}")

    @property
    def name(self) -> str:
        return f"视频文件 {self._path.name}"

    @property
    def fps(self) -> float:
        return self._actual_fps * self._speed

    @property
    def loop_count(self) -> int:
        return self._loops

    def _create_capture(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(str(self._path))

    def read(self) -> Frame | None:
        if self._capture is None:
            return None
        ok, image = self._capture.read()
        if not ok or image is None:
            if not self._loop:
                return None
            self._loops += 1
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = self._capture.read()
            if not ok or image is None:
                return None
            logger.debug("视频循环第 %d 次", self._loops)

        # 用"帧号 ÷ 视频帧率"作为时间戳，使时序判定与播放速率一致
        timestamp = self._index / max(1e-6, self.fps)
        frame = Frame(index=self._index, timestamp=timestamp, image=image)
        self._index += 1
        return frame


# ===========================================================================
class ImageSequenceFrameSource(FrameSource):
    """图片序列适配器（离线批量分析）。

    时间戳按 ``1/fps`` 递增，使时序规则对静态图片也能产生确定的行为，
    便于回归测试与参数调优。
    """

    def __init__(
        self,
        paths: Iterable[str | Path],
        *,
        fps: float = 25.0,
        repeat: int = 1,
        suffix_in_index: bool = True,
    ) -> None:
        self._paths = [Path(p) for p in paths]
        if not self._paths:
            raise FrameSourceError("图片序列为空")
        self._fps = max(1e-6, fps)
        self._repeat = max(1, repeat)
        self._suffix = suffix_in_index
        self._index = 0
        self._missing: list[Path] = []

    @property
    def name(self) -> str:
        return f"图片序列（{len(self._paths)} 张，循环 {self._repeat} 次）"

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def missing_files(self) -> Sequence[Path]:
        return tuple(self._missing)

    def open(self) -> None:
        missing = [p for p in self._paths if not p.exists()]
        if len(missing) == len(self._paths):
            raise FrameSourceError("图片序列中的文件全部不存在")
        self._missing = missing
        if missing:
            logger.warning("有 %d 张图片不存在，已跳过：%s", len(missing), missing[:3])

    def read(self) -> Frame | None:
        total = len(self._paths) * self._repeat
        while self._index < total:
            path = self._paths[self._index % len(self._paths)]
            current = self._index
            self._index += 1
            image = cv2.imread(str(path))
            if image is None:
                logger.warning("读取图片失败，已跳过：%s", path)
                continue
            return Frame(index=current, timestamp=current / self._fps, image=image)
        return None

    def close(self) -> None:
        self._missing = []


# ===========================================================================
class SyntheticFrameSource(FrameSource):
    """合成画面数据源：程序生成"驾驶舱"示意画面。

    用途
    ----
    * **演示与答辩**：没有摄像头也能跑通完整界面与流水线；
    * **回归测试**：画面确定可复现，便于比对流水线耗时与界面刷新；
    * **压力测试**：可以指定极高的目标帧率来衡量纯处理吞吐上限。

    画面内容：座舱背景 + 驾驶员头部轮廓 + 随时间摆动的姿态示意 +
    实时 HUD 文本。它 **不产生真实人脸**，因此需配合
    :class:`~driver_guard.infrastructure.landmarks.replay_provider.ReplayLandmarkProvider`
    使用才能在无摄像头环境下验证"检测—告警"全链路。
    """

    def __init__(
        self,
        *,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        duration: float = 0.0,
        seed: int = 20240920,
    ) -> None:
        self._width = width
        self._height = height
        self._fps = max(1e-6, fps)
        self._duration = duration
        self._rng = np.random.default_rng(seed)
        self._index = 0
        self._opened = False
        self._background: np.ndarray | None = None

    @property
    def name(self) -> str:
        return f"合成画面 {self._width}x{self._height}@{self._fps:g}fps"

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self) -> tuple[int, int]:
        return self._width, self._height

    def open(self) -> None:
        self._index = 0
        self._opened = True
        self._background = self._make_background()

    def close(self) -> None:
        self._opened = False
        self._background = None

    def _make_background(self) -> np.ndarray:
        """生成一次性的静态背景（座舱氛围），逐帧只做增量绘制以省 CPU。"""
        image = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        # 由上到下的深色渐变
        gradient = np.linspace(28, 12, self._height, dtype=np.uint8)
        image[:, :, 0] = gradient[:, None]
        image[:, :, 1] = (gradient * 1.15).clip(0, 255).astype(np.uint8)[:, None]
        image[:, :, 2] = (gradient * 1.45).clip(0, 255).astype(np.uint8)[:, None]
        # 仪表台
        cv2.rectangle(image, (0, int(self._height * 0.72)), (self._width, self._height), (42, 42, 46), -1)
        cv2.line(image, (0, int(self._height * 0.72)), (self._width, int(self._height * 0.72)), (90, 90, 96), 2)
        return image

    def read(self) -> Frame | None:
        if not self._opened:
            return None
        if self._duration > 0 and self._index >= int(self._duration * self._fps):
            return None

        timestamp = self._index / self._fps
        image = self._background.copy() if self._background is not None else self._make_background()
        self._draw_driver(image, timestamp)
        self._draw_hud(image, timestamp)

        frame = Frame(index=self._index, timestamp=timestamp, image=image)
        self._index += 1
        return frame

    # ------------------------------------------------------------------
    def _draw_driver(self, image: np.ndarray, timestamp: float) -> None:
        """绘制一个随正弦摆动的驾驶员示意轮廓。"""
        cx = self._width // 2 + int(40 * math.sin(timestamp * 0.7))
        cy = int(self._height * 0.46) + int(14 * math.sin(timestamp * 1.3))
        radius = int(min(self._width, self._height) * 0.16)

        cv2.ellipse(image, (cx, cy), (radius, int(radius * 1.22)), 0, 0, 360, (150, 158, 170), -1)
        # 眼睛：亮度随时间变化，模拟眨眼
        blink = 0.5 + 0.5 * math.sin(timestamp * 3.1)
        eye_open = max(1, int(radius * 0.10 * blink))
        for sign in (-1, 1):
            ex = cx + sign * int(radius * 0.38)
            ey = cy - int(radius * 0.18)
            cv2.ellipse(image, (ex, ey), (int(radius * 0.16), eye_open), 0, 0, 360, (60, 60, 68), -1)
        # 嘴：张口幅度随时间变化，模拟说话/打哈欠
        mouth = int(radius * (0.10 + 0.22 * abs(math.sin(timestamp * 1.9))))
        cv2.ellipse(image, (cx, cy + int(radius * 0.42)), (int(radius * 0.24), mouth), 0, 0, 360, (70, 60, 66), -1)

    def _draw_hud(self, image: np.ndarray, timestamp: float) -> None:
        height = self._height
        cv2.putText(
            image,
            "SYNTHETIC DRIVER FEED  (no real face)",
            (24, height - 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (170, 170, 176),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"t = {timestamp:7.2f}s   frame #{self._index:06d}",
            (24, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (140, 200, 140),
            1,
            cv2.LINE_AA,
        )
