"""Haar 级联降级感知适配器（优雅降级的落地实现）。

设计意图
--------
本适配器存在的意义 **不是** 替代 MediaPipe，而是证明架构上的一个性质：

    **同一个业务规则，可以在完全不修改规则代码的前提下，
      被一个技术上截然不同的感知实现驱动。**

它不产生 478 点关键点（Haar 给不出），却仍能驱动 :class:`EyesClosedRule`：
做法是把"睁眼检测结果"映射到规则已经在用的第二条证据通道
``blendshapes["eyeBlinkLeft"/"eyeBlinkRight"]`` 上。这正是
:class:`~driver_guard.application.ports.LandmarkProvider` 端口抽象的价值，
也是策略（Strategy）模式带来的可替换性。

能力边界（实测与原理）
----------------------
==============  ==================================================  ==========
能力              实现方式                                          精度
==============  ==================================================  ==========
驾驶员在座        正面人脸级联命中                                  高
驾驶员离位        连续未命中                                        高
闭眼/疲劳        眼级联在上半脸区域命中数 < 2 ⇒ 判为闭眼           中（受光照影响）
==============  ==================================================  ==========
头姿 / 打哈欠 / 打电话 / 抽烟 —— **不支持**，相关规则自动跳过（不会误报）。

依赖
----
需要 ``assets/cascades/`` 下的两个 XML。``opencv-contrib-python`` 5.0 起
不再自带，请先运行 ``python tools/fetch_cascades.py``。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from ...application.ports import LandmarkProvider
from ...domain.models import BoundingBox, Frame, Observation

logger = logging.getLogger(__name__)

__all__ = ["HaarLandmarkProvider"]


class HaarLandmarkProvider(LandmarkProvider):
    """基于 Haar 级联的受限感知实现。

    Args:
        cascade_dir: 级联文件目录。
        face_cascade: 人脸级联文件名。
        eye_cascade: 眼级联文件名。
        scale_factor / min_neighbors / min_size_ratio: 级联检测参数。
            参数偏保守（``min_neighbors`` 较大）以降低误检。
        eye_region_ratio: 在检测到的人脸框上部多大的纵向比例内搜索眼睛
            （眼睛通常位于人脸框上半部，取 0.62 可覆盖眉眼区域）。
        closed_blink_score: 判定为闭眼时写入的 ``eyeBlink`` 分值（0~1）。
    """

    def __init__(
        self,
        *,
        cascade_dir: str | Path = "assets/cascades",
        face_cascade: str = "haarcascade_frontalface_default.xml",
        eye_cascade: str = "haarcascade_eye.xml",
        scale_factor: float = 1.1,
        min_neighbors: int = 6,
        min_size_ratio: float = 0.12,
        eye_region_ratio: float = 0.62,
        closed_blink_score: float = 0.85,
    ) -> None:
        self._dir = Path(cascade_dir)
        self._face_file = self._dir / face_cascade
        self._eye_file = self._dir / eye_cascade
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size_ratio = min_size_ratio
        self._eye_region_ratio = eye_region_ratio
        self._closed_score = closed_blink_score

        self._face_detector: cv2.CascadeClassifier | None = None
        self._eye_detector: cv2.CascadeClassifier | None = None
        self._degraded = True
        self._degraded_reason = "尚未加载级联文件"

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "haar-cascade（降级模式）"

    @property
    def ready(self) -> bool:
        return self._face_detector is not None and not self._face_detector.empty()

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def degraded_reason(self) -> str:
        return self._degraded_reason

    @property
    def eye_detection_available(self) -> bool:
        """眼级联是否可用；不可用时只能做"在座/离位"判定。"""
        return self._eye_detector is not None and not self._eye_detector.empty()

    def missing_files(self) -> tuple[Path, ...]:
        return tuple(p for p in (self._face_file, self._eye_file) if not p.exists())

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._face_detector is not None:
            return
        if not self._face_file.exists():
            raise FileNotFoundError(
                f"缺少人脸级联文件 {self._face_file}\n"
                "请先运行： python tools/fetch_cascades.py"
            )

        face = cv2.CascadeClassifier(str(self._face_file))
        if face.empty():
            raise FileNotFoundError(f"级联文件无法解析（可能已损坏）：{self._face_file}")
        self._face_detector = face
        logger.info("已加载 Haar 人脸级联：%s", self._face_file.name)

        if self._eye_file.exists():
            eye = cv2.CascadeClassifier(str(self._eye_file))
            if not eye.empty():
                self._eye_detector = eye
                self._degraded = False
                self._degraded_reason = ""
                logger.info("已加载 Haar 眼级联：%s", self._eye_file.name)
            else:
                self._mark_degraded(f"眼级联无法解析：{self._eye_file.name}")
        else:
            self._mark_degraded(
                f"未找到眼级联 {self._eye_file.name}，闭眼检测不可用（仅支持在座/离位判定）"
            )

    def _mark_degraded(self, reason: str) -> None:
        self._degraded = True
        self._degraded_reason = reason
        logger.warning("Haar 通道降级：%s", reason)

    def warmup(self) -> None:
        self._ensure_loaded()
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        try:
            self.infer(Frame(index=0, timestamp=0.0, image=blank))
        except Exception:  # noqa: BLE001
            logger.exception("Haar 预热失败（已忽略）")

    def close(self) -> None:
        self._face_detector = None
        self._eye_detector = None

    # ------------------------------------------------------------------
    def infer(self, frame: Frame) -> Observation:
        self._ensure_loaded()
        started = time.perf_counter()
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # 光照归一化，显著提升级联稳定性

        height, width = gray.shape[:2]
        min_size = max(24, int(min(height, width) * self._min_size_ratio))
        faces = self._face_detector.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=(min_size, min_size),
        )
        latency = (time.perf_counter() - started) * 1000.0

        if len(faces) == 0:
            # 未检测到人脸：交给 DriverAbsentRule 处理
            return Observation(provider=self.name, latency_ms=latency)

        # 取面积最大的脸（驾驶员应是最显著的人脸）
        x, y, w, h = max(faces, key=lambda rect: int(rect[2]) * int(rect[3]))
        box = BoundingBox(
            x_min=x / width,
            y_min=y / height,
            x_max=(x + w) / width,
            y_max=(y + h) / height,
        )

        blendshapes: dict[str, float] = {}
        if self.eye_detection_available:
            score = self._estimate_blink(gray, x, y, w, h, min_size)
            blendshapes = {"eyeBlinkLeft": score, "eyeBlinkRight": score}

        return Observation(
            face_box=box,
            blendshapes=blendshapes,
            provider=self.name,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    # ------------------------------------------------------------------
    def _estimate_blink(
        self, gray: np.ndarray, x: int, y: int, w: int, h: int, min_size: int
    ) -> float:
        """用"上半脸能否找到两只眼睛"估计眨眼/闭眼程度。

        原理：睁眼时眼级联通常能在人脸框上部命中 2 个目标；闭眼时眼睑
        覆盖巩膜，级联特征消失，命中数降为 0~1。该判据对光照敏感，
        因此只作为 **降级通道** 使用，且需要时序确认层过滤瞬时抖动。
        """
        assert self._eye_detector is not None
        region_height = int(h * self._eye_region_ratio)
        region = gray[y : y + region_height, x : x + w]
        if region.size == 0:
            return 0.0

        eye_min = max(12, int(min_size * 0.35))
        eyes = self._eye_detector.detectMultiScale(
            region,
            scaleFactor=self._scale_factor,
            minNeighbors=max(3, self._min_neighbors // 2),
            minSize=(eye_min, eye_min),
        )
        found = len(eyes)
        if found >= 2:
            return 0.0
        if found == 1:
            # 单眼命中：可能是侧脸或单眼闭合，给一个中间值而非直接判闭眼
            return self._closed_score * 0.5
        return self._closed_score
