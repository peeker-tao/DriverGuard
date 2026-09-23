"""画面叠加渲染（Overlay）。

把分析结果画到视频帧上：人脸框、关键点、当前行为状态、指标 HUD。

设计约束
--------
* 只在 **工作线程** 中调用，产出的是普通 ``numpy`` 数组，不涉及任何 Qt 类型，
  因此与界面框架解耦、可单独测试；
* 所有绘制基于 `cv2`，开销可控（720p 下单帧 < 2ms）；
* 中文通过 :class:`ChineseTextRenderer` 渲染——OpenCV 内置 ``putText``
  无法绘制汉字，若直接传中文只会显示为 ``???``，这是实战中极易忽略的坑。
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..domain.geometry import FaceMeshIndex
from ..domain.models import BehaviorEvent, BehaviorType, Observation, Severity
from .theme import BEHAVIOR_COLORS, Palette, SEVERITY_COLORS

logger = logging.getLogger(__name__)

__all__ = ["draw_overlay", "ChineseTextRenderer", "OverlayOptions"]


# ===========================================================================
class ChineseTextRenderer:
    """用 PIL + 系统字体绘制中文，解决 OpenCV ``putText`` 不支持汉字的问题。

    字体按候选列表探测，找到第一个可用的即缓存；全部失败时退化为
    英文标签（见 :data:`_ASCII_FALLBACK`），保证界面不出现乱码方块。
    """

    #: Windows / Linux / macOS 上常见的 CJK 字体候选
    FONT_CANDIDATES: tuple[str, ...] = (
        "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑
        "C:/Windows/Fonts/msyhbd.ttc",     # 微软雅黑 Bold
        "C:/Windows/Fonts/simhei.ttf",     # 黑体
        "C:/Windows/Fonts/simsun.ttc",     # 宋体
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    )

    def __init__(self) -> None:
        self._font_path: str | None = None
        self._cache: dict[int, object] = {}
        self._warned = False
        self._resolve_font()

    def _resolve_font(self) -> None:
        for candidate in self.FONT_CANDIDATES:
            if Path(candidate).exists():
                self._font_path = candidate
                logger.debug("中文渲染字体：%s", candidate)
                return
        logger.warning("未找到中文字体，界面上的中文将退化为英文标签")

    @property
    def available(self) -> bool:
        return self._font_path is not None

    def _font(self, size: int):
        try:
            from PIL import ImageFont
        except ImportError:  # pragma: no cover - Pillow 是 mediapipe 的依赖，一般存在
            return None
        font = self._cache.get(size)
        if font is None:
            try:
                font = ImageFont.truetype(self._font_path, size)
            except Exception:  # noqa: BLE001
                if not self._warned:
                    self._warned = True
                    logger.exception("加载中文字体失败，改用英文标签")
                return None
            self._cache[size] = font
        return font

    def draw(
        self,
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        *,
        size: int = 20,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """在 ``image``（BGR）上的 ``origin``（左下角基线）处绘制文字。

        若字体不可用或文字为纯 ASCII，则回退到 ``cv2.putText``（更快）。
        """
        if text.isascii():
            cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                        size / 30.0, color, max(1, size // 16), cv2.LINE_AA)
            return

        font = self._font(size)
        if font is None:
            cv2.putText(image, _to_ascii(text), origin, cv2.FONT_HERSHEY_SIMPLEX,
                        size / 30.0, color, max(1, size // 16), cv2.LINE_AA)
            return

        try:
            from PIL import Image, ImageDraw
        except ImportError:  # pragma: no cover
            cv2.putText(image, _to_ascii(text), origin, cv2.FONT_HERSHEY_SIMPLEX,
                        size / 30.0, color, max(1, size // 16), cv2.LINE_AA)
            return

        # PIL 的坐标原点在左上角，且 draw.text 的 y 是基线；这里用 anchor="ls"
        # 明确指定"左基线"，与 OpenCV 的语义对齐，避免文字上下偏移。
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        drawer = ImageDraw.Draw(pil)
        drawer.text(origin, text, font=font, fill=(color[2], color[1], color[0]), anchor="ls")
        image[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

    def measure(self, text: str, size: int) -> tuple[int, int]:
        """估算文字宽高（像素），用于绘制背景条。"""
        if text.isascii():
            scale = size / 30.0
            (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale,
                                                 max(1, size // 16))
            return width, height
        font = self._font(size)
        if font is None:
            return len(text) * size, size
        try:
            left, top, right, bottom = font.getbbox(text)
            return right - left, bottom - top
        except Exception:  # noqa: BLE001
            return len(text) * size, size


#: 中文字体缺失时的英文兜底标签
_ASCII_FALLBACK: dict[str, str] = {
    "闭眼疲劳": "EYES CLOSED",
    "打哈欠": "YAWNING",
    "低头": "HEAD DOWN",
    "视线偏离": "GAZE OFF ROAD",
    "打电话": "PHONE CALL",
    "抽烟": "SMOKING",
    "驾驶员离位": "DRIVER ABSENT",
    "提示": "INFO",
    "轻微": "LOW",
    "中等": "MEDIUM",
    "严重": "HIGH",
    "危急": "CRITICAL",
    "正常驾驶": "NORMAL",
    "未检测到驾驶员": "NO DRIVER",
    "标定中": "CALIBRATING",
}


def _to_ascii(text: str) -> str:
    return _ASCII_FALLBACK.get(text, text.encode("ascii", "replace").decode("ascii"))


# ===========================================================================
class OverlayOptions:
    """叠加层的开关与外观参数（由界面配置注入）。"""

    __slots__ = ("show_landmarks", "show_face_box", "show_hud", "show_hand_points")

    def __init__(
        self,
        *,
        show_landmarks: bool = True,
        show_face_box: bool = True,
        show_hud: bool = True,
        show_hand_points: bool = True,
    ) -> None:
        self.show_landmarks = show_landmarks
        self.show_face_box = show_face_box
        self.show_hud = show_hud
        self.show_hand_points = show_hand_points


#: 进程级共享的字体渲染器（构造一次，避免每帧探测字体）
_RENDERER = ChineseTextRenderer()

#: 需要重点标注的面部关键点（眉眼口鼻），其余点略去以保持画面清爽
_KEY_FACE_POINTS: tuple[int, ...] = (
    33, 133, 159, 145, 160, 158, 153, 144,      # 右眼
    362, 263, 386, 374, 385, 387, 373, 380,     # 左眼
    13, 14, 78, 308, 61, 291,                   # 嘴
    1, 4, 168, 6, 197,                          # 鼻
    10, 152, 234, 454,                          # 额/下巴/颧
)


# ===========================================================================
def draw_overlay(
    image: np.ndarray,
    observation: Observation,
    *,
    options: OverlayOptions | None = None,
    active_events: tuple[BehaviorEvent, ...] = (),
    metrics: dict[str, float] | None = None,
    calibration_progress: float = 1.0,
    banner: str = "",
    latency_ms: float = 0.0,
) -> np.ndarray:
    """在图像副本上绘制叠加层并返回新数组。

    Args:
        image: BGR 帧（不会被修改）。
        observation: 感知结果。
        options: 显示开关。
        active_events: 当前处于激活状态的行为告警，用于绘制状态横幅。
        metrics: 指标字典，用于右侧 HUD。
        calibration_progress: 中立位标定进度（0~1），未完成时显示进度条。
        banner: 顶部提示文字（如"回放模式"）。
        latency_ms: 感知耗时，显示在 HUD 中。

    Returns:
        绘制后的新图像（BGR）。
    """
    opts = options or OverlayOptions()
    canvas = image.copy()

    if opts.show_face_box and observation.face_box is not None:
        _draw_face_box(canvas, observation, active_events)

    if opts.show_landmarks and observation.face:
        _draw_landmarks(canvas, observation)

    if opts.show_hand_points and observation.hands:
        _draw_hands(canvas, observation)

    if opts.show_hud:
        _draw_hud(canvas, metrics or {}, latency_ms=latency_ms)

    _draw_status_banner(canvas, active_events)
    if calibration_progress < 1.0:
        _draw_calibration(canvas, calibration_progress)
    if banner:
        _draw_banner(canvas, banner)

    return canvas


# ---------------------------------------------------------------------------
#  各图层
# ---------------------------------------------------------------------------
def _primary_color(active_events: tuple[BehaviorEvent, ...]) -> tuple[int, int, int]:
    """当前最严重告警的颜色（BGR）；无告警时用中性蓝。"""
    if not active_events:
        return _hex_to_bgr(Palette.ACCENT)
    worst = max(active_events, key=lambda event: event.severity)
    return _hex_to_bgr(SEVERITY_COLORS[worst.severity])


def _draw_face_box(
    canvas: np.ndarray, observation: Observation, active_events: tuple[BehaviorEvent, ...]
) -> None:
    box = observation.face_box
    if box is None:
        return
    height, width = canvas.shape[:2]
    x1, y1, x2, y2 = box.as_pixels(width, height)
    color = _primary_color(active_events)

    # 四角描边：比整框更轻量，不遮挡面部
    arm = max(10, int(min(width, height) * 0.035))
    thickness = max(2, int(min(width, height) / 320))
    for (px, py, dx, dy) in (
        (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
    ):
        cv2.line(canvas, (px, py), (px + dx * arm, py), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (px, py), (px, py + dy * arm), color, thickness, cv2.LINE_AA)


def _draw_landmarks(canvas: np.ndarray, observation: Observation) -> None:
    height, width = canvas.shape[:2]
    face = observation.face
    for index in _KEY_FACE_POINTS:
        if index >= len(face):
            continue
        point = face[index]
        cv2.circle(canvas, (int(point.x * width), int(point.y * height)), 1,
                   (120, 255, 170), -1, cv2.LINE_AA)


def _draw_hands(canvas: np.ndarray, observation: Observation) -> None:
    height, width = canvas.shape[:2]
    for hand in observation.hands:
        for point in hand:
            cv2.circle(canvas, (int(point.x * width), int(point.y * height)), 1,
                       (255, 190, 120), -1, cv2.LINE_AA)


def _draw_hud(canvas: np.ndarray, metrics: dict[str, float], *, latency_ms: float) -> None:
    """右上角指标面板：EAR / MAR / 头姿 / 手部距离。"""
    rows: list[tuple[str, str]] = []
    if (ear := metrics.get("ear")) is not None:
        rows.append(("EAR", f"{ear:.3f}"))
    if (mar := metrics.get("mar")) is not None:
        rows.append(("MAR", f"{mar:.3f}"))
    if (pitch := metrics.get("pitch")) is not None:
        rows.append(("Pitch", f"{pitch:+.1f}°"))
    if (yaw := metrics.get("yaw")) is not None:
        rows.append(("Yaw", f"{yaw:+.1f}°"))
    if (roll := metrics.get("roll")) is not None:
        rows.append(("Roll", f"{roll:+.1f}°"))
    if (blink := metrics.get("eye_blink")) is not None:
        rows.append(("Blink", f"{blink:.2f}"))
    if (jaw := metrics.get("jaw_open")) is not None:
        rows.append(("Jaw", f"{jaw:.2f}"))
    if (ratio := metrics.get("hand_face_ratio")) is not None:
        rows.append(("Hand", f"{ratio:.2f}"))
    if latency_ms > 0:
        rows.append(("Infer", f"{latency_ms:.0f}ms"))
    if not rows:
        return

    height, width = canvas.shape[:2]
    font_size = max(12, int(min(width, height) / 46))
    line_height = font_size + 8
    text_width = max(_RENDERER.measure(f"{k}  {v}", font_size)[0] for k, v in rows)
    panel_w = text_width + 24
    panel_h = line_height * len(rows) + 16
    x0, y0 = max(0, width - panel_w - 12), 12

    _draw_glass_panel(canvas, x0, y0, panel_w, panel_h)

    for index, (key, value) in enumerate(rows):
        y = y0 + 12 + line_height * index + font_size
        _RENDERER.draw(canvas, key, (x0 + 10, y), size=font_size, color=(178, 190, 202))
        value_width = _RENDERER.measure(value, font_size)[0]
        _RENDERER.draw(canvas, value, (x0 + panel_w - value_width - 10, y),
                       size=font_size, color=(235, 242, 248))


def _draw_status_banner(canvas: np.ndarray, active_events: tuple[BehaviorEvent, ...]) -> None:
    """底部状态条：显示当前所有激活的危险行为。"""
    height, width = canvas.shape[:2]
    font_size = max(16, int(min(width, height) / 30))
    line_height = font_size + 10

    if not active_events:
        texts = [("正常驾驶", _hex_to_bgr(Palette.OK))]
    else:
        ordered = sorted(active_events, key=lambda e: (-int(e.severity), e.behavior.value))
        texts = [
            (f"{event.behavior.label}  {event.confidence:.0%}",
             _hex_to_bgr(SEVERITY_COLORS[event.severity]))
            for event in ordered
        ]

    panel_h = line_height * len(texts) + 16
    y0 = height - panel_h - 12
    _draw_glass_panel(canvas, 12, y0, max(
        _RENDERER.measure(text, font_size)[0] for text, _ in texts
    ) + 28, panel_h)

    for index, (text, color) in enumerate(texts):
        y = y0 + 12 + line_height * index + font_size
        if active_events:
            cv2.rectangle(canvas, (18, y - font_size), (24, y + 4), color, -1)
        _RENDERER.draw(canvas, text, (34, y), size=font_size, color=color)


def _draw_calibration(canvas: np.ndarray, progress: float) -> None:
    """顶部标定进度条。"""
    height, width = canvas.shape[:2]
    bar_w = int(width * 0.34)
    bar_h = 8
    x0 = (width - bar_w) // 2
    y0 = 18
    font_size = max(13, int(min(width, height) / 44))

    label = "中立位标定中，请保持正常坐姿…"
    label_w = _RENDERER.measure(label, font_size)[0]
    _draw_glass_panel(canvas, x0 - 12, y0 - 6, bar_w + 24, bar_h + font_size + 30)
    _RENDERER.draw(canvas, label, ((width - label_w) // 2, y0 + font_size),
                   size=font_size, color=(230, 238, 246))

    bar_y = y0 + font_size + 10
    cv2.rectangle(canvas, (x0, bar_y), (x0 + bar_w, bar_y + bar_h), (60, 70, 82), -1)
    cv2.rectangle(canvas, (x0, bar_y), (x0 + int(bar_w * progress), bar_y + bar_h),
                  _hex_to_bgr(Palette.ACCENT), -1)


def _draw_banner(canvas: np.ndarray, text: str) -> None:
    """左上角提示条（用于标注"回放模式"等状态）。"""
    font_size = max(13, int(min(canvas.shape[:2]) / 42))
    text_w, _ = _RENDERER.measure(text, font_size)
    _draw_glass_panel(canvas, 12, 12, text_w + 24, font_size + 20)
    _RENDERER.draw(canvas, text, (24, 12 + font_size + 8), size=font_size,
                   color=(250, 214, 140))


def _draw_glass_panel(canvas: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    """绘制半透明深色底板，保证文字在任何背景上都可读。"""
    height, width = canvas.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    if x2 <= x1 or y2 <= y1:
        return
    region = canvas[y1:y2, x1:x2]
    overlay = np.zeros_like(region)
    overlay[:] = (16, 20, 26)
    cv2.addWeighted(overlay, 0.62, region, 0.38, 0, region)


def _hex_to_bgr(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return b, g, r
