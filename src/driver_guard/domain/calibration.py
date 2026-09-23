"""个人中立位标定。

会话开始时采集若干有效帧，用 **中位数** 估计驾驶员的自然坐姿与头部朝向，
作为后续所有头部姿态判定的基准。

为什么用中位数而不是均值
------------------------
标定窗口内不可避免会混入眨眼、说话、短暂偏头等异常帧。均值会被这些
离群值拉偏，而中位数对离群值不敏感（稳健估计）。

为什么需要标定
--------------
摄像头安装角度、座椅高度、驾驶员身高都会整体平移姿态角读数。
标定把"个体差异"从"行为差异"中剥离出来，使同一组阈值适用于所有人。
"""

from __future__ import annotations

import logging
import statistics

from .models import DriverMetrics, HeadPose, NeutralProfile

logger = logging.getLogger(__name__)

__all__ = ["NeutralPoseCalibrator"]


class NeutralPoseCalibrator:
    """从连续若干帧的指标中估计个人中立位。

    Args:
        required_samples: 判定标定完成所需的有效样本数（默认 30 帧 ≈ 1 秒 @30fps）。
        max_samples: 样本缓冲区上限，防止长时间停留导致内存增长。
        warmup_frames: 预热帧数，前若干帧模型尚未稳定，直接丢弃。
        max_pitch_ratio_std: ``face_pitch_ratio`` 的标准差上限，超过则认为
            标定期间驾驶员动作过大，本次标定结果标记为"低可信"。
    """

    def __init__(
        self,
        *,
        required_samples: int = 30,
        max_samples: int = 240,
        warmup_frames: int = 5,
        max_pitch_ratio_std: float = 0.05,
    ) -> None:
        if required_samples <= 0:
            raise ValueError("required_samples 必须为正整数")
        self._required = required_samples
        self._max_samples = max(max_samples, required_samples)
        self._warmup = warmup_frames
        self._max_std = max_pitch_ratio_std

        self._seen_frames = 0
        self._pitches: list[float] = []
        self._yaws: list[float] = []
        self._rolls: list[float] = []
        self._ratios: list[float] = []
        self._profile: NeutralProfile | None = None

    # ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        """是否已完成标定。"""
        return self._profile is not None and self._profile.ready

    @property
    def progress(self) -> float:
        """标定进度 0~1，供界面显示进度条。"""
        if self.ready:
            return 1.0
        return min(1.0, len(self._pitches) / self._required)

    @property
    def profile(self) -> NeutralProfile | None:
        return self._profile

    # ------------------------------------------------------------------
    def feed(self, metrics: DriverMetrics) -> NeutralProfile | None:
        """喂入一帧指标；返回最新标定档案（未完成则为 ``None``）。

        只有同时具备人脸与姿态数据的帧才计入，因此驾驶员尚未就座、
        镜头被遮挡等情况不会污染标定。
        """
        if self.ready:
            return self._profile

        self._seen_frames += 1
        if self._seen_frames <= self._warmup:
            return None

        if not metrics.face_detected:
            return None
        if metrics.pitch is None or metrics.yaw is None or metrics.roll is None:
            return None
        # 明显偏头/低头时不参与标定，否则基准会被污染
        if max(abs(metrics.pitch), abs(metrics.yaw), abs(metrics.roll)) > 35.0:
            return None

        self._pitches.append(metrics.pitch)
        self._yaws.append(metrics.yaw)
        self._rolls.append(metrics.roll)
        if metrics.face_pitch_ratio is not None:
            self._ratios.append(metrics.face_pitch_ratio)

        if self._trim():
            logger.warning("标定样本超过上限 %d，已裁剪最早的样本", self._max_samples)

        if len(self._pitches) >= self._required:
            self._finalize()
        return self._profile

    def reset(self) -> None:
        """清空标定状态（用于"重新标定"功能）。"""
        self._seen_frames = 0
        self._pitches.clear()
        self._yaws.clear()
        self._rolls.clear()
        self._ratios.clear()
        self._profile = None

    # ------------------------------------------------------------------
    def _trim(self) -> bool:
        """裁剪超限样本，返回是否发生了裁剪。"""
        overflow = len(self._pitches) - self._max_samples
        if overflow <= 0:
            return False
        del self._pitches[:overflow]
        del self._yaws[:overflow]
        del self._rolls[:overflow]
        if len(self._ratios) > self._max_samples:
            del self._ratios[: len(self._ratios) - self._max_samples]
        return True

    def _finalize(self) -> None:
        head_pose = HeadPose(
            pitch=statistics.median(self._pitches),
            yaw=statistics.median(self._yaws),
            roll=statistics.median(self._rolls),
        )
        ratio = statistics.median(self._ratios) if self._ratios else None

        if len(self._ratios) >= 3:
            std = statistics.pstdev(self._ratios)
            if std > self._max_std:
                logger.warning(
                    "标定期间 face_pitch_ratio 波动较大（σ=%.4f > %.4f），"
                    "本次中立位仅供兜底使用，建议重新标定",
                    std, self._max_std,
                )

        self._profile = NeutralProfile(
            head_pose=head_pose,
            pitch_ratio=ratio,
            samples=len(self._pitches),
        )
        logger.info(
            "中立位标定完成：pitch=%.2f° yaw=%.2f° roll=%.2f° ratio=%s（%d 帧）",
            head_pose.pitch, head_pose.yaw, head_pose.roll,
            f"{ratio:.4f}" if ratio is not None else "N/A",
            len(self._pitches),
        )
