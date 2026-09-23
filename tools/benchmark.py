"""性能基准：测量各阶段耗时，为架构文档提供**实测**数据。

为什么需要它
------------
架构文档里的"实时性满足要求"必须是可复现的数字，不能是定性判断。
本工具把性能断言变成可重复执行的测量，从而：

* 支撑质量属性场景（QA-1）的验证；
* 在改阈值/加规则后快速发现性能回退；
* 区分"感知耗时"与"其余三级耗时"，定位瓶颈归属。

用法::

    # 用合成画面（无真实人脸，可测纯流程开销与感知的空跑开销）
    python tools/benchmark.py

    # 用真实人脸图片测感知耗时——这是最有参考价值的数字
    python tools/benchmark.py --image path/to/face.jpg

    # 指定分辨率阶梯与重复次数
    python tools/benchmark.py --image face.jpg --sizes 640x480 1280x720 --repeat 40
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from driver_guard.application.pipeline import DetectionPipeline  # noqa: E402
from driver_guard.bootstrap import _confirmer_configs  # noqa: E402
from driver_guard.config import AppConfig, load_config  # noqa: E402
from driver_guard.domain.confirmation import BehaviorConfirmer  # noqa: E402
from driver_guard.domain.metrics import MetricExtractor  # noqa: E402
from driver_guard.domain.models import Frame  # noqa: E402
from driver_guard.domain.rules import build_engine  # noqa: E402
from driver_guard.infrastructure.landmarks import create_landmark_provider  # noqa: E402

import cv2  # noqa: E402


def _source_image(path: Path | None, size: tuple[int, int]) -> np.ndarray:
    """取得一帧测试图像：优先使用给定图片，否则用带纹理的合成图。"""
    width, height = size
    if path is not None:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"无法读取图片：{path}")
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    # 合成图：渐变 + 随机噪声块，避免"纯黑图导致检测器提前返回"而低估耗时
    image = np.zeros((height, width, 3), dtype=np.uint8)
    gradient = np.linspace(40, 200, height, dtype=np.uint8)
    image[:, :, 0] = gradient[:, None]
    image[:, :, 1] = np.roll(gradient, 30)[:, None]
    image[:, :, 2] = np.roll(gradient, 60)[:, None]
    rng = np.random.default_rng(7)
    for _ in range(24):
        x, y = rng.integers(0, max(1, width - 40)), rng.integers(0, max(1, height - 40))
        cv2.rectangle(image, (int(x), int(y)), (int(x) + 32, int(y) + 32),
                      tuple(int(v) for v in rng.integers(60, 240, 3)), -1)
    return image


def _percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    def pick(q: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return ordered[index]

    return {
        "mean": statistics.fmean(ordered),
        "p50": pick(0.50),
        "p95": pick(0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def benchmark(
    *,
    config: AppConfig,
    provider_name: str,
    size: tuple[int, int],
    repeat: int,
    image_path: Path | None,
) -> dict | None:
    """在单一分辨率下跑 ``repeat`` 次，返回各阶段耗时统计（毫秒）。"""
    provider = create_landmark_provider(provider_name, config.perception.provider_options())
    if not provider.ready:
        missing = getattr(provider, "missing_models", lambda: ())()
        print(f"  感知实现 {provider_name} 未就绪，跳过。缺失：{list(missing)}")
        return None

    engine = build_engine(config.rules)
    pipeline = DetectionPipeline(
        provider=provider,
        extractor=MetricExtractor(),
        engine=engine,
        confirmer=BehaviorConfirmer(_confirmer_configs(engine)),
        calibrator=None,
        smoothing_window=config.smoothing.window,
    )

    image = _source_image(image_path, size)
    stages: dict[str, list[float]] = {}
    wall: list[float] = []
    hits = 0

    pipeline.warmup()
    try:
        for index in range(repeat):
            frame = Frame(index=index, timestamp=index / 30.0, image=image)
            started = time.perf_counter()
            analysis = pipeline.process(frame)
            wall.append((time.perf_counter() - started) * 1000.0)
            if analysis.observation.has_face:
                hits += 1
            for stage, elapsed in analysis.timings_ms.items():
                stages.setdefault(stage, []).append(float(elapsed))
    finally:
        pipeline.close()

    return {
        "size": size,
        "provider": provider_name,
        "repeat": repeat,
        "face_rate": hits / max(1, repeat),
        "wall": _percentiles(wall),
        "stages": {name: _percentiles(values) for name, values in stages.items()},
    }


def _print_result(result: dict) -> None:
    width, height = result["size"]
    print(f"\n分辨率 {width}x{height}  感知={result['provider']}  "
          f"重复={result['repeat']}  人脸检出率={result['face_rate']:.0%}")
    print(f"  {'阶段':<12}{'均值':>9}{'P50':>9}{'P95':>9}{'最小':>9}{'最大':>9}   (ms)")
    print("  " + "-" * 62)
    for stage, stats in result["stages"].items():
        print(f"  {stage:<12}{stats['mean']:>9.2f}{stats['p50']:>9.2f}"
              f"{stats['p95']:>9.2f}{stats['min']:>9.2f}{stats['max']:>9.2f}")
    wall = result["wall"]
    print(f"  {'合计(墙钟)':<10}{wall['mean']:>9.2f}{wall['p50']:>9.2f}"
          f"{wall['p95']:>9.2f}{wall['min']:>9.2f}{wall['max']:>9.2f}")
    theoretical = 1000.0 / wall["p50"] if wall["p50"] > 0 else float("inf")
    print(f"  → 单线程理论吞吐上限约 {theoretical:.1f} FPS（P50 口径）")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="流水线性能基准")
    parser.add_argument("--image", type=Path, help="测试图片（建议含真实人脸）")
    parser.add_argument("--provider", default="mediapipe",
                        choices=["mediapipe", "haar", "replay"])
    parser.add_argument("--sizes", nargs="+", default=["640x480", "960x540", "1280x720"],
                        help="分辨率列表，如 640x480 1280x720")
    parser.add_argument("--repeat", type=int, default=30, help="每个分辨率重复次数")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "default.json")
    args = parser.parse_args(argv)

    config = load_config(args.config, validate_paths=False)
    sizes: list[tuple[int, int]] = []
    for item in args.sizes:
        width, _, height = item.partition("x")
        sizes.append((int(width), int(height)))

    print("流水线性能基准")
    print("=" * 70)
    print(f"感知实现：{args.provider}"
          + (f"　测试图片：{args.image}" if args.image else "　测试图片：内置合成图"))
    print(f"每分辨率重复 {args.repeat} 次（首帧已通过 warmup 排除）")

    results = []
    for size in sizes:
        result = benchmark(
            config=config, provider_name=args.provider, size=size,
            repeat=args.repeat, image_path=args.image,
        )
        if result is None:
            return 1
        _print_result(result)
        results.append(result)

    if results:
        print("\n结论")
        print("-" * 70)
        best = results[0]
        p50 = sum(stats["p50"] for stats in best["stages"].values())
        print(f"  最小分辨率 {best['size'][0]}x{best['size'][1]} 的四阶段 P50 合计 "
              f"{p50:.2f} ms")
        print(f"  实时性目标（单帧 ≤ 80 ms）在最小分辨率下"
              f"{'满足' if p50 <= 80 else '不满足'}；分辨率越高感知耗时近似线性增长。")
        if best["face_rate"] == 0:
            print("  注意：测试图像中未检出人脸，感知耗时被低估；"
                  "请用 --image 指定含真实人脸的图片以获得有参考价值的数字。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
