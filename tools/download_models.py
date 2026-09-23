"""下载 MediaPipe Tasks 所需的模型包（.task）。

MediaPipe 1.x 移除了旧的 ``mp.solutions`` API，改用 Tasks API，模型以独立
``.task`` 文件发布。本脚本负责一次性把这些模型拉到 ``assets/models`` 下，
使系统在无网络环境下也能运行。

用法::

    python tools/download_models.py            # 下载缺失的模型
    python tools/download_models.py --force    # 强制重新下载
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "assets" / "models"

_BASE = "https://storage.googleapis.com/mediapipe-models"


@dataclass(frozen=True)
class ModelSpec:
    """一个模型包的下载描述。"""

    filename: str
    url: str
    description: str
    approx_size_mb: float


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        filename="face_landmarker.task",
        url=f"{_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        description="人脸 478 关键点 + 头姿矩阵 + 表情系数（疲劳/分心检测核心）",
        approx_size_mb=3.7,
    ),
    ModelSpec(
        filename="hand_landmarker.task",
        url=f"{_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        description="手部 21 关键点（打电话/抽烟等手部危险行为）",
        approx_size_mb=7.8,
    ),
)


def _human(size: int) -> str:
    return f"{size / 1024 / 1024:.2f} MB"


def _download(spec: ModelSpec, target: Path) -> None:
    """带进度输出的流式下载。"""
    tmp = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(spec.url, headers={"User-Agent": "driver-guard/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - 固定可信域名
        total = int(response.headers.get("Content-Length") or 0)
        digest = hashlib.sha256()
        done = 0
        with tmp.open("wb") as handle:
            while chunk := response.read(1 << 16):
                handle.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    sys.stdout.write(f"\r  {spec.filename}: {pct:5.1f}%  ({_human(done)}/{_human(total)})")
                else:
                    sys.stdout.write(f"\r  {spec.filename}: {_human(done)}")
                sys.stdout.flush()
    sys.stdout.write("\n")
    tmp.replace(target)
    print(f"  ✓ 已保存 {target.name}  sha256={digest.hexdigest()[:16]}…")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载 MediaPipe .task 模型")
    parser.add_argument("--force", action="store_true", help="已存在时也重新下载")
    parser.add_argument("--dir", type=Path, default=MODEL_DIR, help="模型输出目录")
    args = parser.parse_args(argv)

    args.dir.mkdir(parents=True, exist_ok=True)
    print(f"模型目录: {args.dir}\n")

    failed: list[str] = []
    for spec in MODELS:
        target = args.dir / spec.filename
        if target.exists() and not args.force:
            print(f"• {spec.filename} 已存在（{_human(target.stat().st_size)}），跳过")
            continue
        print(f"• 下载 {spec.filename} — {spec.description}")
        print(f"  {spec.url}")
        try:
            _download(spec, target)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            failed.append(f"{spec.filename}: {exc}")
            print(f"  ✗ 失败: {exc}")
        print()

    if failed:
        print("以下模型下载失败：")
        for item in failed:
            print(f"  - {item}")
        print("\n提示：可手动下载后放入 assets/models/ 目录。")
        return 1

    print("全部模型就绪。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
