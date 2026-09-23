"""下载 OpenCV Haar 级联文件（降级感知通道使用）。

背景
----
``opencv-contrib-python`` 5.0 起，``cv2.data.haarcascades`` 目录不再随包附带
级联 XML（实测该目录下只有 ``__init__.py``）。因此使用 Haar 降级通道前，
需要先把级联文件放到 ``assets/cascades/``。

用法::

    python tools/fetch_cascades.py

级联文件来自 OpenCV 官方仓库 ``data/haarcascades/``，采用 Apache-2.0 许可。
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = PROJECT_ROOT / "assets" / "cascades"

_BASE = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades"


@dataclass(frozen=True)
class CascadeSpec:
    filename: str
    description: str


CASCADES: tuple[CascadeSpec, ...] = (
    CascadeSpec("haarcascade_frontalface_default.xml", "正面人脸检测"),
    CascadeSpec("haarcascade_eye.xml", "睁眼检测（找不到眼睛即判为闭眼）"),
)


def _download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "driver-guard/1.0"})
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as handle:  # noqa: S310
        while chunk := response.read(1 << 16):
            handle.write(chunk)
    if tmp.stat().st_size < 1024:
        tmp.unlink(missing_ok=True)
        raise OSError("下载内容过小，疑似 404 页面")
    tmp.replace(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载 Haar 级联文件")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    args.dir.mkdir(parents=True, exist_ok=True)
    print(f"目标目录: {args.dir}\n")

    failures: list[str] = []
    for spec in CASCADES:
        target = args.dir / spec.filename
        if target.exists() and not args.force:
            print(f"• {spec.filename} 已存在，跳过")
            continue
        print(f"• 下载 {spec.filename} — {spec.description}")
        try:
            _download(f"{_BASE}/{spec.filename}", target)
            print(f"  ✓ {target.stat().st_size / 1024:.1f} KB")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            failures.append(f"{spec.filename}: {exc}")
            print(f"  ✗ 失败: {exc}")

    if failures:
        print("\n以下文件下载失败：")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("\n级联文件就绪，可使用 --provider haar 启动降级模式。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
