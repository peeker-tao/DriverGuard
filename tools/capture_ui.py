"""界面冒烟测试 + 截图产出（离屏运行，无需真实显示器）。

双重用途
--------
1. **回归验证**：在 CI/无显示器环境里把主窗口完整跑一遍，确保
   "信号—槽—渲染"链路没有断裂；
2. **文档素材**：在回放脚本的不同阶段抓取界面截图，直接用作
   课程文档中的"系统界面原型"配图。

用法::

    python tools/capture_ui.py                          # 默认 5 张，间隔 4 秒
    python tools/capture_ui.py --scenario fatigue --shots 6 --interval 4
    python tools/capture_ui.py --out docs/images

说明：通过 ``QT_QPA_PLATFORM=offscreen`` 强制离屏渲染，
因此可在无图形界面的服务器上运行。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 必须在导入 PySide6 之前设置，否则 Qt 会尝试连接真实显示服务
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from driver_guard.app import create_qapplication, setup_logging  # noqa: E402
from driver_guard.config import load_config  # noqa: E402
from driver_guard.presentation.main_window import MainWindow  # noqa: E402
from driver_guard.presentation.viewmodels.monitor_viewmodel import MonitorViewModel  # noqa: E402


def build_config(scenario: str):
    config = load_config(PROJECT_ROOT / "config" / "default.json")
    source = replace(
        config.source,
        type="replay",
        width=1280,
        height=720,
        fps=30.0,
        synthetic_duration=0.0,
    )
    perception = replace(config.perception, provider="replay", scenario=scenario)
    alert = replace(
        config.alert,
        sound_enabled=False,   # 离屏环境通常没有音频设备
        snapshot_enabled=True,
        csv_enabled=True,
    )
    return config.with_overrides(source=source, perception=perception, alert=alert)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离屏运行界面并截图")
    parser.add_argument("--scenario", default="mixed", help="回放脚本名")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "images")
    parser.add_argument("--shots", type=int, default=5, help="截图张数")
    parser.add_argument("--interval", type=float, default=4.0, help="截图间隔（秒）")
    parser.add_argument("--warmup", type=float, default=2.0,
                        help="第一张截图前的等待时间（秒），用于等模型与标定就绪")
    parser.add_argument("--width", type=int, default=1560)
    parser.add_argument("--height", type=int, default=940)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    config = build_config(args.scenario)
    setup_logging(config, console_level="WARNING")

    app: QApplication = create_qapplication([])
    view_model = MonitorViewModel(config)
    window = MainWindow(view_model)
    window.resize(args.width, args.height)
    window.show()

    if not view_model.initialize():
        print("初始化失败，无法截图")
        return 1
    if not view_model.start():
        print("无法启动检测")
        return 1

    state = {"taken": 0}

    def capture() -> None:
        state["taken"] += 1
        index = state["taken"]
        path = args.out / f"ui_{args.scenario}_{index:02d}.png"
        pixmap = window.grab()
        ok = pixmap.save(str(path), "PNG")
        print(f"[{index}/{args.shots}] {'已保存' if ok else '保存失败'} {path}")
        if state["taken"] >= args.shots:
            view_model.shutdown()
            app.quit()

    timer = QTimer()
    timer.setInterval(int(args.interval * 1000))
    timer.timeout.connect(capture)
    QTimer.singleShot(int(args.warmup * 1000), timer.start)

    # 安全网：无论如何不超过 (warmup + shots*interval + 8) 秒
    guard_ms = int((args.warmup + args.shots * args.interval + 8) * 1000)
    QTimer.singleShot(guard_ms, lambda: (view_model.shutdown(), app.quit()))

    app.exec()
    print(f"\n完成，共 {state['taken']} 张，输出目录：{args.out}")
    return 0 if state["taken"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
