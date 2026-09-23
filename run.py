#!/usr/bin/env python
"""驾驶员危险行为检测系统 —— 启动脚本。

无需安装包，直接运行（自动把 ``src`` 加入模块搜索路径）::

    python run.py                              # 图形界面（默认摄像头）
    python run.py --source replay              # 无摄像头演示（内置行为脚本）
    python run.py --no-gui --source replay --max-frames 900   # 无界面批量
    python run.py --self-check                 # 环境自检
    python run.py --list-scenarios             # 查看内置演示脚本

完整参数见 ``python run.py --help``。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
for path in (SRC, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from driver_guard.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
