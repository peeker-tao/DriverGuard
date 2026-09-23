"""命令行入口。

两种运行方式
------------
::

    # 图形界面（默认）
    python run.py
    python run.py --source camera --provider mediapipe

    # 无界面批量分析（用于生成测试数据、跑回归、服务器端分析）
    python run.py --no-gui --source replay --provider replay --max-frames 900
    python run.py --no-gui --source video --video assets/samples/xxx.mp4

常用参数
--------
``--source``     ``camera`` / ``video`` / ``synthetic`` / ``images`` / ``replay``
``--provider``   ``mediapipe`` / ``haar`` / ``replay``
``--scenario``   回放脚本名（``normal``/``fatigue``/``distraction``/``violation``/``absent``/``mixed``）
``--camera``     摄像头序号（默认 0）
``--max-frames`` 最多处理帧数（``0`` 为不限）
``--list-scenarios`` 列出内置演示脚本
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

# 允许直接从源码目录运行（无需 pip install -e .）
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from driver_guard.config import AppConfig, load_config  # noqa: E402
from driver_guard.app import run_gui, setup_logging  # noqa: E402

logger = logging.getLogger("driver_guard.main")

__all__ = ["build_parser", "main"]


# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="driver-guard",
        description="驾驶员危险行为检测系统 —— 实时检测疲劳/分心/违规驾驶行为",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  实时检测        python run.py\n"
            "  无摄像头演示    python run.py --source replay --provider replay\n"
            "  离线分析录像    python run.py --no-gui --source video --video a.mp4\n"
        ),
    )
    parser.add_argument("--config", type=Path, default=Path("config/default.json"),
                        help="配置文件路径（默认 config/default.json）")
    parser.add_argument("--source", choices=["camera", "video", "synthetic", "images", "replay"],
                        help="数据源类型")
    parser.add_argument("--provider", choices=["mediapipe", "haar", "replay"],
                        help="感知实现；合成/回放演示请用 replay")
    parser.add_argument("--scenario", help="回放脚本名（配合 --source replay）")
    parser.add_argument("--camera", type=int, help="摄像头序号")
    parser.add_argument("--video", type=Path, help="视频文件路径（配合 --source video）")
    parser.add_argument("--images", type=Path, help="图片目录（配合 --source images）")
    parser.add_argument("--width", type=int, help="采集宽度")
    parser.add_argument("--height", type=int, help="采集高度")
    parser.add_argument("--fps", type=float, help="目标帧率")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="最多处理帧数（0 为不限；无界面模式建议设定）")
    parser.add_argument("--no-gui", action="store_true", help="不启动界面，直接在终端输出结果")
    parser.add_argument("--persist", action="store_true",
                        help="无界面模式下也写入 CSV/快照（默认不写，避免污染演示数据）")
    parser.add_argument("--log-level", help="日志级别：DEBUG/INFO/WARNING/ERROR")
    parser.add_argument("--list-scenarios", action="store_true", help="列出内置回放脚本后退出")
    parser.add_argument("--self-check", action="store_true",
                        help="检查模型与级联文件是否就绪后退出")
    parser.add_argument("--replay-port", action="store_true",
                        help=argparse.SUPPRESS)  # 兼容旧脚本，等价于 --source replay
    return parser


# ===========================================================================
def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """把命令行参数叠加到配置上（命令行优先）。"""
    source = replace(
        config.source,
        type=args.source or config.source.type,
        camera_index=args.camera if args.camera is not None else config.source.camera_index,
        width=args.width or config.source.width,
        height=args.height or config.source.height,
        fps=args.fps or config.source.fps,
    )
    if args.video is not None:
        source = replace(source, video_path=str(args.video))
    if args.images is not None:
        source = replace(source, image_dir=str(args.images))

    perception = config.perception
    if args.provider:
        perception = replace(perception, provider=args.provider)

    # 合成画面里没有真实人脸：若用户没显式指定感知实现，自动切到回放，
    # 否则界面会一直显示"未检测到驾驶员"，是最常见的演示踩坑点。
    if source.type in {"replay", "synthetic"} and args.provider is None:
        if perception.provider != "replay":
            logger.warning(
                "数据源为 %s 但感知实现是 %s；合成画面不含真实人脸，"
                "已自动切换为 replay 以演示完整告警链路",
                source.type, perception.provider,
            )
            perception = replace(perception, provider="replay")

    if args.replay_port:
        source = replace(source, type="replay")
        perception = replace(perception, provider="replay")

    if args.scenario:
        perception = replace(perception, scenario=args.scenario)

    return config.with_overrides(source=source, perception=perception)


# ===========================================================================
def _print_scenarios() -> int:
    from driver_guard.infrastructure.landmarks import SCENARIOS

    print("内置回放脚本：\n")
    for name, segments in sorted(SCENARIOS.items()):
        span = max(segment.end for segment in segments)
        print(f"  {name:<12} 周期 {span:5.1f}s   共 {len(segments)} 段")
        for segment in segments:
            print(f"      {segment.start:5.1f}s ~ {segment.end:5.1f}s  {segment.label}")
        print()
    print("用法： python run.py --source replay --provider replay --scenario fatigue")
    return 0


def _self_check(config: AppConfig, *, no_gui: bool = False) -> int:
    """检查依赖、模型与级联是否就绪。

    返回值的语义：``0`` = **当前配置所需的一切都已就绪**；``1`` = 存在
    **会阻碍当前配置运行**的问题。

    刻意区分"阻塞问题"与"可选缺失"：只有当前 ``provider`` 实际需要的模型/
    级联缺失才算阻塞。否则用回放模式在无模型环境演示时，自检会错误地报失败
    （而实际系统完全可用）。
    """
    problems: list[str] = []     # 会阻碍当前配置运行
    optional: list[str] = []     # 可选通道缺失，仅提示
    notes: list[str] = []

    provider = (config.perception.provider or "mediapipe").strip().lower()
    model_dir = config.resolved_model_dir()

    face_path = model_dir / config.perception.face_model
    if face_path.exists():
        notes.append(f"✓ 模型 {face_path}")
    elif provider == "mediapipe":
        problems.append(f"✗ 模型 {face_path}（provider=mediapipe 时必需）")
    else:
        optional.append(f"· 模型 {face_path}（provider={provider}，当前不需要）")

    hand_path = model_dir / config.perception.hand_model
    if hand_path.exists():
        notes.append(f"✓ 模型 {hand_path}")
    elif provider == "mediapipe" and config.perception.enable_hands:
        optional.append(f"· 模型 {hand_path}（手部通道会自动关闭：打电话/抽烟将不可用）")
    else:
        optional.append(f"· 模型 {hand_path}（当前未启用手部通道）")

    cascade_dir = config.resolved_cascade_dir()
    for name in ("haarcascade_frontalface_default.xml", "haarcascade_eye.xml"):
        path = cascade_dir / name
        if path.exists():
            notes.append(f"✓ 级联 {path}")
        elif provider == "haar":
            problems.append(f"✗ 级联 {path}（provider=haar 时必需）")
        else:
            optional.append(f"· 级联 {path}（仅 haar 降级模式需要）")

    try:
        import cv2
        notes.append(f"✓ OpenCV {cv2.__version__}")
    except ImportError as exc:  # pragma: no cover
        problems.append(f"✗ OpenCV 不可用：{exc}")

    if provider in {"mediapipe", "haar"}:
        try:
            import mediapipe as mp
            notes.append(f"✓ MediaPipe {mp.__version__}")
        except ImportError as exc:  # pragma: no cover
            problems.append(f"✗ MediaPipe 不可用：{exc}")

    try:
        import PySide6
        notes.append(f"✓ PySide6 {PySide6.__version__}")
    except ImportError as exc:  # pragma: no cover
        if no_gui:
            optional.append(f"· PySide6 不可用（已指定 --no-gui，不影响）：{exc}")
        else:
            problems.append(f"✗ PySide6 不可用（可加 --no-gui 仅用命令行）：{exc}")

    for name in ("events", "snapshots"):
        directory = config.output.resolve(name)
        directory.mkdir(parents=True, exist_ok=True)
        notes.append(f"✓ 输出目录 {directory}")

    print("环境自检")
    print("-" * 60)
    for line in notes:
        print("  " + line)
    if optional:
        print("\n可选（缺失不影响当前配置运行）：")
        for line in optional:
            print("  " + line)
        print("  按需下载： python tools/download_models.py / python tools/fetch_cascades.py")
    if problems:
        print("\n需要处理：")
        for line in problems:
            print("  " + line)
        return 1
    print("\n全部就绪。")
    return 0


def _run_headless(config: AppConfig, args: argparse.Namespace) -> int:
    """无界面模式：跑完会话并打印汇总。"""
    from driver_guard.bootstrap import build_application

    app = build_application(
        config,
        persist_events=args.persist,
        max_frames=args.max_frames,
    )
    print(f"数据源：{app.source.name}")
    print(f"感知　：{app.provider.name}")
    print(f"规则　：{', '.join(r.behavior.value for r in app.pipeline.engine.rules if r.enabled)}")
    for warning in app.warnings:
        print(f"[提示] {warning}")

    try:
        summary = app.detection_service.run()
    except KeyboardInterrupt:
        app.detection_service.stop()
        print("\n已被用户中断")
        app.shutdown()
        return 130

    print()
    print("会话汇总")
    print("-" * 52)
    print(f"  处理帧数    : {summary.frames}")
    print(f"  时长        : {summary.duration:.2f} s")
    print(f"  平均帧率    : {summary.avg_fps:.2f} FPS")
    print(f"  触发帧数    : {summary.triggered_frames}")
    print(f"  风险指数    : {summary.risk_index} / 100")
    if summary.stage_avg_ms:
        stages = "  ".join(f"{k}={v:.2f}ms" for k, v in summary.stage_avg_ms.items())
        print(f"  阶段耗时    : {stages}")
    print("  事件统计    :")
    if summary.event_counts:
        for behavior, count in sorted(summary.event_counts.items(), key=lambda kv: -kv[1]):
            print(f"      {behavior.label:<8}[{behavior.category.label}] × {count}")
    else:
        print("      未产生告警")

    app.write_session_summary(summary)
    print(f"\n事件明细：{config.output.resolve(config.output.events_csv)}")
    if app.snapshot_sink is not None:
        print(f"现场快照：{app.snapshot_sink.archiver.directory}")
    app.shutdown()
    return 0


# ===========================================================================
def main(argv: list[str] | None = None) -> int:
    """程序入口，返回进程退出码。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_scenarios:
        return _print_scenarios()

    try:
        config = load_config(args.config)
    except (ValueError, OSError) as exc:
        print(f"配置加载失败：{exc}", file=sys.stderr)
        return 2

    log_path = setup_logging(config, console_level=args.log_level)
    if log_path is not None:
        logger.debug("日志文件：%s", log_path)

    if args.self_check:
        return _self_check(config, no_gui=args.no_gui)

    config = apply_overrides(config, args)

    try:
        if args.no_gui:
            return _run_headless(config, args)
        return run_gui(config)
    except KeyboardInterrupt:
        print("\n已退出")
        return 130
    except ImportError as exc:
        print(
            f"缺少依赖：{exc}\n"
            "请先安装： pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
