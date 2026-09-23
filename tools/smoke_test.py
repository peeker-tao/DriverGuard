"""端到端冒烟测试：无需摄像头与界面，跑通「取帧 → 检测 → 告警 → 落库」。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driver_guard.bootstrap import build_application  # noqa: E402
from driver_guard.config import load_config  # noqa: E402


def main() -> int:
    config = load_config("config/default.json")
    # 用「合成画面 + 回放感知」，无需任何硬件；关闭声音以免占用控制台
    config = config.with_overrides(
        source=type(config.source)(
            type="replay", fps=30.0, width=960, height=540, synthetic_duration=0.0
        ),
        perception=type(config.perception)(provider="replay"),
        alert=type(config.alert)(
            sound_enabled=False,
            snapshot_enabled=True,
            csv_enabled=True,
            log_console=True,
        ),
    )

    app = build_application(config, persist_events=True, max_frames=30 * 32)
    print("装配完成")
    print(f"  数据源: {app.source.name}")
    print(f"  感知  : {app.provider.name}")
    print(f"  规则  : {[r.behavior.value for r in app.pipeline.engine.rules]}")
    if app.warnings:
        for w in app.warnings:
            print(f"  [警告] {w}")

    print("\n开始采集（合成 30 秒脚本循环，共 960 帧；演示时不节流）…")
    app.detection_service.limit_frames(30 * 32)
    app.detection_service.set_target_fps(0.0)
    summary = app.detection_service.run()

    print("\n会话汇总")
    print(f"  处理帧数   : {summary.frames}")
    print(f"  时长       : {summary.duration:.2f}s   平均 FPS: {summary.avg_fps:.1f}")
    print(f"  触发帧数   : {summary.triggered_frames}")
    print(f"  风险指数   : {summary.risk_index}")
    print("  各阶段耗时 : " + ", ".join(f"{k}={v:.2f}ms" for k, v in summary.stage_avg_ms.items()))
    print("  事件计数   :")
    for behavior, count in summary.event_counts.items():
        print(f"    {behavior.label:<10} x{count}")

    print("\n统计服务（按大类）")
    for category, count in app.statistics.category_counts().items():
        print(f"    {category.label}: {count}")

    print("\n最近 5 条事件")
    for event in list(app.repository.all())[-5:]:
        print(f"    {event.wall_clock:%H:%M:%S} {event.summary}")

    app.write_session_summary(summary)
    app.shutdown()
    print("\n冒烟测试通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
