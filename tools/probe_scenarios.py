"""离线诊断：让每条内置脚本走一遍流水线，检查是否触发了预期规则。

用途
----
* 调参时的"体检工具"：一眼看出某条规则在当前阈值下是否可达；
* 答辩前自检：确认演示脚本确实能产生告警，避免现场空转；
* 回归基线：输出可复现，便于改阈值前后对比。

用法::

    python tools/probe_scenarios.py                 # 检查全部脚本
    python tools/probe_scenarios.py mixed fatigue   # 只检查指定脚本
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driver_guard.application.pipeline import DetectionPipeline  # noqa: E402
from driver_guard.config import default_config  # noqa: E402
from driver_guard.domain.calibration import NeutralPoseCalibrator  # noqa: E402
from driver_guard.domain.confirmation import BehaviorConfirmer  # noqa: E402
from driver_guard.domain.metrics import MetricExtractor  # noqa: E402
from driver_guard.domain.models import BehaviorType, Frame  # noqa: E402
from driver_guard.domain.rules import build_engine  # noqa: E402
from driver_guard.infrastructure.landmarks import ReplayLandmarkProvider, SCENARIOS  # noqa: E402
from driver_guard.infrastructure.sources import SyntheticFrameSource  # noqa: E402
from driver_guard.bootstrap import _confirmer_configs  # noqa: E402

FPS = 30.0
#: 每个脚本预期的触发行为（用于自动判定通过/失败）
EXPECTED: dict[str, set[str]] = {
    "normal": set(),
    "fatigue": {"eyes_closed", "yawning", "head_down"},
    "distraction": {"gaze_off_road", "head_down"},
    "violation": {"phone_call", "smoking"},
    "absent": {"driver_absent"},
    "mixed": {"eyes_closed", "gaze_off_road", "phone_call", "yawning"},
}


def run_scenario(name: str) -> tuple[Counter, Counter, dict[str, float]]:
    config = default_config()
    engine = build_engine(config.rules)
    provider = ReplayLandmarkProvider(name)
    pipeline = DetectionPipeline(
        provider=provider,
        extractor=MetricExtractor(),
        engine=engine,
        confirmer=BehaviorConfirmer(_confirmer_configs(engine)),
        calibrator=NeutralPoseCalibrator(
            required_samples=config.calibration.required_samples,
            warmup_frames=config.calibration.warmup_frames,
        ),
        smoothing_window=config.smoothing.window,
    )
    source = SyntheticFrameSource(width=320, height=180, fps=FPS)

    raw_counter: Counter[BehaviorType] = Counter()
    confirmed_counter: Counter[BehaviorType] = Counter()
    last_metrics: dict[str, float] = {}

    with source:
        while (frame := source.read()) is not None:
            if frame.timestamp > provider.duration:
                break
            analysis = pipeline.process(frame)
            for outcome in analysis.triggered:
                raw_counter[outcome.behavior] += 1
            for event in analysis.events:
                confirmed_counter[event.behavior] += 1
            last_metrics = analysis.metrics.as_dict()

    pipeline.close()
    return raw_counter, confirmed_counter, last_metrics


def main(argv: list[str]) -> int:
    names = argv or sorted(SCENARIOS)
    exit_code = 0

    print(f"{'脚本':<12}{'结论':<6}{'瞬时(帧)':<34}{'已确认(次)':<26}说明")
    print("-" * 112)
    for name in names:
        raw, confirmed, _ = run_scenario(name)
        expected = EXPECTED.get(name, set())
        got = {b.value for b in confirmed}
        missing = expected - got
        unexpected = got - expected

        if not expected and got:
            verdict, exit_code = "误报", 1
        elif missing:
            verdict, exit_code = "缺", 1
        elif unexpected:
            verdict, exit_code = "多", 1
        else:
            verdict = "OK"

        raw_text = ", ".join(f"{b.label}:{c}" for b, c in raw.most_common()) or "—"
        conf_text = ", ".join(f"{b.label}:{c}" for b, c in confirmed.most_common()) or "—"
        note = ""
        if missing:
            note = f"缺少 {', '.join(sorted(missing))}"
        if unexpected:
            note += ("；" if note else "") + f"多出 {', '.join(sorted(unexpected))}"
        if not note:
            note = "符合预期"
        print(f"{name:<12}{verdict:<6}{raw_text:<34}{conf_text:<26}{note}")

    print()
    print("结论：" + ("全部脚本行为符合预期" if exit_code == 0 else "存在脚本与预期不符，见上表"))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
