"""Stage ablation: the one benchmark this repo can honestly measure itself.

Cross-vendor comparisons need API access to closed models and a shared eval
set. What *is* measurable on your own machine is whether the staged pipeline
actually improves the checkpoint you are running it on — the same prompt, the
same seed, the same weights, with stages switched on one at a time.

That is what this produces: a chart of measured numbers where the bars differ
only by pipeline configuration. If a stage does not earn its runtime on your
content, this is where you will see it.

    python -m seedance.cli ablate --model Wan-AI/Wan2.1-T2V-1.3B \
        --prompt "a glass marble rolls off a table and bounces twice"
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .leaderboard import Benchmark, Entry

logger = logging.getLogger(__name__)

__all__ = ["AblationStep", "DEFAULT_STEPS", "run_ablation", "METRIC_WEIGHTS"]


@dataclass
class AblationStep:
    """One pipeline configuration to measure."""

    key: str
    name: str
    description: str = ""
    apply: Optional[Callable[[Any], None]] = None  # mutates StageSettings
    highlight: bool = False


def _baseline(settings) -> None:
    settings.enhance_prompt = False
    settings.polish.color_stabilize = False
    settings.polish.interpolate = 1
    settings.polish.upscale_factor = 1
    settings.polish.face_restore = False
    settings.use_depth = settings.use_pose = settings.use_flow = False
    settings.physics = None


def _prompt_only(settings) -> None:
    _baseline(settings)
    settings.enhance_prompt = True


def _prompt_identity(settings) -> None:
    _prompt_only(settings)
    settings.measure_identity_drift = True


def _prompt_polish(settings) -> None:
    _prompt_only(settings)
    settings.polish.color_stabilize = True


def _full(settings) -> None:
    _prompt_polish(settings)
    settings.polish.interpolate = 2
    settings.measure_identity_drift = True


DEFAULT_STEPS: List[AblationStep] = [
    AblationStep("baseline", "Wan baseline", "raw checkpoint, every stage off", _baseline),
    AblationStep("s1", "+ Stage 1 prompt", "dense-caption rewriting", _prompt_only),
    AblationStep("s1s2", "+ Stage 2 identity", "subject locking / references", _prompt_identity),
    AblationStep("s1s4", "+ Stage 4 colour", "colour stabilisation", _prompt_polish),
    AblationStep("full", "Seedance staged (full)", "all stages", _full, highlight=True),
]

# How the composite score is built. Weights favour the failure modes the stages
# are supposed to fix, so the composite cannot be gamed by a stage that only
# sharpens frames.
METRIC_WEIGHTS: Dict[str, float] = {
    "temporal_smoothness": 0.3,
    "flicker": 0.2,
    "color_stability": 0.25,
    "structural_persistence": 0.15,
    "sharpness": 0.1,
}


def _composite(qa: Dict[str, float]) -> float:
    total = sum(METRIC_WEIGHTS.get(k, 0.0) * float(v) for k, v in qa.items() if k in METRIC_WEIGHTS)
    norm = sum(METRIC_WEIGHTS[k] for k in qa if k in METRIC_WEIGHTS) or 1.0
    return round(100.0 * total / norm, 1)


def run_ablation(
    model: str = "Wan-AI/Wan2.1-T2V-1.3B",
    *,
    prompt: str = "a glass marble rolls off a wooden table and bounces twice on tile",
    steps: Optional[Sequence[AblationStep]] = None,
    size: Optional[str] = None,
    num_frames: int = 49,
    sampling_steps: Optional[int] = None,
    seed: int = 1234,
    output_dir: str = "bench_out",
    backend: Any = None,
    device_id: int = 0,
    progress: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    """Run each configuration and score the result. Returns benchmarks + raw QA.

    Every configuration uses the same seed and the same backend instance, so the
    only variable is the pipeline. That is what makes the deltas meaningful.
    """
    from ..pipelines.staged import SeedancePipeline, StageSettings
    from ..runtime.io import save_video
    from ..stages.polish import temporal_qa

    steps = list(steps or DEFAULT_STEPS)
    emit = progress or (lambda stage, msg: logger.info("[%s] %s", stage, msg))
    os.makedirs(output_dir, exist_ok=True)

    pipe = SeedancePipeline(model, backend=backend, device_id=device_id, plan_memory=False)
    shared_backend = pipe.backend  # load once, reuse across configurations

    rows: List[Dict[str, Any]] = []
    for step in steps:
        settings = StageSettings.preset("fast")
        settings.enhancer_offline_only = True
        if step.apply:
            step.apply(settings)
        pipe.settings = settings

        emit("ablate", f"{step.name}: {step.description}")
        result = pipe.generate(
            prompt,
            size=size,
            num_frames=num_frames,
            steps=sampling_steps,
            seed=seed,
            output=os.path.join(output_dir, f"ablation_{step.key}.mp4"),
        )
        qa = result.report.get("polish", {}).get("qa") or temporal_qa(result.video)
        qa = {k: float(v) for k, v in qa.items() if isinstance(v, (int, float))}
        rows.append(
            {
                "key": step.key,
                "name": step.name,
                "highlight": step.highlight,
                "composite": _composite(qa),
                "qa": qa,
                "seconds": round(sum(result.timings.values()), 1),
                "path": result.path,
            }
        )
        emit("ablate", f"{step.name}: composite {rows[-1]['composite']}")

    benchmarks = [
        Benchmark(
            title="Seedance staged pipeline",
            subtitle=f"composite quality score · {model} · same seed, same weights",
            unit="",
            higher_is_better=True,
            footnote=(
                "Composite = weighted temporal smoothness, flicker, colour stability, "
                "structural persistence and sharpness. Measured locally; not comparable "
                "to VBench or Arena scores."
            ),
            entries=[
                Entry(
                    name=row["name"],
                    value=row["composite"],
                    logo="seedance" if row["highlight"] else "wan",
                    source="measured",
                    highlight=row["highlight"],
                    note=row["key"],
                )
                for row in rows
            ],
        )
    ]

    for metric in ("temporal_smoothness", "color_stability", "structural_persistence"):
        if not all(metric in row["qa"] for row in rows):
            continue
        benchmarks.append(
            Benchmark(
                title=metric.replace("_", " ").title(),
                subtitle=f"{model} · higher is better",
                higher_is_better=True,
                entries=[
                    Entry(
                        name=row["name"],
                        value=round(100 * row["qa"][metric], 1),
                        logo="seedance" if row["highlight"] else "wan",
                        source="measured",
                        highlight=row["highlight"],
                    )
                    for row in rows
                ],
            )
        )

    return {"model": model, "prompt": prompt, "rows": rows, "benchmarks": benchmarks}
