"""Physics-IQ style evaluation.

``arXiv:2501.09038`` (Motamed et al., INSAIT & Google DeepMind) is the reference
benchmark: 396 real videos across 66 scenarios, ~8 s each, filmed from three
perspectives. Its finding is the one to keep in mind before trying to close the
physics gap -- "physical understanding is severely limited, and unrelated to
visual realism" across Sora, Runway, Pika, Lumiere, SVD and VideoPoet, with
VideoPoet scoring 24.1%. Physics-IQ Verified (arXiv:2606.18943) refines 57.6%
of samples and 34.8% of prompts.

The official dataset is not redistributed here. What this module provides is
the *protocol*: given a real continuation and a generated one, compute the four
comparison metrics the benchmark defines, plus a harness that runs a generator
over a scenario directory. Point ``dataset_dir`` at your copy of the data and
the numbers are comparable; without it, use :func:`synthetic_scenarios` for a
smoke test of the harness itself (those numbers are not Physics-IQ scores and
must not be reported as such).

Metric definitions used here:

* **Spatial IoU** -- overlap of *where* action happens, ignoring when.
* **Spatiotemporal IoU** -- overlap of where *and when*, the strictest one.
* **Weighted spatial IoU** -- spatial IoU weighted by how much action there is,
  so a model that moves the right region by the right amount scores higher.
* **MSE** -- raw pixel error, reported for completeness; it rewards blur.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "action_mask",
    "spatial_iou",
    "spatiotemporal_iou",
    "weighted_spatial_iou",
    "frame_mse",
    "score_pair",
    "PhysicsIQScenario",
    "run_benchmark",
    "synthetic_scenarios",
]


def action_mask(video, threshold: float = 0.1, blur: int = 3):
    """Binary per-frame mask of where motion occurs.

    Motion is measured against the *first* frame rather than the previous one:
    Physics-IQ asks where the scene changed relative to the conditioning frame,
    and successive differencing would erase slow, sustained motion.
    """
    import torch
    import torch.nn.functional as F

    v = (video[0] if video.dim() == 5 else video).float()
    gray = v.mean(dim=0)  # [T, H, W]
    diff = (gray - gray[:1]).abs()
    if blur > 1:
        diff = F.avg_pool2d(diff.unsqueeze(1), blur, stride=1, padding=blur // 2).squeeze(1)
    scale = diff.amax().clamp(min=1e-6)
    return (diff / scale) > threshold


def spatial_iou(real, generated, threshold: float = 0.1) -> float:
    """IoU of the union-over-time action regions."""
    a = action_mask(real, threshold).any(dim=0)
    b = action_mask(generated, threshold).any(dim=0)
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return inter / union if union else 1.0


def spatiotemporal_iou(real, generated, threshold: float = 0.1) -> float:
    """IoU over the full space-time volume of action."""
    a = action_mask(real, threshold)
    b = action_mask(generated, threshold)
    t = min(a.shape[0], b.shape[0])
    a, b = a[:t], b[:t]
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return inter / union if union else 1.0


def weighted_spatial_iou(real, generated, threshold: float = 0.1) -> float:
    """Spatial IoU weighted by action magnitude."""
    import torch

    def weights(video):
        v = (video[0] if video.dim() == 5 else video).float()
        gray = v.mean(dim=0)
        diff = (gray - gray[:1]).abs().amax(dim=0)
        return diff / diff.amax().clamp(min=1e-6)

    wa, wb = weights(real), weights(generated)
    ma = action_mask(real, threshold).any(dim=0)
    mb = action_mask(generated, threshold).any(dim=0)
    inter = (torch.minimum(wa, wb) * (ma & mb)).sum().item()
    union = (torch.maximum(wa, wb) * (ma | mb)).sum().item()
    return inter / union if union else 1.0


def frame_mse(real, generated) -> float:
    """Mean squared error over the overlapping frames, in ``[0, 1]`` units."""
    a = (real[0] if real.dim() == 5 else real).float()
    b = (generated[0] if generated.dim() == 5 else generated).float()
    t = min(a.shape[1], b.shape[1])
    a, b = (a[:, :t] + 1) / 2, (b[:, :t] + 1) / 2
    if a.shape[-2:] != b.shape[-2:]:
        import torch.nn.functional as F

        b = F.interpolate(
            b.permute(1, 0, 2, 3), size=a.shape[-2:], mode="bilinear", align_corners=False
        ).permute(1, 0, 2, 3)
    return float(((a - b) ** 2).mean())


def score_pair(real, generated, threshold: float = 0.1) -> Dict[str, float]:
    """All four metrics for one (real, generated) continuation pair."""
    return {
        "spatial_iou": spatial_iou(real, generated, threshold),
        "spatiotemporal_iou": spatiotemporal_iou(real, generated, threshold),
        "weighted_spatial_iou": weighted_spatial_iou(real, generated, threshold),
        "mse": frame_mse(real, generated),
    }


@dataclass
class PhysicsIQScenario:
    """One benchmark item: a conditioning frame and the real continuation."""

    name: str
    prompt: str
    conditioning_frame: str  # image path
    real_continuation: str  # video path
    perspective: str = "center"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dir(cls, directory: str) -> List["PhysicsIQScenario"]:
        """Load scenarios from a directory of ``<name>.json`` descriptors."""
        out: List[PhysicsIQScenario] = []
        for entry in sorted(os.listdir(directory)):
            if not entry.endswith(".json"):
                continue
            with open(os.path.join(directory, entry), "r", encoding="utf-8") as handle:
                data = json.load(handle)
            base = os.path.dirname(os.path.join(directory, entry))
            out.append(
                cls(
                    name=data.get("name", entry[:-5]),
                    prompt=data["prompt"],
                    conditioning_frame=os.path.join(base, data["conditioning_frame"]),
                    real_continuation=os.path.join(base, data["real_continuation"]),
                    perspective=data.get("perspective", "center"),
                    metadata=data.get("metadata", {}),
                )
            )
        if not out:
            raise FileNotFoundError(f"no scenario descriptors (*.json) found in {directory}")
        return out


def run_benchmark(
    generate_fn: Callable[[str, Any], Any],
    scenarios: Sequence[PhysicsIQScenario],
    *,
    max_frames: int = 121,
    threshold: float = 0.1,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Run a generator over scenarios and aggregate the metrics.

    ``generate_fn(prompt, conditioning_image) -> [C, T, H, W]``.
    """
    from ..runtime.io import load_image, load_video

    rows: List[Dict[str, Any]] = []
    for i, scenario in enumerate(scenarios):
        if progress:
            progress(i + 1, len(scenarios), scenario.name)
        image = load_image(scenario.conditioning_frame)
        generated = generate_fn(scenario.prompt, image)
        real = load_video(scenario.real_continuation, max_frames=max_frames)
        metrics = score_pair(real, generated, threshold)
        rows.append({"scenario": scenario.name, "perspective": scenario.perspective, **metrics})

    if not rows:
        return {"scenarios": [], "aggregate": {}}
    keys = ("spatial_iou", "spatiotemporal_iou", "weighted_spatial_iou", "mse")
    aggregate = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    # The benchmark's headline number combines the IoU family; MSE is reported
    # separately because lower is better and it rewards blur.
    aggregate["physics_iq_proxy"] = (
        aggregate["spatial_iou"]
        + aggregate["spatiotemporal_iou"]
        + aggregate["weighted_spatial_iou"]
    ) / 3
    return {
        "scenarios": rows,
        "aggregate": aggregate,
        "note": (
            "Comparable to published Physics-IQ numbers only if `scenarios` came "
            "from the official dataset. VideoPoet scored 24.1% for reference."
        ),
    }


def synthetic_scenarios(directory: str, count: int = 3, fps: int = 16) -> List[PhysicsIQScenario]:
    """Generate a tiny synthetic scenario set to exercise the harness.

    Uses the built-in analytic physics proxy as "ground truth". These numbers
    are a plumbing check, not a benchmark result.
    """
    from ..runtime.io import save_frames, save_video
    from ..stages.physics import PhysicsProxyRenderer, PhysicsScene

    os.makedirs(directory, exist_ok=True)
    renderer = PhysicsProxyRenderer("analytic", height=128, width=224)
    out: List[PhysicsIQScenario] = []
    for i in range(count):
        scene = PhysicsScene.bouncing_ball(height=1.2 + 0.4 * i, fps=fps, duration_s=2.0)
        video = renderer.render(scene)
        name = f"synthetic_{i:02d}"
        video_path = os.path.join(directory, f"{name}.mp4")
        save_video(video, video_path, fps=fps)
        frame_dir = os.path.join(directory, name)
        frames = save_frames(video[:, :1], frame_dir, prefix="cond")
        descriptor = {
            "name": name,
            "prompt": "a ball falls and bounces on the ground",
            "conditioning_frame": os.path.relpath(frames[0], directory),
            "real_continuation": os.path.basename(video_path),
        }
        with open(os.path.join(directory, f"{name}.json"), "w", encoding="utf-8") as handle:
            json.dump(descriptor, handle, indent=2)
        out.append(
            PhysicsIQScenario(
                name=name,
                prompt=descriptor["prompt"],
                conditioning_frame=frames[0],
                real_continuation=video_path,
            )
        )
    return out
