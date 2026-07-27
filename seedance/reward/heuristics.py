"""Weightless reward proxies.

RewardDance-style generative reward models need a multi-billion-parameter VLM.
That is the right thing in production and useless for getting an RL loop
running, so this module provides cheap, deterministic, download-free proxies
for the dimensions the Seedance reports name: motion quality, structural
coherence and visual fidelity.

These are *proxies*. They correlate with the failure modes they are named after
and they are wired into the same interface as the real reward models, so an
experiment can be built and debugged here and then re-scored with
:class:`~seedance.reward.reward_dance.RewardDance` without changing the loop.
Do not report numbers from these as quality results.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

__all__ = [
    "temporal_smoothness",
    "flicker_score",
    "sharpness",
    "color_drift",
    "structural_persistence",
    "saturation_sanity",
    "HeuristicRewardModel",
]


def _to_bcthw(video: torch.Tensor) -> torch.Tensor:
    """Accept ``[B,C,T,H,W]`` or ``[C,T,H,W]`` in ``[-1,1]``; return 5D float."""
    if video.dim() == 4:
        video = video.unsqueeze(0)
    if video.dim() != 5:
        raise ValueError(f"expected 4D or 5D video tensor, got {tuple(video.shape)}")
    return video.float()


def _gray(video: torch.Tensor) -> torch.Tensor:
    w = torch.tensor([0.299, 0.587, 0.114], device=video.device).view(1, 3, 1, 1, 1)
    return (video * w).sum(dim=1, keepdim=True)


def temporal_smoothness(video: torch.Tensor) -> torch.Tensor:
    """Higher is smoother. Penalises large frame-to-frame changes.

    Uses the *second* difference, not the first: a pan produces a large first
    difference but a small second one, whereas judder and popping show up in
    acceleration. Rewarding low first difference would just reward static video.
    """
    v = _gray(_to_bcthw(video))
    if v.shape[2] < 3:
        return torch.ones(v.shape[0], device=v.device)
    accel = v[:, :, 2:] - 2 * v[:, :, 1:-1] + v[:, :, :-2]
    magnitude = accel.abs().flatten(1).mean(dim=1)
    return torch.exp(-magnitude * 4.0)


def flicker_score(video: torch.Tensor) -> torch.Tensor:
    """Higher is better. Penalises global brightness oscillation."""
    v = _gray(_to_bcthw(video))
    means = v.flatten(3).mean(dim=-1).squeeze(1)  # [B, T]
    if means.shape[1] < 3:
        return torch.ones(means.shape[0], device=means.device)
    d = means[:, 1:] - means[:, :-1]
    sign_flips = ((d[:, 1:] * d[:, :-1]) < 0).float().mean(dim=1)
    amplitude = d.abs().mean(dim=1)
    return torch.exp(-(sign_flips * amplitude * 20.0))


def sharpness(video: torch.Tensor) -> torch.Tensor:
    """Higher is sharper. Laplacian energy, averaged over frames."""
    v = _gray(_to_bcthw(video))
    b, _, t, h, w = v.shape
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=v.device
    ).view(1, 1, 3, 3)
    flat = v.permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w)
    lap = F.conv2d(flat, kernel, padding=1)
    energy = lap.pow(2).flatten(1).mean(dim=1).view(b, t).mean(dim=1)
    return (energy * 50.0).clamp(max=1.0)


def color_drift(video: torch.Tensor) -> torch.Tensor:
    """Higher is more stable. Penalises per-channel mean drift over time.

    This is the failure mode that shows up when long video is generated in
    chunks: each chunk is individually fine and the sequence slowly turns green.
    """
    v = _to_bcthw(video)
    means = v.flatten(3).mean(dim=-1)  # [B, C, T]
    if means.shape[2] < 2:
        return torch.ones(means.shape[0], device=means.device)
    drift = (means[:, :, -1] - means[:, :, 0]).abs().mean(dim=1)
    return torch.exp(-drift * 3.0)


def structural_persistence(video: torch.Tensor, patch: int = 32) -> torch.Tensor:
    """Higher means the scene keeps its structure. Cheap identity-drift proxy.

    Correlation of a downsampled centre crop between the first frame and every
    later frame. It cannot tell you a face changed identity -- it can tell you
    the subject dissolved, which is the deformation failure this stands in for.
    """
    v = _gray(_to_bcthw(video))
    b, _, t, h, w = v.shape
    if t < 2:
        return torch.ones(b, device=v.device)
    ch, cw = h // 2, w // 2
    half = min(h, w) // 4
    crop = v[:, :, :, max(0, ch - half) : ch + half, max(0, cw - half) : cw + half]
    small = F.adaptive_avg_pool3d(crop, (t, patch, patch)).flatten(2)  # [B,1,T*?]
    small = small.view(b, t, -1)
    ref = small[:, :1]
    a = ref - ref.mean(dim=-1, keepdim=True)
    c = small - small.mean(dim=-1, keepdim=True)
    corr = (a * c).sum(-1) / (a.norm(dim=-1) * c.norm(dim=-1)).clamp(min=1e-6)
    return corr[:, 1:].mean(dim=1).clamp(0, 1)


def saturation_sanity(video: torch.Tensor) -> torch.Tensor:
    """Higher is better. Penalises clipped highlights/shadows and neon casts."""
    v = _to_bcthw(video)
    clipped = ((v.abs() > 0.98).float().flatten(1).mean(dim=1))
    sat = (v.amax(dim=1) - v.amin(dim=1)).flatten(1).mean(dim=1)
    return (torch.exp(-clipped * 6.0) * torch.exp(-(sat - 0.5).clamp(min=0) * 2.0)).clamp(0, 1)


class HeuristicRewardModel:
    """Bundles the proxies into the reward-model interface used by GRPO."""

    DIMENSIONS = {
        "motion_quality": lambda v: 0.6 * temporal_smoothness(v) + 0.4 * flicker_score(v),
        "structural_coherence": structural_persistence,
        "visual_fidelity": lambda v: 0.6 * sharpness(v) + 0.4 * saturation_sanity(v),
        "color_stability": color_drift,
    }

    def __init__(self, dimensions: Optional[Sequence[str]] = None, weights: Optional[Dict[str, float]] = None):
        self.dimensions = list(dimensions or self.DIMENSIONS.keys())
        unknown = set(self.dimensions) - set(self.DIMENSIONS)
        if unknown:
            raise ValueError(f"unknown reward dimensions: {sorted(unknown)}")
        self.weights = weights or {d: 1.0 for d in self.dimensions}
        self.name = "heuristic"

    def score(
        self, videos: torch.Tensor, prompts: Optional[Sequence[str]] = None
    ) -> torch.Tensor:
        """Weighted scalar reward per sample, in roughly ``[0, 1]``."""
        per_dim = self.score_dimensions(videos, prompts)
        total = None
        norm = sum(abs(self.weights.get(d, 1.0)) for d in self.dimensions) or 1.0
        for dim, value in per_dim.items():
            contribution = value * self.weights.get(dim, 1.0)
            total = contribution if total is None else total + contribution
        assert total is not None
        return total / norm

    def score_dimensions(
        self, videos: torch.Tensor, prompts: Optional[Sequence[str]] = None
    ) -> Dict[str, torch.Tensor]:
        video = _to_bcthw(videos)
        return {d: self.DIMENSIONS[d](video) for d in self.dimensions}

    def report(self, videos: torch.Tensor) -> List[Dict[str, float]]:
        dims = self.score_dimensions(videos)
        total = self.score(videos)
        out = []
        for i in range(total.shape[0]):
            row = {d: round(float(v[i]), 4) for d, v in dims.items()}
            row["total"] = round(float(total[i]), 4)
            out.append(row)
        return out
