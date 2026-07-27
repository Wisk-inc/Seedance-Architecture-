"""Unified task formulation and multimodal reference packing.

``[1.0]``: "noisy inputs are concatenated with cleaned or zero-padded frames
along the channel dimension, with binary masks indicating which frames are
conditioning instructions, unifying T2I/T2V/I2V; task proportions are
controlled by adjusting the conditional inputs during training."

That single mechanism covers every task in this file. The reference bundle and
``@mention`` resolution below are ``[product-level]``: Seedance 2.0 exposes
them through its API, but no paper describes the token-injection mechanism, so
this is our own scheme built on the same channel-concat contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from ..story import Reference, ReferenceBundle, Shot, Storyboard, parse_mentions

__all__ = [
    "TASKS",
    "ConditionTensors",
    "build_condition",
    # re-exported from seedance.story, which stays torch-free
    "Reference",
    "ReferenceBundle",
    "parse_mentions",
    "Shot",
    "Storyboard",
]

TASKS = (
    "t2i",  # single frame, no conditioning frames
    "t2v",  # no conditioning frames
    "i2v",  # first frame clean
    "flf2v",  # first and last frames clean
    "v2v",  # all frames clean (structure/edit conditioning), noise on top
    "extend",  # leading frames clean, tail generated
    "ref2v",  # references only, no in-place frames
    "inpaint",  # spatial mask per frame
)


@dataclass
class ConditionTensors:
    """Channel-concatenated conditioning for the DiT."""

    condition: torch.Tensor  # [B, C, T, H, W] clean latents, zeros where absent
    mask: torch.Tensor  # [B, 1, T, H, W] 1 where the latent is an instruction
    task: str = "t2v"

    def as_input(self, noisy: torch.Tensor) -> torch.Tensor:
        """Concatenate ``[noisy, condition, mask]`` along the channel axis."""
        if noisy.shape[0] != self.condition.shape[0]:
            raise ValueError("batch mismatch between noisy latents and condition")
        if noisy.shape[2:] != self.condition.shape[2:]:
            raise ValueError(
                f"spatial mismatch: noisy {tuple(noisy.shape)} vs "
                f"condition {tuple(self.condition.shape)}"
            )
        return torch.cat(
            [noisy, self.condition.to(noisy.dtype), self.mask.to(noisy.dtype)], dim=1
        )

    def conditioned_frames(self) -> List[int]:
        """Indices of the latent frames that carry an instruction."""
        # [B, 1, T, H, W] -> per-frame max over space, for the first sample
        per_frame = self.mask[0, 0].flatten(1).amax(dim=-1)
        return [i for i, v in enumerate(per_frame.tolist()) if v > 0]


def build_condition(
    shape: Tuple[int, int, int, int, int],
    task: str = "t2v",
    *,
    clean_latents: Optional[torch.Tensor] = None,
    frame_indices: Optional[Sequence[int]] = None,
    spatial_mask: Optional[torch.Tensor] = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> ConditionTensors:
    """Build the ``(condition, mask)`` pair for one task.

    Args:
        shape: latent shape ``(B, C, T, H, W)``.
        task: one of :data:`TASKS`.
        clean_latents: VAE latents of the conditioning frames, ``[B, C, n, H, W]``
            where ``n == len(frame_indices)``, or ``[B, C, T, H, W]`` for
            ``v2v``/``inpaint``.
        frame_indices: which latent frames the clean latents occupy. Defaults
            follow the task: ``i2v`` -> ``[0]``, ``flf2v`` -> ``[0, T-1]``,
            ``extend`` -> the leading ``n`` frames.
        spatial_mask: ``[B, 1, T, H, W]`` for ``inpaint``; 1 marks *known*
            (conditioning) regions.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    b, c, t, h, w = shape
    device = torch.device(device)
    condition = torch.zeros((b, c, t, h, w), device=device, dtype=dtype)
    mask = torch.zeros((b, 1, t, h, w), device=device, dtype=dtype)

    if task in ("t2i", "t2v", "ref2v"):
        return ConditionTensors(condition, mask, task)

    if clean_latents is None:
        raise ValueError(f"task {task!r} requires clean_latents")
    clean_latents = clean_latents.to(device=device, dtype=dtype)

    if task == "inpaint":
        if spatial_mask is None:
            raise ValueError("task 'inpaint' requires spatial_mask")
        spatial_mask = spatial_mask.to(device=device, dtype=dtype)
        condition = clean_latents * spatial_mask
        return ConditionTensors(condition, spatial_mask.expand(b, 1, t, h, w).clone(), task)

    if task == "v2v":
        if clean_latents.shape[2] != t:
            raise ValueError("task 'v2v' expects clean latents for every frame")
        condition = clean_latents
        mask.fill_(1.0)
        return ConditionTensors(condition, mask, task)

    n = clean_latents.shape[2]
    if frame_indices is None:
        if task == "i2v":
            frame_indices = [0]
        elif task == "flf2v":
            frame_indices = [0, t - 1]
        elif task == "extend":
            frame_indices = list(range(n))
        else:  # pragma: no cover - guarded by TASKS check above
            raise ValueError(f"frame_indices required for task {task!r}")
    frame_indices = list(frame_indices)
    if len(frame_indices) != n:
        raise ValueError(
            f"clean_latents has {n} frames but {len(frame_indices)} indices were given"
        )
    for slot, idx in enumerate(frame_indices):
        if not 0 <= idx < t:
            raise ValueError(f"frame index {idx} out of range for T={t}")
        condition[:, :, idx] = clean_latents[:, :, slot]
        mask[:, :, idx] = 1.0
    return ConditionTensors(condition, mask, task)
