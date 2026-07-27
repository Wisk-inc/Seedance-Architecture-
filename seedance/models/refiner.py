"""Cascaded high-resolution refiner.

``[1.0]``: "The base model generates 480p first; a learned diffusion refiner
(initialized from the base, conditioned on upsampled low-res video
concatenated with noise along the channel dimension) upscales to 720p/1080p."

The refiner is architecturally the same DiT, so this module is thin: it wraps a
:class:`~seedance.models.dit.SeedanceDiT`, upsamples the low-res latent to the
target grid, and runs a partial-noise flow trajectory. Starting at
``noise_strength < 1`` is what keeps the base composition intact while adding
high-frequency detail -- a full-noise restart would re-generate the scene.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DiTConfig, RefinerConfig
from .conditioning import ConditionTensors
from .dit import SeedanceDiT

__all__ = ["CascadedRefiner", "upsample_latents"]


def upsample_latents(
    latents: torch.Tensor, target_hw: Tuple[int, int], mode: str = "trilinear"
) -> torch.Tensor:
    """Resize ``[B, C, T, H, W]`` latents spatially, leaving time untouched."""
    t = latents.shape[2]
    if mode == "trilinear":
        return F.interpolate(
            latents, size=(t, target_hw[0], target_hw[1]), mode="trilinear", align_corners=False
        )
    b, c, _, _, _ = latents.shape
    flat = latents.transpose(1, 2).reshape(b * t, c, latents.shape[3], latents.shape[4])
    out = F.interpolate(flat, size=target_hw, mode=mode, align_corners=False)
    return out.view(b, t, c, *target_hw).transpose(1, 2)


class CascadedRefiner(nn.Module):
    """Latent-space super-resolution stage sharing the DiT architecture."""

    def __init__(
        self,
        dit_cfg: DiTConfig,
        cfg: Optional[RefinerConfig] = None,
        dit: Optional[SeedanceDiT] = None,
    ):
        super().__init__()
        self.cfg = cfg or RefinerConfig()
        self.dit = dit if dit is not None else SeedanceDiT(dit_cfg)

    @classmethod
    def from_base(
        cls, base: SeedanceDiT, cfg: Optional[RefinerConfig] = None, share_weights: bool = False
    ) -> "CascadedRefiner":
        """Initialise from a trained base model, as the report describes."""
        cfg = cfg or RefinerConfig()
        if share_weights:
            return cls(base.cfg, cfg, dit=base)
        refiner = SeedanceDiT(base.cfg)
        refiner.load_state_dict(base.state_dict())
        return cls(base.cfg, cfg, dit=refiner)

    @torch.no_grad()
    def refine(
        self,
        latents: torch.Tensor,
        *,
        text: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        target_latent_hw: Optional[Tuple[int, int]] = None,
        num_steps: Optional[int] = None,
        noise_strength: Optional[float] = None,
        guidance_scale: float = 1.0,
        negative_text: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> torch.Tensor:
        """Upscale ``latents`` and run a partial rectified-flow trajectory."""
        cfg = self.cfg
        steps = num_steps or cfg.num_steps
        strength = cfg.noise_strength if noise_strength is None else noise_strength
        if target_latent_hw is None:
            scale = cfg.target_resolution[0] / cfg.base_resolution[0]
            target_latent_hw = (
                int(round(latents.shape[3] * scale)),
                int(round(latents.shape[4] * scale)),
            )
        upsampled = upsample_latents(latents, target_latent_hw)

        condition = ConditionTensors(
            condition=upsampled,
            mask=torch.ones(
                (upsampled.shape[0], 1, *upsampled.shape[2:]),
                device=upsampled.device,
                dtype=upsampled.dtype,
            ),
            task="v2v",
        )

        noise = torch.randn(
            upsampled.shape, device=upsampled.device, dtype=torch.float32, generator=generator
        )
        # rectified flow: x_t = (1 - t) * x_0 + t * noise
        x = (1 - strength) * upsampled.float() + strength * noise
        ts = torch.linspace(strength, 0.0, steps + 1, device=upsampled.device)
        for i in range(steps):
            t = ts[i].expand(x.shape[0])
            dt = ts[i] - ts[i + 1]
            v = self.dit(
                x.to(upsampled.dtype),
                t * 1000.0,
                text,
                text_mask=text_mask,
                condition=condition,
            ).float()
            if guidance_scale != 1.0 and negative_text is not None:
                v_uncond = self.dit(
                    x.to(upsampled.dtype),
                    t * 1000.0,
                    negative_text,
                    condition=condition,
                ).float()
                v = v_uncond + guidance_scale * (v - v_uncond)
            x = x - dt * v
            if progress is not None:
                progress(i + 1, steps)
        return x

    def training_batch(
        self,
        high_res_latents: torch.Tensor,
        *,
        downscale: int = 2,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, ConditionTensors]:
        """Build a (target, condition) pair for refiner training.

        The low-res condition is produced by degrading the HR latent, matching
        what the base model would have emitted -- close enough for training and
        far cheaper than running the base model per sample.
        """
        h, w = high_res_latents.shape[3], high_res_latents.shape[4]
        low = upsample_latents(high_res_latents, (max(1, h // downscale), max(1, w // downscale)))
        back = upsample_latents(low, (h, w))
        cond = ConditionTensors(
            condition=back,
            mask=torch.ones(
                (back.shape[0], 1, *back.shape[2:]), device=back.device, dtype=back.dtype
            ),
            task="v2v",
        )
        return high_res_latents, cond
