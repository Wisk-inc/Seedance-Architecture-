"""Rectified flow / flow matching.

``[1.5/2.0]`` states the backbone is MMDiT + rectified flow, citing SD3. The
schedule details here -- logit-normal timestep sampling, resolution-dependent
timestep shifting, CFG configuration -- are ``[inferred]`` SD3 conventions and
are *not* disclosed for Seedance. They are the defaults that make a rectified-
flow video model train and sample sanely, not claims about ByteDance's setup.

Convention used everywhere in this package:

    x_t = (1 - t) * x_0 + t * eps,    t in [0, 1],  eps ~ N(0, I)
    v_target = eps - x_0 = dx_t / dt

so sampling integrates *downward* from t=1 (noise) to t=0 (data) with
``x <- x - dt * v``. Getting this sign wrong is the classic flow-matching bug;
:func:`self_consistency_check` exists to catch it in tests.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from ..config import FlowConfig

__all__ = ["RectifiedFlow", "apply_cfg", "rescale_noise_cfg"]


def _time_shift(mu: float, sigma: torch.Tensor, shift: float = 1.0) -> torch.Tensor:
    """SD3 timestep shift: pushes sampling effort toward high-noise steps."""
    return math.exp(mu) / (math.exp(mu) + (1 / sigma - 1) ** shift)


def rescale_noise_cfg(
    guided: torch.Tensor, cond: torch.Tensor, rescale: float = 0.7
) -> torch.Tensor:
    """Rescale the guided prediction to the conditional std (Lin et al. 2024).

    High guidance scales inflate the prediction's variance, which shows up as
    oversaturated, contrast-crushed video. Rescaling recovers the conditional
    statistics without giving up prompt adherence.
    """
    dims = list(range(1, guided.dim()))
    std_cond = cond.std(dim=dims, keepdim=True)
    std_guided = guided.std(dim=dims, keepdim=True).clamp(min=1e-8)
    return rescale * (guided * (std_cond / std_guided)) + (1 - rescale) * guided


def apply_cfg(
    cond: torch.Tensor,
    uncond: torch.Tensor,
    scale: float,
    *,
    rescale: float = 0.0,
    zero_init: bool = False,
) -> torch.Tensor:
    """Classifier-free guidance, optionally CFG-Zero* style.

    ``zero_init`` implements the CFG-Zero* observation that at the very first
    steps the unconditional branch is uninformative, so guiding against it
    injects noise; returning the conditional prediction alone is better.
    """
    if zero_init:
        return cond
    out = uncond + scale * (cond - uncond)
    if rescale > 0:
        out = rescale_noise_cfg(out, cond, rescale)
    return out


class RectifiedFlow:
    """Training targets, noise schedule and samplers for the flow backbone."""

    def __init__(self, cfg: Optional[FlowConfig] = None):
        self.cfg = cfg or FlowConfig()

    # ------------------------------------------------------------- schedule
    def mu_for_sequence(self, seq_len: int) -> float:
        """Resolution-dependent shift (SD3 dynamic shifting)."""
        cfg = self.cfg
        if not cfg.dynamic_shift:
            return math.log(cfg.shift)
        lo, hi = cfg.base_seq_len, cfg.max_seq_len
        slope = (cfg.max_shift - cfg.base_shift) / max(hi - lo, 1)
        intercept = cfg.base_shift - slope * lo
        return slope * seq_len + intercept

    def sigmas(
        self,
        num_steps: Optional[int] = None,
        *,
        seq_len: Optional[int] = None,
        device: torch.device | str = "cpu",
        strength: float = 1.0,
    ) -> torch.Tensor:
        """Descending sigma (== t) schedule, ending at 0.

        ``strength < 1`` starts partway down the trajectory, which is how the
        refiner and any img2img-style pass are driven.
        """
        cfg = self.cfg
        steps = num_steps or cfg.num_inference_steps
        sigmas = torch.linspace(strength, 1.0 / steps, steps, device=device)
        if seq_len is not None and cfg.dynamic_shift:
            sigmas = _time_shift(self.mu_for_sequence(seq_len), sigmas)
        elif cfg.shift != 1.0:
            s = cfg.shift
            sigmas = s * sigmas / (1 + (s - 1) * sigmas)
        return torch.cat([sigmas, sigmas.new_zeros(1)])

    def timesteps(self, sigmas: torch.Tensor) -> torch.Tensor:
        """Map sigmas to the model's timestep scale ``[0, num_train_timesteps]``."""
        return sigmas[:-1] * self.cfg.num_train_timesteps

    # -------------------------------------------------------------- training
    def sample_timesteps(
        self,
        batch: int,
        *,
        device: torch.device | str = "cpu",
        seq_len: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        cfg = self.cfg
        if cfg.timestep_sampling == "uniform":
            t = torch.rand(batch, device=device, generator=generator)
        else:
            normal = torch.randn(batch, device=device, generator=generator)
            t = torch.sigmoid(cfg.logit_mean + cfg.logit_std * normal)
        if seq_len is not None and cfg.dynamic_shift:
            t = _time_shift(self.mu_for_sequence(seq_len), t.clamp(1e-4, 1 - 1e-4))
        return t.clamp(1e-4, 1 - 1e-4)

    def add_noise(
        self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        t = self._broadcast(t, x0)
        return (1 - t) * x0 + t * noise

    def velocity_target(self, x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return noise - x0

    def loss(
        self,
        model_output: torch.Tensor,
        x0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
        *,
        weighting: str = "none",
    ) -> torch.Tensor:
        """Flow-matching MSE with optional per-timestep weighting."""
        target = self.velocity_target(x0, noise)
        per_sample = (model_output.float() - target.float()).pow(2).flatten(1).mean(dim=1)
        if weighting == "sigma_sqrt":
            w = 1.0 / t.clamp(min=1e-3).sqrt()
        elif weighting == "cosmap":
            # CosMap weighting (SD3): emphasises mid-trajectory timesteps.
            w = 2.0 / (math.pi * (1 - 2 * t + 2 * t.pow(2)).clamp(min=1e-6))
        else:
            w = torch.ones_like(t)
        return (per_sample * w.to(per_sample.dtype)).mean()

    def x0_from_velocity(
        self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Single-step estimate of the clean sample, used by distillation."""
        t = self._broadcast(t, x_t)
        return x_t - t * v

    def noise_from_velocity(
        self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        t = self._broadcast(t, x_t)
        return x_t + (1 - t) * v

    # -------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: Sequence[int],
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        num_steps: Optional[int] = None,
        seq_len: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        uncond_model: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        solver: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        strength: float = 1.0,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        """Integrate the flow from noise to data.

        ``model`` and ``uncond_model`` take ``(x_t, timestep)`` and return a
        velocity. Keeping them as callables (rather than an nn.Module) is what
        lets the same sampler drive the native DiT, a distilled student, or a
        wrapped third-party backbone.
        """
        cfg = self.cfg
        scale = cfg.guidance_scale if guidance_scale is None else guidance_scale
        solver = solver or cfg.solver
        sigmas = self.sigmas(num_steps, seq_len=seq_len, device=device, strength=strength)
        steps = len(sigmas) - 1

        if latents is None:
            x = torch.randn(tuple(shape), device=device, dtype=torch.float32, generator=generator)
        else:
            x = latents.float()

        for i in range(steps):
            t = sigmas[i]
            dt = sigmas[i] - sigmas[i + 1]
            ts = (t * cfg.num_train_timesteps).expand(x.shape[0])
            v = self._eval(model, uncond_model, x.to(dtype), ts, scale, step=i)
            if solver == "heun" and i < steps - 1:
                x_mid = x - dt * v
                ts_next = (sigmas[i + 1] * cfg.num_train_timesteps).expand(x.shape[0])
                v_next = self._eval(
                    model, uncond_model, x_mid.to(dtype), ts_next, scale, step=i + 1
                )
                v = 0.5 * (v + v_next)
            x = x - dt * v
            if callback is not None:
                callback(i + 1, steps, x)
        return x

    def _eval(
        self,
        model: Callable,
        uncond_model: Optional[Callable],
        x: torch.Tensor,
        ts: torch.Tensor,
        scale: float,
        step: int,
    ) -> torch.Tensor:
        v = model(x, ts).float()
        if scale == 1.0 or uncond_model is None:
            return v
        v_uncond = uncond_model(x, ts).float()
        return apply_cfg(
            v,
            v_uncond,
            scale,
            rescale=self.cfg.guidance_rescale,
            zero_init=step < self.cfg.cfg_zero_init_steps,
        )

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _broadcast(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        t = t.to(like.device, dtype=torch.float32)
        while t.dim() < like.dim():
            t = t.unsqueeze(-1)
        return t

    def self_consistency_check(self, dims: Tuple[int, ...] = (2, 4, 3, 8, 8)) -> float:
        """Verify that the analytic velocity integrates back to the data point.

        Returns the max absolute error, which should be ~0 up to float error.
        A regression here means a sign or scale slip in the schedule.
        """
        torch.manual_seed(0)
        x0 = torch.randn(dims)
        noise = torch.randn(dims)

        def model(x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
            return (noise - x0).expand_as(x)

        out = self.sample(model, dims, num_steps=8, guidance_scale=1.0, latents=noise)
        return float((out - x0).abs().max())
