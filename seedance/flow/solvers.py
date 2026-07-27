"""Solver adapters.

The native sampler in :mod:`seedance.flow.rectified_flow` is Euler/Heun, which
is correct but needs many steps. When the Wan repo is importable (it is, in
this tree) its UniPC and DPM++ flow-matching schedulers are reused instead --
same rectified-flow parameterisation, far better step efficiency. This keeps
one sampling code path for both the native DiT and any Wan backbone.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import torch

__all__ = ["available_solvers", "make_scheduler", "solve"]

_NATIVE = ("euler", "heun")
_WAN = ("unipc", "dpm++")


def available_solvers() -> Tuple[str, ...]:
    out = list(_NATIVE)
    try:  # pragma: no cover - depends on the host tree
        import wan.utils.fm_solvers_unipc  # noqa: F401
        import wan.utils.fm_solvers  # noqa: F401

        out.extend(_WAN)
    except Exception:
        pass
    return tuple(out)


def make_scheduler(
    name: str,
    num_steps: int,
    *,
    shift: float = 5.0,
    num_train_timesteps: int = 1000,
    device: torch.device | str = "cpu",
):
    """Return ``(scheduler, timesteps)`` for a Wan-family solver.

    Raises ``ValueError`` for the native solvers -- they do not need a
    scheduler object, :func:`solve` handles them directly.
    """
    if name not in _WAN:
        raise ValueError(f"{name!r} is not a scheduler-based solver; use solve()")
    if name == "unipc":
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=num_train_timesteps, shift=1, use_dynamic_shifting=False
        )
        scheduler.set_timesteps(num_steps, device=device, shift=shift)
        return scheduler, scheduler.timesteps

    from wan.utils.fm_solvers import (
        FlowDPMSolverMultistepScheduler,
        get_sampling_sigmas,
        retrieve_timesteps,
    )

    scheduler = FlowDPMSolverMultistepScheduler(
        num_train_timesteps=num_train_timesteps, shift=1, use_dynamic_shifting=False
    )
    sigmas = get_sampling_sigmas(num_steps, shift)
    timesteps, _ = retrieve_timesteps(scheduler, device=device, sigmas=sigmas)
    return scheduler, timesteps


@torch.no_grad()
def solve(
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    latents: torch.Tensor,
    *,
    solver: str = "unipc",
    num_steps: int = 50,
    shift: float = 5.0,
    num_train_timesteps: int = 1000,
    uncond_model: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    guidance_scale: float = 5.0,
    generator: Optional[torch.Generator] = None,
    callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    flow_cfg=None,
) -> torch.Tensor:
    """Sample with the requested solver, falling back to Euler if unavailable."""
    if solver in _WAN and solver in available_solvers():
        scheduler, timesteps = make_scheduler(
            solver,
            num_steps,
            shift=shift,
            num_train_timesteps=num_train_timesteps,
            device=latents.device,
        )
        x = latents
        total = len(timesteps)
        for i, t in enumerate(timesteps):
            ts = t.expand(x.shape[0]) if torch.is_tensor(t) else torch.tensor([t] * x.shape[0])
            v = model(x, ts.to(x.device)).float()
            if guidance_scale != 1.0 and uncond_model is not None:
                v_u = uncond_model(x, ts.to(x.device)).float()
                v = v_u + guidance_scale * (v - v_u)
            x = scheduler.step(v, t, x, return_dict=False, generator=generator)[0]
            if callback is not None:
                callback(i + 1, total, x)
        return x

    from .rectified_flow import RectifiedFlow

    flow = RectifiedFlow(flow_cfg)
    return flow.sample(
        model,
        latents.shape,
        device=latents.device,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        uncond_model=uncond_model,
        solver=solver if solver in _NATIVE else "euler",
        generator=generator,
        latents=latents,
        callback=callback,
    )
