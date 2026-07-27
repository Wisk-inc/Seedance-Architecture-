"""DanceGRPO: Group Relative Policy Optimization for rectified flows.

``arXiv:2505.07818`` (Xue et al., ByteDance Seed & HKU) is the first framework
to adapt GRPO to visual generation, working across both diffusion and
rectified flows and across T2I/T2V/I2V. The three ideas implemented here:

1. **Denoising as an MDP.** Each sampler step is an action; the policy is the
   Gaussian induced by injecting noise at each step (an SDE sampler), which is
   what gives a tractable per-step log-probability.
2. **Group-relative advantages.** ``G`` samples are drawn per prompt and
   advantages are the group-normalised rewards -- no learned value function,
   which is what makes this affordable for video.
3. **Normalised multi-reward summation.** Advantages from up to five reward
   models are normalised *then* summed, rather than summing raw scores.

Reported effect on HunyuanVideo: +56% visual quality and +181% motion quality
on the VideoAlign reward.

Why an SDE sampler: a deterministic ODE sampler has a degenerate policy (zero
variance), so there is no log-probability to differentiate and no exploration.
:meth:`DanceGRPO.rollout` therefore uses the stochastic variant, and the noise
level is annealed by ``noise_scale``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..flow.rectified_flow import RectifiedFlow

__all__ = ["GRPOConfig", "Trajectory", "DanceGRPO"]

VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class GRPOConfig:
    group_size: int = 8  # G samples per prompt
    num_steps: int = 16  # sampler steps per rollout (short: RL is expensive)
    clip_range: float = 0.2
    kl_coef: float = 0.0  # optional KL to the reference policy
    noise_scale: float = 0.7  # SDE exploration noise, in [0, 1]
    advantage_clip: float = 5.0
    timestep_fraction: float = 0.6  # fraction of steps kept for the gradient
    guidance_scale: float = 1.0
    normalize_per_reward: bool = True
    seed: Optional[int] = None


@dataclass
class Trajectory:
    """One rollout: the states visited and the policy's log-probs."""

    latents: List[torch.Tensor] = field(default_factory=list)  # x_t at each step
    next_latents: List[torch.Tensor] = field(default_factory=list)
    timesteps: List[torch.Tensor] = field(default_factory=list)
    log_probs: List[torch.Tensor] = field(default_factory=list)
    sample: Optional[torch.Tensor] = None  # final x_0
    advantages: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.latents)


def _gaussian_log_prob(
    x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Diagonal Gaussian log-prob, summed over all non-batch dims."""
    var = (std**2).clamp(min=1e-12)
    lp = -0.5 * ((x - mean) ** 2 / var) - torch.log(std.clamp(min=1e-6)) - 0.5 * math.log(2 * math.pi)
    return lp.flatten(1).sum(dim=1)


class DanceGRPO:
    """GRPO trainer for a rectified-flow video policy."""

    def __init__(
        self,
        policy: nn.Module,
        reward_model,
        *,
        flow: Optional[RectifiedFlow] = None,
        cfg: Optional[GRPOConfig] = None,
        reference_policy: Optional[nn.Module] = None,
        decode_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """
        Args:
            policy: velocity model ``(x_t, timestep) -> v``; the thing trained.
            reward_model: exposes ``score(videos, prompts) -> [B]`` (see
                :mod:`seedance.reward`). Videos are pixel-space when
                ``decode_fn`` is given, latents otherwise.
            decode_fn: latents -> pixels. Rewards on latents are cheaper but
                a VLM reward model needs pixels.
        """
        self.policy = policy
        self.reward_model = reward_model
        self.flow = flow or RectifiedFlow()
        self.cfg = cfg or GRPOConfig()
        self.reference_policy = reference_policy
        self.decode_fn = decode_fn

    # ---------------------------------------------------------------- rollout
    @torch.no_grad()
    def rollout(
        self,
        shape: Sequence[int],
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
        policy: Optional[VelocityFn] = None,
    ) -> Trajectory:
        """Sample ``shape[0]`` trajectories with the stochastic sampler."""
        cfg = self.cfg
        flow = self.flow
        model = policy or self.policy
        sigmas = flow.sigmas(cfg.num_steps, device=device)
        x = torch.randn(tuple(shape), device=device, dtype=torch.float32, generator=generator)
        traj = Trajectory()

        for i in range(cfg.num_steps):
            t, t_next = sigmas[i], sigmas[i + 1]
            dt = t - t_next
            ts = (t * flow.cfg.num_train_timesteps).expand(x.shape[0])
            v = model(x.to(dtype), ts).float()
            mean = x - dt * v
            # SDE term: variance scaled by the step size and the remaining noise
            std = cfg.noise_scale * torch.sqrt(dt.clamp(min=0) * t_next.clamp(min=1e-6))
            std = std.expand_as(mean) if std.dim() else torch.full_like(mean, float(std))
            noise = torch.randn(mean.shape, device=device, generator=generator)
            x_next = mean + std * noise
            traj.latents.append(x)
            traj.next_latents.append(x_next)
            traj.timesteps.append(ts)
            traj.log_probs.append(_gaussian_log_prob(x_next, mean, std))
            x = x_next

        traj.sample = x
        return traj

    # --------------------------------------------------------------- rewards
    def compute_advantages(
        self,
        traj: Trajectory,
        prompts: Optional[Sequence[str]] = None,
        *,
        group_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Group-normalised advantages for the trajectory's final samples."""
        assert traj.sample is not None
        cfg = self.cfg
        g = group_size or cfg.group_size
        videos = self.decode_fn(traj.sample) if self.decode_fn else traj.sample

        if cfg.normalize_per_reward and hasattr(self.reward_model, "normalized_advantage"):
            adv = self.reward_model.normalized_advantage(videos, prompts, group_size=g)
        else:
            rewards = self.reward_model.score(videos, prompts).float()
            b = rewards.shape[0]
            if b % g:
                raise ValueError(f"batch {b} is not divisible by group size {g}")
            grouped = rewards.view(-1, g)
            mean = grouped.mean(dim=1, keepdim=True)
            std = grouped.std(dim=1, keepdim=True).clamp(min=1e-4)
            adv = ((grouped - mean) / std).reshape(-1)

        adv = adv.clamp(-cfg.advantage_clip, cfg.advantage_clip)
        traj.advantages = adv
        return adv

    # ------------------------------------------------------------------ loss
    def loss(
        self,
        traj: Trajectory,
        *,
        step_indices: Optional[Sequence[int]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Ratio-clipped surrogate over a subset of the trajectory's steps.

        Only a fraction of steps is used per update (``timestep_fraction``):
        gradients from adjacent denoising steps are highly correlated, so
        subsampling buys memory back at almost no cost in signal.
        """
        if traj.advantages is None:
            raise ValueError("call compute_advantages() before loss()")
        cfg = self.cfg
        flow = self.flow
        n = len(traj)
        if step_indices is None:
            keep = max(1, int(round(n * cfg.timestep_fraction)))
            step_indices = torch.randperm(n)[:keep].tolist()

        total = torch.zeros((), device=traj.advantages.device)
        kl_total = torch.zeros((), device=traj.advantages.device)
        clipped = 0.0
        for idx in step_indices:
            x = traj.latents[idx]
            x_next = traj.next_latents[idx]
            ts = traj.timesteps[idx]
            sigmas = flow.sigmas(cfg.num_steps, device=x.device)
            t, t_next = sigmas[idx], sigmas[idx + 1]
            dt = t - t_next

            v = self.policy(x.to(dtype), ts).float()
            mean = x - dt * v
            std = cfg.noise_scale * torch.sqrt(dt.clamp(min=0) * t_next.clamp(min=1e-6))
            std = torch.full_like(mean, float(std))
            log_prob = _gaussian_log_prob(x_next, mean, std)

            ratio = torch.exp((log_prob - traj.log_probs[idx]).clamp(-10, 10))
            adv = traj.advantages
            unclipped = ratio * adv
            clamped = ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range) * adv
            total = total - torch.minimum(unclipped, clamped).mean()
            clipped += float((ratio - 1).abs().gt(cfg.clip_range).float().mean())

            if cfg.kl_coef and self.reference_policy is not None:
                with torch.no_grad():
                    v_ref = self.reference_policy(x.to(dtype), ts).float()
                    mean_ref = x - dt * v_ref
                kl = ((mean - mean_ref) ** 2 / (2 * std**2)).flatten(1).sum(dim=1).mean()
                kl_total = kl_total + kl

        steps = max(1, len(step_indices))
        loss = total / steps + cfg.kl_coef * (kl_total / steps)
        logs = {
            "grpo_loss": float(loss.detach()),
            "advantage_mean": float(traj.advantages.mean()),
            "advantage_std": float(traj.advantages.std()),
            "clip_fraction": clipped / steps,
        }
        if cfg.kl_coef:
            logs["kl"] = float((kl_total / steps).detach())
        return loss, logs

    # ------------------------------------------------------------------ train
    def train_step(
        self,
        shape: Sequence[int],
        optimizer: torch.optim.Optimizer,
        *,
        prompts: Optional[Sequence[str]] = None,
        device: torch.device | str = "cuda",
        inner_epochs: int = 1,
        grad_clip: float = 1.0,
    ) -> Dict[str, float]:
        """One full GRPO iteration: rollout, score, then ``inner_epochs`` updates."""
        traj = self.rollout(shape, device=device)
        self.compute_advantages(traj, prompts)
        logs: Dict[str, float] = {}
        for _ in range(inner_epochs):
            loss, logs = self.loss(traj)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip)
            optimizer.step()
        return logs
