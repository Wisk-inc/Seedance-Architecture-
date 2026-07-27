"""Multi-stage inference distillation.

``[1.0]``: "Multi-stage distillation: trajectory-segmented consistency
distillation (TSCD), score distillation, and an adversarial distillation
strategy extended to a multi-step setting incorporating human-preference data
for supervision -- reducing the number of function evaluations (NFE) while
balancing quality and speed." Together with the Thin VAE decoder and kernel
work this is what produces the reported >10x end-to-end speedup (5 s of 1080p
in 41.4 s on an L20).

This module is training-side. It implements the three objectives with their
real formulations and a stage runner that sequences them; it does not ship
distilled weights. Each class is usable against any velocity model with the
signature ``(x_t, timestep) -> velocity``, including a wrapped Wan backbone.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rectified_flow import RectifiedFlow

__all__ = [
    "TSCDDistiller",
    "ScoreDistiller",
    "AdversarialDistiller",
    "LatentDiscriminator",
    "DistillationPlan",
    "run_plan",
]

VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _ema_update(target: nn.Module, source: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for p_t, p_s in zip(target.parameters(), source.parameters()):
            p_t.mul_(decay).add_(p_s.detach(), alpha=1 - decay)
        for b_t, b_s in zip(target.buffers(), source.buffers()):
            b_t.copy_(b_s)


class TSCDDistiller:
    """Trajectory-segmented consistency distillation.

    Plain consistency distillation forces one map from every ``t`` to ``t=0``,
    which is a hard target early in training. TSCD splits ``[0, 1]`` into ``k``
    segments and only requires consistency *within* a segment: the student at
    ``t`` must agree with its EMA copy at ``t'``, where ``t'`` is one teacher
    ODE step closer to the segment's lower boundary. Shrinking ``k`` over
    training (``k = 8 -> 4 -> 2 -> 1``) anneals toward full consistency, and
    each ``k`` is a usable few-step sampler in its own right.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: VelocityFn,
        flow: Optional[RectifiedFlow] = None,
        *,
        segments: int = 8,
        ema_decay: float = 0.95,
        teacher_steps: int = 1,
        loss_type: str = "huber",
    ):
        self.student = student
        self.teacher = teacher
        self.flow = flow or RectifiedFlow()
        self.segments = segments
        self.ema_decay = ema_decay
        self.teacher_steps = teacher_steps
        self.loss_type = loss_type
        self.target = copy.deepcopy(student).eval().requires_grad_(False)

    def set_segments(self, segments: int) -> None:
        """Anneal the segment count; the paper's curriculum halves it."""
        self.segments = max(1, segments)

    def _segment_bounds(self, t: torch.Tensor) -> torch.Tensor:
        """Lower boundary of the segment each ``t`` falls into."""
        idx = torch.floor(t * self.segments).clamp(0, self.segments - 1)
        return idx / self.segments

    def _distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l2":
            return (a - b).pow(2).flatten(1).mean(dim=1)
        # Pseudo-Huber, as in LCM/TSCD: tolerant of the outliers that dominate
        # early consistency training.
        c = 0.001 * (a[0].numel() ** 0.5)
        return torch.sqrt((a - b).pow(2).flatten(1).mean(dim=1) + c**2) - c

    def step(
        self,
        x0: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """One distillation loss on a batch of clean latents."""
        flow = self.flow
        b = x0.shape[0]
        noise = torch.randn(x0.shape, device=x0.device, dtype=torch.float32, generator=generator)
        t = torch.rand(b, device=x0.device, generator=generator).clamp(1e-3, 1.0)
        lower = self._segment_bounds(t)
        x_t = flow.add_noise(x0, noise, t)

        # teacher ODE step(s) toward the segment boundary
        with torch.no_grad():
            x_prev = x_t.clone()
            t_cur = t.clone()
            for _ in range(self.teacher_steps):
                t_next = torch.maximum(t_cur - (t - lower) / self.teacher_steps, lower)
                v = self.teacher(x_prev, t_cur * flow.cfg.num_train_timesteps).float()
                dt = (t_cur - t_next).view(-1, *([1] * (x0.dim() - 1)))
                x_prev = x_prev - dt * v
                t_cur = t_next
            v_target = self.target(x_prev, t_cur * flow.cfg.num_train_timesteps).float()
            target_x0 = flow.x0_from_velocity(x_prev, v_target, t_cur)

        v_student = self.student(x_t, t * flow.cfg.num_train_timesteps).float()
        student_x0 = flow.x0_from_velocity(x_t, v_student, t)
        loss = self._distance(student_x0, target_x0).mean()
        return loss, {"tscd": float(loss.detach()), "segments": float(self.segments)}

    def after_step(self) -> None:
        _ema_update(self.target, self.student, self.ema_decay)


class ScoreDistiller:
    """Distribution-matching score distillation for a few-step generator.

    The student produces ``x0_hat`` in one (or few) steps. That sample is
    re-noised to a random ``t``, and the gradient is the difference between the
    *teacher's* score at that point and a learned *fake* score fitted to the
    student's own distribution. The difference is what pushes the student's
    output distribution onto the teacher's instead of onto a single mode --
    which is why plain score distillation collapses and this does not.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: VelocityFn,
        fake_score: nn.Module,
        flow: Optional[RectifiedFlow] = None,
        *,
        t_range: Tuple[float, float] = (0.02, 0.98),
    ):
        self.student = student
        self.teacher = teacher
        self.fake_score = fake_score
        self.flow = flow or RectifiedFlow()
        self.t_range = t_range

    def generate(self, noise: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """Run the student as a few-step generator (differentiable)."""
        flow = self.flow
        x = noise
        ts = torch.linspace(1.0, 0.0, steps + 1, device=noise.device)
        for i in range(steps):
            t = ts[i].expand(x.shape[0])
            v = self.student(x, t * flow.cfg.num_train_timesteps).float()
            dt = (ts[i] - ts[i + 1]).view(1, *([1] * (x.dim() - 1)))
            x = x - dt * v
        return x

    def student_loss(
        self, noise: torch.Tensor, *, steps: int = 1
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        flow = self.flow
        x0_hat = self.generate(noise, steps=steps)
        b = x0_hat.shape[0]
        lo, hi = self.t_range
        t = (torch.rand(b, device=x0_hat.device) * (hi - lo) + lo)
        eps = torch.randn_like(x0_hat)
        x_t = flow.add_noise(x0_hat, eps, t)
        ts = t * flow.cfg.num_train_timesteps
        with torch.no_grad():
            v_real = self.teacher(x_t.detach(), ts).float()
            v_fake = self.fake_score(x_t.detach(), ts).float()
            # DMD gradient, normalised by the teacher residual magnitude so the
            # step size does not swing with timestep.
            grad = (v_fake - v_real)
            grad = grad / grad.abs().flatten(1).mean(dim=1).view(-1, *([1] * (grad.dim() - 1))).clamp(min=1e-6)
        # surrogate whose gradient wrt x0_hat equals `grad`
        loss = (x0_hat * grad.detach()).flatten(1).mean(dim=1).mean()
        return loss, {"score_distill": float(loss.detach())}

    def fake_score_loss(
        self, noise: torch.Tensor, *, steps: int = 1
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Denoising loss that keeps the fake score tracking the student."""
        flow = self.flow
        with torch.no_grad():
            x0_hat = self.generate(noise, steps=steps)
        t = flow.sample_timesteps(x0_hat.shape[0], device=x0_hat.device)
        eps = torch.randn_like(x0_hat)
        x_t = flow.add_noise(x0_hat, eps, t)
        v = self.fake_score(x_t, t * flow.cfg.num_train_timesteps)
        loss = flow.loss(v, x0_hat, eps, t)
        return loss, {"fake_score": float(loss.detach())}


class LatentDiscriminator(nn.Module):
    """Small 3D CNN critic over latents, with an optional preference head.

    The preference head is the "human-preference data for supervision" part:
    it is trained to regress a reward-model score, and its output is added to
    the adversarial signal so the critic prefers samples humans prefer rather
    than merely realistic ones.
    """

    def __init__(self, channels: int, base: int = 64, layers: int = 3, preference_head: bool = True):
        super().__init__()
        blocks: List[nn.Module] = []
        ch = channels
        for i in range(layers):
            out = min(base * (2**i), base * 8)
            blocks += [
                nn.Conv3d(ch, out, (3, 4, 4), (1 if i == 0 else 2, 2, 2), (1, 1, 1)),
                nn.GroupNorm(1, out),
                nn.LeakyReLU(0.2, True),
            ]
            ch = out
        self.trunk = nn.Sequential(*blocks)
        self.real_head = nn.Conv3d(ch, 1, 3, 1, 1)
        self.pref_head = nn.Conv3d(ch, 1, 3, 1, 1) if preference_head else None

    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.trunk(latents)
        real = self.real_head(h).flatten(1).mean(dim=1)
        pref = self.pref_head(h).flatten(1).mean(dim=1) if self.pref_head is not None else None
        return real, pref


class AdversarialDistiller:
    """Multi-step adversarial distillation with preference supervision."""

    def __init__(
        self,
        student: nn.Module,
        discriminator: LatentDiscriminator,
        flow: Optional[RectifiedFlow] = None,
        *,
        steps: int = 4,
        preference_weight: float = 0.5,
    ):
        self.student = student
        self.disc = discriminator
        self.flow = flow or RectifiedFlow()
        self.steps = steps
        self.preference_weight = preference_weight

    def generate(self, noise: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        flow = self.flow
        steps = steps or self.steps
        x = noise
        ts = torch.linspace(1.0, 0.0, steps + 1, device=noise.device)
        for i in range(steps):
            t = ts[i].expand(x.shape[0])
            v = self.student(x, t * flow.cfg.num_train_timesteps).float()
            dt = (ts[i] - ts[i + 1]).view(1, *([1] * (x.dim() - 1)))
            x = x - dt * v
        return x

    def generator_loss(
        self, noise: torch.Tensor, *, target_reward: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        fake = self.generate(noise)
        logits, pref = self.disc(fake)
        loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        logs = {"adv_g": float(loss.detach())}
        if pref is not None and self.preference_weight:
            # push samples toward high predicted preference
            pref_loss = -pref.mean()
            loss = loss + self.preference_weight * pref_loss
            logs["pref_g"] = float(pref_loss.detach())
        return loss, logs

    def discriminator_loss(
        self,
        real: torch.Tensor,
        noise: torch.Tensor,
        *,
        real_reward: Optional[torch.Tensor] = None,
        fake_reward: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        with torch.no_grad():
            fake = self.generate(noise)
        r_logits, r_pref = self.disc(real)
        f_logits, f_pref = self.disc(fake)
        loss = 0.5 * (
            F.binary_cross_entropy_with_logits(r_logits, torch.ones_like(r_logits))
            + F.binary_cross_entropy_with_logits(f_logits, torch.zeros_like(f_logits))
        )
        logs = {"adv_d": float(loss.detach())}
        if r_pref is not None and real_reward is not None:
            reg = F.mse_loss(r_pref, real_reward.to(r_pref))
            if fake_reward is not None and f_pref is not None:
                reg = reg + F.mse_loss(f_pref, fake_reward.to(f_pref))
            loss = loss + reg
            logs["pref_d"] = float(reg.detach())
        return loss, logs


@dataclass
class DistillationPlan:
    """A sequence of distillation stages, in the order the report describes."""

    stages: List[Dict[str, object]] = field(
        default_factory=lambda: [
            {"name": "tscd-8", "kind": "tscd", "segments": 8, "iters": 2000},
            {"name": "tscd-4", "kind": "tscd", "segments": 4, "iters": 2000},
            {"name": "tscd-2", "kind": "tscd", "segments": 2, "iters": 2000},
            {"name": "score", "kind": "score", "steps": 4, "iters": 2000},
            {"name": "adversarial", "kind": "adversarial", "steps": 4, "iters": 1000},
        ]
    )

    def describe(self) -> List[str]:
        return [f"{s['name']} ({s['kind']}, {s.get('iters')} iters)" for s in self.stages]


def run_plan(
    plan: DistillationPlan,
    *,
    tscd: Optional[TSCDDistiller] = None,
    score: Optional[ScoreDistiller] = None,
    adversarial: Optional[AdversarialDistiller] = None,
    batches: Callable[[], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    disc_optimizer: Optional[torch.optim.Optimizer] = None,
    fake_optimizer: Optional[torch.optim.Optimizer] = None,
    max_iters_override: Optional[int] = None,
    log_every: int = 100,
    logger: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, float]]:
    """Execute a plan. ``batches()`` must yield a batch of clean latents.

    Returns the per-stage final logs. Written as a plain loop rather than a
    Trainer subclass so it can be dropped into an existing training script.
    """
    out: List[Dict[str, float]] = []
    emit = logger or (lambda msg: None)
    for stage in plan.stages:
        kind = stage["kind"]
        iters = int(max_iters_override or stage.get("iters", 100))
        logs: Dict[str, float] = {}
        for it in range(iters):
            x0 = batches()
            if kind == "tscd":
                if tscd is None:
                    raise ValueError("plan requires a TSCDDistiller")
                tscd.set_segments(int(stage.get("segments", 8)))
                loss, logs = tscd.step(x0)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                tscd.after_step()
            elif kind == "score":
                if score is None:
                    raise ValueError("plan requires a ScoreDistiller")
                noise = torch.randn_like(x0)
                if fake_optimizer is not None:
                    f_loss, f_logs = score.fake_score_loss(noise, steps=int(stage.get("steps", 1)))
                    fake_optimizer.zero_grad(set_to_none=True)
                    f_loss.backward()
                    fake_optimizer.step()
                    logs.update(f_logs)
                loss, s_logs = score.student_loss(noise, steps=int(stage.get("steps", 1)))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                logs.update(s_logs)
            elif kind == "adversarial":
                if adversarial is None or disc_optimizer is None:
                    raise ValueError("plan requires an AdversarialDistiller and disc optimizer")
                noise = torch.randn_like(x0)
                d_loss, d_logs = adversarial.discriminator_loss(x0, noise)
                disc_optimizer.zero_grad(set_to_none=True)
                d_loss.backward()
                disc_optimizer.step()
                g_loss, g_logs = adversarial.generator_loss(torch.randn_like(x0))
                optimizer.zero_grad(set_to_none=True)
                g_loss.backward()
                optimizer.step()
                logs.update(d_logs)
                logs.update(g_logs)
            else:
                raise ValueError(f"unknown distillation stage kind {kind!r}")
            if log_every and it % log_every == 0:
                emit(f"[{stage['name']}] iter {it}: {logs}")
        out.append({"stage": stage["name"], **logs})
    return out
