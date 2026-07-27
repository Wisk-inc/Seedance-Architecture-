"""Training-side tests: distillation objectives and GRPO.

None of these train anything to convergence -- they check that the objectives
are well-formed, produce finite non-zero losses, and deliver gradients to the
student. A freshly initialised DiT emits exactly zero (adaLN-zero + zero head),
which would make every loss trivially zero, so the models are woken up first.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from seedance.config import DiTConfig  # noqa: E402
from seedance.flow.distill import (  # noqa: E402
    AdversarialDistiller,
    DistillationPlan,
    LatentDiscriminator,
    ScoreDistiller,
    TSCDDistiller,
)
from seedance.flow.rectified_flow import RectifiedFlow  # noqa: E402
from seedance.models.dit import SeedanceDiT  # noqa: E402
from seedance.models.norm import AdaLNModulation  # noqa: E402
from seedance.reward.dance_grpo import DanceGRPO, GRPOConfig  # noqa: E402
from seedance.reward.heuristics import HeuristicRewardModel  # noqa: E402
from seedance.reward.reward_dance import RewardEnsemble, build_reward_model  # noqa: E402

CFG = DiTConfig(
    dim=64,
    ffn_dim=128,
    num_heads=4,
    num_spatial_layers=1,
    num_temporal_layers=1,
    text_dim=64,
    in_channels=8,
    out_channels=8,
    condition_channels=8,
    temporal_window=(2, 2),
)


def _wake(model: SeedanceDiT) -> SeedanceDiT:
    for module in model.modules():
        if isinstance(module, AdaLNModulation):
            torch.nn.init.normal_(module.proj[1].weight, std=0.02)
    torch.nn.init.normal_(model.head.proj.weight, std=0.02)
    return model


def _velocity_fn(model: SeedanceDiT):
    text = torch.randn(1, 4, CFG.text_dim)

    def fn(x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        return model(x, ts, text.expand(x.shape[0], -1, -1))

    return fn


@pytest.fixture
def models():
    torch.manual_seed(0)
    teacher = _wake(SeedanceDiT(CFG)).eval()
    student = _wake(SeedanceDiT(CFG))
    return teacher, student


def test_tscd_produces_a_gradient(models):
    teacher, student = models
    distiller = TSCDDistiller(student, _velocity_fn(teacher), segments=4)
    distiller.student = _velocity_fn(student)
    distiller.target = _velocity_fn(teacher)
    loss, logs = distiller.step(torch.randn(2, 8, 2, 4, 4))
    loss.backward()
    assert torch.isfinite(loss) and float(loss.detach()) > 0
    assert logs["segments"] == 4
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())


def test_tscd_segments_anneal(models):
    teacher, student = models
    distiller = TSCDDistiller(student, _velocity_fn(teacher), segments=8)
    distiller.student = _velocity_fn(student)
    distiller.target = _velocity_fn(teacher)
    bounds = distiller._segment_bounds(torch.tensor([0.05, 0.3, 0.99]))
    assert bounds.tolist() == [0.0, 0.25, 0.875]
    distiller.set_segments(2)
    bounds = distiller._segment_bounds(torch.tensor([0.05, 0.6]))
    assert bounds.tolist() == [0.0, 0.5]


def test_score_distillation_trains_both_networks(models):
    teacher, student = models
    fake = _wake(SeedanceDiT(CFG))
    distiller = ScoreDistiller(_velocity_fn(student), _velocity_fn(teacher), _velocity_fn(fake))
    noise = torch.randn(2, 8, 2, 4, 4)

    fake_loss, fake_logs = distiller.fake_score_loss(noise)
    fake_loss.backward()
    assert float(fake_logs["fake_score"]) > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in fake.parameters())

    student_loss, _ = distiller.student_loss(noise)
    student_loss.backward()
    assert torch.isfinite(student_loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())


def test_adversarial_distillation_with_preference_head(models):
    _, student = models
    disc = LatentDiscriminator(8, base=8, layers=2)
    distiller = AdversarialDistiller(_velocity_fn(student), disc, steps=2)
    real = torch.randn(2, 8, 2, 8, 8)
    d_loss, d_logs = distiller.discriminator_loss(
        real, torch.randn_like(real), real_reward=torch.rand(2)
    )
    d_loss.backward()
    assert "pref_d" in d_logs
    g_loss, g_logs = distiller.generator_loss(torch.randn_like(real))
    g_loss.backward()
    assert "pref_g" in g_logs
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())


def test_distillation_plan_is_ordered_as_the_report_describes():
    kinds = [stage["kind"] for stage in DistillationPlan().stages]
    assert kinds[0] == "tscd" and kinds[-1] == "adversarial"
    assert "score" in kinds
    segments = [s["segments"] for s in DistillationPlan().stages if s["kind"] == "tscd"]
    assert segments == sorted(segments, reverse=True), "segments must anneal downward"


# ----------------------------------------------------------------------- GRPO

def test_grpo_rollout_records_a_differentiable_policy(models):
    _, student = models
    grpo = DanceGRPO(
        _velocity_fn(student),
        HeuristicRewardModel(),
        cfg=GRPOConfig(group_size=4, num_steps=4, noise_scale=0.5),
        decode_fn=lambda z: z[:, :3].clamp(-1, 1),
    )
    traj = grpo.rollout((4, 8, 3, 8, 8), device="cpu")
    assert len(traj) == 4
    assert traj.sample is not None
    assert all(lp.shape == (4,) for lp in traj.log_probs)
    assert all(torch.isfinite(lp).all() for lp in traj.log_probs)


def test_grpo_advantages_are_group_normalised(models):
    _, student = models
    grpo = DanceGRPO(
        _velocity_fn(student),
        HeuristicRewardModel(),
        cfg=GRPOConfig(group_size=4, num_steps=3),
        decode_fn=lambda z: z[:, :3].clamp(-1, 1),
    )
    traj = grpo.rollout((4, 8, 3, 8, 8), device="cpu")
    adv = grpo.compute_advantages(traj)
    assert adv.shape == (4,)
    assert abs(float(adv.mean())) < 1e-4, "group-normalised advantages are zero-mean"


def test_grpo_loss_backpropagates(models):
    _, student = models
    grpo = DanceGRPO(
        _velocity_fn(student),
        HeuristicRewardModel(),
        cfg=GRPOConfig(group_size=4, num_steps=4, timestep_fraction=1.0),
        decode_fn=lambda z: z[:, :3].clamp(-1, 1),
    )
    traj = grpo.rollout((4, 8, 3, 8, 8), device="cpu")
    grpo.compute_advantages(traj)
    loss, logs = grpo.loss(traj)
    loss.backward()
    assert torch.isfinite(loss)
    assert 0.0 <= logs["clip_fraction"] <= 1.0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())


def test_grpo_requires_advantages_first(models):
    _, student = models
    grpo = DanceGRPO(_velocity_fn(student), HeuristicRewardModel(), cfg=GRPOConfig(num_steps=2))
    traj = grpo.rollout((2, 8, 2, 4, 4), device="cpu")
    with pytest.raises(ValueError):
        grpo.loss(traj)


# --------------------------------------------------------------------- reward

def test_heuristic_reward_ranks_clean_video_above_noise():
    rm = HeuristicRewardModel()
    smooth = torch.linspace(-0.4, 0.4, 12).view(1, 1, 12, 1, 1).expand(1, 3, 12, 16, 16)
    noise = torch.randn(1, 3, 12, 16, 16).clamp(-1, 1)
    assert float(rm.score(smooth)[0]) > float(rm.score(noise)[0])


def test_reward_ensemble_normalises_per_model_before_summing():
    class Wide:
        name = "wide"

        def score(self, videos, prompts=None):
            return torch.tensor([0.0, 1000.0, 2000.0, 3000.0])

    class Narrow:
        name = "narrow"

        def score(self, videos, prompts=None):
            return torch.tensor([0.3, 0.2, 0.1, 0.0])

    ensemble = RewardEnsemble([Wide(), Narrow()])
    videos = torch.randn(4, 3, 4, 8, 8)
    adv = ensemble.normalized_advantage(videos, group_size=4)
    # the wide-range model must not dominate: normalisation cancels the two
    # opposing rankings almost exactly
    assert abs(float(adv.abs().max())) < 0.2


def test_build_reward_model_falls_back_to_heuristics():
    rm = build_reward_model("definitely/not-a-real-vlm", allow_fallback=True)
    assert isinstance(rm, HeuristicRewardModel)
    with pytest.raises(Exception):
        build_reward_model("definitely/not-a-real-vlm", allow_fallback=False)
