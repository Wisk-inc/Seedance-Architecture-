"""Pipeline tests, driven by the weightless mock backend."""

from __future__ import annotations

import os

import pytest

# ------------------------------------------------------------- torch required

torch = pytest.importorskip("torch")

from seedance.backends.mock import MockBackend  # noqa: E402
from seedance.stages.physics import PhysicsScene  # noqa: E402
from seedance.pipelines.staged import SeedancePipeline, StageSettings  # noqa: E402
from seedance.stages.polish import (  # noqa: E402
    PolishConfig,
    adain_normalize,
    interpolate_frames,
    match_colors,
    temporal_qa,
)
from seedance.story import Storyboard  # noqa: E402


def _pipe(**kwargs):
    settings = StageSettings.preset("fast")
    settings.enhancer_offline_only = True
    for key, value in kwargs.items():
        setattr(settings, key, value)
    return SeedancePipeline(
        "mock", backend=MockBackend().load(), settings=settings, plan_memory=False
    )


def test_end_to_end_generate_and_save(tmp_path):
    out = tmp_path / "clip.mp4"
    pipe = _pipe()
    result = pipe.generate(
        "a paper boat drifting down a gutter",
        size="64*64",
        num_frames=9,
        output=str(out) if _has_imageio() else None,
    )
    assert result.video.shape[0] == 3
    assert result.video.shape[1] == 9
    assert result.enhanced_prompt != result.prompt, "stage 1 should have rewritten it"
    assert "prompt" in result.report["stages"]
    assert "polish" in result.report["stages"]
    assert result.report["polish"]["qa"]["sharpness"] >= 0
    if _has_imageio():
        assert os.path.exists(result.path)


def test_multishot_chaining_produces_one_longer_clip():
    pipe = _pipe(transition_frames=2)
    board = Storyboard.from_lines("a chase", ["shot one", "shot two"], 9)
    result = pipe.generate("a chase", size="64*64", num_frames=9, storyboard=board)
    # two 9-frame shots minus the 2-frame crossfade overlap
    assert result.video.shape[1] == 16
    assert len(result.report["shots"]) == 2


def test_physics_stage_builds_a_guide_and_says_what_it_is():
    settings = StageSettings.preset("fast")
    settings.enhancer_offline_only = True
    settings.physics = PhysicsScene.bouncing_ball(fps=16, duration_s=1.0)
    pipe = SeedancePipeline(
        "mock",
        backend=MockBackend(tasks=("t2v", "vace")).load(),
        settings=settings,
        plan_memory=False,
    )
    result = pipe.generate("a ball bounces", size="64*64", num_frames=9)
    assert result.report["physics"]["engine"] in ("analytic", "pybullet", "mujoco")
    assert "no physics engine" in result.report["physics"]["note"]


def test_control_guides_warn_when_the_checkpoint_cannot_use_them():
    settings = StageSettings.preset("fast")
    settings.enhancer_offline_only = True
    settings.physics = PhysicsScene.bouncing_ball(fps=16, duration_s=1.0)
    pipe = SeedancePipeline(
        "mock",
        backend=MockBackend(tasks=("t2v",)).load(),  # no control path
        settings=settings,
        plan_memory=False,
    )
    result = pipe.generate("a ball bounces", size="64*64", num_frames=9)
    assert any("control" in w for w in result.warnings)


def test_capability_report_is_explicit_about_the_gap():
    pipe = _pipe()
    report = pipe.capability_report()
    assert any("audio" in item for item in report["not_reproducible"])
    assert any("@mention" in item for item in report["not_reproducible"])


def test_reference_images_are_dropped_loudly_without_a_reference_path():
    from seedance.stages.identity import IdentityConfig

    settings = StageSettings.preset("fast")
    settings.enhancer_offline_only = True
    settings.identity = IdentityConfig(method="auto", subject_description="a tall man")
    pipe = SeedancePipeline(
        "mock", backend=MockBackend(tasks=("t2v",)).load(), settings=settings, plan_memory=False
    )
    result = pipe.generate("he walks", size="64*64", num_frames=9)
    assert result.report["identity"]["method"] == "caption_only"


# ------------------------------------------------------------------- stage 4

def test_color_matching_removes_drift():
    base = torch.randn(3, 8, 16, 16) * 0.2
    drifting = base.clone()
    for i in range(8):
        drifting[1, i] += 0.08 * i  # a green cast that grows
    from seedance.reward.heuristics import color_drift

    before = float(color_drift(drifting.unsqueeze(0))[0])
    after = float(color_drift(match_colors(drifting).unsqueeze(0))[0])
    assert after > before


def test_adain_matches_statistics():
    video = torch.randn(3, 6, 8, 8)
    video[:, 3:] *= 3.0
    out = adain_normalize(video)
    stds = [float(out[:, i].std()) for i in range(6)]
    assert max(stds) / min(stds) < 2.0


def test_interpolation_increases_the_frame_count():
    video = torch.randn(3, 5, 16, 16).clamp(-1, 1)
    out, method = interpolate_frames(video, 2, method="blend")
    assert out.shape[1] == 9  # 5 frames -> 4 gaps -> 9
    assert method == "blend"


def test_temporal_qa_reports_all_metrics():
    report = temporal_qa(torch.randn(3, 8, 16, 16).clamp(-1, 1))
    for key in ("temporal_smoothness", "flicker", "sharpness", "color_stability"):
        assert 0.0 <= report[key] <= 1.0


def test_qa_prefers_smooth_video_over_noise():
    from seedance.reward.heuristics import temporal_smoothness

    frames = torch.linspace(-0.5, 0.5, 12).view(1, 12, 1, 1).expand(3, 12, 16, 16)
    noise = torch.randn(3, 12, 16, 16).clamp(-1, 1)
    assert float(temporal_smoothness(frames.unsqueeze(0))[0]) > float(
        temporal_smoothness(noise.unsqueeze(0))[0]
    )


# --------------------------------------------------------------------- bench

def test_physics_iq_metrics_are_perfect_on_identical_video():
    from seedance.bench.physics_iq import score_pair

    video = torch.randn(3, 8, 16, 16).clamp(-1, 1)
    scores = score_pair(video, video)
    assert scores["spatial_iou"] == pytest.approx(1.0)
    assert scores["spatiotemporal_iou"] == pytest.approx(1.0)
    assert scores["mse"] == pytest.approx(0.0, abs=1e-6)


def test_physics_iq_metrics_penalise_the_wrong_region():
    from seedance.bench.physics_iq import spatial_iou

    real = torch.zeros(3, 8, 32, 32)
    fake = torch.zeros(3, 8, 32, 32)
    for i in range(8):
        real[:, i, 4:12, 4:12] = i / 8  # motion top-left
        fake[:, i, 20:28, 20:28] = i / 8  # motion bottom-right
    assert spatial_iou(real, fake) < 0.05


# ---------------------------------------------------------------------- utils

def _has_imageio() -> bool:
    try:
        import imageio  # noqa: F401

        return True
    except Exception:
        return False
