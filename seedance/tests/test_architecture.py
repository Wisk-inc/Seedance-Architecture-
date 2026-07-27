"""Architecture tests. Skipped entirely when torch is not installed.

Everything runs at the ``dev-tiny`` scale on CPU in a few seconds. The point is
not quality -- there are no weights -- it is that shapes, causality, masking and
the flow parameterisation are correct, which is what a reference implementation
owes you.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from seedance.config import get_config  # noqa: E402
from seedance.flow.rectified_flow import RectifiedFlow  # noqa: E402
from seedance.models.attention import window_partition, window_unpartition  # noqa: E402
from seedance.models.audio import alignment_mask  # noqa: E402
from seedance.models.conditioning import build_condition  # noqa: E402
from seedance.models.dit import SeedanceDiT, layer_schedule  # noqa: E402
from seedance.models.rope import MMRoPE, axial_split, shot_time_positions  # noqa: E402
from seedance.models.vae3d import TemporallyCausalVAE  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return get_config("dev-tiny")


# ------------------------------------------------------------------- windows

def test_window_partition_roundtrips_with_padding():
    x = torch.randn(2, 5, 7, 9, 16)  # H=7, W=9 are not multiples of the window
    parts, shape = window_partition(x, (4, 4))
    back = window_unpartition(parts, shape, 2, 5, 7, 9)
    assert back.shape == x.shape
    assert torch.allclose(back, x, atol=1e-6)


def test_window_partition_keeps_the_full_time_axis():
    x = torch.randn(1, 6, 8, 8, 4)
    parts, _ = window_partition(x, (4, 4))
    # every window row must contain all 6 frames -> global temporal receptive field
    assert parts.shape[1] == 6 * 4 * 4


# ---------------------------------------------------------------------- rope

def test_axial_split_sums_and_stays_even():
    for dim in (32, 64, 96, 128):
        dt, dh, dw = axial_split(dim)
        assert dt + dh + dw == dim
        assert all(d % 2 == 0 for d in (dt, dh, dw))


def test_rope_preserves_norm():
    rope = MMRoPE(head_dim=32).visual(3, 4, 4)
    q = torch.randn(2, 3 * 4 * 4, 2, 32)
    out = rope(q)
    assert out.shape == q.shape
    assert torch.allclose(out.norm(dim=-1), q.norm(dim=-1), atol=1e-4)


def test_multishot_positions_leave_gaps_between_shots():
    pos = shot_time_positions([3, 2, 4], gap=8)
    assert pos.tolist() == [0, 1, 2, 11, 12, 21, 22, 23, 24]


def test_text_positions_do_not_alias_visual_positions():
    rope = MMRoPE(head_dim=32)
    visual = rope.visual(4, 4, 4)
    text = rope.text(16)
    assert visual.freqs.shape[0] == 64
    assert text.freqs.shape[1] * 2 == 32
    # separate shots get separate caption slots
    assert not torch.allclose(rope.text(4, shot_index=0).freqs, rope.text(4, shot_index=1).freqs)


# --------------------------------------------------------------- conditioning

def test_unified_task_formulation_masks():
    shape = (1, 8, 5, 4, 4)
    clean = torch.randn(1, 8, 1, 4, 4)
    cond = build_condition(shape, "i2v", clean_latents=clean)
    assert cond.conditioned_frames() == [0]
    assert torch.allclose(cond.condition[:, :, 0], clean[:, :, 0])
    assert cond.condition[:, :, 1:].abs().sum() == 0

    flf = build_condition(shape, "flf2v", clean_latents=torch.randn(1, 8, 2, 4, 4))
    assert flf.conditioned_frames() == [0, 4]

    t2v = build_condition(shape, "t2v")
    assert t2v.mask.sum() == 0

    noisy = torch.randn(shape)
    assert cond.as_input(noisy).shape[1] == 8 + 8 + 1


def test_condition_rejects_mismatched_inputs():
    shape = (1, 8, 5, 4, 4)
    with pytest.raises(ValueError):
        build_condition(shape, "i2v")  # no clean latents
    with pytest.raises(ValueError):
        build_condition(shape, "not_a_task")
    cond = build_condition(shape, "t2v")
    with pytest.raises(ValueError):
        cond.as_input(torch.randn(1, 8, 5, 8, 8))  # wrong spatial size


# ----------------------------------------------------------------------- DiT

def test_layer_schedule_counts_and_interleaves(cfg):
    schedule = layer_schedule(cfg.dit)
    assert schedule.count("spatial") == cfg.dit.num_spatial_layers
    assert schedule.count("temporal") == cfg.dit.num_temporal_layers
    assert schedule[0] == "spatial" and schedule[1] == "temporal"


def test_dit_forward_shape_and_finiteness(cfg):
    dit = SeedanceDiT(cfg.dit).eval()
    b, c, t, h, w = 1, cfg.dit.in_channels, 3, 8, 8
    x = torch.randn(b, c, t, h, w)
    text = torch.randn(b, 12, cfg.dit.text_dim)
    mask = torch.ones(b, 12, dtype=torch.bool)
    with torch.no_grad():
        out = dit(x, torch.tensor([500.0]), text, text_mask=mask)
    assert out.shape == (b, cfg.dit.out_channels, t, h, w)
    assert torch.isfinite(out).all()


def test_dit_accepts_conditioning_and_shot_lengths(cfg):
    dit = SeedanceDiT(cfg.dit).eval()
    shape = (1, cfg.dit.in_channels, 4, 8, 8)
    x = torch.randn(shape)
    cond = build_condition(shape, "i2v", clean_latents=torch.randn(1, cfg.dit.in_channels, 1, 8, 8))
    text = torch.randn(1, 8, cfg.dit.text_dim)
    with torch.no_grad():
        out = dit(x, torch.tensor([100.0]), text, condition=cond, shot_lengths=[2, 2])
    assert out.shape == shape
    with pytest.raises(ValueError):
        dit(x, torch.tensor([100.0]), text, shot_lengths=[3, 3])


def test_dit_is_zero_initialised_at_the_output(cfg):
    """adaLN-zero + zero head: a fresh model predicts exactly zero."""
    dit = SeedanceDiT(cfg.dit).eval()
    with torch.no_grad():
        out = dit(
            torch.randn(1, cfg.dit.in_channels, 3, 8, 8),
            torch.tensor([10.0]),
            torch.randn(1, 6, cfg.dit.text_dim),
        )
    assert out.abs().max() == 0


def test_dit_gradients_flow(cfg):
    dit = SeedanceDiT(cfg.dit).train()
    x = torch.randn(1, cfg.dit.in_channels, 3, 8, 8, requires_grad=True)
    out = dit(x, torch.tensor([300.0]), torch.randn(1, 6, cfg.dit.text_dim))
    out.sum().backward()
    grads = [p.grad for p in dit.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert grads, "no parameter received a gradient"


def _wake_up(dit: SeedanceDiT) -> SeedanceDiT:
    """Undo adaLN-zero and the zero head so the model actually computes.

    A freshly initialised DiT outputs exactly zero by design, which would make
    any "does X change the output" test vacuously pass.
    """
    from seedance.models.norm import AdaLNModulation

    for module in dit.modules():
        if isinstance(module, AdaLNModulation):
            torch.nn.init.normal_(module.proj[1].weight, std=0.02)
            torch.nn.init.normal_(module.proj[1].bias, std=0.02)
    torch.nn.init.normal_(dit.head.proj.weight, std=0.02)
    return dit


def test_text_mask_changes_the_result(cfg):
    dit = _wake_up(SeedanceDiT(cfg.dit).eval())
    x = torch.randn(1, cfg.dit.in_channels, 2, 8, 8)
    text = torch.randn(1, 10, cfg.dit.text_dim)
    full = torch.ones(1, 10, dtype=torch.bool)
    partial = full.clone()
    partial[:, 5:] = False
    with torch.no_grad():
        a = dit(x, torch.tensor([200.0]), text, text_mask=full)
        b = dit(x, torch.tensor([200.0]), text, text_mask=partial)
    assert not torch.allclose(a, b)


# ----------------------------------------------------------------------- VAE

def test_vae_shape_contract(cfg):
    vae = TemporallyCausalVAE(cfg.vae).eval()
    frames = 1 + 4 * 2  # 9 pixel frames -> 3 latent frames
    video = torch.randn(1, 3, frames, 32, 32)
    with torch.no_grad():
        z = vae.encode(video, sample=False)
    assert z.shape[0] == 1
    assert z.shape[1] == cfg.vae.latent_channels
    assert z.shape[2] == vae.latent_frames(frames)
    assert z.shape[3] == 32 // cfg.vae.spatial_downsample
    with torch.no_grad():
        recon = vae.decode(z)
    assert recon.shape[1] == 3
    assert recon.min() >= -1 and recon.max() <= 1


def test_vae_rejects_invalid_frame_counts(cfg):
    vae = TemporallyCausalVAE(cfg.vae)
    with pytest.raises(ValueError):
        vae.latent_frames(8)
    assert vae.pixel_frames(vae.latent_frames(9)) == 9


def test_vae_is_temporally_causal(cfg):
    """Changing the last frame must not change earlier latents."""
    vae = TemporallyCausalVAE(cfg.vae).eval()
    video = torch.randn(1, 3, 9, 32, 32)
    altered = video.clone()
    altered[:, :, -1] = torch.randn_like(altered[:, :, -1])
    with torch.no_grad():
        a = vae.encode(video, sample=False)
        b = vae.encode(altered, sample=False)
    assert torch.allclose(a[:, :, :-1], b[:, :, :-1], atol=1e-4)
    assert not torch.allclose(a[:, :, -1], b[:, :, -1], atol=1e-4)


def test_thin_decoder_is_smaller_but_same_shape(cfg):
    vae = TemporallyCausalVAE(cfg.vae).eval()
    assert vae.thin_decoder is not None
    thin = sum(p.numel() for p in vae.thin_decoder.parameters())
    full = sum(p.numel() for p in vae.decoder.parameters())
    assert thin < full
    z = torch.randn(1, cfg.vae.latent_channels, 3, 4, 4)
    with torch.no_grad():
        vae.use_thin_decoder(False)
        a = vae.decode(z)
        vae.use_thin_decoder(True)
        b = vae.decode(z)
    assert a.shape == b.shape


def test_latent_stats_whiten(cfg):
    vae = TemporallyCausalVAE(cfg.vae).eval()
    videos = [torch.randn(1, 3, 9, 32, 32) for _ in range(2)]
    vae.fit_latent_stats(videos)
    with torch.no_grad():
        z = vae.encode(videos[0], sample=False)
    assert abs(float(z.mean())) < 1.0
    assert 0.2 < float(z.std()) < 5.0


# ---------------------------------------------------------------------- flow

def test_flow_integrates_back_to_the_data_point():
    """Catches sign/scale slips in the rectified-flow parameterisation."""
    flow = RectifiedFlow()
    assert flow.self_consistency_check() < 1e-4


def test_add_noise_endpoints():
    flow = RectifiedFlow()
    x0, noise = torch.randn(2, 4, 3, 4, 4), torch.randn(2, 4, 3, 4, 4)
    at_zero = flow.add_noise(x0, noise, torch.zeros(2))
    at_one = flow.add_noise(x0, noise, torch.ones(2))
    assert torch.allclose(at_zero, x0, atol=1e-6)
    assert torch.allclose(at_one, noise, atol=1e-6)


def test_x0_recovery_from_velocity():
    flow = RectifiedFlow()
    x0, noise = torch.randn(1, 4, 2, 4, 4), torch.randn(1, 4, 2, 4, 4)
    t = torch.tensor([0.37])
    x_t = flow.add_noise(x0, noise, t)
    v = flow.velocity_target(x0, noise)
    assert torch.allclose(flow.x0_from_velocity(x_t, v, t), x0, atol=1e-5)
    assert torch.allclose(flow.noise_from_velocity(x_t, v, t), noise, atol=1e-5)


def test_sigma_schedule_is_descending_and_ends_at_zero():
    flow = RectifiedFlow()
    sigmas = flow.sigmas(10, seq_len=1024)
    assert sigmas[0] > sigmas[-2] > sigmas[-1] == 0
    assert len(sigmas) == 11


def test_cfg_zero_init_returns_the_conditional_branch():
    from seedance.flow.rectified_flow import apply_cfg

    cond, uncond = torch.ones(2, 3), torch.zeros(2, 3)
    assert torch.allclose(apply_cfg(cond, uncond, 5.0, zero_init=True), cond)
    assert torch.allclose(apply_cfg(cond, uncond, 5.0), uncond + 5.0 * (cond - uncond))


# --------------------------------------------------------------------- audio

def test_alignment_mask_binds_only_nearby_events():
    mask = alignment_mask(200, 48, audio_rate=100.0, video_fps=24.0, tolerance_s=0.1)
    assert mask.shape == (200, 48)
    assert bool(mask[0, 0])            # t=0 audio binds to frame 0
    assert not bool(mask[0, 47])       # ... and not to a frame 2 s later
    assert mask.any(dim=1).all()       # no audio token is left unbound


def test_audio_bridge_starts_as_a_no_op():
    from seedance.models.audio import CrossModalJointModule

    bridge = CrossModalJointModule(video_dim=32, audio_dim=16, num_heads=4)
    visual = torch.randn(1, 4, 9, 32)
    audio = torch.randn(1, 20, 16)
    out_v, out_a = bridge(visual, audio, torch.randn(1, 32))
    # gates are zero-initialised, so a pretrained video backbone is unperturbed
    assert torch.allclose(out_v, visual, atol=1e-6)
    assert torch.allclose(out_a, audio, atol=1e-6)


def test_audio_tokenizer_rate_and_shape():
    from seedance.config import AudioConfig
    from seedance.models.audio import AudioTokenizer

    cfg = AudioConfig(sample_rate=16000, hop_length=160, n_fft=512, n_mels=32, dim=64,
                      waveform_patch=160, channels=1)
    tok = AudioTokenizer(cfg)
    assert tok.token_rate == 100.0
    tokens = tok(torch.randn(1, 1, 16000))
    assert tokens.shape[0] == 1 and tokens.shape[2] == 64
    assert 95 <= tokens.shape[1] <= 105  # ~1 s at 100 tokens/s
