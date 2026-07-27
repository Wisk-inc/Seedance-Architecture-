"""Torch-free tests: config, registry, checkpoint detection, storyboards.

These run on a bare interpreter with no ML stack, which is the point -- model
discovery and request validation must not require a 3 GB wheel.
"""

from __future__ import annotations

import json
import os

import pytest

from seedance.backends.base import GenerationRequest, parse_size
from seedance.backends.registry import (
    MODELS,
    detect_layout,
    list_models,
    recommended_size,
    resolve,
)
from seedance.config import get_config, list_variants
from seedance.story import Reference, ReferenceBundle, Shot, Storyboard, parse_mentions


# --------------------------------------------------------------------- config

def test_all_variants_build():
    for name in list_variants():
        cfg = get_config(name)
        assert cfg.dit.dim % cfg.dit.num_heads == 0
        assert cfg.dit.head_dim % 2 == 0, "RoPE needs an even head dim"
        assert cfg.dit.in_channels == cfg.vae.latent_channels


def test_seedance_vae_matches_the_report():
    cfg = get_config("seedance-1.0")
    assert cfg.vae.stride == (4, 16, 16)
    assert cfg.vae.latent_channels == 48
    assert cfg.dit.patch_size == (1, 1, 1), "DC-AE: no patchify on the DiT side"


def test_latent_shape_and_frame_rounding():
    cfg = get_config("seedance-2.0")
    c, t, h, w = cfg.latent_shape(97, 480, 832)
    assert (c, t, h, w) == (48, 25, 30, 52)
    with pytest.raises(ValueError):
        cfg.latent_shape(96, 480, 832)  # (n-1) % 4 != 0
    with pytest.raises(ValueError):
        cfg.latent_shape(97, 481, 832)  # not a multiple of 16
    assert (cfg.frames_for_duration(5.0) - 1) % 4 == 0


def test_config_json_roundtrip():
    cfg = get_config("seedance-1.5-pro")
    restored = type(cfg).from_dict(json.loads(cfg.to_json()))
    assert restored.vae.stride == cfg.vae.stride
    assert restored.dit.temporal_window == cfg.dit.temporal_window
    assert restored.audio.enabled == cfg.audio.enabled


# ------------------------------------------------------------------- registry

def test_known_models_resolve_by_alias_and_repo():
    for spec in MODELS:
        assert resolve(spec.alias)[0] == spec.repo_id
        assert resolve(spec.repo_id)[0] == spec.repo_id
        assert spec.default_size in spec.sizes


def test_unknown_repo_passes_through_and_garbage_raises():
    assert resolve("some-org/brand-new-wan")[0] == "some-org/brand-new-wan"
    with pytest.raises(KeyError):
        resolve("definitely-not-a-model")


def test_recommended_size_downgrades_when_vram_is_tight():
    spec = next(m for m in MODELS if m.alias == "t2v-14B")
    assert recommended_size(spec, None, 80) == spec.default_size
    assert "480" in recommended_size(spec, None, 12)


def _mk(tmp_path, name, files, cfg=None):
    root = tmp_path / name
    for rel in files:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    if cfg is not None:
        (root / "config.json").write_text(json.dumps(cfg))
    return str(root)


def test_detects_native_t2v(tmp_path):
    root = _mk(
        tmp_path,
        "Wan2.1-T2V-1.3B",
        ["Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "diffusion_pytorch_model.safetensors"],
        {"model_type": "t2v", "dim": 1536, "num_layers": 30, "in_dim": 16},
    )
    layout = detect_layout(root, repo_id="Wan-AI/Wan2.1-T2V-1.3B")
    assert layout.loader == "wan_native"
    assert layout.task == "t2v"
    assert layout.dim == 1536
    assert layout.vae_checkpoint == "Wan2.1_VAE.pth"
    assert not layout.warnings


def test_detects_i2v_from_clip_weights_without_a_name_hint(tmp_path):
    root = _mk(
        tmp_path,
        "mirror-of-some-model",
        [
            "Wan2.1_VAE.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        ],
        {"dim": 5120, "in_dim": 36},
    )
    layout = detect_layout(root, repo_id="mirror-of-some-model")
    assert layout.task == "i2v"
    assert layout.clip_checkpoint


def test_detects_diffusers_moe(tmp_path):
    root = _mk(
        tmp_path,
        "Wan2.2-T2V-A14B",
        ["model_index.json", "transformer/config.json", "high_noise_model/w.safetensors"],
    )
    layout = detect_layout(root, repo_id="Wan-AI/Wan2.2-T2V-A14B")
    assert layout.loader == "diffusers"
    assert layout.moe
    assert layout.family == "wan2.2"


def test_detects_gguf_and_warns(tmp_path):
    root = _mk(tmp_path, "wan-gguf", ["wan2.2-Q4_K.gguf"])
    layout = detect_layout(root, repo_id="quantstack/Wan2.2-T2V-GGUF")
    assert layout.quantization == "gguf"
    assert any("GGUF" in w for w in layout.warnings)


def test_incomplete_native_checkpoint_is_flagged(tmp_path):
    root = _mk(tmp_path, "partial", ["Wan2.1_VAE.pth"], {"model_type": "t2v", "dim": 1536})
    layout = detect_layout(root, repo_id="partial")
    assert any("missing" in w for w in layout.warnings)


# -------------------------------------------------------------------- request

def test_parse_size_variants():
    assert parse_size("832*480") == (832, 480)
    assert parse_size("1280x720") == (1280, 720)
    with pytest.raises(ValueError):
        parse_size("720p")


def test_frame_normalisation_is_valid_for_the_vae():
    req = GenerationRequest(prompt="x", num_frames=80)
    assert (req.normalized_frames(4) - 1) % 4 == 0
    assert GenerationRequest(prompt="x", num_frames=1).normalized_frames(4) >= 5


# ------------------------------------------------------------------- story

def test_reference_budget_is_enforced():
    bundle = ReferenceBundle(max_images=2)
    bundle.add(Reference("image", "@a"))
    bundle.add(Reference("image", "@b"))
    with pytest.raises(ValueError):
        bundle.add(Reference("image", "@c"))
    with pytest.raises(ValueError):
        bundle.add(Reference("video", "@a"))  # duplicate handle


def test_mentions_are_resolved_into_plain_language():
    bundle = ReferenceBundle()
    bundle.add(Reference("image", "@hero", role="subject"))
    bundle.add(Reference("video", "@swing", role="motion"))
    text, used = bundle.resolve("@hero performs @swing at dusk")
    assert "@" not in text
    assert len(used) == 2
    assert parse_mentions("@hero and [Image1]") == ["@hero", "[Image1]"]


def test_storyboard_limits_and_totals():
    board = Storyboard.from_lines("a chase", ["shot one", "shot two", "shot three"], 17)
    assert len(board.shots) == 3
    assert board.total_frames == 51
    board.max_shots = 3
    with pytest.raises(ValueError):
        board.add(Shot("shot four", 17))


# ------------------------------------------------------- offline Wan backend

def test_wan_backend_reconstructs_a_config_without_the_wan_stack(tmp_path):
    """`inspect` and config building must work with no torch/diffusers/easydict."""
    from seedance.backends.wan_backend import WanBackend

    root = _mk(
        tmp_path,
        "Wan2.1-I2V-14B-480P",
        [
            "Wan2.1_VAE.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        ],
        {"model_type": "i2v", "dim": 5120, "num_layers": 40, "in_dim": 36},
    )
    backend = WanBackend(root, download=False)
    info = backend.describe()
    assert info["task"] == "i2v"
    assert backend.capabilities().tasks == ("i2v",)
    assert backend.defaults()["size"] == "832*480"  # the 480P hint wins over size

    backend.config = backend._build_config()
    assert backend.config.vae_checkpoint == "Wan2.1_VAE.pth"
    assert backend.config.clip_checkpoint.startswith("models_clip")
    assert backend._missing_files() == []


def test_wan_backend_reports_missing_weights_instead_of_failing_late(tmp_path):
    from seedance.backends.wan_backend import WanBackend

    root = _mk(
        tmp_path,
        "half-a-checkpoint",
        ["Wan2.1_VAE.pth"],
        {"model_type": "t2v", "dim": 1536},
    )
    backend = WanBackend(root, download=False)
    backend.describe()
    backend.config = backend._build_config()
    missing = backend._missing_files()
    assert any("umt5" in name for name in missing)


def test_native_loader_refuses_a_diffusers_checkpoint(tmp_path):
    from seedance.backends.wan_backend import WanBackend

    root = _mk(tmp_path, "Wan2.2-T2V-A14B", ["model_index.json", "transformer/config.json"])
    backend = WanBackend(root, download=False)
    with pytest.raises(RuntimeError, match="diffusers"):
        backend.load()


def test_mirrored_wan_configs_match_the_real_ones_when_available():
    """If `wan.configs` imports here, the mirrored table must agree with it."""
    wan_configs = pytest.importorskip("wan.configs")
    from seedance.backends.wan_config import base_config

    for key, attr in (
        ("t2v-1.3B", "t2v_1_3B"),
        ("t2v-14B", "t2v_14B"),
        ("i2v-14B", "i2v_14B"),
    ):
        real = getattr(wan_configs, attr)
        mirrored = base_config(key, prefer_wan=False)
        for field in ("dim", "ffn_dim", "num_heads", "num_layers", "vae_stride",
                      "patch_size", "text_len", "sample_neg_prompt"):
            assert tuple(mirrored[field]) == tuple(real[field]) if isinstance(
                real[field], (list, tuple)
            ) else mirrored[field] == real[field], f"{key}.{field} drifted"
