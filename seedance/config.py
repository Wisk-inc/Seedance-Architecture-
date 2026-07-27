"""Configuration objects for the Seedance architecture.

Every field carries a provenance tag in its comment:

* ``[1.0]``      -- stated in the Seedance 1.0 technical report (arXiv:2506.09113).
* ``[1.5/2.0]``  -- stated in the abstract/introduction of the 1.5-Pro / 2.0 reports.
* ``[inferred]`` -- not disclosed anywhere; a defensible reconstruction following
  SD3 / rectified-flow conventions. Do not cite these as Seedance facts.

Nothing in this module imports torch, so it can be inspected and tested on a
machine with no ML stack installed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "VAEConfig",
    "DiTConfig",
    "AudioConfig",
    "TextEncoderConfig",
    "FlowConfig",
    "RefinerConfig",
    "SeedanceConfig",
    "VARIANTS",
    "get_config",
    "list_variants",
]


@dataclass
class VAEConfig:
    """Temporally-causal 3D VAE (Seedance 1.0, following MAGVIT)."""

    # [1.0] downsampling ratios (r_t, r_h, r_w) = (4, 16, 16)
    temporal_downsample: int = 4
    spatial_downsample: int = 16
    # [1.0] C = 48 latent channels
    latent_channels: int = 48
    base_channels: int = 128
    # channel multipliers per stage; 4 spatial stages => 16x, 2 temporal
    # downsamples among them => 4x temporal.
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4, 4)
    temporal_downsample_stages: Tuple[bool, ...] = (False, True, True, False, False)
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int, ...] = ()
    dropout: float = 0.0
    # [1.0] Thin VAE decoder narrows the stages closest to pixel space.
    thin_decoder: bool = True
    thin_decoder_shrink: Tuple[float, ...] = (0.25, 0.5, 1.0, 1.0, 1.0)
    # [1.0] loss weights: L1 + KL + LPIPS + adversarial (PatchGAN hybrid)
    loss_l1: float = 1.0
    loss_kl: float = 1e-6
    loss_lpips: float = 0.5
    loss_adv: float = 0.1
    # tiling for long/high-res video (memory bound, not a paper claim)
    tile_spatial: int = 0
    tile_temporal: int = 0

    @property
    def stride(self) -> Tuple[int, int, int]:
        return (self.temporal_downsample, self.spatial_downsample, self.spatial_downsample)


@dataclass
class DiTConfig:
    """Decoupled spatial/temporal MMDiT backbone."""

    dim: int = 2048
    ffn_dim: int = 8192
    num_heads: int = 16
    # [1.0] decoupled layers: spatial (intra-frame, multimodal) and temporal
    # (inter-frame, visual-only). They are interleaved 1:1 by default.
    num_spatial_layers: int = 24
    num_temporal_layers: int = 24
    interleave: str = "alternating"  # "alternating" | "spatial_first" | "blocked"
    blocked_group_size: int = 4
    # [1.0] patchification on the DiT side is removed (following DC-AE) because
    # the VAE already downsamples 16x spatially.
    patch_size: Tuple[int, int, int] = (1, 1, 1)
    in_channels: int = 48
    out_channels: int = 48
    # [1.0] unified task formulation: noisy latents are concatenated with clean
    # or zero-padded conditioning frames plus a binary mask channel.
    condition_channels: int = 48
    mask_channels: int = 1
    text_dim: int = 4096
    text_len: int = 512
    # [1.0] window partitioning inside temporal layers, applied within frames.
    temporal_window: Tuple[int, int] = (8, 8)  # (h, w) window over latent grid
    # [1.0] Q/K are normalized before attention to prevent training instability.
    qk_norm: bool = True
    eps: float = 1e-6
    rope_theta: float = 10000.0
    # [1.0] multishot MM-RoPE: 3D RoPE for visual tokens + 1D for text tokens.
    max_shots: int = 8
    shot_position_stride: int = 1024
    # [1.5/2.0] cross-modal joint module cadence for the audio branch: every Nth
    # spatial layer exchanges tokens with the audio branch.
    audio_bridge_every: int = 4
    grad_checkpointing: bool = False

    @property
    def head_dim(self) -> int:
        assert self.dim % self.num_heads == 0, "dim must be divisible by num_heads"
        return self.dim // self.num_heads

    @property
    def total_in_channels(self) -> int:
        return self.in_channels + self.condition_channels + self.mask_channels


@dataclass
class AudioConfig:
    """Audio branch of the dual-branch DiT.

    ``[1.5/2.0]`` establishes that a second branch exists and is fused by
    attention. The tokenizer split below (waveform + spectral) is ``[inferred]``
    from secondary sources and is implemented as a configurable choice, not as
    a claim about the real model.
    """

    enabled: bool = False
    sample_rate: int = 48000
    channels: int = 2  # [2.0] binaural / dual-channel output
    n_mels: int = 128
    hop_length: int = 480  # 100 Hz frame rate at 48 kHz
    n_fft: int = 2048
    dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 8
    num_layers: int = 16
    latent_channels: int = 32
    waveform_patch: int = 480
    # [2.0] multi-track: background audio / ambient SFX / character narration
    tracks: Tuple[str, ...] = ("music", "ambience", "voice")
    tokenizer: str = "hybrid"  # "mel" | "waveform" | "hybrid"


@dataclass
class TextEncoderConfig:
    """[1.0] a fine-tuned decoder-only LLM is used as the text encoder."""

    # Any HF causal LM works; hidden states from ``layer_index`` are used.
    model_id: str = "Qwen/Qwen2.5-1.5B"
    layer_index: int = -2
    max_length: int = 512
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    # [1.0] separate prompt-engineering model, initialised from Qwen2.5-14B.
    prompt_engineer_model_id: str = "Qwen/Qwen2.5-14B-Instruct"
    # fall back to a deterministic template rewriter when weights are absent.
    prompt_engineer_offline_ok: bool = True


@dataclass
class FlowConfig:
    """Rectified-flow / flow-matching schedule.

    ``[1.5/2.0]`` states the backbone is MMDiT + rectified flow (citing SD3).
    Timestep shifting and the CFG configuration are ``[inferred]`` from SD3
    conventions -- they are not disclosed for Seedance.
    """

    num_train_timesteps: int = 1000
    # SD3-style resolution-dependent shift
    shift: float = 5.0
    dynamic_shift: bool = True
    base_shift: float = 0.5
    max_shift: float = 1.15
    base_seq_len: int = 256
    max_seq_len: int = 4096
    # logit-normal timestep sampling (SD3)
    timestep_sampling: str = "logit_normal"  # "uniform" | "logit_normal"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    guidance_scale: float = 5.0
    guidance_rescale: float = 0.0
    cfg_zero_init_steps: int = 0  # CFG-Zero* style
    solver: str = "unipc"  # "unipc" | "dpm++" | "euler" | "heun"
    num_inference_steps: int = 50


@dataclass
class RefinerConfig:
    """[1.0] cascaded HR generation: 480p base -> learned diffusion refiner."""

    enabled: bool = True
    base_resolution: Tuple[int, int] = (480, 832)
    target_resolution: Tuple[int, int] = (720, 1280)
    num_steps: int = 15
    # refiner is initialised from the base model and conditioned on the
    # channel-concatenated upsampled low-res video.
    init_from_base: bool = True
    noise_strength: float = 0.5


@dataclass
class SeedanceConfig:
    """Top-level configuration for a Seedance-shaped model."""

    name: str = "seedance-dev"
    dit: DiTConfig = field(default_factory=DiTConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    refiner: RefinerConfig = field(default_factory=RefinerConfig)
    fps: int = 24
    # [2.0] 4-15 s duration, native 480p/720p
    min_duration_s: float = 4.0
    max_duration_s: float = 15.0
    # [2.0] reference budget: up to 3 videos / 9 images / 3 audio clips
    max_ref_videos: int = 3
    max_ref_images: int = 9
    max_ref_audio: int = 3
    notes: str = ""

    def __post_init__(self) -> None:
        # keep the DiT input contract consistent with the VAE latent width
        if self.dit.in_channels != self.vae.latent_channels:
            self.dit.in_channels = self.vae.latent_channels
            self.dit.out_channels = self.vae.latent_channels
            self.dit.condition_channels = self.vae.latent_channels

    # ---------------------------------------------------------------- helpers
    def latent_shape(
        self, num_frames: int, height: int, width: int
    ) -> Tuple[int, int, int, int]:
        """Return ``(C, T, H, W)`` in latent space for a pixel-space request."""
        rt, rh, rw = self.vae.stride
        if (num_frames - 1) % rt != 0:
            raise ValueError(
                f"num_frames must satisfy (n-1) % {rt} == 0, got {num_frames}"
            )
        if height % rh or width % rw:
            raise ValueError(f"height/width must be multiples of {rh}")
        return (
            self.vae.latent_channels,
            (num_frames - 1) // rt + 1,
            height // rh,
            width // rw,
        )

    def frames_for_duration(self, seconds: float) -> int:
        rt = self.vae.temporal_downsample
        n = int(round(seconds * self.fps))
        return ((n - 1) // rt) * rt + 1

    def replace(self, **kwargs: Any) -> "SeedanceConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeedanceConfig":
        sub = {
            "dit": DiTConfig,
            "vae": VAEConfig,
            "audio": AudioConfig,
            "text_encoder": TextEncoderConfig,
            "flow": FlowConfig,
            "refiner": RefinerConfig,
        }
        kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            if key in sub and isinstance(value, dict):
                fields = {f for f in sub[key].__dataclass_fields__}
                clean = {k: _retuple(v) for k, v in value.items() if k in fields}
                kwargs[key] = sub[key](**clean)
            elif key in cls.__dataclass_fields__:
                kwargs[key] = _retuple(value)
        return cls(**kwargs)

    @classmethod
    def from_json_file(cls, path: str) -> "SeedanceConfig":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def param_estimate(self) -> Dict[str, float]:
        """Rough parameter counts (millions), for capacity planning."""
        d, f = self.dit.dim, self.dit.ffn_dim
        # attention (qkvo) + ffn per layer; spatial layers carry duplicated
        # modality-specific weights for the text stream (MMDiT).
        per_temporal = 4 * d * d + 2 * d * f
        per_spatial = per_temporal + (4 * d * d + 2 * d * f)
        total = (
            per_spatial * self.dit.num_spatial_layers
            + per_temporal * self.dit.num_temporal_layers
        )
        audio = 0
        if self.audio.enabled:
            ad, af = self.audio.dim, self.audio.ffn_dim
            audio = (4 * ad * ad + 2 * ad * af) * self.audio.num_layers
        return {
            "dit_m": round(total / 1e6, 1),
            "audio_branch_m": round(audio / 1e6, 1),
            "total_m": round((total + audio) / 1e6, 1),
        }


def _retuple(value: Any) -> Any:
    """JSON round-trips tuples to lists; restore lists of scalars to tuples."""
    if isinstance(value, list) and all(not isinstance(v, (dict, list)) for v in value):
        return tuple(value)
    return value


# --------------------------------------------------------------------- variants

def _dev_tiny() -> SeedanceConfig:
    """CPU-runnable smoke-test size. Not a Seedance claim -- a test fixture."""
    return SeedanceConfig(
        name="seedance-dev-tiny",
        dit=DiTConfig(
            dim=128,
            ffn_dim=256,
            num_heads=4,
            num_spatial_layers=2,
            num_temporal_layers=2,
            text_dim=256,
            text_len=32,
            temporal_window=(2, 2),
            in_channels=8,
            out_channels=8,
            condition_channels=8,
        ),
        vae=VAEConfig(
            latent_channels=8,
            base_channels=16,
            channel_multipliers=(1, 2, 2, 2),
            temporal_downsample_stages=(False, True, True, False),
            spatial_downsample=8,
            num_res_blocks=1,
        ),
        text_encoder=TextEncoderConfig(model_id="", max_length=32),
        flow=FlowConfig(num_inference_steps=4, solver="euler"),
        refiner=RefinerConfig(enabled=False),
        notes="tiny fixture for tests; random init",
    )


def _seedance_1_0() -> SeedanceConfig:
    return SeedanceConfig(
        name="seedance-1.0",
        dit=DiTConfig(
            dim=3072,
            ffn_dim=12288,
            num_heads=24,
            num_spatial_layers=24,
            num_temporal_layers=24,
        ),
        vae=VAEConfig(),
        audio=AudioConfig(enabled=False),
        flow=FlowConfig(num_inference_steps=50),
        refiner=RefinerConfig(enabled=True),
        notes=(
            "Layer widths/counts are NOT disclosed in arXiv:2506.09113; the "
            "shapes here are inferred. VAE ratios (4,16,16)/C=48, the decoupled "
            "spatial/temporal MMDiT split, MM-RoPE, the unified task "
            "formulation, the Thin decoder and the refiner ARE from the report."
        ),
    )


def _seedance_1_5_pro() -> SeedanceConfig:
    cfg = _seedance_1_0()
    cfg.name = "seedance-1.5-pro"
    cfg.audio = AudioConfig(enabled=True)
    cfg.notes = (
        "Dual-branch DiT with a cross-modal joint module on an MMDiT/rectified-"
        "flow backbone is stated in arXiv:2512.13507; every internal dimension "
        "here is inferred."
    )
    return cfg


def _seedance_2_0() -> SeedanceConfig:
    cfg = _seedance_1_5_pro()
    cfg.name = "seedance-2.0"
    cfg.dit = DiTConfig(
        dim=3584,
        ffn_dim=14336,
        num_heads=28,
        num_spatial_layers=28,
        num_temporal_layers=28,
        max_shots=12,
    )
    cfg.audio = AudioConfig(enabled=True, tracks=("music", "ambience", "voice"))
    cfg.max_duration_s = 15.0
    cfg.notes = (
        "arXiv:2604.14148 is a capability/evaluation report with no model-design "
        "section. Duration (4-15 s), native 480p/720p, multi-track binaural "
        "audio and the 3 video / 9 image / 3 audio reference budget are product-"
        "level facts; all layer shapes are inferred."
    )
    return cfg


VARIANTS = {
    "dev-tiny": _dev_tiny,
    "seedance-1.0": _seedance_1_0,
    "seedance-1.5-pro": _seedance_1_5_pro,
    "seedance-2.0": _seedance_2_0,
}


def get_config(name: str = "seedance-2.0") -> SeedanceConfig:
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; choose from {sorted(VARIANTS)}")
    return VARIANTS[name]()


def list_variants() -> List[str]:
    return sorted(VARIANTS)


def describe_variants() -> Sequence[Dict[str, Any]]:
    rows = []
    for name in list_variants():
        cfg = get_config(name)
        rows.append(
            {
                "name": name,
                "audio": cfg.audio.enabled,
                "vae_stride": cfg.vae.stride,
                "latent_channels": cfg.vae.latent_channels,
                "spatial_layers": cfg.dit.num_spatial_layers,
                "temporal_layers": cfg.dit.num_temporal_layers,
                **cfg.param_estimate(),
            }
        )
    return rows


def resolve_optional(value: Optional[str], default: str) -> str:
    return default if value in (None, "") else value
