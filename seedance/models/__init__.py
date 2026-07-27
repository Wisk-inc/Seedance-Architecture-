"""Seedance model components.

Import order matters only in that everything here needs torch; the package root
(:mod:`seedance`) is deliberately torch-free so config/registry code can be
used without an ML stack installed.
"""

from .attention import (
    CrossAttention,
    JointAttention,
    SelfAttention,
    attention,
    available_backends,
    window_partition,
    window_unpartition,
)
from .audio import (
    AudioBranchDiT,
    AudioTokenizer,
    CrossModalJointModule,
    GriffinLimVocoder,
    alignment_mask,
    build_audio_bridge,
)
from .conditioning import (
    ConditionTensors,
    Reference,
    ReferenceBundle,
    Shot,
    Storyboard,
    build_condition,
    parse_mentions,
)
from .dit import SeedanceDiT, SpatialMMDiTBlock, TemporalDiTBlock, layer_schedule
from .norm import AdaLNModulation, FP32LayerNorm, RMSNorm, TimestepEmbedding
from .refiner import CascadedRefiner, upsample_latents
from .rope import MMRoPE, RoPE, axial_split, shot_time_positions
from .text_encoder import (
    DecoderOnlyLLMTextEncoder,
    HashingTextEncoder,
    TextEncoderOutput,
    build_text_encoder,
)
from .vae3d import (
    CausalConv3d,
    DiagonalGaussian,
    HybridPatchDiscriminator,
    TemporallyCausalVAE,
)

__all__ = [
    "attention",
    "available_backends",
    "window_partition",
    "window_unpartition",
    "SelfAttention",
    "JointAttention",
    "CrossAttention",
    "RMSNorm",
    "FP32LayerNorm",
    "AdaLNModulation",
    "TimestepEmbedding",
    "MMRoPE",
    "RoPE",
    "axial_split",
    "shot_time_positions",
    "SeedanceDiT",
    "SpatialMMDiTBlock",
    "TemporalDiTBlock",
    "layer_schedule",
    "ConditionTensors",
    "build_condition",
    "Reference",
    "ReferenceBundle",
    "parse_mentions",
    "Shot",
    "Storyboard",
    "TemporallyCausalVAE",
    "CausalConv3d",
    "DiagonalGaussian",
    "HybridPatchDiscriminator",
    "AudioTokenizer",
    "AudioBranchDiT",
    "CrossModalJointModule",
    "GriffinLimVocoder",
    "build_audio_bridge",
    "alignment_mask",
    "DecoderOnlyLLMTextEncoder",
    "HashingTextEncoder",
    "TextEncoderOutput",
    "build_text_encoder",
    "CascadedRefiner",
    "upsample_latents",
]
