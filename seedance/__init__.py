"""Seedance architecture -- a reference implementation plus a runnable pipeline.

Two halves, and it matters which is which:

**The architecture** (:mod:`seedance.models`, :mod:`seedance.flow`,
:mod:`seedance.reward`) is a faithful implementation of what ByteDance actually
published about Seedance: the decoupled spatial/temporal MMDiT, the
temporally-causal (4,16,16)/C=48 VAE with its Thin decoder, multishot MM-RoPE,
the unified channel-concat task formulation, the cascaded refiner, the
dual-branch audio design, TSCD/score/adversarial distillation, and
RewardDance/DanceGRPO. Every field and class carries a provenance tag --
``[1.0]`` for the technical report, ``[1.5/2.0]`` for the later abstracts,
``[inferred]`` for defensible reconstruction. **There are no public Seedance
weights**, so this half runs but does not produce video.

**The pipeline** (:mod:`seedance.pipelines`, :mod:`seedance.backends`,
:mod:`seedance.stages`) applies the same four-stage recipe to open weights: it
downloads any Wan checkpoint from Hugging Face, works out what it is, and drives
it through prompt rewriting, identity/motion control, physics-proxy
conditioning and post-processing. This half produces video.

Quick start::

    from seedance import SeedancePipeline

    pipe = SeedancePipeline("Wan-AI/Wan2.1-T2V-1.3B")
    result = pipe.generate("a red fox trotting through snow", output="fox.mp4")
    print(result.summary())

This module imports nothing heavy; ``seedance.config``, ``seedance.story`` and
the model registry work with no ML stack installed.
"""

from __future__ import annotations

from .config import (
    AudioConfig,
    DiTConfig,
    FlowConfig,
    RefinerConfig,
    SeedanceConfig,
    TextEncoderConfig,
    VAEConfig,
    describe_variants,
    get_config,
    list_variants,
)
from .story import Reference, ReferenceBundle, Shot, Storyboard, parse_mentions
from .version import SOURCES, __version__

__all__ = [
    "__version__",
    "SOURCES",
    "SeedanceConfig",
    "DiTConfig",
    "VAEConfig",
    "AudioConfig",
    "FlowConfig",
    "RefinerConfig",
    "TextEncoderConfig",
    "get_config",
    "list_variants",
    "describe_variants",
    "Shot",
    "Storyboard",
    "Reference",
    "ReferenceBundle",
    "parse_mentions",
    # lazily resolved below
    "SeedancePipeline",
    "StageSettings",
    "PipelineResult",
    "SeedanceNativePipeline",
    "GenerationRequest",
    "auto_backend",
    "list_models",
]

_LAZY = {
    "SeedancePipeline": ("seedance.pipelines.staged", "SeedancePipeline"),
    "StageSettings": ("seedance.pipelines.staged", "StageSettings"),
    "PipelineResult": ("seedance.pipelines.base", "PipelineResult"),
    "SeedanceNativePipeline": ("seedance.pipelines.native", "SeedanceNativePipeline"),
    "GenerationRequest": ("seedance.backends.base", "GenerationRequest"),
    "auto_backend": ("seedance.backends", "auto_backend"),
    "list_models": ("seedance.backends.registry", "list_models"),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
