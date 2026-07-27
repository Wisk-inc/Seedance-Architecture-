"""Pipelines: the staged open-weights one, and the native reference one."""

from .base import PipelineResult, StageTimer
from .staged import SeedancePipeline, StageSettings

__all__ = ["SeedancePipeline", "StageSettings", "PipelineResult", "StageTimer"]


def __getattr__(name: str):
    # The native pipeline imports torch at module import time; keep it lazy so
    # `from seedance.pipelines import SeedancePipeline` works without it.
    if name == "SeedanceNativePipeline":
        from .native import SeedanceNativePipeline

        return SeedanceNativePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
