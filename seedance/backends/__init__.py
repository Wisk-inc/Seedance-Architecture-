"""Pluggable video-generation backends."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from .base import (
    BackendCapabilities,
    GenerationRequest,
    VideoBackend,
    download_model,
    parse_size,
)
from .diffusers_backend import DiffusersWanBackend
from .mock import MockBackend
from .registry import (
    ALIASES,
    MODELS,
    DetectedLayout,
    ModelSpec,
    detect_layout,
    list_models,
    recommended_size,
    resolve,
)
from .wan_backend import WanBackend

__all__ = [
    "GenerationRequest",
    "BackendCapabilities",
    "VideoBackend",
    "WanBackend",
    "DiffusersWanBackend",
    "MockBackend",
    "auto_backend",
    "download_model",
    "parse_size",
    "resolve",
    "detect_layout",
    "list_models",
    "recommended_size",
    "MODELS",
    "ALIASES",
    "ModelSpec",
    "DetectedLayout",
]

logger = logging.getLogger(__name__)


def auto_backend(
    model: str,
    *,
    prefer: str = "auto",
    download: bool = True,
    **kwargs: Any,
) -> VideoBackend:
    """Return a loaded-capable backend appropriate to ``model``.

    The checkpoint is downloaded (or located) and inspected first, so the
    choice is based on the files that are actually there rather than on the
    repo name. ``prefer`` may be ``"native"`` or ``"diffusers"`` to override.
    """
    if model in ("mock", "mock:") or model.startswith("mock:"):
        # escape hatch for smoke tests: no download, no GPU, no weights
        return MockBackend(**_filter(kwargs, MockBackend))
    if prefer == "native":
        return WanBackend(model, download=download, **_filter(kwargs, WanBackend))
    if prefer == "diffusers":
        return DiffusersWanBackend(model, download=download, **_filter(kwargs, DiffusersWanBackend))

    probe = WanBackend(model, download=download, **_filter(kwargs, WanBackend))
    path = probe.ensure_weights()
    layout = probe.layout
    assert layout is not None
    if layout.loader == "diffusers":
        logger.info("%s is a diffusers checkpoint; using DiffusersWanBackend", model)
        return DiffusersWanBackend(
            model,
            checkpoint_dir=path,
            download=False,
            **_filter(kwargs, DiffusersWanBackend),
        )

    # A native checkpoint still needs CUDA; fall back rather than crash later.
    try:
        import torch

        if not torch.cuda.is_available():
            logger.warning(
                "no CUDA device: the native Wan loader cannot run. Falling back "
                "to the diffusers backend, which will be extremely slow on CPU."
            )
            return DiffusersWanBackend(
                model, checkpoint_dir=path, download=False, device="cpu",
                **_filter(kwargs, DiffusersWanBackend, exclude={"device"}),
            )
    except ImportError:
        pass
    return probe


def _filter(kwargs: dict, cls: type, exclude: Optional[set] = None) -> dict:
    """Keep only the kwargs a backend's ``__init__`` accepts."""
    import inspect

    allowed = set(inspect.signature(cls.__init__).parameters) - {"self", "model"}
    if exclude:
        allowed -= exclude
    return {k: v for k, v in kwargs.items() if k in allowed}
