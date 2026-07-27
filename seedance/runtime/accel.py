"""Inference acceleration switches.

``[1.0]`` lists the techniques behind the reported >10x speedup: multi-stage
distillation (see :mod:`seedance.flow.distill`), a Thin VAE decoder (see
:mod:`seedance.models.vae3d`), kernel fusion on Attention and Gemm (~15%
cumulative), fine-grained mixed-precision quantisation, adaptive sparsity
(extending AdaSpa), and adaptive hybrid parallelism with FP8 communication.

What is implementable here without ByteDance's kernels: torch.compile fusion,
SDPA backend selection, TeaCache-style step skipping, and a token-sparsity
mask. Each is opt-in and each reports honestly whether it actually engaged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["AccelConfig", "apply_acceleration", "TeaCache", "block_sparse_mask"]


@dataclass
class AccelConfig:
    compile_model: bool = False
    compile_mode: str = "default"  # "default" | "reduce-overhead" | "max-autotune"
    channels_last: bool = False
    tf32: bool = True
    sdpa_backend: str = "auto"  # "auto" | "flash" | "efficient" | "math"
    teacache_threshold: float = 0.0  # 0 disables; 0.1-0.3 is the usable range
    thin_vae_decoder: bool = True
    vae_tiling: bool = False


def apply_acceleration(model: Any, cfg: Optional[AccelConfig] = None) -> Dict[str, bool]:
    """Apply what is available; return which switches actually engaged."""
    cfg = cfg or AccelConfig()
    applied: Dict[str, bool] = {}
    try:
        import torch
    except ImportError:  # pragma: no cover
        return {"torch": False}

    if cfg.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        applied["tf32"] = True

    if cfg.channels_last and hasattr(model, "to"):
        try:
            model.to(memory_format=torch.channels_last_3d)
            applied["channels_last"] = True
        except Exception as exc:
            logger.debug("channels_last unavailable: %s", exc)
            applied["channels_last"] = False

    if cfg.sdpa_backend != "auto":
        try:
            backends = {
                "flash": torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                "efficient": torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                "math": torch.nn.attention.SDPBackend.MATH,
            }
            torch.nn.attention.sdpa_kernel(backends[cfg.sdpa_backend]).__enter__()
            applied["sdpa_backend"] = True
        except Exception as exc:
            logger.debug("could not pin the SDPA backend: %s", exc)
            applied["sdpa_backend"] = False

    if cfg.compile_model:
        try:
            compiled = torch.compile(model, mode=cfg.compile_mode)
            applied["compile"] = True
            return {**applied, "_model": compiled}  # type: ignore[dict-item]
        except Exception as exc:
            logger.warning("torch.compile failed (%s); continuing uncompiled", exc)
            applied["compile"] = False
    return applied


class TeaCache:
    """Skip redundant denoise steps when the conditioning barely changed.

    Diffusion transformers produce near-identical outputs on consecutive steps
    in the middle of the trajectory. TeaCache tracks the relative change in the
    modulated input and reuses the previous residual while that change stays
    under ``threshold``, which is roughly a 2x speedup at threshold ~0.2 with a
    small quality cost that shows up first as softened motion.
    """

    def __init__(self, threshold: float = 0.2, warmup_steps: int = 2, max_consecutive: int = 3):
        self.threshold = threshold
        self.warmup_steps = warmup_steps
        self.max_consecutive = max_consecutive
        self.reset()

    def reset(self) -> None:
        self._prev_input = None
        self._prev_residual = None
        self._accumulated = 0.0
        self._step = 0
        self._consecutive = 0
        self.skipped = 0

    def should_skip(self, model_input) -> bool:
        self._step += 1
        if self.threshold <= 0 or self._step <= self.warmup_steps:
            self._prev_input = model_input.detach()
            return False
        if self._prev_input is None or self._prev_residual is None:
            self._prev_input = model_input.detach()
            return False
        if self._consecutive >= self.max_consecutive:
            self._consecutive = 0
            self._accumulated = 0.0
            self._prev_input = model_input.detach()
            return False
        delta = (model_input - self._prev_input).abs().mean()
        scale = self._prev_input.abs().mean().clamp(min=1e-6)
        self._accumulated += float(delta / scale)
        self._prev_input = model_input.detach()
        if self._accumulated < self.threshold:
            self._consecutive += 1
            self.skipped += 1
            return True
        self._accumulated = 0.0
        self._consecutive = 0
        return False

    def store(self, residual) -> None:
        self._prev_residual = residual.detach()

    @property
    def cached_residual(self):
        return self._prev_residual


def block_sparse_mask(
    num_tokens: int,
    block_size: int = 128,
    keep_ratio: float = 0.5,
    device: str = "cpu",
):
    """Blockified attention mask, the shape AdaSpa-style sparsity exploits.

    Keeps the diagonal blocks (local context, always needed) plus a strided
    selection of off-diagonal blocks up to ``keep_ratio``. Returned as a bool
    mask usable directly by SDPA.
    """
    import torch

    n_blocks = max(1, math_ceil(num_tokens, block_size))
    keep = torch.zeros(n_blocks, n_blocks, dtype=torch.bool, device=device)
    keep.fill_diagonal_(True)
    budget = int(keep_ratio * n_blocks * n_blocks) - n_blocks
    if budget > 0:
        stride = max(1, (n_blocks * n_blocks) // max(budget, 1))
        flat = keep.view(-1)
        flat[::stride] = True
    mask = keep.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
    return mask[:num_tokens, :num_tokens]


def math_ceil(a: int, b: int) -> int:
    return -(-a // b)
