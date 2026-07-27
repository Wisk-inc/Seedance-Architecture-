"""Attention kernels, window partitioning and the MMDiT joint-attention block.

Backend dispatch order: flash-attn 3 -> flash-attn 2 -> PyTorch SDPA. SDPA is
always available, so nothing here hard-requires a compiled kernel; the flash
paths are opportunistic speedups.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm

__all__ = [
    "attention",
    "available_backends",
    "window_partition",
    "window_unpartition",
    "SelfAttention",
    "JointAttention",
    "CrossAttention",
]

_FLASH_2 = None
_FLASH_3 = None


def available_backends() -> Tuple[str, ...]:
    """Report which attention implementations can be used in this process."""
    global _FLASH_2, _FLASH_3
    if _FLASH_2 is None:
        try:  # pragma: no cover - depends on the host
            import flash_attn  # noqa: F401

            _FLASH_2 = True
        except Exception:
            _FLASH_2 = False
    if _FLASH_3 is None:
        try:  # pragma: no cover
            import flash_attn_interface  # noqa: F401

            _FLASH_3 = True
        except Exception:
            _FLASH_3 = False
    out = ["sdpa"]
    if _FLASH_2:
        out.append("flash_attn_2")
    if _FLASH_3:
        out.append("flash_attn_3")
    return tuple(out)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    backend: str = "auto",
) -> torch.Tensor:
    """Multi-head attention on ``[B, L, H, D]`` tensors, returning ``[B, L, H, D]``.

    The ``[B, L, H, D]`` layout (rather than ``[B, H, L, D]``) matches flash-attn
    and avoids a transpose on the hot path when flash is present.
    """
    if backend == "auto":
        backends = available_backends()
        backend = "flash_attn_3" if "flash_attn_3" in backends else (
            "flash_attn_2" if "flash_attn_2" in backends else "sdpa"
        )

    if backend.startswith("flash") and attn_mask is None and q.dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        try:  # pragma: no cover - requires GPU kernels
            if backend == "flash_attn_3":
                from flash_attn_interface import flash_attn_func
            else:
                from flash_attn import flash_attn_func
            return flash_attn_func(q, k, v, dropout_p=dropout_p, causal=causal)
        except Exception:
            pass  # fall through to SDPA

    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=causal,
    )
    return out.transpose(1, 2)


# --------------------------------------------------------------- windowing

def window_partition(
    x: torch.Tensor, window: Tuple[int, int]
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """Partition ``[B, T, H, W, C]`` into ``[B * nW, T * wh * ww, C]`` windows.

    ``[1.0]``: temporal layers apply window partitioning *within each frame* so
    that attention keeps a global receptive field along time while staying
    local in space. Zero padding is added when H/W are not multiples of the
    window; ``window_unpartition`` removes it.
    """
    b, t, h, w, c = x.shape
    wh, ww = window
    wh, ww = min(wh, h) or 1, min(ww, w) or 1
    pad_h = (wh - h % wh) % wh
    pad_w = (ww - w % ww) % ww
    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    hp, wp = h + pad_h, w + pad_w
    x = x.view(b, t, hp // wh, wh, wp // ww, ww, c)
    # -> [B, nH, nW, T, wh, ww, C]
    x = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous()
    x = x.view(b * (hp // wh) * (wp // ww), t * wh * ww, c)
    return x, (hp, wp, wh, ww)


def window_unpartition(
    x: torch.Tensor,
    shape: Tuple[int, int, int, int],
    b: int,
    t: int,
    h: int,
    w: int,
) -> torch.Tensor:
    """Inverse of :func:`window_partition`, cropping any padding away."""
    hp, wp, wh, ww = shape
    nh, nw = hp // wh, wp // ww
    c = x.shape[-1]
    x = x.view(b, nh, nw, t, wh, ww, c)
    x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(b, t, hp, wp, c)
    return x[:, :, :h, :w]


# ------------------------------------------------------------------- blocks

class SelfAttention(nn.Module):
    """Single-stream self-attention with optional Q/K normalisation.

    ``[1.0]``: "Q and K embeddings are normalized before attention to prevent
    training instability."
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(self.head_dim, eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(self.head_dim, eps) if qk_norm else nn.Identity()

    def qkv_split(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, l, _ = x.shape
        qkv = self.qkv(x).view(b, l, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        return self.norm_q(q), self.norm_k(k), v

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[object] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, k, v = self.qkv_split(x)
        if rope is not None:
            q, k = rope(q), rope(k)
        out = attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(out.flatten(2))


class JointAttention(nn.Module):
    """MMDiT joint attention over concatenated visual and text tokens.

    ``[1.0]``: "a multi-modality self-attention layer is applied exclusively in
    spatial layers to integrate visual and textual tokens" with "separate
    weights (adaptive layer norm, QKV projection, MLP) for the two modalities".
    Concretely: each stream owns its QKV/output projections, the streams are
    concatenated for one attention op, then split back.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        text_dim: Optional[int] = None,
        qk_norm: bool = True,
        eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()
        text_dim = text_dim or dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.qkv_visual = nn.Linear(dim, dim * 3)
        self.qkv_text = nn.Linear(text_dim, dim * 3)
        self.proj_visual = nn.Linear(dim, dim)
        self.proj_text = nn.Linear(dim, text_dim)

        self.norm_q_v = RMSNorm(self.head_dim, eps) if qk_norm else nn.Identity()
        self.norm_k_v = RMSNorm(self.head_dim, eps) if qk_norm else nn.Identity()
        self.norm_q_t = RMSNorm(self.head_dim, eps) if qk_norm else nn.Identity()
        self.norm_k_t = RMSNorm(self.head_dim, eps) if qk_norm else nn.Identity()

    def _split(
        self, proj: nn.Linear, x: torch.Tensor, nq: nn.Module, nk: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, l, _ = x.shape
        qkv = proj(x).view(b, l, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        return nq(q), nk(k), v

    def forward(
        self,
        visual: torch.Tensor,
        text: torch.Tensor,
        visual_rope: Optional[object] = None,
        text_rope: Optional[object] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        qv, kv, vv = self._split(self.qkv_visual, visual, self.norm_q_v, self.norm_k_v)
        qt, kt, vt = self._split(self.qkv_text, text, self.norm_q_t, self.norm_k_t)
        if visual_rope is not None:
            qv, kv = visual_rope(qv), visual_rope(kv)
        if text_rope is not None:
            qt, kt = text_rope(qt), text_rope(kt)

        lv = visual.shape[1]
        q = torch.cat([qv, qt], dim=1)
        k = torch.cat([kv, kt], dim=1)
        v = torch.cat([vv, vt], dim=1)

        attn_mask = None
        if text_mask is not None:
            # text_mask: [B, Lt] with True for real tokens.
            total = k.shape[1]
            keep = torch.ones(
                text_mask.shape[0], total, dtype=torch.bool, device=text_mask.device
            )
            keep[:, lv:] = text_mask
            attn_mask = keep[:, None, None, :]

        out = attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        out_v, out_t = out[:, :lv], out[:, lv:]
        return self.proj_visual(out_v.flatten(2)), self.proj_text(out_t.flatten(2))


class CrossAttention(nn.Module):
    """Cross-attention used by the audio<->video bridge and by adapters."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kv_dim: Optional[int] = None,
        qk_norm: bool = True,
        eps: float = 1e-6,
        gated: bool = False,
    ):
        super().__init__()
        kv_dim = kv_dim or dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(kv_dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(self.head_dim, eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(self.head_dim, eps) if qk_norm else nn.Identity()
        # zero-initialised gate: the bridge starts as a no-op, so adding it to a
        # pretrained single-modality backbone does not perturb it.
        self.gate = nn.Parameter(torch.zeros(1)) if gated else None

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, l, _ = x.shape
        q = self.norm_q(self.q(x).view(b, l, self.num_heads, self.head_dim))
        kv = self.kv(context).view(b, context.shape[1], 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k = self.norm_k(k)
        out = attention(q, k, v, attn_mask=attn_mask)
        out = self.proj(out.flatten(2))
        if self.gate is not None:
            out = out * torch.tanh(self.gate)
        return out
