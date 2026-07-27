"""Normalisation and modulation primitives.

Kept separate from ``attention.py`` because the VAE, the DiT and the audio
branch all share these, and because the fp32 discipline around adaptive layer
norm is the single most common source of training instability in DiTs.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

__all__ = [
    "RMSNorm",
    "FP32LayerNorm",
    "AdaLNModulation",
    "modulate",
    "gate",
    "TimestepEmbedding",
]


class RMSNorm(nn.Module):
    """Root-mean-square norm computed in fp32, cast back to the input dtype."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = out.type_as(x)
        return out * self.weight if self.weight is not None else out


class FP32LayerNorm(nn.LayerNorm):
    """LayerNorm that always accumulates in fp32."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)


def modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """``x * (1 + scale) + shift`` with broadcasting over the token axis."""
    while shift.dim() < x.dim():
        shift = shift.unsqueeze(-2)
        scale = scale.unsqueeze(-2)
    return x.float() * (1 + scale.float()) + shift.float()


def gate(g: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Reshape an adaLN gate ``[B, D]`` to broadcast against ``like``.

    Inserting the axes explicitly (rather than relying on right-aligned
    broadcasting) is what keeps batch > 1 correct: ``[B, D] * [B, T, S, D]``
    silently aligns ``B`` with ``T`` and either errors or, worse, computes
    something wrong when they happen to match.
    """
    while g.dim() < like.dim():
        g = g.unsqueeze(-2)
    return g


class AdaLNModulation(nn.Module):
    """Produces the six adaLN-zero parameters from a conditioning vector.

    ``[1.0]`` MMDiT keeps *separate* modulation weights per modality inside the
    spatial layers, so one of these is instantiated per stream.
    """

    def __init__(self, dim: int, cond_dim: int | None = None, chunks: int = 6):
        super().__init__()
        self.chunks = chunks
        self.dim = dim
        cond_dim = cond_dim or dim
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * chunks))
        # adaLN-zero: start as the identity so early training is stable.
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def forward(self, cond: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        params = self.proj(cond.float())
        return params.chunk(self.chunks, dim=-1)


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep features followed by an MLP."""

    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_embedding(t, self.freq_dim).to(self.mlp[0].weight.dtype))


def sinusoidal_embedding(position: torch.Tensor, dim: int) -> torch.Tensor:
    assert dim % 2 == 0, "sinusoidal dim must be even"
    half = dim // 2
    pos = position.to(torch.float64).flatten()
    freqs = torch.pow(
        10000, -torch.arange(half, device=pos.device, dtype=torch.float64) / half
    )
    angles = torch.outer(pos, freqs)
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=1).float()
