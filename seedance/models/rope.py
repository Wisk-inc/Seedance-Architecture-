"""Multishot MM-RoPE.

``[1.0]``: "3D RoPE for visual tokens plus extra 1D positional encoding for
textual tokens (following Seaweed and LCT); supports interleaved visual/text
sequences and multi-shot training where shots are organized in temporal order
of actions and each shot has its own detailed caption."

The implementation splits the head dimension into three axial groups (t, h, w),
exactly the split Wan/Seedream-lineage models use, and lets callers supply
arbitrary integer position grids. That generality is what makes the awkward
cases work: window-partitioned temporal layers reorder tokens, and multi-shot
sequences need per-shot temporal offsets so shot boundaries are representable
without inventing cut tokens.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

__all__ = ["axial_split", "RoPE", "MMRoPE", "shot_time_positions"]


def axial_split(head_dim: int) -> Tuple[int, int, int]:
    """Split ``head_dim`` into (t, h, w) rotary sub-dimensions.

    Each returned value counts *real* dimensions and is even, so the three sum
    to ``head_dim``. h and w get ``2 * (d // 6)`` each; t takes the remainder,
    which biases capacity toward time -- the axis with the longest range.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    d = head_dim
    dh = 2 * (d // 6)
    dw = 2 * (d // 6)
    dt = d - dh - dw
    assert dt % 2 == 0, "axial split produced an odd temporal dim"
    return dt, dh, dw


def _freqs(dim: int, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """Complex rotary factors of shape ``[L, dim // 2]``."""
    half = dim // 2
    if half == 0:
        return torch.ones(positions.numel(), 0, dtype=torch.complex64, device=positions.device)
    inv = 1.0 / torch.pow(
        theta,
        torch.arange(0, half, device=positions.device, dtype=torch.float64) / half,
    )
    angles = torch.outer(positions.to(torch.float64), inv)
    return torch.polar(torch.ones_like(angles), angles).to(torch.complex64)


class RoPE:
    """Applies precomputed rotary factors to ``[B, L, H, D]`` q/k tensors."""

    __slots__ = ("freqs",)

    def __init__(self, freqs: torch.Tensor):
        # freqs: [L, D // 2] complex
        self.freqs = freqs

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        b, l, h, d = x.shape
        f = self.freqs
        if f.shape[0] < l:
            raise ValueError(f"RoPE built for {f.shape[0]} positions, got {l}")
        f = f[:l]
        if f.shape[1] * 2 != d:
            raise ValueError(f"RoPE dim {f.shape[1] * 2} != head dim {d}")
        xc = torch.view_as_complex(x.float().reshape(b, l, h, d // 2, 2))
        out = torch.view_as_real(xc * f.to(xc.device)[None, :, None, :]).flatten(3)
        return out.type_as(x)

    def to(self, device: torch.device | str) -> "RoPE":
        return RoPE(self.freqs.to(device))


def shot_time_positions(
    shot_lengths: Sequence[int], gap: int = 0
) -> torch.Tensor:
    """Temporal positions for a multi-shot latent sequence.

    Shots are laid out in temporal order of the action; ``gap`` inserts unused
    position indices between shots so a cut is visible to the model as a
    discontinuity in the rotary phase rather than as a special token.
    """
    positions: List[int] = []
    cursor = 0
    for length in shot_lengths:
        positions.extend(range(cursor, cursor + length))
        cursor += length + gap
    return torch.tensor(positions, dtype=torch.long)


class MMRoPE(nn.Module):
    """Builds visual (3D) and text (1D) rotary factors for a request."""

    def __init__(self, head_dim: int, theta: float = 10000.0, text_offset: int = 1 << 20):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        # Text positions live in a disjoint range so text and visual tokens
        # never alias each other inside the joint attention op.
        self.text_offset = text_offset
        self.dt, self.dh, self.dw = axial_split(head_dim)

    # ------------------------------------------------------------------ visual
    def visual(
        self,
        num_frames: int,
        height: int,
        width: int,
        *,
        device: torch.device | str = "cpu",
        order: str = "thw",
        time_positions: Optional[torch.Tensor] = None,
        window: Optional[Tuple[int, int]] = None,
    ) -> RoPE:
        """Rotary factors for a visual token grid.

        Args:
            order: ``"thw"`` flattens as ``(t, h, w)`` -- the layout used by
                spatial layers and by the unpartitioned path. ``"window"``
                matches :func:`~seedance.models.attention.window_partition`,
                where tokens inside one window are ordered ``(t, wh, ww)``.
            time_positions: per-frame positions, e.g. from
                :func:`shot_time_positions` for multi-shot sequences.
            window: ``(wh, ww)`` required when ``order == "window"``.
        """
        device = torch.device(device)
        if time_positions is None:
            t_pos = torch.arange(num_frames, device=device)
        else:
            t_pos = time_positions.to(device)
            if t_pos.numel() != num_frames:
                raise ValueError("time_positions length must equal num_frames")

        if order == "thw":
            t_idx = t_pos.view(-1, 1, 1).expand(num_frames, height, width)
            h_idx = torch.arange(height, device=device).view(1, -1, 1).expand_as(t_idx)
            w_idx = torch.arange(width, device=device).view(1, 1, -1).expand_as(t_idx)
        elif order == "window":
            if window is None:
                raise ValueError("window=(wh, ww) is required for order='window'")
            wh, ww = window
            wh, ww = min(wh, height) or 1, min(ww, width) or 1
            # Positions for a single window at the origin; every window shares
            # the same relative layout, and RoPE is relative, so one grid is
            # enough regardless of which window a token came from.
            t_idx = t_pos.view(-1, 1, 1).expand(num_frames, wh, ww)
            h_idx = torch.arange(wh, device=device).view(1, -1, 1).expand_as(t_idx)
            w_idx = torch.arange(ww, device=device).view(1, 1, -1).expand_as(t_idx)
        else:
            raise ValueError(f"unknown order {order!r}")

        freqs = torch.cat(
            [
                _freqs(self.dt, t_idx.reshape(-1), self.theta),
                _freqs(self.dh, h_idx.reshape(-1), self.theta),
                _freqs(self.dw, w_idx.reshape(-1), self.theta),
            ],
            dim=-1,
        )
        return RoPE(freqs)

    # -------------------------------------------------------------------- text
    def text(
        self,
        length: int,
        *,
        device: torch.device | str = "cpu",
        shot_index: int = 0,
        shot_stride: int = 1024,
    ) -> RoPE:
        """1D rotary factors for text tokens.

        ``shot_index``/``shot_stride`` give each shot's caption its own slot in
        the text position range, which is how per-shot captions stay separable
        in an interleaved visual/text sequence.
        """
        device = torch.device(device)
        base = self.text_offset + shot_index * shot_stride
        pos = torch.arange(base, base + length, device=device)
        # Text uses the full head dim as a single 1D axis.
        return RoPE(_freqs(self.head_dim, pos, self.theta))

    def interleaved(
        self,
        shots: Sequence[Tuple[int, int, int]],
        caption_lengths: Sequence[int],
        *,
        device: torch.device | str = "cpu",
        gap: int = 0,
    ) -> Tuple[RoPE, RoPE, List[Tuple[int, int]]]:
        """Rotary factors for a full multi-shot interleaved sequence.

        Args:
            shots: one ``(num_frames, height, width)`` per shot. Height/width
                must match across shots (they share a latent canvas).
            caption_lengths: per-shot caption token counts.
        Returns:
            ``(visual_rope, text_rope, spans)`` where ``spans`` gives the
            ``(start, end)`` visual-token range of each shot so downstream code
            can apply per-shot conditioning.
        """
        if len(shots) != len(caption_lengths):
            raise ValueError("one caption length per shot is required")
        heights = {s[1] for s in shots}
        widths = {s[2] for s in shots}
        if len(heights) > 1 or len(widths) > 1:
            raise ValueError("all shots must share the same latent H/W")
        height, width = shots[0][1], shots[0][2]
        lengths = [s[0] for s in shots]
        t_pos = shot_time_positions(lengths, gap=gap)
        visual = self.visual(
            sum(lengths), height, width, device=device, time_positions=t_pos
        )

        text_freqs = []
        for i, n in enumerate(caption_lengths):
            text_freqs.append(
                self.text(n, device=device, shot_index=i).freqs
            )
        text = RoPE(torch.cat(text_freqs, dim=0)) if text_freqs else RoPE(
            torch.zeros(0, self.head_dim // 2, dtype=torch.complex64)
        )

        spans: List[Tuple[int, int]] = []
        cursor = 0
        per_frame = height * width
        for n in lengths:
            spans.append((cursor, cursor + n * per_frame))
            cursor += n * per_frame
        return visual, text, spans
