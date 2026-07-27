"""Multi-shot generation by chaining.

``[1.0]`` handles multiple shots *natively*: shots are ordered segments inside
one position-encoded sequence, each with its own caption, and MM-RoPE keeps
them separable. Open weights have no such path, so shots are generated one at a
time and stitched.

Chaining strategies, strongest first:

* ``flf2v`` -- the previous shot's last frame becomes the next shot's first
  frame *and* a target last frame is supplied, pinning both boundaries.
* ``i2v``   -- the previous shot's last frame seeds the next shot. This is the
  workhorse: available on every I2V checkpoint and it holds identity well
  within a continuous action.
* ``anchor``-- for a hard cut, the previous frame is not passed at all; identity
  is carried by the reference images and the locked caption instead.

Colour continuity across the seam is handled explicitly: each new shot is
histogram-matched to the previous shot's tail before concatenation, because
independent sampling runs drift apart even when the conditioning is identical.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = ["ShotResult", "ShotChainer", "crossfade", "concat_shots"]


@dataclass
class ShotResult:
    index: int
    caption: str
    video: Any = None
    mode: str = "t2v"
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)


def crossfade(a, b, frames: int = 4):
    """Blend the tail of ``a`` into the head of ``b`` over ``frames``.

    Only sensible for a continuous action; on a hard cut a crossfade reads as a
    dissolve transition, which is usually not what was asked for.
    """
    import torch

    if frames <= 0:
        return torch.cat([a, b], dim=1)
    frames = min(frames, a.shape[1], b.shape[1])
    weights = torch.linspace(0, 1, frames, device=a.device).view(1, -1, 1, 1)
    tail = a[:, -frames:]
    head = b[:, :frames]
    blended = tail * (1 - weights) + head * weights
    return torch.cat([a[:, :-frames], blended, b[:, frames:]], dim=1)


def concat_shots(videos: Sequence, *, transition_frames: int = 0, color_match: bool = True):
    """Join shots into one clip, optionally colour-matching and crossfading."""
    import torch

    from .polish import match_colors

    if not videos:
        raise ValueError("no shots to concatenate")
    out = videos[0]
    for nxt in videos[1:]:
        if color_match:
            # Match the incoming shot to the outgoing shot's final frame.
            joined = torch.cat([out[:, -1:], nxt], dim=1)
            joined = match_colors(joined, reference_index=0, strength=0.8)
            nxt = joined[:, 1:]
        out = crossfade(out, nxt, transition_frames) if transition_frames else torch.cat(
            [out, nxt], dim=1
        )
    return out


class ShotChainer:
    """Generates a storyboard shot by shot and stitches the result."""

    def __init__(
        self,
        generate_fn: Callable[..., Any],
        *,
        mode: str = "auto",  # auto | flf2v | i2v | anchor
        transition_frames: int = 0,
        color_match: bool = True,
        anchor_offset: int = -1,
    ):
        """
        Args:
            generate_fn: called as ``generate_fn(prompt, image=..., last_image=...,
                num_frames=..., shot_index=...)`` and returns ``[C, T, H, W]``.
            anchor_offset: which frame of the previous shot seeds the next one.
                ``-1`` is the last frame; ``-3`` or so can help when the final
                frame has motion blur.
        """
        self.generate_fn = generate_fn
        self.mode = mode
        self.transition_frames = transition_frames
        self.color_match = color_match
        self.anchor_offset = anchor_offset

    def _mode_for(self, index: int, capabilities) -> str:
        if index == 0:
            return "t2v"
        if self.mode != "auto":
            return self.mode
        tasks = getattr(capabilities, "tasks", ("t2v",))
        if "flf2v" in tasks:
            return "flf2v"
        if "i2v" in tasks:
            return "i2v"
        return "anchor"

    def run(
        self,
        storyboard,
        *,
        capabilities=None,
        fps: int = 16,
        last_frames: Optional[Sequence[Any]] = None,
        progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """Generate every shot and return the stitched clip plus a report."""
        from .identity import IdentityController

        shots = storyboard.shots
        results: List[ShotResult] = []
        videos: List[Any] = []
        previous = None

        for i, shot in enumerate(shots):
            mode = self._mode_for(i, capabilities)
            kwargs: Dict[str, Any] = {
                "num_frames": shot.num_frames,
                "shot_index": i,
            }
            if mode in ("i2v", "flf2v") and previous is not None:
                kwargs["image"] = IdentityController.anchor_frame(previous, self.anchor_offset)
            if mode == "flf2v" and last_frames and i < len(last_frames) and last_frames[i]:
                kwargs["last_image"] = last_frames[i]
            elif mode == "flf2v":
                # No explicit target frame: fall back to I2V rather than
                # inventing one, which would pin the shot to a guess.
                mode = "i2v"
            kwargs["task"] = mode if mode != "anchor" else "t2v"

            if progress:
                progress(i + 1, len(shots), shot.caption)
            video = self.generate_fn(shot.caption, **kwargs)
            videos.append(video)
            previous = video
            results.append(
                ShotResult(
                    index=i,
                    caption=shot.caption,
                    video=video,
                    mode=mode,
                    seconds=shot.num_frames / max(fps, 1),
                )
            )

        stitched = concat_shots(
            videos, transition_frames=self.transition_frames, color_match=self.color_match
        )
        return {
            "video": stitched,
            "shots": results,
            "total_frames": int(stitched.shape[1]),
            "seconds": float(stitched.shape[1]) / max(fps, 1),
        }
