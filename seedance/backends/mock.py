"""A backend that generates without weights.

Useful for three things: exercising the staged pipeline end to end with no
download, testing new stages, and reproducing a bug report without a GPU. It
renders a deterministic moving-gradient pattern keyed to the prompt hash, so
output is stable across runs and different prompts look different.

It is not a video model. Nothing it produces means anything.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict

from .base import BackendCapabilities, GenerationRequest, VideoBackend, parse_size

__all__ = ["MockBackend"]


class MockBackend(VideoBackend):
    name = "mock"

    def __init__(self, *, tasks=("t2v", "i2v", "flf2v", "vace", "v2v", "ref2v"), audio: bool = False, **_: Any):
        # extra kwargs are swallowed so `auto_backend('mock', device_id=..., ...)` works
        self._tasks = tuple(tasks)
        self._audio = audio
        self.loaded = False
        self.calls: list = []

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            tasks=self._tasks,
            max_frames=241,
            sizes=(),
            audio=self._audio,
            reference_images="vace" in self._tasks,
            control_video="vace" in self._tasks,
            lora=True,
            notes="synthetic pattern generator; produces no meaningful video",
        )

    def defaults(self) -> Dict[str, Any]:
        return {
            "size": "128*128",
            "steps": 4,
            "shift": 5.0,
            "guidance_scale": 5.0,
            "fps": 16,
            "num_frames": 17,
        }

    def load(self) -> "MockBackend":
        self.loaded = True
        return self

    def unload(self) -> None:
        self.loaded = False

    def generate(self, request: GenerationRequest):
        import torch

        self.load()
        self.calls.append(request)
        width, height = parse_size(request.size)
        frames = request.normalized_frames(4)
        seed = int(hashlib.blake2b(request.prompt.encode(), digest_size=4).hexdigest(), 16)
        if request.seed >= 0:
            seed ^= request.seed
        generator = torch.Generator().manual_seed(seed % (2**31))

        t = torch.linspace(0, 1, frames).view(1, -1, 1, 1)
        ys = torch.linspace(-1, 1, height).view(1, 1, -1, 1)
        xs = torch.linspace(-1, 1, width).view(1, 1, 1, -1)
        phase = (seed % 360) * math.pi / 180
        base = torch.sin(3 * xs + 6 * t + phase) * torch.cos(3 * ys - 4 * t)
        colour = torch.tensor([1.0, 0.6, 0.35]).view(3, 1, 1, 1)
        video = (base * colour).expand(3, frames, height, width).clone()
        video += 0.05 * torch.randn(video.shape, generator=generator)
        return video.clamp(-1, 1)
