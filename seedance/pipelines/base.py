"""Shared pipeline types."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["PipelineResult", "StageTimer"]


@dataclass
class PipelineResult:
    """What a pipeline returns: the video, where it went, and what happened."""

    video: Any = None  # [C, T, H, W] in [-1, 1]
    path: Optional[str] = None
    audio_path: Optional[str] = None
    fps: int = 16
    prompt: str = ""
    enhanced_prompt: str = ""
    seed: int = -1
    report: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return int(self.video.shape[1]) if self.video is not None else 0

    @property
    def duration_s(self) -> float:
        return self.num_frames / max(self.fps, 1)

    def summary(self) -> str:
        lines = [
            f"video: {self.num_frames} frames @ {self.fps} fps ({self.duration_s:.1f}s)",
        ]
        if self.path:
            lines.append(f"saved: {self.path}")
        if self.enhanced_prompt and self.enhanced_prompt != self.prompt:
            lines.append(f"prompt ({self.report.get('prompt_source', '?')}): {self.enhanced_prompt}")
        if self.timings:
            total = sum(self.timings.values())
            parts = ", ".join(f"{k} {v:.1f}s" for k, v in self.timings.items())
            lines.append(f"timing: total {total:.1f}s ({parts})")
        qa = self.report.get("polish", {}).get("qa") if self.report else None
        if qa:
            metrics = ", ".join(
                f"{k}={v:.2f}" for k, v in qa.items() if isinstance(v, (int, float))
            )
            lines.append(f"qa: {metrics}")
        for warning in self.warnings:
            lines.append(f"warning: {warning}")
        return "\n".join(lines)

    def to_json(self, indent: int = 2) -> str:
        payload = {
            "path": self.path,
            "audio_path": self.audio_path,
            "fps": self.fps,
            "frames": self.num_frames,
            "seed": self.seed,
            "prompt": self.prompt,
            "enhanced_prompt": self.enhanced_prompt,
            "report": self.report,
            "timings": self.timings,
            "warnings": self.warnings,
        }
        return json.dumps(payload, indent=indent, default=str)


class StageTimer:
    """Accumulates per-stage wall time into a dict."""

    def __init__(self) -> None:
        self.timings: Dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = self.timings.get(name, 0.0) + (time.perf_counter() - start)
