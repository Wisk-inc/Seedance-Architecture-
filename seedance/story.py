"""Storyboards and multimodal references.

Deliberately torch-free: the CLI, the registry and the staged pipeline all need
these types, and none of them should pull in an ML stack to parse a prompt.
:mod:`seedance.models.conditioning` re-exports everything here.

The ``@mention`` / ``[Image1]`` reference syntax is ``[product-level]``: Seedance
2.0 exposes it through its API (up to 3 video / 9 image / 3 audio references,
with subject / motion / style / effect roles), but no paper describes the
token-injection mechanism. This is our own scheme with the same surface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

__all__ = [
    "Reference",
    "ReferenceBundle",
    "parse_mentions",
    "Shot",
    "Storyboard",
]

REFERENCE_KINDS = ("image", "video", "audio")
REFERENCE_ROLES = ("subject", "motion", "style", "effect", "audio", "scene")


@dataclass
class Reference:
    """One multimodal reference slot."""

    kind: str
    handle: str  # "@hero", "[Image1]" ...
    payload: object = None  # PIL.Image / path / tensor / waveform
    role: str = "subject"
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in REFERENCE_KINDS:
            raise ValueError(f"unknown reference kind {self.kind!r}")
        if self.role not in REFERENCE_ROLES:
            raise ValueError(f"unknown reference role {self.role!r}")


@dataclass
class ReferenceBundle:
    """A validated set of references, budgeted like the 2.0 product limits."""

    references: List[Reference] = field(default_factory=list)
    max_images: int = 9
    max_videos: int = 3
    max_audio: int = 3

    def add(self, ref: Reference) -> "ReferenceBundle":
        counts = self.counts()
        limits = {"image": self.max_images, "video": self.max_videos, "audio": self.max_audio}
        if counts.get(ref.kind, 0) >= limits[ref.kind]:
            raise ValueError(
                f"reference budget exceeded: at most {limits[ref.kind]} {ref.kind} refs"
            )
        if any(r.handle == ref.handle for r in self.references):
            raise ValueError(f"duplicate reference handle {ref.handle!r}")
        self.references.append(ref)
        return self

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for ref in self.references:
            out[ref.kind] = out.get(ref.kind, 0) + 1
        return out

    def by_handle(self) -> Dict[str, Reference]:
        return {r.handle: r for r in self.references}

    def of_kind(self, kind: str) -> List[Reference]:
        return [r for r in self.references if r.kind == kind]

    def of_role(self, role: str) -> List[Reference]:
        return [r for r in self.references if r.role == role]

    def resolve(self, prompt: str) -> Tuple[str, List[Reference]]:
        """Strip reference handles from ``prompt``; return the text and refs used.

        Each handle becomes a plain noun phrase, so a text encoder that has
        never seen the syntax still receives a grammatical sentence.
        """
        table = self.by_handle()
        used: List[Reference] = []
        out = prompt
        for handle in sorted(table, key=len, reverse=True):
            if handle in out:
                ref = table[handle]
                used.append(ref)
                out = out.replace(handle, _noun_for(ref))
        return re.sub(r"\s+", " ", out).strip(), used


def _noun_for(ref: Reference) -> str:
    label = ref.handle.lstrip("@").strip("[]")
    return {
        "subject": f"the referenced subject ({label})",
        "style": f"in the referenced visual style ({label})",
        "motion": f"following the referenced motion ({label})",
        "effect": f"with the referenced visual effect ({label})",
        "scene": f"in the referenced scene ({label})",
        "audio": f"with the referenced audio ({label})",
    }[ref.role]


_MENTION_RE = re.compile(r"(@[A-Za-z0-9_\-]+|\[(?:Image|Video|Audio)\d+\])")


def parse_mentions(prompt: str) -> List[str]:
    """Extract ``@name`` / ``[Image1]`` handles from a prompt, in order."""
    seen: List[str] = []
    for match in _MENTION_RE.findall(prompt):
        if match not in seen:
            seen.append(match)
    return seen


@dataclass
class Shot:
    """One shot of a multi-shot sequence, with its own dense caption.

    ``[1.0]``: shots are organised in temporal order of the action and each
    shot carries its own detailed caption.
    """

    caption: str
    num_frames: int
    camera: str = ""
    references: List[str] = field(default_factory=list)  # handles into a bundle

    def __post_init__(self) -> None:
        if self.num_frames <= 0:
            raise ValueError("shot num_frames must be positive")


@dataclass
class Storyboard:
    """An ordered list of shots plus the global prompt."""

    prompt: str
    shots: List[Shot] = field(default_factory=list)
    max_shots: int = 8

    def add(self, shot: Shot) -> "Storyboard":
        if len(self.shots) >= self.max_shots:
            raise ValueError(f"at most {self.max_shots} shots are supported")
        self.shots.append(shot)
        return self

    @property
    def total_frames(self) -> int:
        return sum(s.num_frames for s in self.shots)

    def lengths(self) -> List[int]:
        return [s.num_frames for s in self.shots]

    def captions(self) -> List[str]:
        return [s.caption for s in self.shots]

    @classmethod
    def single(cls, prompt: str, num_frames: int) -> "Storyboard":
        return cls(prompt=prompt, shots=[Shot(caption=prompt, num_frames=num_frames)])

    @classmethod
    def from_lines(cls, prompt: str, lines: Iterable[str], frames_per_shot: int) -> "Storyboard":
        """Build a storyboard from one caption per line."""
        board = cls(prompt=prompt)
        for line in lines:
            line = line.strip()
            if line:
                board.add(Shot(caption=line, num_frames=frames_per_shot))
        if not board.shots:
            board.add(Shot(caption=prompt, num_frames=frames_per_shot))
        return board
