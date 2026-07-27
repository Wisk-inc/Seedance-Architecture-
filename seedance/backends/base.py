"""Backend interface shared by every video generator this package can drive.

A backend is anything that turns a :class:`GenerationRequest` into a pixel
tensor. The staged Seedance pipeline is written against this interface only, so
the same Stage 1-4 orchestration runs over the native Wan loader, a diffusers
pipeline, or the reference Seedance DiT without changes.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "GenerationRequest",
    "BackendCapabilities",
    "VideoBackend",
    "download_model",
    "parse_size",
]


def parse_size(size: str) -> Tuple[int, int]:
    """``"832*480"`` -> ``(832, 480)`` as ``(width, height)``.

    Accepts ``*``, ``x`` or ``X`` separators because every tool in this
    ecosystem picked a different one.
    """
    for sep in ("*", "x", "X", ","):
        if sep in size:
            w, h = size.split(sep, 1)
            return int(w.strip()), int(h.strip())
    raise ValueError(f"cannot parse size {size!r}; expected e.g. '832*480'")


@dataclass
class GenerationRequest:
    """Everything a backend needs for one clip.

    Fields a given backend cannot honour are reported through
    :meth:`VideoBackend.validate` rather than silently ignored -- silently
    dropping a control signal is the failure mode that wastes an hour of GPU
    time before anyone notices.
    """

    prompt: str
    negative_prompt: str = ""
    size: str = "832*480"
    num_frames: int = 81
    fps: int = 16
    steps: int = 50
    guidance_scale: float = 5.0
    shift: float = 5.0
    seed: int = -1
    solver: str = "unipc"
    task: str = "t2v"

    # conditioning inputs
    image: Optional[Any] = None  # PIL.Image for i2v / first frame
    last_image: Optional[Any] = None  # PIL.Image for flf2v
    reference_images: List[Any] = field(default_factory=list)  # VACE / Phantom refs
    source_video: Optional[str] = None  # path, for v2v / control
    source_mask: Optional[str] = None
    control_video: Optional[str] = None  # depth/pose/physics guide frames
    control_scale: float = 1.0
    audio: Optional[Any] = None

    # runtime
    offload_model: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return parse_size(self.size)[0]

    @property
    def height(self) -> int:
        return parse_size(self.size)[1]

    @property
    def duration_s(self) -> float:
        return self.num_frames / max(self.fps, 1)

    def with_(self, **kwargs: Any) -> "GenerationRequest":
        data = {**self.__dict__, **kwargs}
        return GenerationRequest(**data)

    def normalized_frames(self, temporal_stride: int = 4) -> int:
        """Round to the nearest valid ``1 + k * stride`` frame count."""
        n = max(1 + temporal_stride, self.num_frames)
        return ((n - 1) // temporal_stride) * temporal_stride + 1


@dataclass
class BackendCapabilities:
    tasks: Tuple[str, ...] = ("t2v",)
    max_frames: int = 81
    sizes: Tuple[str, ...] = ()
    audio: bool = False
    reference_images: bool = False
    control_video: bool = False
    lora: bool = False
    multi_gpu: bool = False
    notes: str = ""

    def supports(self, task: str) -> bool:
        return task in self.tasks


class VideoBackend(ABC):
    """Abstract video generation backend."""

    name: str = "backend"

    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """What this backend can actually do, once loaded."""

    @abstractmethod
    def load(self) -> "VideoBackend":
        """Materialise weights on the target device. Idempotent."""

    @abstractmethod
    def generate(self, request: GenerationRequest):
        """Return a video tensor ``[C, T, H, W]`` in ``[-1, 1]``."""

    def unload(self) -> None:
        """Release GPU memory. Default is a no-op."""

    def validate(self, request: GenerationRequest) -> List[str]:
        """Return human-readable problems with ``request`` for this backend."""
        caps = self.capabilities()
        problems: List[str] = []
        if not caps.supports(request.task):
            problems.append(
                f"{self.name} does not support task {request.task!r} "
                f"(supported: {', '.join(caps.tasks)})"
            )
        if caps.sizes and request.size not in caps.sizes:
            problems.append(
                f"size {request.size} is untested for {self.name} "
                f"(known-good: {', '.join(caps.sizes)})"
            )
        if request.num_frames > caps.max_frames:
            problems.append(
                f"{request.num_frames} frames exceeds this backend's tested maximum "
                f"of {caps.max_frames}"
            )
        if request.audio is not None and not caps.audio:
            problems.append(f"{self.name} cannot generate audio; the audio input is ignored")
        if request.reference_images and not caps.reference_images:
            problems.append(
                f"{self.name} ignores reference images; use a VACE/Phantom checkpoint"
            )
        if request.control_video and not caps.control_video:
            problems.append(f"{self.name} ignores control video; use a VACE checkpoint")
        return problems

    def __enter__(self) -> "VideoBackend":
        return self.load()

    def __exit__(self, *exc: object) -> None:
        self.unload()


def download_model(
    repo_id: str,
    *,
    local_dir: Optional[str] = None,
    allow_patterns: Optional[Sequence[str]] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    progress: bool = True,
) -> str:
    """Fetch a Hugging Face repo and return the local path.

    A local directory is returned untouched, so ``--model /path/to/weights``
    works the same as ``--model Wan-AI/Wan2.1-T2V-1.3B``.
    """
    if os.path.isdir(repo_id):
        return repo_id
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required to download models: pip install huggingface_hub"
        ) from exc

    return snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=list(allow_patterns) if allow_patterns else None,
        revision=revision,
        token=token or os.environ.get("HF_TOKEN"),
        tqdm_class=None if progress else _SilentTqdm,
    )


class _SilentTqdm:  # pragma: no cover - trivial
    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
