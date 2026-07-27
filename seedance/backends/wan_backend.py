"""Wan backend: download any Wan checkpoint from Hugging Face and run it.

This is the runnable half of the package. The Seedance modules elsewhere are a
faithful reference implementation with no public weights; Wan is the closest
open-weights analog (3D causal VAE, DiT, flow matching, T2V/I2V/FLF2V/VACE), so
the staged pipeline drives *it* and gets real video out.

What this backend adds over calling ``wan`` directly:

* **Any repo, not just the seven known ones.** The checkpoint is downloaded,
  inspected by :func:`~seedance.backends.registry.detect_layout`, and the Wan
  config is reconstructed from the files actually present -- so community
  fine-tunes and re-uploads with different file names still load.
* **Honest capability reporting.** ``validate()`` says up front which parts of
  a request the loaded checkpoint will ignore.
* **One uniform ``generate``** across t2v / i2v / flf2v / vace, returning the
  same ``[C, T, H, W]`` tensor in ``[-1, 1]``.
"""

from __future__ import annotations

import copy
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import (
    BackendCapabilities,
    GenerationRequest,
    VideoBackend,
    download_model,
    parse_size,
)
from .registry import DetectedLayout, ModelSpec, detect_layout, resolve

logger = logging.getLogger(__name__)

__all__ = ["WanBackend", "WAN_ALLOW_PATTERNS"]

# Everything a native Wan checkpoint needs. Kept explicit so a partial download
# is a clear error rather than a confusing missing-key traceback later.
WAN_ALLOW_PATTERNS: Tuple[str, ...] = (
    "*.json",
    "*.txt",
    "*.py",
    "*.md",
    "*.pth",
    "*.safetensors",
    "*.model",
    "google/*",
    "xlm-roberta-large/*",
)


class WanBackend(VideoBackend):
    """Runs a Wan checkpoint through its native loader."""

    name = "wan"

    def __init__(
        self,
        model: str = "Wan-AI/Wan2.1-T2V-1.3B",
        *,
        device_id: int = 0,
        cache_dir: Optional[str] = None,
        offload_model: Optional[bool] = None,
        t5_cpu: bool = False,
        t5_fsdp: bool = False,
        dit_fsdp: bool = False,
        use_usp: bool = False,
        rank: int = 0,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        download: bool = True,
    ):
        self.model_spec_str = model
        self.device_id = device_id
        self.cache_dir = cache_dir
        self.offload_model = offload_model
        self.t5_cpu = t5_cpu
        self.t5_fsdp = t5_fsdp
        self.dit_fsdp = dit_fsdp
        self.use_usp = use_usp
        self.rank = rank
        self.revision = revision
        self.token = token
        self.download = download

        self.repo_id, self.known = resolve(model)
        self.checkpoint_dir: Optional[str] = None
        self.layout: Optional[DetectedLayout] = None
        self.config = None
        self.pipeline = None

    # ------------------------------------------------------------- discovery
    def ensure_weights(self) -> str:
        """Download (or locate) the checkpoint and classify it."""
        if self.checkpoint_dir is not None:
            return self.checkpoint_dir
        if os.path.isdir(self.repo_id):
            path = self.repo_id
        elif self.download:
            logger.info("downloading %s", self.repo_id)
            path = download_model(
                self.repo_id,
                local_dir=self.cache_dir,
                allow_patterns=WAN_ALLOW_PATTERNS,
                revision=self.revision,
                token=self.token,
            )
        else:
            raise FileNotFoundError(
                f"{self.repo_id} is not a local directory and download=False"
            )
        self.checkpoint_dir = path
        self.layout = detect_layout(path, repo_id=self.repo_id)
        for warning in self.layout.warnings:
            logger.warning("%s: %s", self.repo_id, warning)
        logger.info("detected %s", self.layout.summary())
        return path

    def describe(self) -> Dict[str, Any]:
        """Metadata about the resolved checkpoint, safe to call before load."""
        self.ensure_weights()
        assert self.layout is not None
        return {
            "repo_id": self.repo_id,
            "path": self.checkpoint_dir,
            "task": self.layout.task,
            "family": self.layout.family,
            "loader": self.layout.loader,
            "dim": self.layout.dim,
            "num_layers": self.layout.num_layers,
            "moe": self.layout.moe,
            "quantization": self.layout.quantization,
            "known_alias": self.known.alias if self.known else None,
            "warnings": self.layout.warnings,
        }

    # ----------------------------------------------------------------- config
    def _build_config(self):
        """Reconstruct a Wan config for the detected checkpoint.

        Uses ``wan.configs`` when it imports, and a mirrored table otherwise, so
        inspection works on a machine without the full Wan dependency stack.
        """
        from .wan_config import base_config

        assert self.layout is not None
        layout = self.layout
        dim = layout.dim or (1536 if "1.3b" in self.repo_id.lower() else 5120)
        small = dim <= 2048

        if layout.task == "i2v":
            key = "i2v-14B"
        elif layout.task == "flf2v":
            key = "flf2v-14B"
        else:
            key = "t2v-1.3B" if small else "t2v-14B"
        cfg = base_config(key)

        # Point the config at the files this repo actually ships.
        if layout.t5_checkpoint:
            cfg.t5_checkpoint = layout.t5_checkpoint
        if layout.vae_checkpoint:
            cfg.vae_checkpoint = layout.vae_checkpoint
        if layout.clip_checkpoint:
            cfg.clip_checkpoint = layout.clip_checkpoint
        # tokenizers ship as subdirectories of the repo
        for attr, folder in (("t5_tokenizer", "google/umt5-xxl"), ("clip_tokenizer", "xlm-roberta-large")):
            value = getattr(cfg, attr, None)
            if value and not os.path.isdir(os.path.join(self.checkpoint_dir or "", value)):
                fallback = os.path.join(self.checkpoint_dir or "", os.path.basename(folder))
                if os.path.isdir(fallback):
                    setattr(cfg, attr, os.path.basename(folder))

        # Wan 2.2's VAE compresses 4x more aggressively in space.
        if layout.family == "wan2.2" and layout.task == "ti2v":
            cfg.vae_stride = (4, 16, 16)
        return cfg

    def _missing_files(self) -> List[str]:
        assert self.layout is not None and self.checkpoint_dir is not None
        cfg = self.config
        missing = []
        for attr in ("t5_checkpoint", "vae_checkpoint", "clip_checkpoint"):
            name = getattr(cfg, attr, None)
            if not name:
                continue
            if attr == "clip_checkpoint" and self.layout.task not in ("i2v", "flf2v"):
                continue
            if not os.path.exists(os.path.join(self.checkpoint_dir, name)):
                missing.append(name)
        return missing

    # ------------------------------------------------------------------ load
    def load(self) -> "WanBackend":
        if self.pipeline is not None:
            return self
        import torch

        self.ensure_weights()
        assert self.layout is not None
        if self.layout.loader == "diffusers":
            raise RuntimeError(
                f"{self.repo_id} ships a diffusers-format checkpoint "
                f"({self.layout.summary()}). Use DiffusersWanBackend instead -- "
                "seedance.backends.auto_backend() picks the right one for you."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "the native Wan loader requires CUDA (it places modules on "
                "cuda:N unconditionally). Use a GPU host, or the diffusers "
                "backend with device='cpu' for a very slow smoke test."
            )

        self.config = self._build_config()
        missing = self._missing_files()
        if missing:
            raise FileNotFoundError(
                f"{self.repo_id} is missing required weight files: {missing}. "
                "Re-download without allow_patterns filtering, or pass a "
                "complete local directory."
            )

        import wan

        task = self.layout.task
        common = dict(
            config=self.config,
            checkpoint_dir=self.checkpoint_dir,
            device_id=self.device_id,
            rank=self.rank,
            t5_fsdp=self.t5_fsdp,
            dit_fsdp=self.dit_fsdp,
            use_usp=self.use_usp,
            t5_cpu=self.t5_cpu,
        )
        logger.info("loading %s as task=%s", self.repo_id, task)
        if task in ("t2v", "t2i"):
            self.pipeline = wan.WanT2V(**common)
        elif task == "i2v":
            self.pipeline = wan.WanI2V(**common)
        elif task == "flf2v":
            self.pipeline = wan.WanFLF2V(**common)
        elif task == "vace":
            self.pipeline = wan.WanVace(**common)
        else:
            raise ValueError(f"unsupported task {task!r} for the native Wan loader")
        return self

    def unload(self) -> None:
        if self.pipeline is None:
            return
        import gc

        import torch

        self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------------------------------------- capabilities
    def capabilities(self) -> BackendCapabilities:
        task = self.layout.task if self.layout else (self.known.task if self.known else "t2v")
        sizes = self.known.sizes if self.known else ()
        if task == "vace":
            tasks = ("vace", "t2v", "v2v", "ref2v")
        elif task == "i2v":
            tasks = ("i2v",)
        elif task == "flf2v":
            tasks = ("flf2v", "i2v")
        else:
            tasks = ("t2v", "t2i")
        return BackendCapabilities(
            tasks=tasks,
            max_frames=81,
            sizes=sizes,
            audio=False,
            reference_images=(task == "vace"),
            control_video=(task == "vace"),
            lora=False,
            multi_gpu=True,
            notes=(
                "Wan2.1 has no audio branch; use the staged pipeline's audio "
                "stage to attach a separately generated track."
            ),
        )

    def defaults(self) -> Dict[str, Any]:
        """Sampling defaults appropriate to the resolved checkpoint."""
        if self.known:
            return {
                "size": self.known.default_size,
                "steps": self.known.default_steps,
                "shift": self.known.default_shift,
                "guidance_scale": self.known.default_guidance,
                "fps": self.known.default_fps,
                "num_frames": self.known.default_frames,
            }
        task = self.layout.task if self.layout else "t2v"
        small = (self.layout.dim or 5120) <= 2048 if self.layout else False
        # An unrecognised repo still names its training resolution most of the
        # time; running a 480p checkpoint at 720p is off-distribution, so trust
        # the hint over the parameter count.
        hint = (self.repo_id or "").lower()
        if "480" in hint:
            size = "832*480"
        elif "720" in hint or "1080" in hint:
            size = "1280*720"
        else:
            size = "832*480" if small else "1280*720"
        return {
            "size": size,
            "steps": 40 if task == "i2v" else 50,
            "shift": 16.0 if task == "flf2v" else (3.0 if small else 5.0),
            "guidance_scale": 5.5 if task == "flf2v" else 5.0,
            "fps": 16,
            "num_frames": 81,
        }

    # -------------------------------------------------------------- generate
    def generate(self, request: GenerationRequest):
        """Run the checkpoint. Returns ``[C, T, H, W]`` in ``[-1, 1]``."""
        if self.pipeline is None:
            self.load()
        assert self.layout is not None and self.pipeline is not None

        for problem in self.validate(request):
            logger.warning("%s", problem)

        width, height = parse_size(request.size)
        frames = request.normalized_frames(4)
        offload = self.offload_model if self.offload_model is not None else request.offload_model
        task = self.layout.task
        seed = request.seed

        common = dict(
            shift=request.shift,
            sample_solver=request.solver,
            sampling_steps=request.steps,
            guide_scale=request.guidance_scale,
            n_prompt=request.negative_prompt,
            seed=seed,
            offload_model=offload,
        )

        if task in ("t2v", "t2i"):
            video = self.pipeline.generate(
                request.prompt, size=(width, height), frame_num=frames, **common
            )
        elif task == "i2v":
            if request.image is None:
                raise ValueError("this checkpoint is I2V: request.image is required")
            video = self.pipeline.generate(
                request.prompt,
                _as_pil(request.image),
                max_area=width * height,
                frame_num=frames,
                **common,
            )
        elif task == "flf2v":
            if request.image is None or request.last_image is None:
                raise ValueError(
                    "this checkpoint is FLF2V: both request.image and "
                    "request.last_image are required"
                )
            video = self.pipeline.generate(
                request.prompt,
                _as_pil(request.image),
                _as_pil(request.last_image),
                max_area=width * height,
                frame_num=frames,
                **common,
            )
        elif task == "vace":
            video = self._generate_vace(request, width, height, frames, common)
        else:
            raise ValueError(f"unsupported task {task!r}")
        return video

    def _generate_vace(
        self,
        request: GenerationRequest,
        width: int,
        height: int,
        frames: int,
        common: Dict[str, Any],
    ):
        import torch

        device = torch.device(f"cuda:{self.device_id}")
        # VACE consumes a control video, an optional mask and reference images.
        # The staged pipeline's motion/physics stages write exactly this.
        src_video = [request.control_video or request.source_video]
        src_mask = [request.source_mask]
        refs = [_as_pil(r) for r in request.reference_images] or None
        src_ref_images = [refs]

        frames_t, masks_t, ref_t = self.pipeline.prepare_source(
            src_video,
            src_mask,
            src_ref_images,
            frames,
            (height, width),
            device,
        )
        return self.pipeline.generate(
            request.prompt,
            frames_t,
            masks_t,
            ref_t,
            size=(width, height),
            frame_num=frames,
            context_scale=request.control_scale,
            **common,
        )


def _as_pil(image: Any):
    """Accept a path, a PIL image or a tensor and return a PIL RGB image."""
    if image is None:
        return None
    if hasattr(image, "convert") and hasattr(image, "size"):
        return image.convert("RGB")
    if isinstance(image, (str, os.PathLike)):
        from PIL import Image

        return Image.open(image).convert("RGB")
    try:
        import numpy as np
        import torch
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise TypeError(f"cannot convert {type(image)} to PIL") from exc
    if isinstance(image, torch.Tensor):
        arr = image.detach().float().cpu()
        if arr.dim() == 4:
            arr = arr[:, 0] if arr.shape[1] > 3 else arr[0]
        if arr.shape[0] in (1, 3):
            arr = arr.permute(1, 2, 0)
        if arr.min() < 0:
            arr = (arr + 1) / 2
        return Image.fromarray((arr.clamp(0, 1).numpy() * 255).astype(np.uint8)).convert("RGB")
    raise TypeError(f"unsupported image type {type(image)}")
