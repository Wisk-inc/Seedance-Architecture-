"""Diffusers backend for Wan checkpoints published in diffusers format.

Wan 2.2 (MoE), the 5B TI2V model and later releases ship as diffusers
repositories (``model_index.json`` + subfolders), which the native Wan loader
in this tree cannot read. This backend covers them, and also serves as the
forward-compatible path: a Wan release that appears after this code was
written will still load here as long as diffusers has a pipeline class for it.

It is also the only path that will run (very slowly) on CPU, which makes it the
one usable smoke test on a machine without a GPU.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import BackendCapabilities, GenerationRequest, VideoBackend, download_model, parse_size
from .registry import DetectedLayout, ModelSpec, detect_layout, resolve

logger = logging.getLogger(__name__)

__all__ = ["DiffusersWanBackend"]


class DiffusersWanBackend(VideoBackend):
    """Runs any diffusers-format Wan pipeline."""

    name = "diffusers-wan"

    def __init__(
        self,
        model: str = "Wan-AI/Wan2.2-TI2V-5B",
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        cache_dir: Optional[str] = None,
        cpu_offload: bool = True,
        sequential_offload: bool = False,
        vae_tiling: bool = True,
        attention_slicing: bool = False,
        lora_paths: Sequence[str] = (),
        lora_scales: Sequence[float] = (),
        revision: Optional[str] = None,
        token: Optional[str] = None,
        download: bool = True,
        checkpoint_dir: Optional[str] = None,
    ):
        self.model_spec_str = model
        self.device = device
        self.dtype_str = dtype
        self.cache_dir = cache_dir
        self.cpu_offload = cpu_offload
        self.sequential_offload = sequential_offload
        self.vae_tiling = vae_tiling
        self.attention_slicing = attention_slicing
        self.lora_paths = list(lora_paths)
        self.lora_scales = list(lora_scales)
        self.revision = revision
        self.token = token
        self.download = download

        self.repo_id, self.known = resolve(model)
        self.checkpoint_dir = checkpoint_dir
        self.layout: Optional[DetectedLayout] = None
        self.pipe = None

    # ------------------------------------------------------------- discovery
    def ensure_weights(self) -> str:
        if self.checkpoint_dir and self.layout:
            return self.checkpoint_dir
        if self.checkpoint_dir is None:
            if os.path.isdir(self.repo_id):
                self.checkpoint_dir = self.repo_id
            elif self.download:
                self.checkpoint_dir = download_model(
                    self.repo_id,
                    local_dir=self.cache_dir,
                    revision=self.revision,
                    token=self.token,
                )
            else:
                raise FileNotFoundError(f"{self.repo_id} not found locally and download=False")
        self.layout = detect_layout(self.checkpoint_dir, repo_id=self.repo_id)
        for warning in self.layout.warnings:
            logger.warning("%s: %s", self.repo_id, warning)
        return self.checkpoint_dir

    # ------------------------------------------------------------------ load
    def load(self) -> "DiffusersWanBackend":
        if self.pipe is not None:
            return self
        import torch

        self.ensure_weights()
        assert self.layout is not None
        dtype = getattr(torch, self.dtype_str, torch.bfloat16)
        if self.device == "cpu":
            dtype = torch.float32  # bf16 on CPU is slower than it looks

        pipeline_cls, kwargs = self._pipeline_class()
        logger.info("loading %s with %s", self.repo_id, pipeline_cls.__name__)
        self.pipe = pipeline_cls.from_pretrained(
            self.checkpoint_dir, torch_dtype=dtype, **kwargs
        )

        for path, scale in zip(self.lora_paths, self.lora_scales or [1.0] * len(self.lora_paths)):
            logger.info("loading LoRA %s (scale %.2f)", path, scale)
            self.pipe.load_lora_weights(path)
            try:
                self.pipe.fuse_lora(lora_scale=scale)
            except Exception:  # some pipelines do not implement fuse
                logger.warning("could not fuse LoRA %s; it stays applied unfused", path)

        if self.vae_tiling and hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_tiling"):
            self.pipe.vae.enable_tiling()
        if self.attention_slicing and hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()

        if self.device.startswith("cuda") and self.sequential_offload:
            self.pipe.enable_sequential_cpu_offload()
        elif self.device.startswith("cuda") and self.cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)
        return self

    def _pipeline_class(self):
        """Pick the diffusers pipeline class for this checkpoint."""
        import diffusers

        assert self.layout is not None
        kwargs: Dict[str, Any] = {}

        # A float32 VAE is the documented Wan recommendation; bf16 latents
        # decode with visible banding.
        try:
            import torch
            from diffusers import AutoencoderKLWan

            kwargs["vae"] = AutoencoderKLWan.from_pretrained(
                self.checkpoint_dir, subfolder="vae", torch_dtype=torch.float32
            )
        except Exception as exc:
            logger.debug("no AutoencoderKLWan override (%s)", exc)

        index = os.path.join(self.checkpoint_dir or "", "model_index.json")
        class_name = ""
        if os.path.exists(index):
            import json

            try:
                with open(index, "r", encoding="utf-8") as handle:
                    class_name = json.load(handle).get("_class_name", "")
            except Exception:
                class_name = ""

        if class_name and hasattr(diffusers, class_name):
            return getattr(diffusers, class_name), kwargs

        task = self.layout.task
        candidates = {
            "i2v": ("WanImageToVideoPipeline", "WanPipeline"),
            "ti2v": ("WanImageToVideoPipeline", "WanPipeline"),
            "flf2v": ("WanImageToVideoPipeline", "WanPipeline"),
            "vace": ("WanVACEPipeline", "WanPipeline"),
        }.get(task, ("WanPipeline",))
        for name in candidates:
            if hasattr(diffusers, name):
                return getattr(diffusers, name), kwargs
        raise RuntimeError(
            f"installed diffusers ({diffusers.__version__}) has no Wan pipeline "
            f"class for task {task!r}; upgrade diffusers"
        )

    def unload(self) -> None:
        if self.pipe is None:
            return
        import gc

        import torch

        self.pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------------------------------------- capabilities
    def capabilities(self) -> BackendCapabilities:
        task = self.layout.task if self.layout else (self.known.task if self.known else "t2v")
        tasks = {
            "i2v": ("i2v",),
            "ti2v": ("t2v", "i2v"),
            "flf2v": ("flf2v", "i2v"),
            "vace": ("vace", "t2v", "v2v", "ref2v"),
        }.get(task, ("t2v", "t2i"))
        return BackendCapabilities(
            tasks=tasks,
            max_frames=121,
            sizes=self.known.sizes if self.known else (),
            audio=False,
            reference_images=(task == "vace"),
            control_video=(task == "vace"),
            lora=True,
            multi_gpu=False,
            notes="diffusers pipeline; supports LoRA, CPU offload and VAE tiling",
        )

    def defaults(self) -> Dict[str, Any]:
        if self.known:
            return {
                "size": self.known.default_size,
                "steps": self.known.default_steps,
                "shift": self.known.default_shift,
                "guidance_scale": self.known.default_guidance,
                "fps": self.known.default_fps,
                "num_frames": self.known.default_frames,
            }
        return {
            "size": "832*480",
            "steps": 40,
            "shift": 5.0,
            "guidance_scale": 5.0,
            "fps": 16,
            "num_frames": 81,
        }

    # -------------------------------------------------------------- generate
    def generate(self, request: GenerationRequest):
        import torch

        if self.pipe is None:
            self.load()
        assert self.pipe is not None
        for problem in self.validate(request):
            logger.warning("%s", problem)

        self._set_shift(request.shift)
        width, height = parse_size(request.size)
        generator = None
        if request.seed >= 0:
            generator = torch.Generator(device="cpu").manual_seed(request.seed)

        kwargs: Dict[str, Any] = dict(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            height=height,
            width=width,
            num_frames=request.normalized_frames(4),
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
        )
        if request.image is not None and self.capabilities().supports("i2v"):
            from .wan_backend import _as_pil

            kwargs["image"] = _as_pil(request.image)
            if request.last_image is not None:
                kwargs["last_image"] = _as_pil(request.last_image)

        result = self.pipe(**kwargs)
        frames = result.frames[0]
        return _frames_to_tensor(frames)

    def _set_shift(self, shift: float) -> None:
        """Apply the flow-matching shift to the scheduler, if it takes one."""
        try:
            from diffusers import UniPCMultistepScheduler

            config = dict(self.pipe.scheduler.config)
            if "flow_shift" in config or "shift" in config:
                self.pipe.scheduler = UniPCMultistepScheduler.from_config(
                    self.pipe.scheduler.config, flow_shift=shift
                )
        except Exception as exc:
            logger.debug("could not set flow shift: %s", exc)


def _frames_to_tensor(frames):
    """PIL frame list (or array) -> ``[C, T, H, W]`` tensor in ``[-1, 1]``."""
    import numpy as np
    import torch

    if torch.is_tensor(frames):
        video = frames.float()
        if video.dim() == 4 and video.shape[-1] in (1, 3):  # T,H,W,C
            video = video.permute(3, 0, 1, 2)
        if video.max() > 1.5:
            video = video / 255.0
        return video * 2 - 1 if video.min() >= 0 else video
    arr = np.stack([np.asarray(f.convert("RGB"), dtype=np.float32) / 255.0 for f in frames])
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2)  # C,T,H,W
    return tensor * 2 - 1
