"""VRAM planning and memory-saving switches.

The point of this module is to answer, *before* a 20-minute download and a
crash at step 3: will this model, at this resolution, fit on this GPU -- and if
not, what is the cheapest thing to change?

The estimates are deliberately coarse (order-of-magnitude, as the source
material's own numbers are) and err on the pessimistic side.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["DeviceInfo", "device_info", "VRAMPlan", "plan_vram", "apply_memory_plan", "bytes_per_param"]


@dataclass
class DeviceInfo:
    kind: str  # "cuda" | "mps" | "cpu"
    name: str = "cpu"
    total_vram_gb: float = 0.0
    free_vram_gb: float = 0.0
    count: int = 0
    system_ram_gb: float = 0.0

    @property
    def usable_vram_gb(self) -> float:
        # leave headroom for fragmentation and the CUDA context
        return max(0.0, self.free_vram_gb - 1.5)


def device_info(device_id: int = 0) -> DeviceInfo:
    info = DeviceInfo(kind="cpu")
    try:
        info.system_ram_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        pass
    try:
        import torch
    except ImportError:
        return info
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device_id)
        free, total = torch.cuda.mem_get_info(device_id)
        return DeviceInfo(
            kind="cuda",
            name=props.name,
            total_vram_gb=round(total / 1e9, 1),
            free_vram_gb=round(free / 1e9, 1),
            count=torch.cuda.device_count(),
            system_ram_gb=info.system_ram_gb,
        )
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return DeviceInfo(kind="mps", name="mps", count=1, system_ram_gb=info.system_ram_gb)
    return info


def bytes_per_param(dtype: str) -> float:
    return {"float32": 4.0, "fp32": 4.0, "bfloat16": 2.0, "bf16": 2.0,
            "float16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0,
            "gguf_q4": 0.6, "gguf_q3": 0.5}.get(dtype, 2.0)


@dataclass
class VRAMPlan:
    fits: bool
    estimated_gb: float
    available_gb: float
    dtype: str
    offload_model: bool = False
    t5_cpu: bool = False
    sequential_offload: bool = False
    vae_tiling: bool = False
    suggested_size: Optional[str] = None
    suggested_frames: Optional[int] = None
    notes: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []

    def summary(self) -> str:
        head = "fits" if self.fits else "DOES NOT FIT"
        return (
            f"{head}: ~{self.estimated_gb:.1f} GB needed vs {self.available_gb:.1f} GB "
            f"available ({self.dtype})"
        )


def _param_count(params: str) -> float:
    """'1.3B' / '14B' / '27B MoE (14B active)' -> active parameter count."""
    import re

    matches = re.findall(r"(\d+(?:\.\d+)?)\s*B", params)
    if not matches:
        return 14e9
    # for MoE strings the *last* number is the active count
    return float(matches[-1]) * 1e9


def plan_vram(
    *,
    params: str = "14B",
    dtype: str = "bfloat16",
    width: int = 832,
    height: int = 480,
    num_frames: int = 81,
    device_id: int = 0,
    latent_channels: int = 16,
    vae_stride: Tuple[int, int, int] = (4, 8, 8),
    text_encoder_gb: float = 11.0,
    available_gb: Optional[float] = None,
) -> VRAMPlan:
    """Estimate peak VRAM and pick memory switches to make it fit."""
    info = device_info(device_id)
    avail = available_gb if available_gb is not None else info.usable_vram_gb

    weights_gb = _param_count(params) * bytes_per_param(dtype) / 1e9
    rt, rh, rw = vae_stride
    latent_frames = (num_frames - 1) // rt + 1
    tokens = latent_frames * (height // rh) * (width // rw)
    # activations: attention working set scales with tokens; the constant is
    # fitted loosely to observed 14B numbers (~10 GB at 480p/81f in bf16).
    act_gb = tokens * 1.6e-5 * (bytes_per_param(dtype) / 2)
    vae_gb = 0.5 + num_frames * height * width * 3 * 4 / 1e9 * 2
    total = weights_gb + act_gb + vae_gb + text_encoder_gb

    plan = VRAMPlan(
        fits=total <= avail,
        estimated_gb=round(total, 1),
        available_gb=round(avail, 1),
        dtype=dtype,
    )
    if plan.fits:
        return plan

    # Cheapest fixes first, each re-checked against the budget.
    plan.t5_cpu = True
    total -= text_encoder_gb
    plan.notes.append("moved the text encoder to CPU (-%.1f GB)" % text_encoder_gb)
    if total <= avail:
        plan.fits, plan.estimated_gb = True, round(total, 1)
        return plan

    plan.vae_tiling = True
    total -= vae_gb * 0.6
    plan.notes.append("enabled VAE tiling")
    if total <= avail:
        plan.fits, plan.estimated_gb = True, round(total, 1)
        return plan

    plan.offload_model = True
    total -= weights_gb * 0.35
    plan.notes.append("enabled model offload between denoise and decode")
    if total <= avail:
        plan.fits, plan.estimated_gb = True, round(total, 1)
        return plan

    if dtype in ("bfloat16", "float16", "bf16", "fp16"):
        saved = weights_gb * 0.5
        weights_gb -= saved
        total -= saved
        plan.dtype = "fp8"
        plan.notes.append("switched weights to fp8 (quality cost is small, VRAM halves)")
        if total <= avail:
            plan.fits, plan.estimated_gb = True, round(total, 1)
            return plan

    if height * width > 480 * 832:
        scale = (480 * 832) / (height * width)
        total -= act_gb * (1 - scale)
        plan.suggested_size = "832*480"
        plan.notes.append("dropped to 480p")
        if total <= avail:
            plan.fits, plan.estimated_gb = True, round(total, 1)
            return plan

    if num_frames > 49:
        total -= act_gb * (1 - 49 / num_frames) * 0.5
        plan.suggested_frames = 49
        plan.notes.append("shortened the clip to 49 frames")

    plan.sequential_offload = True
    plan.notes.append(
        "enabled sequential CPU offload -- this is slow; if it still does not "
        "fit, use a smaller checkpoint (t2v-1.3B) or GGUF Q4 weights"
    )
    plan.estimated_gb = round(total, 1)
    plan.fits = total <= avail
    return plan


def apply_memory_plan(plan: VRAMPlan, request, backend_kwargs: Dict) -> Dict:
    """Fold a plan's decisions into backend kwargs and a request, in place."""
    backend_kwargs = dict(backend_kwargs)
    if plan.t5_cpu:
        backend_kwargs["t5_cpu"] = True
    if plan.offload_model:
        backend_kwargs["offload_model"] = True
        request.offload_model = True
    if plan.sequential_offload:
        backend_kwargs["sequential_offload"] = True
    if plan.vae_tiling:
        backend_kwargs["vae_tiling"] = True
    if plan.dtype == "fp8":
        backend_kwargs["dtype"] = "float8_e4m3fn"
    if plan.suggested_size:
        request.size = plan.suggested_size
    if plan.suggested_frames:
        request.num_frames = plan.suggested_frames
    return backend_kwargs
