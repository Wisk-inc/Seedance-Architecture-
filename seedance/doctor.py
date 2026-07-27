"""Environment diagnostics.

``python -m seedance.cli doctor`` answers, in one screen: can this machine run
a Wan model, how big a one, and which optional stages will actually engage
rather than silently falling back.
"""

from __future__ import annotations

import importlib
import platform
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Check", "run_checks", "format_report"]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    severity: str = "required"  # required | recommended | optional
    fix: str = ""

    @property
    def symbol(self) -> str:
        if self.ok:
            return "ok  "
        return {"required": "FAIL", "recommended": "warn", "optional": "--  "}[self.severity]


def _version(module: str) -> Optional[str]:
    try:
        mod = importlib.import_module(module)
    except Exception:
        return None
    return str(getattr(mod, "__version__", "installed"))


def run_checks() -> List[Check]:
    checks: List[Check] = []

    py_ok = sys.version_info >= (3, 10)
    checks.append(
        Check(
            "python >= 3.10",
            py_ok,
            platform.python_version(),
            fix="Wan and diffusers both require 3.10+",
        )
    )

    torch_version = _version("torch")
    checks.append(
        Check("torch", torch_version is not None, torch_version or "not installed",
              fix="pip install torch>=2.4")
    )

    cuda_detail, cuda_ok = "no CUDA device", False
    if torch_version:
        import torch

        if torch.cuda.is_available():
            cuda_ok = True
            free, total = torch.cuda.mem_get_info(0)
            cuda_detail = (
                f"{torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}, "
                f"{total / 1e9:.1f} GB total / {free / 1e9:.1f} GB free, "
                f"CUDA {torch.version.cuda}"
            )
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            cuda_detail = "MPS available (the native Wan loader still needs CUDA)"
    checks.append(
        Check(
            "CUDA GPU",
            cuda_ok,
            cuda_detail,
            severity="recommended",
            fix="the native Wan loader requires CUDA; without it only the diffusers backend runs (very slowly)",
        )
    )

    for module, severity, fix in (
        ("huggingface_hub", "required", "pip install huggingface_hub"),
        ("transformers", "required", "pip install transformers>=4.49"),
        ("diffusers", "recommended", "pip install diffusers>=0.31 (needed for Wan2.2+ checkpoints)"),
        ("accelerate", "recommended", "pip install accelerate"),
        ("imageio", "required", "pip install imageio imageio-ffmpeg"),
        ("einops", "optional", "pip install einops"),
        ("flash_attn", "optional", "pip install flash_attn (falls back to PyTorch SDPA)"),
        ("PIL", "required", "pip install pillow"),
        ("cv2", "recommended", "pip install opencv-python (flow interpolation, Farneback)"),
    ):
        version = _version(module)
        checks.append(
            Check(module, version is not None, version or "not installed",
                  severity=severity, fix=fix)
        )

    checks.append(
        Check(
            "wan package",
            _version("wan") is not None or importlib.util.find_spec("wan") is not None,
            "importable" if importlib.util.find_spec("wan") else "not importable",
            fix="run from the Wan2.1 repo root, or pip install -e .",
        )
    )

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    checks.append(
        Check("ffmpeg", ffmpeg is not None, ffmpeg or "not found", severity="recommended",
              fix="needed to mux audio and concatenate clips")
    )

    # optional stage dependencies -- each one that is missing means a fallback
    for module, stage in (
        ("pybullet", "stage 3 physics (rigid-body proxy)"),
        ("mujoco", "stage 3 physics (alternative engine)"),
        ("insightface", "stage 2 face crop and ArcFace identity drift"),
        ("controlnet_aux", "stage 2 pose guides (DWPose/OpenPose)"),
        ("torchvision", "stage 2 optical flow (RAFT)"),
        ("gfpgan", "stage 4 face restoration"),
    ):
        version = _version(module)
        checks.append(
            Check(module, version is not None, version or f"not installed -> {stage} falls back",
                  severity="optional", fix=f"pip install {module}")
        )

    try:
        from .models.attention import available_backends

        checks.append(
            Check("attention backends", True, ", ".join(available_backends()), severity="optional")
        )
    except Exception as exc:
        checks.append(Check("attention backends", False, str(exc), severity="optional"))

    try:
        from .flow.solvers import available_solvers

        checks.append(
            Check("flow solvers", True, ", ".join(available_solvers()), severity="optional")
        )
    except Exception as exc:
        checks.append(Check("flow solvers", False, str(exc), severity="optional"))

    try:
        from .stages.physics import available_engines

        checks.append(
            Check("physics engines", True, ", ".join(available_engines()), severity="optional")
        )
    except Exception as exc:
        checks.append(Check("physics engines", False, str(exc), severity="optional"))

    return checks


def recommend_model() -> str:
    """Suggest a checkpoint that fits this machine."""
    try:
        from .runtime.memory import device_info

        info = device_info()
    except Exception:
        return "Wan-AI/Wan2.1-T2V-1.3B (could not probe the device)"
    if info.kind != "cuda":
        return "no GPU: generation is not practical here. Use the CLI to inspect models only."
    vram = info.total_vram_gb
    if vram >= 60:
        return "Wan-AI/Wan2.1-T2V-14B or Wan-AI/Wan2.2-T2V-A14B at 720p"
    if vram >= 40:
        return "Wan-AI/Wan2.1-T2V-14B at 480p (fp8 for 720p)"
    if vram >= 20:
        return "Wan-AI/Wan2.2-TI2V-5B, or Wan2.1-T2V-1.3B at 480p"
    if vram >= 10:
        return "Wan-AI/Wan2.1-T2V-1.3B at 480p"
    return "Wan-AI/Wan2.1-T2V-1.3B at 480p with offloading, or GGUF Q4 weights"


def format_report(checks: Optional[List[Check]] = None) -> str:
    checks = checks or run_checks()
    width = max(len(c.name) for c in checks) + 2
    lines = ["seedance doctor", "=" * 60]
    for check in checks:
        lines.append(f"[{check.symbol}] {check.name.ljust(width)} {check.detail}")
        if not check.ok and check.fix and check.severity != "optional":
            lines.append(f"         -> {check.fix}")
    failures = [c for c in checks if not c.ok and c.severity == "required"]
    lines.append("=" * 60)
    lines.append(
        "all required dependencies present"
        if not failures
        else f"{len(failures)} required dependency missing: "
        + ", ".join(c.name for c in failures)
    )
    lines.append(f"suggested model: {recommend_model()}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(format_report())
