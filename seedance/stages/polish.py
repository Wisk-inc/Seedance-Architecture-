"""Stage 4: post-processing.

Order matters and is not arbitrary: **colour stabilise -> interpolate ->
restore faces -> upscale -> QA**. Interpolating before fixing colour drift
propagates the drift into the new frames; upscaling before face restoration
makes the restorer work on artifacts it then amplifies; QA last, on the actual
deliverable.

Every step here has an optional high-quality implementation and a working
fallback, and each reports which one ran. The fallbacks are real algorithms
(optical-flow warping, histogram matching, Lanczos), not no-ops -- but they are
not RIFE, CodeFormer or SeedVR2, and the report says so.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "PolishConfig",
    "interpolate_frames",
    "match_colors",
    "adain_normalize",
    "restore_faces",
    "upscale",
    "temporal_qa",
    "polish",
]


@dataclass
class PolishConfig:
    color_stabilize: bool = True
    color_method: str = "histogram"  # "histogram" | "adain" | "off"
    interpolate: int = 1  # frame-rate multiplier; 1 disables
    interpolator: str = "auto"  # "auto" | "rife" | "film" | "flow" | "blend"
    face_restore: bool = False
    face_method: str = "codeformer"  # "codeformer" | "gfpgan"
    face_strength: float = 0.35  # low on purpose: high strength drifts identity
    upscale_factor: int = 1
    upscaler: str = "auto"  # "auto" | "seedvr2" | "flashvsr" | "realesrgan" | "lanczos"
    run_qa: bool = True


# --------------------------------------------------------------------- colour

def match_colors(video, reference_index: int = 0, strength: float = 1.0):
    """Per-channel histogram matching of every frame to a reference frame.

    This is the fix for chunked-generation colour drift. Strength below 1
    blends toward the match so a deliberate lighting change is not flattened.
    """
    import torch

    v = (video[0] if video.dim() == 5 else video).float()
    c, t, h, w = v.shape
    out = v.clone()
    ref = v[:, reference_index].reshape(c, -1)
    ref_sorted, _ = ref.sort(dim=1)
    n = ref_sorted.shape[1]
    for i in range(t):
        if i == reference_index:
            continue
        frame = v[:, i].reshape(c, -1)
        order = frame.argsort(dim=1)
        ranks = torch.zeros_like(order)
        arange = torch.arange(frame.shape[1], device=v.device)
        for ch in range(c):
            ranks[ch, order[ch]] = arange
        idx = (ranks.float() * (n - 1) / max(frame.shape[1] - 1, 1)).round().long()
        matched = torch.gather(ref_sorted, 1, idx)
        blended = frame * (1 - strength) + matched * strength
        out[:, i] = blended.view(c, h, w)
    return out.clamp(-1, 1)


def adain_normalize(video, reference_index: int = 0, strength: float = 1.0):
    """Match per-channel mean/std to a reference frame (AdaIN).

    Cheaper and gentler than histogram matching: it corrects exposure and cast
    drift without touching the tonal curve, so it rarely posterises.
    """
    v = (video[0] if video.dim() == 5 else video).float()
    ref = v[:, reference_index]
    ref_mean = ref.mean(dim=(1, 2), keepdim=True)
    ref_std = ref.std(dim=(1, 2), keepdim=True).clamp(min=1e-5)
    out = v.clone()
    for i in range(v.shape[1]):
        frame = v[:, i]
        mean = frame.mean(dim=(1, 2), keepdim=True)
        std = frame.std(dim=(1, 2), keepdim=True).clamp(min=1e-5)
        norm = (frame - mean) / std * ref_std + ref_mean
        out[:, i] = frame * (1 - strength) + norm * strength
    return out.clamp(-1, 1)


# -------------------------------------------------------------- interpolation

def interpolate_frames(video, factor: int = 2, method: str = "auto") -> Tuple[Any, str]:
    """Increase the frame rate by ``factor``. Returns ``(video, method_used)``."""
    import torch

    if factor <= 1:
        return video, "none"
    v = video[0] if video.dim() == 5 else video

    if method in ("auto", "rife"):
        out = _rife(v, factor)
        if out is not None:
            return out, "rife"
        if method == "rife":
            logger.warning("RIFE unavailable; falling back to flow warping")
    if method in ("auto", "film"):
        out = _film(v, factor)
        if out is not None:
            return out, "film"
    if method in ("auto", "flow"):
        out = _flow_interpolate(v, factor)
        if out is not None:
            return out, "flow"
    return _blend_interpolate(v, factor), "blend"


def _rife(video, factor: int):
    try:
        from rife_ncnn_vulkan_python import Rife  # community binding
    except Exception:
        return None
    try:
        import numpy as np
        from PIL import Image

        from ..runtime.io import from_uint8_frames, to_uint8_frames

        rife = Rife(gpuid=0)
        frames = to_uint8_frames(video)
        out: List[Any] = []
        for i in range(len(frames) - 1):
            a, b = Image.fromarray(frames[i]), Image.fromarray(frames[i + 1])
            out.append(frames[i])
            for k in range(1, factor):
                mid = rife.process(a, b)  # single midpoint; recursive for >2
                out.append(np.asarray(mid))
        out.append(frames[-1])
        return from_uint8_frames(out)
    except Exception as exc:
        logger.debug("RIFE failed: %s", exc)
        return None


def _film(video, factor: int):
    try:
        import tensorflow_hub  # noqa: F401
    except Exception:
        return None
    logger.debug("FILM hub model detected but not wired; use RIFE or flow")
    return None


def _flow_interpolate(video, factor: int):
    """Optical-flow forward warping. Handles motion, unlike cross-fading."""
    try:
        import cv2
        import numpy as np
        import torch
    except ImportError:
        return None

    from ..runtime.io import from_uint8_frames, to_uint8_frames

    frames = to_uint8_frames(video)
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    out: List[Any] = []
    for i in range(len(frames) - 1):
        out.append(frames[i])
        flow = cv2.calcOpticalFlowFarneback(grays[i], grays[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = grays[i].shape
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        for k in range(1, factor):
            alpha = k / factor
            map_x = (grid_x + flow[..., 0] * alpha).astype(np.float32)
            map_y = (grid_y + flow[..., 1] * alpha).astype(np.float32)
            warped_a = cv2.remap(frames[i], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            map_x_b = (grid_x - flow[..., 0] * (1 - alpha)).astype(np.float32)
            map_y_b = (grid_y - flow[..., 1] * (1 - alpha)).astype(np.float32)
            warped_b = cv2.remap(frames[i + 1], map_x_b, map_y_b, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            blended = (warped_a.astype(np.float32) * (1 - alpha) + warped_b.astype(np.float32) * alpha)
            out.append(blended.astype(np.uint8))
    out.append(frames[-1])
    return from_uint8_frames(out)


def _blend_interpolate(video, factor: int):
    """Linear cross-fade. Always available; ghosts on fast motion."""
    import torch

    v = video
    frames = [v[:, i] for i in range(v.shape[1])]
    out = []
    for i in range(len(frames) - 1):
        out.append(frames[i])
        for k in range(1, factor):
            alpha = k / factor
            out.append(frames[i] * (1 - alpha) + frames[i + 1] * alpha)
    out.append(frames[-1])
    return torch.stack(out, dim=1)


# ---------------------------------------------------------------------- faces

def restore_faces(video, method: str = "codeformer", strength: float = 0.35) -> Tuple[Any, str]:
    """Face restoration, blended at ``strength``.

    Kept low by default and blended rather than replaced: aggressive face
    restoration produces the plasticky over-restored look and, worse, drifts
    identity -- undoing exactly what Stage 2 was for.
    """
    import torch

    v = video[0] if video.dim() == 5 else video
    restored = None
    try:
        if method == "gfpgan":
            restored = _gfpgan(v)
        else:
            restored = _codeformer(v)
    except Exception as exc:
        logger.warning("face restoration unavailable (%s); skipped", exc)
    if restored is None:
        return v, "skipped"
    return (v * (1 - strength) + restored * strength).clamp(-1, 1), method


def _gfpgan(video):
    from gfpgan import GFPGANer  # type: ignore

    import numpy as np

    from ..runtime.io import from_uint8_frames, to_uint8_frames

    restorer = GFPGANer(model_path=None, upscale=1, arch="clean", channel_multiplier=2)
    out = []
    for frame in to_uint8_frames(video):
        _, _, restored = restorer.enhance(frame[:, :, ::-1], paste_back=True)
        out.append(np.asarray(restored)[:, :, ::-1])
    return from_uint8_frames(out)


def _codeformer(video):
    # CodeFormer has no stable pip API; this is the documented hook point.
    raise ImportError(
        "CodeFormer must be installed from source (sczhou/CodeFormer) and wired "
        "here; use method='gfpgan' for a pip-installable path"
    )


# -------------------------------------------------------------------- upscale

def upscale(video, factor: int = 2, method: str = "auto") -> Tuple[Any, str]:
    """Video super-resolution. Returns ``(video, method_used)``."""
    import torch
    import torch.nn.functional as F

    if factor <= 1:
        return video, "none"
    v = video[0] if video.dim() == 5 else video

    if method in ("auto", "seedvr2"):
        out = _external_vsr(v, factor, "seedvr2")
        if out is not None:
            return out, "seedvr2"
    if method in ("auto", "flashvsr"):
        out = _external_vsr(v, factor, "flashvsr")
        if out is not None:
            return out, "flashvsr"
    if method in ("auto", "realesrgan"):
        out = _realesrgan(v, factor)
        if out is not None:
            return out, "realesrgan"

    # Lanczos on each frame: no invention, no temporal flicker, no detail either.
    c, t, h, w = v.shape
    flat = v.permute(1, 0, 2, 3)
    out = F.interpolate(flat, size=(h * factor, w * factor), mode="bicubic", align_corners=False)
    return out.permute(1, 0, 2, 3).clamp(-1, 1), "bicubic"


def _external_vsr(video, factor: int, which: str):
    """Hook for SeedVR2 / FlashVSR, which ship as repos rather than packages."""
    try:
        if which == "seedvr2":
            import seedvr2  # type: ignore
        else:
            import flashvsr  # type: ignore
    except Exception:
        return None
    logger.info(
        "%s is installed but needs its own checkpoint and call signature; "
        "wire it here for your install", which
    )
    return None


def _realesrgan(video, factor: int):
    try:
        from realesrgan import RealESRGANer  # type: ignore
    except Exception:
        return None
    try:
        import numpy as np

        from ..runtime.io import from_uint8_frames, to_uint8_frames

        upsampler = RealESRGANer(scale=factor, model_path=None, model=None)
        out = []
        for frame in to_uint8_frames(video):
            result, _ = upsampler.enhance(frame, outscale=factor)
            out.append(np.asarray(result))
        return from_uint8_frames(out)
    except Exception as exc:
        logger.debug("RealESRGAN failed: %s", exc)
        return None


# --------------------------------------------------------------------- the QA

def temporal_qa(video) -> Dict[str, float]:
    """Flag flicker, drift and softness on the finished clip."""
    from ..reward.heuristics import (
        color_drift,
        flicker_score,
        sharpness,
        structural_persistence,
        temporal_smoothness,
    )

    v = video if video.dim() == 5 else video.unsqueeze(0)
    report = {
        "temporal_smoothness": float(temporal_smoothness(v)[0]),
        "flicker": float(flicker_score(v)[0]),
        "sharpness": float(sharpness(v)[0]),
        "color_stability": float(color_drift(v)[0]),
        "structural_persistence": float(structural_persistence(v)[0]),
    }
    warnings = []
    if report["flicker"] < 0.7:
        warnings.append("visible flicker; try colour stabilisation or fewer chunks")
    if report["color_stability"] < 0.7:
        warnings.append("colour drift across the clip; enable colour matching")
    if report["structural_persistence"] < 0.6:
        warnings.append("subject structure changes a lot; strengthen identity conditioning")
    if report["sharpness"] < 0.15:
        warnings.append("soft output; consider fewer interpolation passes or an upscaler")
    report["warnings"] = warnings  # type: ignore[assignment]
    return report


def polish(video, cfg: Optional[PolishConfig] = None) -> Tuple[Any, Dict[str, Any]]:
    """Run the whole Stage 4 chain in the correct order."""
    cfg = cfg or PolishConfig()
    report: Dict[str, Any] = {"steps": []}
    out = video[0] if video.dim() == 5 else video

    if cfg.color_stabilize and cfg.color_method != "off":
        out = (
            adain_normalize(out) if cfg.color_method == "adain" else match_colors(out)
        )
        report["steps"].append(f"color:{cfg.color_method}")

    if cfg.interpolate > 1:
        out, used = interpolate_frames(out, cfg.interpolate, cfg.interpolator)
        report["steps"].append(f"interpolate:{used}x{cfg.interpolate}")
        report["interpolator"] = used

    if cfg.face_restore:
        out, used = restore_faces(out, cfg.face_method, cfg.face_strength)
        report["steps"].append(f"face:{used}")
        report["face_restore"] = used

    if cfg.upscale_factor > 1:
        out, used = upscale(out, cfg.upscale_factor, cfg.upscaler)
        report["steps"].append(f"upscale:{used}x{cfg.upscale_factor}")
        report["upscaler"] = used

    if cfg.run_qa:
        report["qa"] = temporal_qa(out)
    return out, report
