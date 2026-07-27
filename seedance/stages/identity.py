"""Stage 2a: identity and subject consistency.

Seedance gets identity preservation from multi-shot training with per-shot
captions plus RLHF on structural coherence -- there is **no dedicated identity
adapter** in the reports. Open weights cannot copy that, so this stage bolts on
the explicit mechanisms that do exist:

* **Reference conditioning** (VACE Ref2V, Phantom) -- the strongest option when
  the checkpoint supports it, because the reference enters the model's own
  conditioning path rather than being bolted onto attention.
* **LoRA per character** -- highest fidelity, needs training per identity.
* **IP-Adapter / PuLID / InstantID** -- image-prompt adapters. Community
  framing is "InstantID for stability, PuLID for fidelity"; both are
  SDXL-first, so on video models they are applied per-frame or through a
  ComfyUI graph rather than natively here.
* **Caption locking and anchor frames** -- free, and does more work than people
  expect across shot boundaries.

This module owns the parts that are model-agnostic: preparing references,
choosing anchor frames, and *measuring* identity drift so you know whether any
of the above helped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "IdentityConfig",
    "IdentityController",
    "identity_drift",
    "ADAPTER_NOTES",
]


ADAPTER_NOTES: Dict[str, str] = {
    "vace": (
        "Native to Wan VACE checkpoints. Pass reference images through "
        "GenerationRequest.reference_images; no extra weights needed."
    ),
    "phantom": (
        "Wan-native single/multi-subject reference (1.3B and 14B). Requires the "
        "Phantom checkpoint; not loadable from a stock Wan repo."
    ),
    "ip_adapter": (
        "Decoupled cross-attention injecting CLIP image features as a visual "
        "prompt. Robust to pose and lighting; weaker on exact facial identity."
    ),
    "pulid": (
        "EvaCLIP + face encoder with contrastive alignment and an accurate-ID "
        "loss. Highest face fidelity of the adapter family."
    ),
    "instantid": (
        "ArcFace embedding + ControlNet-style IdentityNet. SDXL-focused; "
        "usually paired with a Canny control."
    ),
    "lora": (
        "Per-character LoRA. Highest fidelity and the only option that also "
        "captures body, clothing and mannerisms; costs a training run each."
    ),
    "fantasyid": "Face-knowledge-enhanced identity preservation for video DiTs.",
    "stableanimator": (
        "Distribution-aware ID adapter (ArcFace + FaceEncoder) that resolves the "
        "conflict between temporal layers and identity injection."
    ),
    "unianimate_dit": "Wan-native pose-guided human animation on 14B-I2V.",
}


@dataclass
class IdentityConfig:
    method: str = "auto"  # auto | vace | phantom | lora | ip_adapter | pulid | caption_only
    reference_images: List[Any] = field(default_factory=list)
    subject_description: str = ""
    lora_paths: List[str] = field(default_factory=list)
    lora_scales: List[float] = field(default_factory=list)
    face_crop: bool = True
    crop_margin: float = 0.4
    anchor_from_previous_shot: bool = True
    drift_threshold: float = 0.6  # below this, warn


class IdentityController:
    """Prepares identity conditioning and measures whether it worked."""

    def __init__(self, cfg: Optional[IdentityConfig] = None):
        self.cfg = cfg or IdentityConfig()
        self._face_app = None
        self._tried_face = False

    # ---------------------------------------------------------------- method
    def choose_method(self, capabilities) -> str:
        """Pick the strongest identity mechanism the backend actually supports."""
        cfg = self.cfg
        if cfg.method != "auto":
            return cfg.method
        if cfg.lora_paths:
            return "lora"
        if cfg.reference_images and getattr(capabilities, "reference_images", False):
            return "vace"
        if cfg.reference_images:
            logger.warning(
                "reference images were given but this checkpoint has no reference "
                "path; falling back to caption locking. Use a VACE or Phantom "
                "checkpoint for real reference conditioning."
            )
        return "caption_only"

    # ------------------------------------------------------------ references
    def prepare_references(self, size: Tuple[int, int] = (512, 512)) -> List[Any]:
        """Normalise reference images: face-crop if possible, then resize."""
        from PIL import Image

        out = []
        for ref in self.cfg.reference_images:
            image = ref if hasattr(ref, "convert") else Image.open(ref)
            image = image.convert("RGB")
            if self.cfg.face_crop:
                box = self._detect_face(image)
                if box:
                    image = image.crop(box)
            out.append(image.resize(size, Image.LANCZOS))
        return out

    def _detect_face(self, image) -> Optional[Tuple[int, int, int, int]]:
        """Face box with margin, via insightface if installed; else None."""
        if not self._tried_face:
            self._tried_face = True
            try:
                from insightface.app import FaceAnalysis

                app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                app.prepare(ctx_id=-1, det_size=(640, 640))
                self._face_app = app
            except Exception as exc:
                logger.debug("insightface unavailable (%s); skipping face crop", exc)
                self._face_app = None
        if self._face_app is None:
            return None
        import numpy as np

        faces = self._face_app.get(np.asarray(image)[:, :, ::-1])
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = face.bbox
        mw, mh = (x2 - x1) * self.cfg.crop_margin, (y2 - y1) * self.cfg.crop_margin
        w, h = image.size
        return (
            int(max(0, x1 - mw)),
            int(max(0, y1 - mh)),
            int(min(w, x2 + mw)),
            int(min(h, y2 + mh)),
        )

    # ---------------------------------------------------------------- prompt
    def lock_caption(self, caption: str) -> str:
        """Prepend the fixed subject description used across every shot."""
        subject = self.cfg.subject_description.strip()
        if not subject:
            return caption
        if subject.lower() in caption.lower():
            return caption
        return f"{subject.rstrip('.')}. {caption}"

    # --------------------------------------------------------------- anchors
    @staticmethod
    def anchor_frame(video, index: int = -1):
        """Extract a frame as a PIL image, for chaining into the next shot."""
        from PIL import Image
        import numpy as np
        import torch

        v = video[0] if video.dim() == 5 else video
        frame = v[:, index].detach().float().cpu()
        if frame.min() < -0.01:
            frame = (frame + 1) / 2
        arr = (frame.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)

    def apply(self, request, capabilities) -> Dict[str, Any]:
        """Mutate ``request`` for the chosen identity method; report what was done."""
        method = self.choose_method(capabilities)
        info: Dict[str, Any] = {"method": method, "note": ADAPTER_NOTES.get(method, "")}
        if method in ("vace", "phantom"):
            refs = self.prepare_references()
            request.reference_images = refs
            request.task = "vace" if method == "vace" else request.task
            info["num_references"] = len(refs)
        elif method == "lora":
            info["lora"] = list(self.cfg.lora_paths)
        request.prompt = self.lock_caption(request.prompt)
        return info


def identity_drift(video, *, reference=None, stride: int = 4) -> Dict[str, float]:
    """Measure how much the subject changes across a clip.

    Uses ArcFace embeddings when insightface is available -- that is the number
    worth quoting. Without it, falls back to a centre-crop feature correlation,
    which detects dissolution and gross morphing but *not* a swap to a
    different person; the returned ``method`` field says which one you got.
    """
    import torch

    v = video[0] if video.dim() == 5 else video
    frames = list(range(0, v.shape[1], max(1, stride)))
    try:
        from insightface.app import FaceAnalysis
        import numpy as np

        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        embeddings = []
        for i in frames:
            frame = ((v[:, i].float().clamp(-1, 1) + 1) * 127.5).byte()
            arr = frame.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
            faces = app.get(arr)
            if faces:
                embeddings.append(torch.tensor(faces[0].normed_embedding))
        if len(embeddings) >= 2:
            stack = torch.stack(embeddings)
            ref = stack[0] if reference is None else reference
            sims = (stack @ ref) / (stack.norm(dim=1) * ref.norm()).clamp(min=1e-6)
            return {
                "method": "arcface",
                "mean_similarity": float(sims.mean()),
                "min_similarity": float(sims.min()),
                "frames_with_face": float(len(embeddings)) / len(frames),
            }
    except Exception as exc:
        logger.debug("ArcFace drift unavailable (%s); using the crop proxy", exc)

    from ..reward.heuristics import structural_persistence

    score = float(structural_persistence(v.unsqueeze(0))[0])
    return {"method": "crop_correlation", "mean_similarity": score, "min_similarity": score}
