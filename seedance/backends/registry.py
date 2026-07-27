"""Registry and auto-detection for Wan checkpoints on Hugging Face.

Two ways in:

* **Known models** -- :data:`MODELS` carries the published Wan repos with their
  task, layout and resource envelope, reachable by short alias (``t2v-1.3B``)
  or full repo id.
* **Anything else** -- :func:`detect_layout` inspects a downloaded directory and
  works out the task, family and loader from the files actually present. That
  is what makes "download any Wan model and run it" true for repos that did not
  exist when this was written, including community fine-tunes and quantised
  re-uploads.

This module is torch-free on purpose: ``seedance models`` must work before a
single wheel is installed.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ModelSpec",
    "MODELS",
    "ALIASES",
    "resolve",
    "list_models",
    "detect_layout",
    "DetectedLayout",
    "recommended_size",
]


@dataclass(frozen=True)
class ModelSpec:
    """A known Wan checkpoint."""

    repo_id: str
    alias: str
    family: str  # "wan2.1" | "wan2.2" | "wan2.5+"
    task: str  # t2v | i2v | flf2v | t2i | vace | ti2v
    params: str
    loader: str  # "wan_native" | "diffusers"
    sizes: Tuple[str, ...]
    default_size: str
    min_vram_gb: float
    notes: str = ""
    default_frames: int = 81
    default_fps: int = 16
    default_steps: int = 50
    default_shift: float = 5.0
    default_guidance: float = 5.0

    @property
    def name(self) -> str:
        return self.alias


def _s(*sizes: str) -> Tuple[str, ...]:
    return tuple(sizes)


MODELS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        repo_id="Wan-AI/Wan2.1-T2V-1.3B",
        alias="t2v-1.3B",
        family="wan2.1",
        task="t2v",
        params="1.3B",
        loader="wan_native",
        sizes=_s("832*480", "480*832"),
        default_size="832*480",
        min_vram_gb=8.2,
        default_shift=3.0,
        notes="Consumer-grade. 480p is where it is trained; 720p is unstable.",
    ),
    ModelSpec(
        repo_id="Wan-AI/Wan2.1-T2V-14B",
        alias="t2v-14B",
        family="wan2.1",
        task="t2v",
        params="14B",
        loader="wan_native",
        sizes=_s("1280*720", "720*1280", "832*480", "480*832"),
        default_size="1280*720",
        min_vram_gb=40.0,
        notes="~69 GB in bf16; ~40-48 GB at 480p / ~65-80 GB at 720p with fp8.",
    ),
    ModelSpec(
        repo_id="Wan-AI/Wan2.1-I2V-14B-480P",
        alias="i2v-14B-480p",
        family="wan2.1",
        task="i2v",
        params="14B",
        loader="wan_native",
        sizes=_s("832*480", "480*832"),
        default_size="832*480",
        min_vram_gb=40.0,
        default_steps=40,
        default_shift=3.0,
    ),
    ModelSpec(
        repo_id="Wan-AI/Wan2.1-I2V-14B-720P",
        alias="i2v-14B-720p",
        family="wan2.1",
        task="i2v",
        params="14B",
        loader="wan_native",
        sizes=_s("1280*720", "720*1280"),
        default_size="1280*720",
        min_vram_gb=48.0,
        default_steps=40,
    ),
    ModelSpec(
        repo_id="Wan-AI/Wan2.1-FLF2V-14B-720P",
        alias="flf2v-14B",
        family="wan2.1",
        task="flf2v",
        params="14B",
        loader="wan_native",
        sizes=_s("1280*720", "720*1280"),
        default_size="1280*720",
        min_vram_gb=48.0,
        default_shift=16.0,
        default_guidance=5.5,
        notes="First-last-frame conditioning; pins t=0 and t=1 boundaries.",
    ),
    ModelSpec(
        repo_id="Wan-AI/Wan2.1-VACE-1.3B",
        alias="vace-1.3B",
        family="wan2.1",
        task="vace",
        params="1.3B",
        loader="wan_native",
        sizes=_s("832*480", "480*832"),
        default_size="832*480",
        min_vram_gb=10.0,
        notes="All-in-one reference/editing: Ref2V, V2V, MV2V.",
    ),
    ModelSpec(
        repo_id="Wan-AI/Wan2.1-VACE-14B",
        alias="vace-14B",
        family="wan2.1",
        task="vace",
        params="14B",
        loader="wan_native",
        sizes=_s("1280*720", "720*1280", "832*480", "480*832"),
        default_size="1280*720",
        min_vram_gb=48.0,
    ),
    ModelSpec(
        repo_id="Wan-AI/Wan2.2-T2V-A14B",
        alias="t2v-a14b",
        family="wan2.2",
        task="t2v",
        params="27B MoE (14B active)",
        loader="diffusers",
        sizes=_s("1280*720", "832*480"),
        default_size="1280*720",
        min_vram_gb=48.0,
        notes="MoE: high-noise expert for layout, low-noise expert for detail.",
    ),
    ModelSpec(
        repo_id="Wan-AI/Wan2.2-I2V-A14B",
        alias="i2v-a14b",
        family="wan2.2",
        task="i2v",
        params="27B MoE (14B active)",
        loader="diffusers",
        sizes=_s("1280*720", "832*480"),
        default_size="1280*720",
        min_vram_gb=48.0,
    ),
    ModelSpec(
        repo_id="Wan-AI/Wan2.2-TI2V-5B",
        alias="ti2v-5B",
        family="wan2.2",
        task="ti2v",
        params="5B",
        loader="diffusers",
        sizes=_s("1280*704", "704*1280"),
        default_size="1280*704",
        min_vram_gb=12.0,
        default_fps=24,
        notes="High-compression VAE (up to 64x); runs on a single 4090.",
    ),
)

ALIASES: Dict[str, str] = {}
for _spec in MODELS:
    ALIASES[_spec.alias.lower()] = _spec.repo_id
    ALIASES[_spec.repo_id.lower()] = _spec.repo_id
# convenience aliases people actually type
ALIASES.update(
    {
        "1.3b": "Wan-AI/Wan2.1-T2V-1.3B",
        "14b": "Wan-AI/Wan2.1-T2V-14B",
        "t2v": "Wan-AI/Wan2.1-T2V-1.3B",
        "i2v": "Wan-AI/Wan2.1-I2V-14B-480P",
        "flf2v": "Wan-AI/Wan2.1-FLF2V-14B-720P",
        "vace": "Wan-AI/Wan2.1-VACE-1.3B",
    }
)


def list_models() -> List[ModelSpec]:
    return list(MODELS)


def resolve(spec: str) -> Tuple[str, Optional[ModelSpec]]:
    """Map a user string to ``(repo_id_or_path, known_spec_or_None)``.

    Accepts an alias, a full repo id, or a local directory. Unknown repo ids
    pass through unchanged -- they are handled by :func:`detect_layout` after
    download, which is the whole point.
    """
    if not spec:
        raise ValueError("model spec is empty")
    if os.path.isdir(spec):
        return spec, None
    key = spec.lower().strip()
    if key in ALIASES:
        repo = ALIASES[key]
        for model in MODELS:
            if model.repo_id == repo:
                return repo, model
        return repo, None
    if "/" in spec:
        return spec, None
    raise KeyError(
        f"unknown model {spec!r}. Pass a Hugging Face repo id (owner/name), a "
        f"local path, or one of: {', '.join(sorted(m.alias for m in MODELS))}"
    )


# ------------------------------------------------------------------ detection

@dataclass
class DetectedLayout:
    """What an on-disk checkpoint actually is."""

    root: str
    loader: str  # "wan_native" | "diffusers"
    task: str  # t2v | i2v | flf2v | vace | ti2v | unknown
    family: str  # wan2.1 | wan2.2 | wan2.x
    dim: Optional[int] = None
    num_layers: Optional[int] = None
    in_dim: Optional[int] = None
    moe: bool = False
    t5_checkpoint: Optional[str] = None
    vae_checkpoint: Optional[str] = None
    clip_checkpoint: Optional[str] = None
    quantization: Optional[str] = None  # "fp8" | "gguf" | None
    files: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        bits = [f"{self.family} {self.task}", f"loader={self.loader}"]
        if self.dim:
            bits.append(f"dim={self.dim}")
        if self.num_layers:
            bits.append(f"layers={self.num_layers}")
        if self.moe:
            bits.append("MoE")
        if self.quantization:
            bits.append(self.quantization)
        return ", ".join(bits)


def _listdir(root: str) -> List[str]:
    out: List[str] = []
    for base, _dirs, files in os.walk(root):
        rel = os.path.relpath(base, root)
        prefix = "" if rel == "." else rel + "/"
        out.extend(prefix + f for f in files)
        if len(out) > 20000:  # pathological repos; enough to classify
            break
    return out


def detect_layout(root: str, repo_id: str = "") -> DetectedLayout:
    """Classify a downloaded checkpoint directory.

    Detection is deliberately file-driven rather than name-driven: a repo can
    be renamed, mirrored or re-quantised, but a native Wan checkpoint always
    ships ``Wan2.x_VAE.pth`` next to a diffusers-style ``config.json``, and a
    diffusers conversion always ships ``model_index.json``.
    """
    files = _listdir(root)
    lower = {f.lower() for f in files}
    layout = DetectedLayout(root=root, loader="wan_native", task="unknown", family="wan2.x", files=files)

    has_model_index = "model_index.json" in lower
    has_transformer_dir = any(f.startswith("transformer/") for f in lower)
    has_native_vae = any(re.match(r"wan2\.\d_vae\.pth", f) for f in lower)
    has_native_t5 = any("umt5" in f and f.endswith(".pth") for f in lower)
    moe = any(f.startswith("high_noise_model/") for f in lower) or any(
        f.startswith("low_noise_model/") for f in lower
    )

    if has_model_index or (has_transformer_dir and not has_native_vae):
        layout.loader = "diffusers"
    elif has_native_vae or has_native_t5:
        layout.loader = "wan_native"
    elif any(f.endswith(".gguf") for f in lower):
        layout.loader = "diffusers"
        layout.quantization = "gguf"
        layout.warnings.append(
            "GGUF weights detected: load these through ComfyUI-GGUF or "
            "gguf-enabled diffusers; the native Wan loader cannot read them."
        )
    else:
        layout.warnings.append(
            "could not identify the checkpoint layout; assuming a native Wan repo"
        )

    layout.moe = moe
    if moe:
        layout.family = "wan2.2"

    # native checkpoint file names
    for name in files:
        base = os.path.basename(name)
        low = base.lower()
        if "umt5" in low and low.endswith(".pth"):
            layout.t5_checkpoint = base
        elif re.match(r"wan2\.\d_vae\.pth", low):
            layout.vae_checkpoint = base
            m = re.match(r"wan(2\.\d)_vae\.pth", low)
            if m:
                layout.family = f"wan{m.group(1)}"
        elif "clip" in low and low.endswith(".pth"):
            layout.clip_checkpoint = base
        elif low.endswith(".safetensors") and "fp8" in low:
            layout.quantization = "fp8"

    # transformer config
    cfg = _read_json(os.path.join(root, "config.json"))
    if not cfg and has_transformer_dir:
        cfg = _read_json(os.path.join(root, "transformer", "config.json"))
    if cfg:
        layout.dim = cfg.get("dim") or cfg.get("num_attention_heads", 0) * cfg.get(
            "attention_head_dim", 0
        ) or None
        layout.num_layers = cfg.get("num_layers") or cfg.get("num_layers", None)
        layout.in_dim = cfg.get("in_dim") or cfg.get("in_channels")
        model_type = cfg.get("model_type")
        if isinstance(model_type, str) and model_type in ("t2v", "i2v", "flf2v", "vace"):
            layout.task = model_type

    # task inference, in decreasing order of reliability
    name_hint = (repo_id or os.path.basename(os.path.normpath(root))).lower()
    if layout.task == "unknown":
        if "vace" in name_hint or any("vace" in f.lower() for f in files):
            layout.task = "vace"
        elif "flf2v" in name_hint:
            layout.task = "flf2v"
        elif "ti2v" in name_hint:
            layout.task = "ti2v"
        elif "i2v" in name_hint:
            layout.task = "i2v"
        elif "t2i" in name_hint:
            layout.task = "t2i"
        elif "t2v" in name_hint:
            layout.task = "t2v"
        elif layout.clip_checkpoint or (layout.in_dim or 0) >= 32:
            # I2V conditions on an extra image latent + mask -> 36 in-channels
            layout.task = "i2v"
        else:
            layout.task = "t2v"

    if "2.2" in name_hint and layout.family == "wan2.x":
        layout.family = "wan2.2"
    elif "2.1" in name_hint and layout.family == "wan2.x":
        layout.family = "wan2.1"

    if layout.loader == "wan_native":
        missing = []
        if not layout.t5_checkpoint:
            missing.append("UMT5 text encoder (.pth)")
        if not layout.vae_checkpoint:
            missing.append("Wan VAE (.pth)")
        if layout.task in ("i2v", "flf2v") and not layout.clip_checkpoint:
            missing.append("CLIP image encoder (.pth)")
        if missing:
            layout.warnings.append(
                "native loader is missing: " + ", ".join(missing)
                + " -- download the full repo, or use the diffusers loader"
            )
    return layout


def _read_json(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def recommended_size(spec: Optional[ModelSpec], layout: Optional[DetectedLayout], vram_gb: float) -> str:
    """Pick a resolution that fits the available VRAM."""
    if spec is not None:
        if vram_gb >= spec.min_vram_gb:
            return spec.default_size
        for size in spec.sizes:
            if "480" in size:
                return size
        return spec.sizes[-1]
    dim = (layout.dim if layout else None) or 0
    if dim and dim >= 5120 and vram_gb < 40:
        return "832*480"
    return "1280*720" if vram_gb >= 40 else "832*480"
