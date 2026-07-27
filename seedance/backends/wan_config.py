"""Wan model configs, reconstructed without importing the `wan` package.

``wan.configs`` is the authority and is used when it imports cleanly. It cannot
always: ``wan/__init__.py`` pulls in the whole stack (diffusers, the T5 encoder,
the VAE), so on a machine that only wants to *inspect* a checkpoint the import
fails on a missing dependency. This module mirrors the same values so
``seedance inspect`` and config reconstruction work with nothing installed, and
defers to the real thing whenever it is importable.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

__all__ = ["WanConfigDict", "base_config", "SHARED", "T2V_1_3B", "T2V_14B", "I2V_14B", "FLF2V_14B"]


class WanConfigDict(dict):
    """Attribute-accessible dict, matching the EasyDict interface Wan expects."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __deepcopy__(self, memo) -> "WanConfigDict":
        return WanConfigDict({k: copy.deepcopy(v, memo) for k, v in self.items()})


NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

SHARED: Dict[str, Any] = {
    "t5_model": "umt5_xxl",
    "text_len": 512,
    "num_train_timesteps": 1000,
    "sample_fps": 16,
    "sample_neg_prompt": NEG_PROMPT,
    "t5_checkpoint": "models_t5_umt5-xxl-enc-bf16.pth",
    "t5_tokenizer": "google/umt5-xxl",
    "vae_checkpoint": "Wan2.1_VAE.pth",
    "vae_stride": (4, 8, 8),
    "patch_size": (1, 2, 2),
    "freq_dim": 256,
    "window_size": (-1, -1),
    "qk_norm": True,
    "cross_attn_norm": True,
    "eps": 1e-6,
}

T2V_1_3B: Dict[str, Any] = {
    **SHARED,
    "__name__": "Config: Wan T2V 1.3B",
    "dim": 1536,
    "ffn_dim": 8960,
    "num_heads": 12,
    "num_layers": 30,
}

T2V_14B: Dict[str, Any] = {
    **SHARED,
    "__name__": "Config: Wan T2V 14B",
    "dim": 5120,
    "ffn_dim": 13824,
    "num_heads": 40,
    "num_layers": 40,
}

I2V_14B: Dict[str, Any] = {
    **T2V_14B,
    "__name__": "Config: Wan I2V 14B",
    "sample_neg_prompt": "镜头晃动，" + NEG_PROMPT,
    "clip_model": "clip_xlm_roberta_vit_h_14",
    "clip_checkpoint": "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    "clip_tokenizer": "xlm-roberta-large",
}

FLF2V_14B: Dict[str, Any] = {
    **I2V_14B,
    "__name__": "Config: Wan FLF2V 14B",
    "sample_neg_prompt": "镜头切换，" + NEG_PROMPT,
}

_TABLE = {
    "t2v-1.3B": T2V_1_3B,
    "t2v-14B": T2V_14B,
    "i2v-14B": I2V_14B,
    "flf2v-14B": FLF2V_14B,
}

# `wan.configs` names for the same four entries
_WAN_NAMES = {
    "t2v-1.3B": "t2v_1_3B",
    "t2v-14B": "t2v_14B",
    "i2v-14B": "i2v_14B",
    "flf2v-14B": "flf2v_14B",
}


def base_config(key: str, *, prefer_wan: bool = True) -> WanConfigDict:
    """Return a fresh config for ``key`` (``t2v-1.3B``, ``i2v-14B``, ...).

    Torch dtypes are attached only when torch is importable — the loader needs
    them, ``seedance inspect`` does not.
    """
    if key not in _TABLE:
        raise KeyError(f"unknown Wan config {key!r}; expected one of {sorted(_TABLE)}")

    if prefer_wan:
        try:
            import wan.configs as wan_configs  # noqa: PLC0415

            cfg = copy.deepcopy(getattr(wan_configs, _WAN_NAMES[key]))
            return WanConfigDict(cfg)
        except Exception:
            pass  # fall through to the mirrored table

    cfg = WanConfigDict(copy.deepcopy(_TABLE[key]))
    try:
        import torch

        cfg.param_dtype = torch.bfloat16
        cfg.t5_dtype = torch.bfloat16
        if "clip_model" in cfg:
            cfg.clip_dtype = torch.float16
    except ImportError:
        pass
    return cfg
