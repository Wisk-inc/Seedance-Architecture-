"""Reference pipeline over the native Seedance architecture.

**There are no public Seedance weights.** Built from scratch this pipeline runs
end to end and produces noise -- that is expected and is the honest state of
affairs. It exists for three real uses:

1. **Architecture validation.** Shape, dtype and gradient flow through the
   decoupled MMDiT, the causal VAE, MM-RoPE, the audio bridge and the refiner.
2. **Training.** :meth:`SeedanceNativePipeline.training_step` gives the
   rectified-flow loss for a batch, so the architecture is trainable as-is.
3. **A drop-in target.** ``load_weights`` accepts a state dict, so the day you
   train (or convert) something, inference already works.

For actual video today, use :class:`~seedance.pipelines.staged.SeedancePipeline`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..config import SeedanceConfig, get_config
from ..flow.rectified_flow import RectifiedFlow
from ..models.audio import AudioBranchDiT, build_audio_bridge
from ..models.conditioning import ConditionTensors, build_condition
from ..story import Storyboard
from ..models.dit import SeedanceDiT
from ..models.refiner import CascadedRefiner
from ..models.text_encoder import TextEncoderOutput, build_text_encoder
from ..models.vae3d import TemporallyCausalVAE
from .base import PipelineResult, StageTimer

logger = logging.getLogger(__name__)

__all__ = ["SeedanceNativePipeline"]


class SeedanceNativePipeline(nn.Module):
    """The Seedance architecture wired end to end."""

    def __init__(
        self,
        config: Optional[SeedanceConfig] = None,
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        build_refiner: bool = False,
        text_encoder_fallback: bool = True,
    ):
        super().__init__()
        self.cfg = config or get_config("seedance-2.0")
        self.device_str = device
        self.dtype = dtype

        self.vae = TemporallyCausalVAE(self.cfg.vae)
        bridge = build_audio_bridge(self.cfg.dit, self.cfg.audio)
        self.dit = SeedanceDiT(self.cfg.dit, audio_bridge=bridge if len(bridge) else None)
        self.audio_branch = (
            AudioBranchDiT(self.cfg.audio, text_dim=self.cfg.dit.text_dim)
            if self.cfg.audio.enabled
            else None
        )
        self.refiner = (
            CascadedRefiner(self.cfg.dit, self.cfg.refiner)
            if (build_refiner and self.cfg.refiner.enabled)
            else None
        )
        self.flow = RectifiedFlow(self.cfg.flow)
        self.text_encoder, self.text_encoder_kind = build_text_encoder(
            self.cfg.text_encoder,
            device=device,
            dim=self.cfg.dit.text_dim,
            allow_fallback=text_encoder_fallback,
        )
        if self.text_encoder_kind == "hashing":
            logger.warning(
                "using the hashing text-encoder fixture: prompts carry no "
                "semantics. Set config.text_encoder.model_id to a real LLM."
            )
        self.to(device)

    # ----------------------------------------------------------------- utils
    @property
    def is_trained(self) -> bool:
        """Whether any weights have been loaded (as opposed to random init)."""
        return getattr(self, "_loaded", False)

    def load_weights(self, state_dict: Dict[str, Any], strict: bool = False) -> Tuple[List[str], List[str]]:
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        self._loaded = True
        if missing:
            logger.warning("%d missing keys (first few: %s)", len(missing), missing[:5])
        if unexpected:
            logger.warning("%d unexpected keys (first few: %s)", len(unexpected), unexpected[:5])
        return list(missing), list(unexpected)

    def parameter_report(self) -> Dict[str, float]:
        def count(module: Optional[nn.Module]) -> float:
            return sum(p.numel() for p in module.parameters()) / 1e6 if module else 0.0

        return {
            "dit_m": round(count(self.dit), 1),
            "vae_m": round(count(self.vae), 1),
            "audio_branch_m": round(count(self.audio_branch), 1),
            "refiner_m": round(count(self.refiner), 1),
            "total_m": round(count(self), 1),
        }

    def encode_text(self, prompts: Sequence[str]) -> TextEncoderOutput:
        return self.text_encoder(list(prompts), device=self.device_str)

    # -------------------------------------------------------------- sampling
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        height: int = 480,
        width: int = 832,
        num_frames: int = 97,
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: int = -1,
        task: str = "t2v",
        conditioning_video: Optional[torch.Tensor] = None,
        storyboard: Optional[Storyboard] = None,
        refine: bool = False,
        decode: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    ) -> PipelineResult:
        """Sample a clip. Without trained weights this returns noise."""
        timer = StageTimer()
        cfg = self.cfg
        device = self.device_str
        generator = None
        if seed >= 0:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        with timer("text"):
            text = self.encode_text([prompt])
            negative = self.encode_text([negative_prompt or ""])

        c, t, h, w = cfg.latent_shape(num_frames, height, width)
        shape = (1, c, t, h, w)

        with timer("condition"):
            condition = None
            if conditioning_video is not None:
                latents = self.vae.encode(conditioning_video.to(device), sample=False)
                condition = build_condition(
                    shape, task, clean_latents=latents, device=device, dtype=torch.float32
                )
            elif task != "t2v":
                condition = build_condition(shape, task, device=device)

        shot_lengths = None
        if storyboard is not None and len(storyboard.shots) > 1:
            rt = cfg.vae.temporal_downsample
            shot_lengths = [max(1, (s.num_frames - 1) // rt + 1) for s in storyboard.shots]
            if sum(shot_lengths) != t:
                shot_lengths[-1] += t - sum(shot_lengths)

        def velocity(embeddings: TextEncoderOutput):
            def fn(x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
                return self.dit(
                    x.to(self.dtype),
                    ts,
                    embeddings.embeddings.to(self.dtype),
                    text_mask=embeddings.mask,
                    condition=condition,
                    shot_lengths=shot_lengths,
                )

            return fn

        with timer("sample"):
            seq_len = t * h * w
            latents = self.flow.sample(
                velocity(text),
                shape,
                device=device,
                dtype=self.dtype,
                num_steps=num_steps,
                seq_len=seq_len,
                guidance_scale=guidance_scale,
                uncond_model=velocity(negative),
                generator=generator,
                callback=callback,
            )

        if refine and self.refiner is not None:
            with timer("refine"):
                latents = self.refiner.refine(
                    latents,
                    text=text.embeddings.to(self.dtype),
                    text_mask=text.mask,
                    generator=generator,
                )

        video = None
        if decode:
            with timer("decode"):
                self.vae.use_thin_decoder(self.cfg.vae.thin_decoder)
                video = self.vae.decode(latents)[0]

        return PipelineResult(
            video=video if video is not None else latents,
            fps=cfg.fps,
            prompt=prompt,
            enhanced_prompt=prompt,
            seed=seed,
            timings=timer.timings,
            report={
                "architecture": "seedance-native",
                "config": cfg.name,
                "text_encoder": self.text_encoder_kind,
                "trained": self.is_trained,
                "parameters_m": self.parameter_report(),
                "latent_shape": list(shape),
            },
            warnings=(
                []
                if self.is_trained
                else ["no trained weights loaded: this output is noise by construction"]
            ),
        )

    # -------------------------------------------------------------- training
    def training_step(
        self,
        video: torch.Tensor,
        prompts: Sequence[str],
        *,
        task: str = "t2v",
        audio: Optional[torch.Tensor] = None,
        weighting: str = "none",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Rectified-flow loss for one batch of clips ``[B, 3, T, H, W]``."""
        device = video.device
        with torch.no_grad():
            x0 = self.vae.encode(video)
        text = self.encode_text(prompts).to(device)

        b = x0.shape[0]
        seq_len = x0.shape[2] * x0.shape[3] * x0.shape[4]
        t = self.flow.sample_timesteps(b, device=device, seq_len=seq_len)
        noise = torch.randn_like(x0)
        x_t = self.flow.add_noise(x0, noise, t)

        condition = None
        if task != "t2v":
            clean = x0[:, :, :1] if task == "i2v" else x0
            condition = build_condition(
                tuple(x0.shape), task, clean_latents=clean, device=device
            )

        audio_tokens = None
        if audio is not None and self.audio_branch is not None:
            audio_tokens = self.audio_branch.tokens(audio)

        v = self.dit(
            x_t.to(self.dtype),
            t * self.flow.cfg.num_train_timesteps,
            text.embeddings.to(self.dtype),
            text_mask=text.mask,
            condition=condition,
            audio_tokens=audio_tokens,
        )
        loss = self.flow.loss(v, x0, noise, t, weighting=weighting)
        return loss, {"flow_loss": float(loss.detach()), "t_mean": float(t.mean())}

    def forward(self, *args, **kwargs):  # pragma: no cover - convenience
        return self.generate(*args, **kwargs)
