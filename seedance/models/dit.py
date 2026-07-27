"""The decoupled spatial/temporal MMDiT backbone.

``[1.0]``, verbatim from arXiv:2506.09113 and implemented here:

* "Decoupled spatial and temporal layers: spatial layers perform attention
  within each frame; temporal layers compute attention across frames."
* "Window partitioning is applied within each frame in the temporal layers to
  give a global receptive field across the temporal dimension while improving
  both training and inference efficiency."
* "A multi-modality self-attention layer is applied exclusively in spatial
  layers to integrate visual and textual tokens; temporal layers process only
  visual tokens."
* "Separate weights (adaptive layer norm, QKV projection, MLP) are used for the
  two modalities in spatial layers."
* "Q and K embeddings are normalized before attention to prevent training
  instability."
* Patchification is removed on the DiT side (following DC-AE) because the VAE
  already compresses 16x spatially -- hence the default ``patch_size=(1,1,1)``.

Two things are *design choices*, not report content, and are flagged inline:
how the text stream is aggregated back after per-frame joint attention, and the
audio bridge placement (``[1.5/2.0]`` says only that a cross-modal joint module
exists).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from ..config import DiTConfig
from .attention import JointAttention, SelfAttention, window_partition, window_unpartition
from .conditioning import ConditionTensors
from .norm import AdaLNModulation, FP32LayerNorm, TimestepEmbedding, gate, modulate
from .rope import MMRoPE, shot_time_positions

__all__ = ["SpatialMMDiTBlock", "TemporalDiTBlock", "SeedanceDiT", "layer_schedule"]


def _ffn(dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim)
    )


def layer_schedule(cfg: DiTConfig) -> List[str]:
    """Order in which spatial and temporal layers are applied."""
    ns, nt = cfg.num_spatial_layers, cfg.num_temporal_layers
    if cfg.interleave == "spatial_first":
        return ["spatial"] * ns + ["temporal"] * nt
    if cfg.interleave == "blocked":
        g = max(1, cfg.blocked_group_size)
        out: List[str] = []
        s_left, t_left = ns, nt
        while s_left or t_left:
            take = min(g, s_left)
            out += ["spatial"] * take
            s_left -= take
            take = min(g, t_left)
            out += ["temporal"] * take
            t_left -= take
        return out
    # "alternating": interleave 1:1, appending whatever is left over.
    out = []
    s_left, t_left = ns, nt
    while s_left or t_left:
        if s_left:
            out.append("spatial")
            s_left -= 1
        if t_left:
            out.append("temporal")
            t_left -= 1
    return out


class SpatialMMDiTBlock(nn.Module):
    """Intra-frame multimodal attention over visual + text tokens."""

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        d, f = cfg.dim, cfg.ffn_dim
        self.norm1_v = FP32LayerNorm(d, cfg.eps)
        self.norm1_t = FP32LayerNorm(d, cfg.eps)
        self.norm2_v = FP32LayerNorm(d, cfg.eps)
        self.norm2_t = FP32LayerNorm(d, cfg.eps)
        self.attn = JointAttention(d, cfg.num_heads, qk_norm=cfg.qk_norm, eps=cfg.eps)
        self.ffn_v = _ffn(d, f)
        self.ffn_t = _ffn(d, f)
        # separate modulation per modality (MMDiT)
        self.mod_v = AdaLNModulation(d)
        self.mod_t = AdaLNModulation(d)

    def forward(
        self,
        visual: torch.Tensor,  # [B, T, S, D]
        text: torch.Tensor,  # [B, Lt, D]
        cond: torch.Tensor,  # [B, D]
        visual_rope: Optional[object] = None,
        text_rope: Optional[object] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, s, d = visual.shape
        sv1, cv1, gv1, sv2, cv2, gv2 = self.mod_v(cond)
        st1, ct1, gt1, st2, ct2, gt2 = self.mod_t(cond)

        # Attention runs per frame: fold T into the batch axis, and let every
        # frame see the full text stream.
        v_in = modulate(self.norm1_v(visual), sv1, cv1).reshape(b * t, s, d).type_as(visual)
        t_in = modulate(self.norm1_t(text), st1, ct1).type_as(text)
        t_rep = t_in.unsqueeze(1).expand(b, t, *t_in.shape[1:]).reshape(b * t, *t_in.shape[1:])
        mask_rep = None
        if text_mask is not None:
            mask_rep = text_mask.unsqueeze(1).expand(b, t, -1).reshape(b * t, -1)

        v_out, t_out = self.attn(
            v_in, t_rep, visual_rope=visual_rope, text_rope=text_rope, text_mask=mask_rep
        )
        v_out = v_out.view(b, t, s, d)
        # DESIGN CHOICE (not in the report): the text stream is shared across
        # frames, so its per-frame updates are averaged before the residual.
        # Mean keeps the update magnitude independent of frame count.
        t_out = t_out.view(b, t, *t_in.shape[1:]).mean(dim=1)

        visual = visual + (gate(gv1, visual) * v_out.float()).type_as(visual)
        text = text + (gate(gt1, text) * t_out.float()).type_as(text)

        v_ffn = self.ffn_v(modulate(self.norm2_v(visual), sv2, cv2).type_as(visual))
        t_ffn = self.ffn_t(modulate(self.norm2_t(text), st2, ct2).type_as(text))
        visual = visual + (gate(gv2, visual) * v_ffn.float()).type_as(visual)
        text = text + (gate(gt2, text) * t_ffn.float()).type_as(text)
        return visual, text


class TemporalDiTBlock(nn.Module):
    """Inter-frame attention over visual tokens only, inside spatial windows."""

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        d, f = cfg.dim, cfg.ffn_dim
        self.window = cfg.temporal_window
        self.norm1 = FP32LayerNorm(d, cfg.eps)
        self.norm2 = FP32LayerNorm(d, cfg.eps)
        self.attn = SelfAttention(d, cfg.num_heads, qk_norm=cfg.qk_norm, eps=cfg.eps)
        self.ffn = _ffn(d, f)
        self.mod = AdaLNModulation(d)

    def forward(
        self,
        visual: torch.Tensor,  # [B, T, S, D]
        cond: torch.Tensor,
        grid: Tuple[int, int],  # (H, W) of the latent grid
        rope: Optional[object] = None,
    ) -> torch.Tensor:
        b, t, s, d = visual.shape
        h, w = grid
        assert h * w == s, f"grid {h}x{w} does not match {s} tokens per frame"
        s1, c1, g1, s2, c2, g2 = self.mod(cond)

        x = modulate(self.norm1(visual), s1, c1).type_as(visual)
        x = x.view(b, t, h, w, d)
        x, shape = window_partition(x, self.window)
        x = self.attn(x, rope=rope)
        x = window_unpartition(x, shape, b, t, h, w).reshape(b, t, s, d)
        visual = visual + (gate(g1, visual) * x.float()).type_as(visual)

        y = self.ffn(modulate(self.norm2(visual), s2, c2).type_as(visual))
        return visual + (gate(g2, visual) * y.float()).type_as(visual)


class Head(nn.Module):
    """Final adaLN + projection back to (patchified) latent channels."""

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.out_channels = cfg.out_channels
        out = math.prod(cfg.patch_size) * cfg.out_channels
        self.norm = FP32LayerNorm(cfg.dim, cfg.eps)
        self.mod = AdaLNModulation(cfg.dim, chunks=2)
        self.proj = nn.Linear(cfg.dim, out)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.mod(cond)
        return self.proj(modulate(self.norm(x), shift, scale).type_as(self.proj.weight))


class SeedanceDiT(nn.Module):
    """Seedance-shaped diffusion transformer.

    Input/output are latent tensors ``[B, C, T, H, W]``; conditioning frames and
    their binary mask arrive as extra channels (see
    :mod:`seedance.models.conditioning`).
    """

    def __init__(self, cfg: DiTConfig, audio_bridge: Optional[nn.ModuleList] = None):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Conv3d(
            cfg.total_in_channels, cfg.dim, kernel_size=cfg.patch_size, stride=cfg.patch_size
        )
        self.text_proj = nn.Sequential(
            nn.Linear(cfg.text_dim, cfg.dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(cfg.dim, cfg.dim),
        )
        self.time_embed = TimestepEmbedding(cfg.dim)
        # pooled text is added to the timestep vector, SD3-style
        self.text_pool_proj = nn.Linear(cfg.text_dim, cfg.dim)

        self.schedule = layer_schedule(cfg)
        blocks: List[nn.Module] = []
        for kind in self.schedule:
            blocks.append(SpatialMMDiTBlock(cfg) if kind == "spatial" else TemporalDiTBlock(cfg))
        self.blocks = nn.ModuleList(blocks)
        self.head = Head(cfg)
        self.rope = MMRoPE(cfg.head_dim, theta=cfg.rope_theta)
        self.audio_bridge = audio_bridge  # ModuleList of CrossModalJointModule
        self._init_weights()

    # ------------------------------------------------------------------ setup
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.patch_embed.weight.flatten(1))
        nn.init.zeros_(self.patch_embed.bias)
        for module in (*self.text_proj.modules(), *self.time_embed.modules()):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
        # re-zero the adaLN-zero and head projections clobbered by the loop above
        for module in self.modules():
            if isinstance(module, AdaLNModulation):
                nn.init.zeros_(module.proj[1].weight)
                nn.init.zeros_(module.proj[1].bias)
        nn.init.zeros_(self.head.proj.weight)
        nn.init.zeros_(self.head.proj.bias)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def enable_grad_checkpointing(self, enable: bool = True) -> None:
        self.cfg.grad_checkpointing = enable

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        text: torch.Tensor,
        *,
        text_mask: Optional[torch.Tensor] = None,
        condition: Optional[ConditionTensors] = None,
        shot_lengths: Optional[Sequence[int]] = None,
        audio_tokens: Optional[torch.Tensor] = None,
        shot_gap: int = 0,
    ) -> torch.Tensor:
        """Predict the flow-matching velocity for ``latents``.

        Args:
            latents: noisy latents ``[B, C, T, H, W]``.
            timestep: ``[B]`` or scalar, in ``[0, num_train_timesteps]``.
            text: text embeddings ``[B, Lt, text_dim]``.
            shot_lengths: per-shot latent frame counts; enables multishot
                MM-RoPE temporal offsets.
            audio_tokens: ``[B, La, audio_dim]`` when the audio branch is used.
        Returns:
            ``[B, C_out, T, H, W]``.
        """
        cfg = self.cfg
        b = latents.shape[0]
        if condition is None:
            zeros = torch.zeros(
                (b, cfg.condition_channels, *latents.shape[2:]),
                device=latents.device,
                dtype=latents.dtype,
            )
            mask = torch.zeros(
                (b, cfg.mask_channels, *latents.shape[2:]),
                device=latents.device,
                dtype=latents.dtype,
            )
            x = torch.cat([latents, zeros, mask], dim=1)
        else:
            x = condition.as_input(latents)

        x = self.patch_embed(x)  # [B, D, T', H', W']
        _, d, t, h, w = x.shape
        visual = x.flatten(3).permute(0, 2, 3, 1).contiguous()  # [B, T, S, D]

        if timestep.dim() == 0:
            timestep = timestep.expand(b)
        cond = self.time_embed(timestep.to(latents.device))
        text_h = self.text_proj(text)
        if text_mask is None:
            pooled = text.mean(dim=1)
        else:
            m = text_mask.to(text.dtype).unsqueeze(-1)
            pooled = (text * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        cond = cond + self.text_pool_proj(pooled).to(cond.dtype)

        device = latents.device
        # Intra-frame attention only ever sees one frame's tokens, so a single
        # t=0 grid is sufficient (RoPE is relative along each axis).
        spatial_rope = self.rope.visual(1, h, w, device=device).to(device)
        text_rope = self.rope.text(text_h.shape[1], device=device).to(device)
        time_positions = None
        if shot_lengths is not None:
            lengths = list(shot_lengths)
            if sum(lengths) != t:
                raise ValueError(
                    f"shot_lengths sum to {sum(lengths)} but latent has {t} frames"
                )
            time_positions = shot_time_positions(lengths, gap=shot_gap)
        temporal_rope = self.rope.visual(
            t,
            h,
            w,
            device=device,
            order="window",
            window=cfg.temporal_window,
            time_positions=time_positions,
        ).to(device)

        audio_idx = 0
        spatial_seen = 0
        for kind, block in zip(self.schedule, self.blocks):
            if kind == "spatial":
                visual, text_h = self._run(
                    block,
                    visual,
                    text_h,
                    cond,
                    spatial_rope,
                    text_rope,
                    text_mask,
                    spatial=True,
                )
                spatial_seen += 1
                if (
                    self.audio_bridge is not None
                    and audio_tokens is not None
                    and spatial_seen % max(1, cfg.audio_bridge_every) == 0
                    and audio_idx < len(self.audio_bridge)
                ):
                    visual, audio_tokens = self.audio_bridge[audio_idx](
                        visual, audio_tokens, cond
                    )
                    audio_idx += 1
            else:
                visual = self._run(
                    block, visual, None, cond, temporal_rope, None, None, spatial=False, grid=(h, w)
                )

        out = self.head(visual.reshape(b, t * h * w, d), cond)
        return self._unpatchify(out, (t, h, w))

    def _run(
        self,
        block: nn.Module,
        visual: torch.Tensor,
        text: Optional[torch.Tensor],
        cond: torch.Tensor,
        rope: object,
        text_rope: Optional[object],
        text_mask: Optional[torch.Tensor],
        *,
        spatial: bool,
        grid: Optional[Tuple[int, int]] = None,
    ):
        use_ckpt = self.cfg.grad_checkpointing and self.training
        if spatial:
            if use_ckpt:
                return checkpoint.checkpoint(
                    block, visual, text, cond, rope, text_rope, text_mask, use_reentrant=False
                )
            return block(visual, text, cond, rope, text_rope, text_mask)
        if use_ckpt:
            return checkpoint.checkpoint(block, visual, cond, grid, rope, use_reentrant=False)
        return block(visual, cond, grid, rope)

    def _unpatchify(self, x: torch.Tensor, grid: Tuple[int, int, int]) -> torch.Tensor:
        pt, ph, pw = self.cfg.patch_size
        t, h, w = grid
        c = self.cfg.out_channels
        b = x.shape[0]
        x = x.view(b, t, h, w, pt, ph, pw, c)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        return x.view(b, c, t * pt, h * ph, w * pw)

    # ------------------------------------------------------------- utilities
    def token_count(self, num_frames: int, height: int, width: int) -> int:
        pt, ph, pw = self.cfg.patch_size
        return (num_frames // pt) * (height // ph) * (width // pw)

    def flops_estimate(self, num_frames: int, height: int, width: int) -> float:
        """Rough forward FLOPs, in TFLOPs, for capacity planning.

        Counts attention and FFN matmuls only; the point of the estimate is the
        spatial/temporal split, which is where the decoupled design pays off
        against full 3D attention.
        """
        cfg = self.cfg
        d = cfg.dim
        s = (height // cfg.patch_size[1]) * (width // cfg.patch_size[2])
        t = num_frames // cfg.patch_size[0]
        wh, ww = cfg.temporal_window
        win = min(wh, height // cfg.patch_size[1]) * min(ww, width // cfg.patch_size[2])

        def block_flops(seq: int, tokens: int, mm: bool) -> float:
            proj = 2 * tokens * (4 * d * d) * (2 if mm else 1)
            attn = 2 * tokens * seq * d * 2
            ffn = 2 * tokens * (2 * d * cfg.ffn_dim) * (2 if mm else 1)
            return proj + attn + ffn

        spatial = block_flops(s, t * s, mm=True) * cfg.num_spatial_layers
        temporal = block_flops(t * win, t * s, mm=False) * cfg.num_temporal_layers
        return (spatial + temporal) / 1e12
