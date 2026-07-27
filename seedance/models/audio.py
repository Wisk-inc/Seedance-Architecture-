"""Audio branch of the dual-branch DiT, and the cross-modal joint module.

``[1.5/2.0]`` is the only support for any of this: Seedance 1.5 Pro is described
as "a dual-branch Diffusion Transformer" integrating "a cross-modal joint
module" on the MMDiT/rectified-flow backbone, processing video and audio in
parallel with synchronisation and semantic consistency mediated through
attention-based fusion; 2.0 adds multi-track binaural output. Everything below
that -- tokenizer split, bridge placement, the sync window -- is a
reconstruction, marked ``[inferred]``. Third-party "waveform vs spectral
branch with a millisecond-level attention bridge" descriptions are marketing
copy, not specification, so the tokenizer here is *configurable* rather than
asserting one design.

The one non-obvious piece is synchronisation. Audio runs at ~100 tokens/s and
video at 24 fps, so a naive full cross-attention lets a footstep attend to a
frame two seconds away. :class:`CrossModalJointModule` therefore builds an
explicit alignment mask from the two frame rates and only allows attention
inside a tolerance window, which is what makes lip-sync and impact sounds land
on the right frame.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AudioConfig, DiTConfig
from .attention import CrossAttention, SelfAttention
from .norm import AdaLNModulation, FP32LayerNorm, TimestepEmbedding, gate, modulate
from .rope import RoPE, _freqs

__all__ = [
    "AudioTokenizer",
    "AudioBranchDiT",
    "CrossModalJointModule",
    "build_audio_bridge",
    "alignment_mask",
]


class AudioTokenizer(nn.Module):
    """Turns a waveform into audio tokens.

    Modes:
        ``"mel"``      -- log-mel spectrogram frames projected to ``dim``.
        ``"waveform"`` -- raw samples patched at ``waveform_patch`` and projected.
        ``"hybrid"``   -- both, concatenated per frame then projected (default).
    """

    def __init__(self, cfg: AudioConfig, dim: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.dim = dim or cfg.dim
        self.register_buffer("mel_fb", _mel_filterbank(cfg), persistent=False)
        mel_in = cfg.n_mels * cfg.channels
        wav_in = cfg.waveform_patch * cfg.channels
        if cfg.tokenizer == "mel":
            self.proj = nn.Linear(mel_in, self.dim)
        elif cfg.tokenizer == "waveform":
            self.proj = nn.Linear(wav_in, self.dim)
        elif cfg.tokenizer == "hybrid":
            self.proj = nn.Linear(mel_in + wav_in, self.dim)
        else:
            raise ValueError(f"unknown audio tokenizer {cfg.tokenizer!r}")

    @property
    def token_rate(self) -> float:
        """Audio tokens per second."""
        return self.cfg.sample_rate / self.cfg.hop_length

    def mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """``[B, C, N]`` waveform -> ``[B, L, C * n_mels]`` log-mel frames."""
        cfg = self.cfg
        b, c, n = waveform.shape
        flat = waveform.reshape(b * c, n)
        window = torch.hann_window(cfg.n_fft, device=waveform.device, dtype=torch.float32)
        spec = torch.stft(
            flat.float(),
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            window=window,
            center=True,
            return_complex=True,
        )
        power = spec.abs().pow(2)  # [B*C, F, L]
        mel = torch.matmul(self.mel_fb.to(power.dtype), power)
        mel = torch.log(mel.clamp(min=1e-5))
        l = mel.shape[-1]
        return mel.view(b, c * cfg.n_mels, l).transpose(1, 2).contiguous()

    def patches(self, waveform: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """``[B, C, N]`` -> ``[B, num_tokens, C * waveform_patch]``."""
        cfg = self.cfg
        b, c, n = waveform.shape
        need = num_tokens * cfg.waveform_patch
        if n < need:
            waveform = F.pad(waveform, (0, need - n))
        else:
            waveform = waveform[:, :, :need]
        return waveform.view(b, c, num_tokens, cfg.waveform_patch).permute(0, 2, 1, 3).reshape(
            b, num_tokens, c * cfg.waveform_patch
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """``[B, C, N]`` waveform in ``[-1, 1]`` -> ``[B, L, dim]`` tokens."""
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.shape[1] != self.cfg.channels:
            waveform = waveform[:, :1].expand(-1, self.cfg.channels, -1)
        mode = self.cfg.tokenizer
        if mode == "mel":
            feats = self.mel(waveform)
        elif mode == "waveform":
            n_tokens = max(1, waveform.shape[-1] // self.cfg.waveform_patch)
            feats = self.patches(waveform, n_tokens)
        else:
            mel = self.mel(waveform)
            feats = torch.cat([mel, self.patches(waveform, mel.shape[1])], dim=-1)
        return self.proj(feats.to(self.proj.weight.dtype))


def _mel_filterbank(cfg: AudioConfig) -> torch.Tensor:
    """Slaney-style triangular mel filterbank, ``[n_mels, n_fft // 2 + 1]``."""
    n_freqs = cfg.n_fft // 2 + 1
    f_max = cfg.sample_rate / 2
    freqs = torch.linspace(0, f_max, n_freqs)

    def hz_to_mel(f: torch.Tensor | float) -> torch.Tensor:
        return 2595.0 * torch.log10(torch.as_tensor(1.0 + f / 700.0))

    def mel_to_hz(m: torch.Tensor) -> torch.Tensor:
        return 700.0 * (torch.pow(10.0, m / 2595.0) - 1.0)

    mels = torch.linspace(float(hz_to_mel(0.0)), float(hz_to_mel(f_max)), cfg.n_mels + 2)
    points = mel_to_hz(mels)
    fb = torch.zeros(cfg.n_mels, n_freqs)
    for i in range(cfg.n_mels):
        left, centre, right = points[i], points[i + 1], points[i + 2]
        rising = (freqs - left) / max(float(centre - left), 1e-6)
        falling = (right - freqs) / max(float(right - centre), 1e-6)
        fb[i] = torch.clamp(torch.minimum(rising, falling), min=0.0)
    return fb


def alignment_mask(
    num_audio_tokens: int,
    num_video_frames: int,
    audio_rate: float,
    video_fps: float,
    tolerance_s: float = 0.12,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Boolean ``[La, Tv]`` mask allowing attention only inside a time window.

    ``tolerance_s`` defaults to 120 ms -- wide enough to cover the perceptual
    audio-visual binding window, narrow enough that a sound cannot bind to an
    event a second away.
    """
    a_t = torch.arange(num_audio_tokens, device=device, dtype=torch.float32) / max(audio_rate, 1e-6)
    v_t = torch.arange(num_video_frames, device=device, dtype=torch.float32) / max(video_fps, 1e-6)
    return (a_t[:, None] - v_t[None, :]).abs() <= tolerance_s


class AudioDiTBlock(nn.Module):
    def __init__(self, cfg: AudioConfig):
        super().__init__()
        self.norm1 = FP32LayerNorm(cfg.dim)
        self.attn = SelfAttention(cfg.dim, cfg.num_heads)
        self.norm2 = FP32LayerNorm(cfg.dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.dim, cfg.ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(cfg.ffn_dim, cfg.dim),
        )
        self.mod = AdaLNModulation(cfg.dim)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, rope: Optional[RoPE] = None
    ) -> torch.Tensor:
        s1, c1, g1, s2, c2, g2 = self.mod(cond)
        h = self.attn(modulate(self.norm1(x), s1, c1).type_as(x), rope=rope)
        x = x + (gate(g1, x) * h.float()).type_as(x)
        h = self.ffn(modulate(self.norm2(x), s2, c2).type_as(x))
        return x + (gate(g2, x) * h.float()).type_as(x)


class AudioBranchDiT(nn.Module):
    """The audio half of the dual-branch DiT.

    Operates on audio latents ``[B, Ca, L]`` (channels-first, time-major),
    predicting a velocity field in the same shape so the audio branch trains
    with the same rectified-flow objective as the video branch.
    """

    def __init__(self, cfg: AudioConfig, text_dim: int = 4096):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.latent_channels, cfg.dim)
        self.text_proj = nn.Linear(text_dim, cfg.dim)
        self.time_embed = TimestepEmbedding(cfg.dim)
        self.track_embed = nn.Embedding(max(1, len(cfg.tracks)), cfg.dim)
        self.blocks = nn.ModuleList([AudioDiTBlock(cfg) for _ in range(cfg.num_layers)])
        self.text_attn = nn.ModuleList(
            [CrossAttention(cfg.dim, cfg.num_heads) for _ in range(cfg.num_layers)]
        )
        self.norm_out = FP32LayerNorm(cfg.dim)
        self.out_proj = nn.Linear(cfg.dim, cfg.latent_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def tokens(self, latents: torch.Tensor) -> torch.Tensor:
        return self.in_proj(latents.transpose(1, 2))

    def forward(
        self,
        latents: torch.Tensor,  # [B, Ca, L]
        timestep: torch.Tensor,
        text: torch.Tensor,  # [B, Lt, text_dim]
        *,
        track: int = 0,
        video_tokens: Optional[torch.Tensor] = None,  # [B, Tv, D] for fusion
        bridge: Optional["CrossModalJointModule"] = None,
        video_fps: float = 24.0,
    ) -> torch.Tensor:
        b = latents.shape[0]
        x = self.tokens(latents)
        if timestep.dim() == 0:
            timestep = timestep.expand(b)
        cond = self.time_embed(timestep.to(latents.device))
        cond = cond + self.track_embed(
            torch.full((b,), track, device=latents.device, dtype=torch.long)
        )
        ctx = self.text_proj(text)
        head_dim = self.cfg.dim // self.cfg.num_heads
        rope = RoPE(
            _freqs(head_dim, torch.arange(x.shape[1], device=latents.device), 10000.0)
        )
        for i, block in enumerate(self.blocks):
            x = block(x, cond, rope)
            x = x + self.text_attn[i](x, ctx)
            if bridge is not None and video_tokens is not None and (i + 1) % 4 == 0:
                video_tokens, x = bridge.fuse_audio_first(
                    video_tokens, x, cond, video_fps=video_fps, audio_rate=self.token_rate
                )
        x = self.norm_out(x)
        return self.out_proj(x).transpose(1, 2)

    @property
    def token_rate(self) -> float:
        return self.cfg.sample_rate / self.cfg.hop_length


class CrossModalJointModule(nn.Module):
    """Attention-based fusion between the video and audio branches.

    Video tokens are pooled per frame before fusion (``[B, T, S, D] ->
    [B, T, D]``): a sound event is bound to a *moment*, not to a pixel, and
    per-frame pooling keeps the bridge's cost linear in frames instead of
    quadratic in tokens. Both directions are gated with a zero-initialised
    ``tanh`` gate, so inserting the bridge into a pretrained video-only
    backbone is a no-op at step 0.
    """

    def __init__(self, video_dim: int, audio_dim: int, num_heads: int = 8, tolerance_s: float = 0.12):
        super().__init__()
        self.tolerance_s = tolerance_s
        self.v2a = CrossAttention(audio_dim, num_heads, kv_dim=video_dim, gated=True)
        self.a2v = CrossAttention(video_dim, num_heads, kv_dim=audio_dim, gated=True)
        self.norm_v = FP32LayerNorm(video_dim)
        self.norm_a = FP32LayerNorm(audio_dim)
        self.mod_v = AdaLNModulation(video_dim, chunks=2)
        self.mod_a = AdaLNModulation(audio_dim, cond_dim=video_dim, chunks=2)

    def _masks(
        self,
        num_audio: int,
        num_frames: int,
        audio_rate: float,
        video_fps: float,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        align = alignment_mask(
            num_audio, num_frames, audio_rate, video_fps, self.tolerance_s, device
        )
        # Guard against an empty row/column (very short clips): fall back to
        # unrestricted attention for those queries rather than producing NaNs.
        a_rows = align.any(dim=1, keepdim=True)
        v_rows = align.any(dim=0, keepdim=True).transpose(0, 1)
        a_mask = torch.where(a_rows, align, torch.ones_like(align))
        v_mask = torch.where(v_rows, align.transpose(0, 1), torch.ones_like(align.transpose(0, 1)))
        return a_mask[None, None], v_mask[None, None]

    def forward(
        self,
        visual: torch.Tensor,  # [B, T, S, D]
        audio: torch.Tensor,  # [B, La, Da]
        cond: torch.Tensor,
        *,
        video_fps: float = 24.0,
        audio_rate: float = 100.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, s, d = visual.shape
        frames = visual.mean(dim=2)  # [B, T, D]
        a_mask, v_mask = self._masks(audio.shape[1], t, audio_rate, video_fps, visual.device)

        sv, cv = self.mod_v(cond)
        sa, ca = self.mod_a(cond)
        frames_n = modulate(self.norm_v(frames), sv, cv).type_as(frames)
        audio_n = modulate(self.norm_a(audio), sa, ca).type_as(audio)

        audio_out = audio + self.v2a(audio_n, frames_n, attn_mask=a_mask)
        frame_out = self.a2v(frames_n, audio_n, attn_mask=v_mask)
        # broadcast the per-frame audio update back over spatial tokens
        visual_out = visual + frame_out.unsqueeze(2).expand_as(visual).type_as(visual)
        return visual_out, audio_out

    def fuse_audio_first(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
        cond: torch.Tensor,
        *,
        video_fps: float = 24.0,
        audio_rate: float = 100.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Same fusion, called from inside the audio branch's layer loop."""
        if visual.dim() == 3:  # already pooled per frame
            visual = visual.unsqueeze(2)
            out_v, out_a = self.forward(
                visual, audio, cond, video_fps=video_fps, audio_rate=audio_rate
            )
            return out_v.squeeze(2), out_a
        return self.forward(visual, audio, cond, video_fps=video_fps, audio_rate=audio_rate)


def build_audio_bridge(dit_cfg: DiTConfig, audio_cfg: AudioConfig) -> nn.ModuleList:
    """One joint module per bridge site along the spatial layer stack."""
    if not audio_cfg.enabled:
        return nn.ModuleList()
    sites = max(1, dit_cfg.num_spatial_layers // max(1, dit_cfg.audio_bridge_every))
    return nn.ModuleList(
        [
            CrossModalJointModule(dit_cfg.dim, audio_cfg.dim, num_heads=audio_cfg.num_heads)
            for _ in range(sites)
        ]
    )


class GriffinLimVocoder(nn.Module):
    """Weightless mel -> waveform inversion, for smoke tests and previews.

    A real deployment swaps this for a neural vocoder; the point of having it
    is that the audio path stays end-to-end runnable with zero downloads.
    """

    def __init__(self, cfg: AudioConfig, iterations: int = 32):
        super().__init__()
        self.cfg = cfg
        self.iterations = iterations
        self.register_buffer("mel_fb", _mel_filterbank(cfg), persistent=False)

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        """``[B, L, n_mels]`` log-mel -> ``[B, 1, N]`` waveform."""
        cfg = self.cfg
        mel = log_mel.exp().transpose(1, 2)  # [B, n_mels, L]
        fb = self.mel_fb.to(mel.dtype)
        # least-squares pseudo-inverse of the filterbank
        inv = torch.linalg.pinv(fb)
        spec = (inv @ mel).clamp(min=0).sqrt()
        window = torch.hann_window(cfg.n_fft, device=mel.device, dtype=torch.float32)
        angles = torch.rand_like(spec) * 2 * math.pi
        for _ in range(self.iterations):
            complex_spec = torch.polar(spec.float(), angles)
            wave = torch.istft(
                complex_spec, n_fft=cfg.n_fft, hop_length=cfg.hop_length, window=window
            )
            rebuilt = torch.stft(
                wave,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                window=window,
                return_complex=True,
            )
            angles = rebuilt.angle()
        complex_spec = torch.polar(spec.float(), angles)
        wave = torch.istft(
            complex_spec, n_fft=cfg.n_fft, hop_length=cfg.hop_length, window=window
        )
        peak = wave.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        return (wave / peak).unsqueeze(1)
