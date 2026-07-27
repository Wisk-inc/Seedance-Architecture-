"""Temporally-causal 3D VAE.

``[1.0]``: "Following MAGVIT, a temporally-causal convolutional encoder/decoder
jointly compresses space and time. Raw pixels ``(T'+1, H', W', 3)`` map to
latents ``(T+1, H, W, C)`` with downsampling ratios ``(r_t, r_h, r_w) =
(4, 16, 16)`` and ``C = 48``." Losses are L1 + KL + LPIPS + adversarial with a
PatchGAN-style hybrid discriminator modelling appearance and motion together.

``[1.0]`` also describes the **Thin VAE decoder**: the stages closest to pixel
space dominate decode latency, so their channel widths are narrowed and the
decoder is retrained with a frozen encoder, giving ~2x faster decode.

Temporal causality is what makes the ``T'+1 -> T+1`` shape work and what allows
streaming/chunked decode of arbitrarily long video: frame ``t`` never sees
frame ``t+1``, so a rolling cache of the last ``kernel-1`` frames is sufficient
state to continue.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import VAEConfig

__all__ = [
    "CausalConv3d",
    "CausalGroupNorm",
    "DiagonalGaussian",
    "TemporallyCausalVAE",
    "HybridPatchDiscriminator",
]


class CausalConv3d(nn.Conv3d):
    """3D convolution with left-only padding in time.

    ``cache`` carries the last ``kt - 1`` input frames so a long video can be
    encoded/decoded in chunks with bit-identical results to a single pass.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size=3, stride=1, dilation=1, **kw):
        kt, kh, kw_ = _triple(kernel_size)
        self._time_pad = (kt - 1) * (dilation if isinstance(dilation, int) else dilation[0])
        self._space_pad = (kh // 2, kw_ // 2)
        super().__init__(
            in_ch,
            out_ch,
            kernel_size=(kt, kh, kw_),
            stride=_triple(stride),
            padding=(0, self._space_pad[0], self._space_pad[1]),
            dilation=dilation,
            **kw,
        )

    def forward(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None):
        """Returns ``(output, new_cache)``; ``new_cache`` is ``None`` when unused."""
        if self._time_pad == 0:
            return super().forward(x), None
        if cache is None:
            pad = x[:, :, :1].repeat(1, 1, self._time_pad, 1, 1) * 0  # zero pad
            x_in = torch.cat([pad, x], dim=2)
        else:
            x_in = torch.cat([cache.to(x.dtype), x], dim=2)
        new_cache = x_in[:, :, -self._time_pad :].detach()
        return super().forward(x_in), new_cache


def _triple(v) -> Tuple[int, int, int]:
    if isinstance(v, int):
        return (v, v, v)
    v = tuple(v)
    assert len(v) == 3
    return v  # type: ignore[return-value]


class _CacheRunner:
    """Threads per-conv caches through a module list during chunked inference."""

    def __init__(self) -> None:
        self.caches: List[Optional[torch.Tensor]] = []
        self.idx = 0
        self.enabled = False

    def reset(self, size: int) -> None:
        self.caches = [None] * size
        self.idx = 0

    def step(self, conv: CausalConv3d, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return conv(x)[0]
        while len(self.caches) <= self.idx:
            self.caches.append(None)
        out, cache = conv(x, self.caches[self.idx])
        self.caches[self.idx] = cache
        self.idx += 1
        return out


class CausalGroupNorm(nn.GroupNorm):
    """GroupNorm that never mixes statistics across time.

    ``nn.GroupNorm`` on a ``[B, C, T, H, W]`` tensor normalises over ``T`` as
    well, which silently breaks temporal causality: frame 0's output would then
    depend on the last frame's statistics. Folding ``T`` into the batch axis
    keeps each frame independent, which is what makes chunked/streaming decode
    equal to a single pass.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            return super().forward(x)
        b, c, t, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        y = super().forward(y)
        return y.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)


def group_norm(channels: int) -> CausalGroupNorm:
    groups = 32 if channels >= 32 and channels % 32 == 0 else 1
    return CausalGroupNorm(groups, channels, eps=1e-6)


class ResnetBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0, runner=None):
        super().__init__()
        self.norm1 = group_norm(in_ch)
        self.conv1 = CausalConv3d(in_ch, out_ch, 3)
        self.norm2 = group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CausalConv3d(out_ch, out_ch, 3)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.runner = runner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.runner.step(self.conv1, h) if self.runner else self.conv1(h)[0]
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.runner.step(self.conv2, h) if self.runner else self.conv2(h)[0]
        return h + self.skip(x)


class AttnBlock2D(nn.Module):
    """Per-frame self-attention; kept 2D so temporal causality is preserved."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch, eps=1e-6) if ch >= 32 else nn.GroupNorm(1, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        y = self.norm(y)
        q, k, v = self.qkv(y).chunk(3, dim=1)
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b * t, c, h, w)
        out = self.proj(out)
        out = out.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return x + out


class Downsample(nn.Module):
    def __init__(self, ch: int, temporal: bool, runner=None):
        super().__init__()
        stride = (2 if temporal else 1, 2, 2)
        self.conv = CausalConv3d(ch, ch, kernel_size=3, stride=stride)
        self.runner = runner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.runner.step(self.conv, x) if self.runner else self.conv(x)[0]


class Upsample(nn.Module):
    """Nearest-neighbour upsample + causal conv.

    Temporal upsampling duplicates each latent frame and then drops the first
    duplicate so that ``T+1`` latent frames expand to ``4T+1`` pixel frames --
    the inverse of the encoder's ``T'+1 -> T+1`` shape contract.
    """

    def __init__(self, ch: int, out_ch: int, temporal: bool, runner=None):
        super().__init__()
        self.temporal = temporal
        self.conv = CausalConv3d(ch, out_ch, 3)
        self.runner = runner

    def forward(self, x: torch.Tensor, first_chunk: bool = True) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="nearest")
        if self.temporal:
            x = x.repeat_interleave(2, dim=2)
            if first_chunk:
                x = x[:, :, 1:]
        return self.runner.step(self.conv, x) if self.runner else self.conv(x)[0]


class Encoder(nn.Module):
    def __init__(self, cfg: VAEConfig, runner=None):
        super().__init__()
        self.cfg = cfg
        chs = [cfg.base_channels * m for m in cfg.channel_multipliers]
        self.conv_in = CausalConv3d(3, chs[0], 3)
        self.runner = runner
        stages: List[nn.Module] = []
        for i in range(len(chs)):
            blocks: List[nn.Module] = []
            in_ch = chs[max(0, i - 1)] if i > 0 else chs[0]
            for j in range(cfg.num_res_blocks):
                blocks.append(ResnetBlock3D(in_ch if j == 0 else chs[i], chs[i], cfg.dropout, runner))
            if i < len(chs) - 1:
                temporal = bool(cfg.temporal_downsample_stages[i]) if i < len(
                    cfg.temporal_downsample_stages
                ) else False
                blocks.append(Downsample(chs[i], temporal, runner))
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        mid = chs[-1]
        self.mid_block1 = ResnetBlock3D(mid, mid, cfg.dropout, runner)
        self.mid_attn = AttnBlock2D(mid)
        self.mid_block2 = ResnetBlock3D(mid, mid, cfg.dropout, runner)
        self.norm_out = group_norm(mid)
        self.conv_out = CausalConv3d(mid, cfg.latent_channels * 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.runner.step(self.conv_in, x) if self.runner else self.conv_in(x)[0]
        for stage in self.stages:
            h = stage(h)
        h = self.mid_block2(self.mid_attn(self.mid_block1(h)))
        h = F.silu(self.norm_out(h))
        return self.runner.step(self.conv_out, h) if self.runner else self.conv_out(h)[0]


class Decoder(nn.Module):
    """Mirror of the encoder, optionally thinned near pixel space."""

    def __init__(self, cfg: VAEConfig, thin: bool = False, runner=None):
        super().__init__()
        self.cfg = cfg
        self.runner = runner
        mult = list(cfg.channel_multipliers)
        chs = [cfg.base_channels * m for m in mult]
        if thin:
            shrink = list(cfg.thin_decoder_shrink)
            shrink += [1.0] * (len(chs) - len(shrink))
            # index 0 is the stage closest to pixel space
            chs = [max(8, int(round(c * shrink[i]))) for i, c in enumerate(chs)]
        self.channels = chs
        mid = chs[-1]
        self.conv_in = CausalConv3d(cfg.latent_channels, mid, 3)
        self.mid_block1 = ResnetBlock3D(mid, mid, cfg.dropout, runner)
        self.mid_attn = AttnBlock2D(mid)
        self.mid_block2 = ResnetBlock3D(mid, mid, cfg.dropout, runner)

        ups: List[nn.Module] = []
        for i in reversed(range(len(chs))):
            blocks: List[nn.Module] = []
            in_ch = chs[i + 1] if i + 1 < len(chs) else mid
            for j in range(cfg.num_res_blocks + 1):
                blocks.append(ResnetBlock3D(in_ch if j == 0 else chs[i], chs[i], cfg.dropout, runner))
            ups.append(nn.ModuleList(blocks))
        self.up_blocks = nn.ModuleList(ups)

        # One upsampler per level transition, in the same order as up_blocks:
        # up_blocks[k] works at level i = len(chs) - 1 - k and emits chs[i]
        # channels, so upsamplers[k] must take chs[i] and invert the encoder's
        # downsample between levels i-1 and i.
        upsamplers: List[nn.Module] = []
        for i in reversed(range(1, len(chs))):
            temporal = (
                bool(cfg.temporal_downsample_stages[i - 1])
                if i - 1 < len(cfg.temporal_downsample_stages)
                else False
            )
            upsamplers.append(Upsample(chs[i], chs[i], temporal, runner))
        self.upsamplers = nn.ModuleList(upsamplers)
        self.norm_out = group_norm(chs[0])
        self.conv_out = CausalConv3d(chs[0], 3, 3)

    def forward(self, z: torch.Tensor, first_chunk: bool = True) -> torch.Tensor:
        h = self.runner.step(self.conv_in, z) if self.runner else self.conv_in(z)[0]
        h = self.mid_block2(self.mid_attn(self.mid_block1(h)))
        n = len(self.up_blocks)
        for i, blocks in enumerate(self.up_blocks):
            for block in blocks:
                h = block(h)
            if i < n - 1:
                h = self.upsamplers[i](h, first_chunk=first_chunk)
        h = F.silu(self.norm_out(h))
        return self.runner.step(self.conv_out, h) if self.runner else self.conv_out(h)[0]


class DiagonalGaussian:
    """Latent distribution with the KL term used in the VAE objective."""

    def __init__(self, parameters: torch.Tensor):
        self.mean, self.logvar = parameters.chunk(2, dim=1)
        self.logvar = self.logvar.clamp(-30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        noise = torch.randn(
            self.mean.shape, device=self.mean.device, dtype=self.mean.dtype, generator=generator
        )
        return self.mean + self.std * noise

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        return 0.5 * torch.sum(
            self.mean.pow(2) + self.logvar.exp() - 1.0 - self.logvar,
            dim=list(range(1, self.mean.dim())),
        )


class TemporallyCausalVAE(nn.Module):
    """The full VAE, with a standard decoder and an optional Thin decoder."""

    def __init__(self, cfg: Optional[VAEConfig] = None):
        super().__init__()
        self.cfg = cfg or VAEConfig()
        self._enc_runner = _CacheRunner()
        self._dec_runner = _CacheRunner()
        self.encoder = Encoder(self.cfg, self._enc_runner)
        self.decoder = Decoder(self.cfg, thin=False, runner=self._dec_runner)
        self.thin_decoder = (
            Decoder(self.cfg, thin=True, runner=self._dec_runner) if self.cfg.thin_decoder else None
        )
        self.register_buffer(
            "latent_mean", torch.zeros(self.cfg.latent_channels), persistent=True
        )
        self.register_buffer(
            "latent_std", torch.ones(self.cfg.latent_channels), persistent=True
        )
        self._use_thin = False

    # ------------------------------------------------------------------ shape
    @property
    def stride(self) -> Tuple[int, int, int]:
        return self.cfg.stride

    def latent_frames(self, pixel_frames: int) -> int:
        rt = self.cfg.temporal_downsample
        if (pixel_frames - 1) % rt != 0:
            raise ValueError(f"pixel frames must be 1 + k*{rt}, got {pixel_frames}")
        return (pixel_frames - 1) // rt + 1

    def pixel_frames(self, latent_frames: int) -> int:
        return (latent_frames - 1) * self.cfg.temporal_downsample + 1

    def use_thin_decoder(self, enable: bool = True) -> None:
        if enable and self.thin_decoder is None:
            raise RuntimeError("this VAE was built without a thin decoder")
        self._use_thin = enable

    def active_decoder(self) -> Decoder:
        return self.thin_decoder if (self._use_thin and self.thin_decoder) else self.decoder

    # --------------------------------------------------------------- encode
    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        mean = self.latent_mean.view(1, -1, 1, 1, 1).to(z)
        std = self.latent_std.view(1, -1, 1, 1, 1).to(z)
        return (z - mean) / std

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        mean = self.latent_mean.view(1, -1, 1, 1, 1).to(z)
        std = self.latent_std.view(1, -1, 1, 1, 1).to(z)
        return z * std + mean

    def encode(
        self,
        video: torch.Tensor,
        *,
        sample: bool = True,
        generator: Optional[torch.Generator] = None,
        chunk_frames: int = 0,
    ) -> torch.Tensor:
        """Encode ``[B, 3, T, H, W]`` in ``[-1, 1]`` to normalised latents."""
        params = self._encode_params(video, chunk_frames=chunk_frames)
        dist = DiagonalGaussian(params)
        z = dist.sample(generator) if sample else dist.mode()
        return self.normalize(z)

    def encode_dist(self, video: torch.Tensor) -> DiagonalGaussian:
        return DiagonalGaussian(self._encode_params(video))

    def _encode_params(self, video: torch.Tensor, chunk_frames: int = 0) -> torch.Tensor:
        rt = self.cfg.temporal_downsample
        chunk_frames = chunk_frames or self.cfg.tile_temporal
        if not chunk_frames:
            self._enc_runner.enabled = False
            return self.encoder(video)
        # Streaming: first chunk carries the +1 frame, later chunks are
        # multiples of r_t so latent frames line up exactly.
        self._enc_runner.enabled = True
        self._enc_runner.reset(0)
        outputs = []
        t = video.shape[2]
        first = min(t, 1 + chunk_frames * rt)
        bounds = [(0, first)]
        cursor = first
        while cursor < t:
            end = min(t, cursor + chunk_frames * rt)
            bounds.append((cursor, end))
            cursor = end
        for start, end in bounds:
            self._enc_runner.idx = 0
            outputs.append(self.encoder(video[:, :, start:end]))
        self._enc_runner.enabled = False
        return torch.cat(outputs, dim=2)

    # --------------------------------------------------------------- decode
    def decode(
        self, z: torch.Tensor, *, chunk_frames: int = 0, normalized: bool = True
    ) -> torch.Tensor:
        """Decode latents to ``[B, 3, T, H, W]`` in ``[-1, 1]``."""
        z = self.denormalize(z) if normalized else z
        decoder = self.active_decoder()
        chunk_frames = chunk_frames or self.cfg.tile_temporal
        if not chunk_frames:
            self._dec_runner.enabled = False
            return decoder(z, first_chunk=True).clamp(-1, 1)
        self._dec_runner.enabled = True
        self._dec_runner.reset(0)
        outputs = []
        t = z.shape[2]
        cursor = 0
        first = True
        while cursor < t:
            end = min(t, cursor + max(1, chunk_frames))
            self._dec_runner.idx = 0
            outputs.append(decoder(z[:, :, cursor:end], first_chunk=first))
            cursor = end
            first = False
        self._dec_runner.enabled = False
        return torch.cat(outputs, dim=2).clamp(-1, 1)

    def forward(
        self, video: torch.Tensor, *, sample: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, DiagonalGaussian]:
        dist = self.encode_dist(video)
        z = dist.sample() if sample else dist.mode()
        recon = self.active_decoder()(z)
        return recon, self.normalize(z), dist

    # ------------------------------------------------------------ statistics
    @torch.no_grad()
    def fit_latent_stats(self, videos: Sequence[torch.Tensor]) -> None:
        """Set the per-channel latent mean/std used by :meth:`normalize`.

        Latent whitening matters more than it looks: the DiT's noise schedule
        assumes unit-variance inputs, and an unwhitened VAE silently shifts the
        effective SNR of every timestep.
        """
        sums = torch.zeros(self.cfg.latent_channels)
        sqs = torch.zeros(self.cfg.latent_channels)
        count = 0
        for video in videos:
            params = self._encode_params(video)
            z = DiagonalGaussian(params).mode().float()
            flat = z.permute(1, 0, 2, 3, 4).reshape(z.shape[1], -1).cpu()
            sums += flat.sum(dim=1)
            sqs += flat.pow(2).sum(dim=1)
            count += flat.shape[1]
        if count == 0:
            return
        mean = sums / count
        var = (sqs / count) - mean.pow(2)
        self.latent_mean.copy_(mean)
        self.latent_std.copy_(var.clamp(min=1e-8).sqrt())

    # ----------------------------------------------------------------- loss
    def loss(
        self,
        video: torch.Tensor,
        recon: torch.Tensor,
        dist: DiagonalGaussian,
        *,
        lpips_fn=None,
        disc_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        cfg = self.cfg
        l1 = (video - recon).abs().mean()
        kl = dist.kl().mean() / video[0].numel()
        total = cfg.loss_l1 * l1 + cfg.loss_kl * kl
        logs = {"l1": float(l1.detach()), "kl": float(kl.detach())}
        if lpips_fn is not None:
            b, c, t, h, w = video.shape
            flat_real = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            flat_fake = recon.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            perceptual = lpips_fn(flat_real, flat_fake).mean()
            total = total + cfg.loss_lpips * perceptual
            logs["lpips"] = float(perceptual.detach())
        if disc_logits is not None:
            # non-saturating generator loss
            adv = F.binary_cross_entropy_with_logits(
                disc_logits, torch.ones_like(disc_logits)
            )
            total = total + cfg.loss_adv * adv
            logs["adv"] = float(adv.detach())
        logs["total"] = float(total.detach())
        return total, logs


class HybridPatchDiscriminator(nn.Module):
    """PatchGAN-style discriminator over appearance *and* motion.

    ``[1.0]`` specifies "a PatchGAN-style hybrid discriminator that models
    appearance and motion simultaneously". Implemented as two heads sharing an
    input: a 2D head over sampled frames (appearance) and a 3D head over the
    clip (motion). Logits are concatenated so the generator sees both.
    """

    def __init__(self, base: int = 64, layers: int = 3, in_ch: int = 3):
        super().__init__()

        def block2d(i: int, o: int, stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(i, o, 4, stride, 1), nn.GroupNorm(1, o), nn.LeakyReLU(0.2, True)
            )

        def block3d(i: int, o: int, stride) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv3d(i, o, (3, 4, 4), stride, (1, 1, 1)),
                nn.GroupNorm(1, o),
                nn.LeakyReLU(0.2, True),
            )

        ch = base
        app: List[nn.Module] = [nn.Conv2d(in_ch, ch, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        mot: List[nn.Module] = [nn.Conv3d(in_ch, ch, (3, 4, 4), (1, 2, 2), (1, 1, 1)), nn.LeakyReLU(0.2, True)]
        for i in range(layers):
            nxt = min(ch * 2, base * 8)
            stride = 2 if i < layers - 1 else 1
            app.append(block2d(ch, nxt, stride))
            mot.append(block3d(ch, nxt, (2 if i < layers - 1 else 1, stride, stride)))
            ch = nxt
        app.append(nn.Conv2d(ch, 1, 4, 1, 1))
        mot.append(nn.Conv3d(ch, 1, (3, 4, 4), 1, (1, 1, 1)))
        self.appearance = nn.Sequential(*app)
        self.motion = nn.Sequential(*mot)

    def forward(self, video: torch.Tensor, frame_stride: int = 4) -> torch.Tensor:
        b, c, t, h, w = video.shape
        frames = video[:, :, ::frame_stride]
        nf = frames.shape[2]
        app_in = frames.permute(0, 2, 1, 3, 4).reshape(b * nf, c, h, w)
        app = self.appearance(app_in).flatten(1)
        mot = self.motion(video).flatten(1)
        return torch.cat([app.view(b, -1), mot.view(b, -1)], dim=1)

    def loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        real = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        fake = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        return 0.5 * (real + fake)
