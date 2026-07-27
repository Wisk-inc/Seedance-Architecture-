"""Exercise the native Seedance architecture: shapes, a training step, sampling.

    python -m seedance.examples.native_architecture --variant dev-tiny

There are no public Seedance weights, so sampling here produces noise. What
this script demonstrates is that the architecture is complete and trainable:
the VAE round-trips, the decoupled MMDiT accepts conditioning and multi-shot
positions, the rectified-flow loss produces gradients, and the audio bridge is
a no-op at initialisation.
"""

from __future__ import annotations

import argparse

import torch

from seedance.config import get_config
from seedance.pipelines.native import SeedanceNativePipeline
from seedance.story import Shot, Storyboard


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="dev-tiny",
                        help="dev-tiny runs on CPU in seconds; the others need a GPU")
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = get_config(args.variant)
    print(f"variant: {cfg.name}")
    print(f"VAE stride {cfg.vae.stride}, C={cfg.vae.latent_channels}, "
          f"thin decoder={cfg.vae.thin_decoder}")
    print(f"DiT: {cfg.dit.num_spatial_layers} spatial (MMDiT) + "
          f"{cfg.dit.num_temporal_layers} temporal (window {cfg.dit.temporal_window})")

    pipe = SeedanceNativePipeline(cfg, device=args.device)
    print("parameters (M):", pipe.parameter_report())
    print("FLOPs estimate (TFLOP/forward):",
          round(pipe.dit.flops_estimate(args.frames, args.height, args.width), 2))

    # --- shapes -----------------------------------------------------------
    c, t, h, w = cfg.latent_shape(args.frames, args.height, args.width)
    print(f"latent for {args.frames}x{args.height}x{args.width}: C={c} T={t} H={h} W={w}")

    video = torch.randn(1, 3, args.frames, args.height, args.width).clamp(-1, 1)
    with torch.no_grad():
        latents = pipe.vae.encode(video, sample=False)
        decoded = pipe.vae.decode(latents)
    print(f"VAE round trip: {tuple(video.shape)} -> {tuple(latents.shape)} -> {tuple(decoded.shape)}")

    # --- one training step -------------------------------------------------
    loss, logs = pipe.training_step(video, ["a test clip"])
    loss.backward()
    grad_norm = sum(
        float(p.grad.norm()) for p in pipe.dit.parameters() if p.grad is not None
    )
    print(f"training step: loss={logs['flow_loss']:.4f}, total grad norm={grad_norm:.4f}")

    # --- multi-shot sampling ----------------------------------------------
    board = Storyboard("two shots")
    board.add(Shot("the door opens", args.frames // 2 + 1))
    board.add(Shot("she steps outside", args.frames // 2))
    result = pipe.generate(
        "a test prompt",
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_steps=args.steps,
        storyboard=board,
        decode=True,
    )
    print(result.summary())
    print("report:", result.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
