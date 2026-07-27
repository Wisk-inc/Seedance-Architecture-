"""Command-line interface.

    seedance doctor                      # what will and will not run here
    seedance models                      # known Wan checkpoints
    seedance inspect <repo>              # download + classify any Wan repo
    seedance generate --model <repo> --prompt "..." --output out.mp4
    seedance arch [--variant seedance-2.0]   # the architecture, summarised
    seedance bench <video.mp4>           # QA metrics on a finished clip

Only ``generate`` and ``bench`` need torch; the rest run on a bare interpreter.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seedance",
        description="Seedance-shaped video generation over open Wan weights",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check this machine's dependencies and GPU")

    p_models = sub.add_parser("models", help="list known Wan checkpoints")
    p_models.add_argument("--json", action="store_true")

    p_inspect = sub.add_parser("inspect", help="download and classify a checkpoint")
    p_inspect.add_argument("model", help="HF repo id, alias or local path")
    p_inspect.add_argument("--no-download", action="store_true")
    p_inspect.add_argument("--cache-dir", default=None)
    p_inspect.add_argument("--json", action="store_true")

    p_arch = sub.add_parser("arch", help="summarise the Seedance architecture config")
    p_arch.add_argument("--variant", default="seedance-2.0")
    p_arch.add_argument("--json", action="store_true")
    p_arch.add_argument("--frames", type=int, default=97)
    p_arch.add_argument("--height", type=int, default=480)
    p_arch.add_argument("--width", type=int, default=832)

    p_gen = sub.add_parser("generate", help="generate a video")
    p_gen.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B")
    p_gen.add_argument("--prompt", required=True)
    p_gen.add_argument("--negative-prompt", default="")
    p_gen.add_argument("--output", "-o", default="seedance_out.mp4")
    p_gen.add_argument("--size", default=None, help="e.g. 832*480")
    p_gen.add_argument("--frames", type=int, default=None)
    p_gen.add_argument("--fps", type=int, default=None)
    p_gen.add_argument("--steps", type=int, default=None)
    p_gen.add_argument("--guidance", type=float, default=None)
    p_gen.add_argument("--shift", type=float, default=None)
    p_gen.add_argument("--seed", type=int, default=-1)
    p_gen.add_argument("--solver", default="unipc", choices=["unipc", "dpm++", "euler", "heun"])
    p_gen.add_argument("--image", default=None, help="first frame, for I2V")
    p_gen.add_argument("--last-image", default=None, help="last frame, for FLF2V")
    p_gen.add_argument("--source-video", default=None, help="control source for depth/pose/flow")
    p_gen.add_argument("--reference", action="append", default=[], help="reference image (repeatable)")
    p_gen.add_argument("--preset", default="fast",
                       choices=["fast", "balanced", "quality", "cinematic"])
    p_gen.add_argument("--no-enhance", action="store_true", help="skip prompt rewriting")
    p_gen.add_argument("--offline-prompt", action="store_true",
                       help="rewrite prompts with templates only, never download an LLM")
    p_gen.add_argument("--subject", default="", help="subject description locked into every shot")
    p_gen.add_argument("--storyboard", default=None,
                       help="file with one shot caption per line")
    p_gen.add_argument("--depth", action="store_true")
    p_gen.add_argument("--pose", action="store_true")
    p_gen.add_argument("--flow", action="store_true")
    p_gen.add_argument("--camera", default=None,
                       choices=["orbit", "dolly", "pan"], help="camera trajectory")
    p_gen.add_argument("--physics", default=None,
                       choices=["bouncing_ball", "collision"], help="physics proxy scene")
    p_gen.add_argument("--physics-engine", default="auto",
                       choices=["auto", "analytic", "pybullet", "mujoco"])
    p_gen.add_argument("--interpolate", type=int, default=None, help="frame-rate multiplier")
    p_gen.add_argument("--upscale", type=int, default=None)
    p_gen.add_argument("--audio", action="append", default=[], help="audio file to mix in")
    p_gen.add_argument("--device-id", type=int, default=0)
    p_gen.add_argument("--cache-dir", default=None)
    p_gen.add_argument("--prefer", default="auto", choices=["auto", "native", "diffusers"])
    p_gen.add_argument("--report", default=None, help="write the JSON report here")
    p_gen.add_argument("--dry-run", action="store_true",
                       help="resolve everything and print the plan without sampling")

    p_bench = sub.add_parser("bench", help="QA metrics for an existing video")
    p_bench.add_argument("video")
    p_bench.add_argument("--json", action="store_true")
    p_bench.add_argument("--max-frames", type=int, default=121)

    return parser


# ------------------------------------------------------------------ commands

def cmd_doctor(args: argparse.Namespace) -> int:
    from .doctor import format_report

    print(format_report())
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from .backends.registry import list_models

    models = list_models()
    if args.json:
        print(json.dumps([m.__dict__ for m in models], indent=2, default=str))
        return 0
    header = f"{'alias':<16}{'repo':<34}{'task':<7}{'params':<22}{'VRAM':<8}sizes"
    print(header)
    print("-" * len(header))
    for m in models:
        print(
            f"{m.alias:<16}{m.repo_id:<34}{m.task:<7}{m.params:<22}"
            f"{m.min_vram_gb:<8.0f}{', '.join(m.sizes)}"
        )
    print()
    print("Any other Hugging Face repo id or local path also works: the checkpoint")
    print("is inspected after download. Try: seedance inspect <owner>/<repo>")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from .backends.wan_backend import WanBackend

    backend = WanBackend(
        args.model, cache_dir=args.cache_dir, download=not args.no_download
    )
    try:
        info = backend.describe()
    except Exception as exc:
        print(f"could not inspect {args.model}: {exc}", file=sys.stderr)
        return 1
    info["defaults"] = backend.defaults()
    caps = backend.capabilities()
    info["capabilities"] = {
        "tasks": list(caps.tasks),
        "reference_images": caps.reference_images,
        "control_video": caps.control_video,
        "audio": caps.audio,
    }
    if args.json:
        print(json.dumps(info, indent=2, default=str))
        return 0
    print(f"repo      : {info['repo_id']}")
    print(f"path      : {info['path']}")
    print(f"family    : {info['family']}")
    print(f"task      : {info['task']}")
    print(f"loader    : {info['loader']}")
    print(f"dim/layers: {info['dim']}/{info['num_layers']}")
    print(f"MoE       : {info['moe']}")
    print(f"quantized : {info['quantization'] or 'no'}")
    print(f"tasks     : {', '.join(caps.tasks)}")
    print(f"defaults  : {info['defaults']}")
    for warning in info.get("warnings", []):
        print(f"warning   : {warning}")
    return 0


def cmd_arch(args: argparse.Namespace) -> int:
    from .config import get_config

    cfg = get_config(args.variant)
    if args.json:
        print(cfg.to_json())
        return 0
    c, t, h, w = cfg.latent_shape(
        cfg.frames_for_duration(args.frames / cfg.fps) if args.frames > 400 else args.frames,
        args.height,
        args.width,
    )
    params = cfg.param_estimate()
    print(f"variant            : {cfg.name}")
    print(f"VAE stride         : {cfg.vae.stride}  latent channels: {cfg.vae.latent_channels}")
    print(f"thin decoder       : {cfg.vae.thin_decoder}")
    print(f"DiT dim/heads      : {cfg.dit.dim}/{cfg.dit.num_heads}")
    print(f"spatial layers     : {cfg.dit.num_spatial_layers} (MMDiT, intra-frame, text+visual)")
    print(f"temporal layers    : {cfg.dit.num_temporal_layers} "
          f"(visual only, window {cfg.dit.temporal_window})")
    print(f"patch size         : {cfg.dit.patch_size} (DC-AE: no DiT-side patchify)")
    print(f"audio branch       : {'on' if cfg.audio.enabled else 'off'}"
          + (f", tracks {cfg.audio.tracks}" if cfg.audio.enabled else ""))
    print(f"flow               : rectified, shift {cfg.flow.shift}, "
          f"{cfg.flow.num_inference_steps} steps, solver {cfg.flow.solver}")
    print(f"refiner            : {'on' if cfg.refiner.enabled else 'off'} "
          f"{cfg.refiner.base_resolution} -> {cfg.refiner.target_resolution}")
    print(f"params (estimate)  : {params['total_m']:.0f} M")
    print(f"latent for {args.frames}f {args.height}x{args.width}: C={c} T={t} H={h} W={w}")
    print()
    print("provenance:")
    print(f"  {cfg.notes}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from .pipelines.staged import SeedancePipeline, StageSettings
    from .stages.audio_stage import AudioTrack
    from .stages.identity import IdentityConfig
    from .stages.motion import CameraTrajectory
    from .stages.physics import PhysicsScene
    from .story import Storyboard

    settings = StageSettings.preset(args.preset)
    settings.enhance_prompt = not args.no_enhance
    settings.enhancer_offline_only = args.offline_prompt
    settings.subject_lock = args.subject
    settings.use_depth = args.depth or settings.use_depth
    settings.use_pose = args.pose
    settings.use_flow = args.flow
    settings.identity = IdentityConfig(
        reference_images=list(args.reference), subject_description=args.subject
    )
    if args.interpolate is not None:
        settings.polish.interpolate = args.interpolate
    if args.upscale is not None:
        settings.polish.upscale_factor = args.upscale
    if args.audio:
        settings.audio_tracks = [
            AudioTrack(name=f"track{i}", source=path) for i, path in enumerate(args.audio)
        ]

    frames = args.frames or 81
    if args.camera:
        settings.camera = getattr(CameraTrajectory, args.camera)(frames)
    if args.physics:
        settings.physics = getattr(PhysicsScene, args.physics)(fps=args.fps or 16)
        settings.physics_engine = args.physics_engine

    storyboard = None
    if args.storyboard:
        with open(args.storyboard, "r", encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]
        per_shot = max(17, ((frames // max(len(lines), 1) - 1) // 4) * 4 + 1)
        storyboard = Storyboard.from_lines(args.prompt, lines, per_shot)

    pipe = SeedancePipeline(
        args.model,
        settings=settings,
        device_id=args.device_id,
        cache_dir=args.cache_dir,
        prefer_backend=args.prefer,
    )

    if args.dry_run:
        info = pipe.capability_report()
        info["settings"] = {
            "preset": args.preset,
            "enhance_prompt": settings.enhance_prompt,
            "guides": {
                "depth": settings.use_depth,
                "pose": settings.use_pose,
                "flow": settings.use_flow,
                "camera": bool(settings.camera),
                "physics": bool(settings.physics),
            },
            "polish": settings.polish.__dict__,
            "shots": len(storyboard.shots) if storyboard else 1,
        }
        print(json.dumps(info, indent=2, default=str))
        return 0

    def progress(stage: str, message: str) -> None:
        print(f"[{stage}] {message}", flush=True)

    result = pipe.generate(
        args.prompt,
        negative_prompt=args.negative_prompt,
        size=args.size,
        num_frames=args.frames,
        fps=args.fps,
        steps=args.steps,
        guidance_scale=args.guidance,
        shift=args.shift,
        seed=args.seed,
        solver=args.solver,
        image=args.image,
        last_image=args.last_image,
        source_video=args.source_video,
        storyboard=storyboard,
        output=args.output,
        progress=progress,
    )
    print()
    print(result.summary())
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(result.to_json())
        print(f"report: {args.report}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .runtime.io import load_video
    from .stages.polish import temporal_qa

    video = load_video(args.video, max_frames=args.max_frames)
    report = temporal_qa(video)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    for key, value in report.items():
        if key == "warnings":
            continue
        print(f"{key:<24}{value:.3f}")
    for warning in report.get("warnings", []):
        print(f"warning: {warning}")
    return 0


COMMANDS = {
    "doctor": cmd_doctor,
    "models": cmd_models,
    "inspect": cmd_inspect,
    "arch": cmd_arch,
    "generate": cmd_generate,
    "bench": cmd_bench,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return COMMANDS[args.command](args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        if args.verbose:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
