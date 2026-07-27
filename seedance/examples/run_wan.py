"""Download a Wan model from Hugging Face and generate through the staged pipeline.

    python -m seedance.examples.run_wan --model Wan-AI/Wan2.1-T2V-1.3B \
        --prompt "a red fox trotting through fresh snow at dawn"

Everything here is also reachable from the CLI (``seedance generate``); this
file exists to show the Python API in one screen.
"""

from __future__ import annotations

import argparse

from seedance import SeedancePipeline
from seedance.pipelines.staged import StageSettings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--prompt", default="a red fox trotting through fresh snow at dawn")
    parser.add_argument("--output", default="fox.mp4")
    parser.add_argument("--preset", default="balanced")
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--size", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    settings = StageSettings.preset(args.preset)
    # Rewrite prompts with templates instead of downloading a 7B model. Drop
    # this line if you want the full Qwen prompt-engineering stage.
    settings.enhancer_offline_only = True

    pipe = SeedancePipeline(args.model, settings=settings)
    print("capabilities:", pipe.capability_report())

    result = pipe.generate(
        args.prompt,
        size=args.size,
        num_frames=args.frames,
        seed=args.seed,
        output=args.output,
        progress=lambda stage, message: print(f"[{stage}] {message}", flush=True),
    )
    print()
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
