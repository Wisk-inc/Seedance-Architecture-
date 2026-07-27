"""Stage 3 in isolation: simulate a scene, render the proxy, condition on it.

    python -m seedance.examples.physics_guided --scene collision --guide-only

``--guide-only`` writes just the proxy video, which is the fastest way to check
that your scene looks the way you intended before spending GPU time on it.

Reminder: this is the PhysGen pattern (arXiv:2409.18964), not a reproduction of
Seedance. Seedance has no physics engine -- its plausibility is emergent from
data and reward modelling, and it still fails on complex fluid and multi-body
scenes.
"""

from __future__ import annotations

import argparse

from seedance.stages.physics import Body, PhysicsProxyRenderer, PhysicsScene, available_engines


def build_scene(name: str, fps: int) -> PhysicsScene:
    if name == "bouncing_ball":
        return PhysicsScene.bouncing_ball(height=2.2, fps=fps, duration_s=3.0)
    if name == "collision":
        return PhysicsScene.collision(fps=fps, duration_s=3.0)
    if name == "break_rack":
        scene = PhysicsScene(fps=fps, duration_s=3.0)
        scene.add(Body(name="cue", position=(-2.0, 0.12, 0.0), velocity=(4.0, 0.0, 0.0),
                       radius=0.12, colour=(0.95, 0.95, 0.9)))
        for i in range(3):
            for j in range(i + 1):
                scene.add(
                    Body(
                        name=f"ball{i}{j}",
                        position=(0.6 + i * 0.22, 0.12, -i * 0.11 + j * 0.22),
                        radius=0.12,
                        colour=(0.85, 0.25 + 0.2 * j, 0.2),
                    )
                )
        return scene
    raise SystemExit(f"unknown scene {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="bouncing_ball",
                        choices=["bouncing_ball", "collision", "break_rack"])
    parser.add_argument("--engine", default="auto")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--guide-output", default="physics_guide.mp4")
    parser.add_argument("--guide-only", action="store_true")
    parser.add_argument("--model", default="Wan-AI/Wan2.1-VACE-1.3B")
    parser.add_argument("--prompt", default="a glossy red ball bouncing on wet asphalt at night")
    parser.add_argument("--output", default="physics_out.mp4")
    args = parser.parse_args()

    from seedance.backends.base import parse_size
    from seedance.runtime.io import save_video

    width, height = parse_size(args.size)
    print("available engines:", ", ".join(available_engines()))

    scene = build_scene(args.scene, args.fps)
    renderer = PhysicsProxyRenderer(args.engine, height=height, width=width)
    print(f"simulating with: {renderer.engine}")
    guide = renderer.guide_for_request(
        scene, num_frames=args.frames, height=height, width=width
    )
    save_video(guide, args.guide_output, fps=args.fps)
    print(f"guide video: {args.guide_output}")
    if args.guide_only:
        return 0

    from seedance import SeedancePipeline
    from seedance.pipelines.staged import StageSettings

    settings = StageSettings.preset("balanced")
    settings.enhancer_offline_only = True
    settings.physics = scene
    settings.physics_engine = args.engine
    settings.physics_weight = 1.0

    pipe = SeedancePipeline(args.model, settings=settings)
    caps = pipe.backend.capabilities()
    if not caps.control_video:
        print(
            f"warning: {args.model} has no control path, so the guide cannot be "
            "applied. Use a VACE checkpoint (Wan-AI/Wan2.1-VACE-1.3B)."
        )
    result = pipe.generate(
        args.prompt,
        size=args.size,
        num_frames=args.frames,
        fps=args.fps,
        output=args.output,
        progress=lambda stage, message: print(f"[{stage}] {message}", flush=True),
    )
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
