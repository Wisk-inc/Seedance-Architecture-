"""Gradio UI for the staged pipeline — locally or on a Hugging Face Space.

    python -m seedance.app_gradio --model Wan-AI/Wan2.1-T2V-1.3B

On a Space, set the model with the ``SEEDANCE_MODEL`` environment variable and
point the Space's entry point at this file. The default is the 1.3B checkpoint,
which is the only one that fits comfortably on the smaller GPU tiers; CPU tiers
cannot run any of them (use ``--model mock`` to check the UI works).

The model is loaded lazily on the first generate, so the Space starts fast and
does not download 69 GB before anyone presses a button.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

DESCRIPTION = """
# Seedance-staged video generation

Runs any **Wan** checkpoint through a Seedance-shaped pipeline: dense-caption
prompt rewriting → identity/motion control → optional physics-proxy
conditioning → colour stabilisation, interpolation and upscaling → QA.

The Seedance *architecture* is implemented in this package too (decoupled
spatial/temporal MMDiT, causal (4,16,16)/C=48 VAE, multishot MM-RoPE,
dual-branch audio, RewardDance/DanceGRPO), but **there are no public Seedance
weights** — what generates here is Wan.
"""


class _State:
    """Holds the lazily-constructed pipeline so the Space boots instantly."""

    def __init__(self, model: str, device_id: int = 0):
        self.model = model
        self.device_id = device_id
        self.pipe = None
        self.preset: Optional[str] = None

    def get(self, preset: str):
        from seedance.pipelines.staged import SeedancePipeline, StageSettings

        if self.pipe is not None and self.preset == preset:
            return self.pipe
        settings = StageSettings.preset(preset)
        settings.enhancer_offline_only = os.environ.get("SEEDANCE_OFFLINE_PROMPT", "1") == "1"
        if self.pipe is None:
            self.pipe = SeedancePipeline(
                self.model, settings=settings, device_id=self.device_id
            )
        else:
            self.pipe.settings = settings
        self.preset = preset
        return self.pipe


def build_ui(state: _State):
    import gradio as gr

    def generate(
        prompt: str,
        negative_prompt: str,
        preset: str,
        size: str,
        frames: int,
        steps: int,
        guidance: float,
        seed: int,
        physics: str,
        interpolate: int,
        upscale: int,
        image,
        progress=gr.Progress(),
    ):
        import tempfile

        from seedance.stages.physics import PhysicsScene

        pipe = state.get(preset)
        if physics != "none":
            pipe.settings.physics = getattr(PhysicsScene, physics)(fps=16)
        else:
            pipe.settings.physics = None
        pipe.settings.polish.interpolate = int(interpolate)
        pipe.settings.polish.upscale_factor = int(upscale)

        out_path = os.path.join(tempfile.mkdtemp(prefix="seedance_"), "out.mp4")
        seen = {"stage": ""}

        def report(stage: str, message: str) -> None:
            seen["stage"] = stage
            progress(0.0, desc=f"{stage}: {message}")

        result = pipe.generate(
            prompt,
            negative_prompt=negative_prompt,
            size=size,
            num_frames=int(frames),
            steps=int(steps),
            guidance_scale=float(guidance),
            seed=int(seed),
            image=image,
            task="i2v" if image is not None else None,
            output=out_path,
            progress=report,
        )
        summary = result.summary()
        if result.warnings:
            summary += "\n\n" + "\n".join(f"- {w}" for w in result.warnings)
        return result.path, result.enhanced_prompt, summary

    with gr.Blocks(title="Seedance staged pipeline") as demo:
        gr.Markdown(DESCRIPTION)
        gr.Markdown(f"**Model:** `{state.model}`")

        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    value="a red fox trotting through fresh snow at dawn, breath visible",
                )
                negative = gr.Textbox(label="Negative prompt", lines=1, value="")
                image = gr.Image(label="First frame (optional — switches to I2V)", type="pil")
            with gr.Column(scale=2):
                preset = gr.Radio(
                    ["fast", "balanced", "quality", "cinematic"],
                    value="balanced",
                    label="Preset",
                    info="controls prompt rewriting and the stage-4 chain",
                )
                size = gr.Dropdown(
                    ["832*480", "480*832", "1280*720", "720*1280"],
                    value="832*480",
                    label="Resolution",
                )
                frames = gr.Slider(17, 121, value=81, step=4, label="Frames (4n+1)")
                steps = gr.Slider(10, 60, value=40, step=1, label="Sampling steps")
                guidance = gr.Slider(1.0, 10.0, value=5.0, step=0.1, label="Guidance")
                seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                with gr.Accordion("Stage options", open=False):
                    physics = gr.Radio(
                        ["none", "bouncing_ball", "collision"],
                        value="none",
                        label="Physics proxy (needs a VACE checkpoint to apply)",
                    )
                    interpolate = gr.Slider(1, 4, value=1, step=1, label="Frame interpolation ×")
                    upscale = gr.Slider(1, 4, value=1, step=1, label="Upscale ×")

        run = gr.Button("Generate", variant="primary")
        with gr.Row():
            video = gr.Video(label="Result")
            with gr.Column():
                enhanced = gr.Textbox(label="Rewritten prompt", lines=4)
                summary = gr.Textbox(label="Report", lines=10)

        run.click(
            generate,
            inputs=[prompt, negative, preset, size, frames, steps, guidance, seed,
                    physics, interpolate, upscale, image],
            outputs=[video, enhanced, summary],
        )

        gr.Examples(
            examples=[
                ["a red fox trotting through fresh snow at dawn", "balanced", "832*480"],
                ["slow dolly through a neon-lit alley after rain", "cinematic", "832*480"],
                ["a glass marble rolls off a table and bounces twice", "balanced", "832*480"],
            ],
            inputs=[prompt, preset, size],
        )
    return demo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("SEEDANCE_MODEL", "Wan-AI/Wan2.1-T2V-1.3B"))
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_ui(_State(args.model, args.device_id))
    demo.queue().launch(
        server_name=args.server_name, server_port=args.server_port, share=args.share
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
