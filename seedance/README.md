# seedance

A Seedance-architecture implementation, plus a pipeline that runs it over any
Wan checkpoint you can download from Hugging Face.

The package has two halves and it matters which is which:

| | what it is | does it make video? |
|---|---|---|
| `seedance.models`, `seedance.flow`, `seedance.reward` | the Seedance architecture, implemented from the published reports | **no** — there are no public Seedance weights |
| `seedance.pipelines`, `seedance.backends`, `seedance.stages` | the same four-stage recipe applied to open Wan weights | **yes** |

Nothing here is a leak or a conversion of ByteDance weights. The architecture
half is what the papers describe, built so it is trainable and inspectable; the
pipeline half is what you run today.

---

## Install

From the repo root (this package lives inside the Wan2.1 tree so it can reuse
Wan's loaders):

```bash
pip install -r requirements.txt              # Wan itself
pip install -r seedance/requirements.txt     # this package's extras
```

Then check the machine:

```bash
python -m seedance.cli doctor
```

It prints what is installed, how much VRAM you have, which optional stages will
engage rather than silently falling back, and which checkpoint to start with.

## Generate

```bash
python -m seedance.cli generate \
  --model Wan-AI/Wan2.1-T2V-1.3B \
  --prompt "a red fox trotting through fresh snow at dawn" \
  --preset balanced \
  --output fox.mp4
```

The model is downloaded, inspected, and driven through all four stages. Any Wan
repo id works — `Wan-AI/Wan2.2-TI2V-5B`, an I2V checkpoint, a VACE checkpoint, a
community fine-tune, or a local directory. From Python:

```python
from seedance import SeedancePipeline

pipe = SeedancePipeline("Wan-AI/Wan2.1-T2V-1.3B")
result = pipe.generate("a red fox trotting through fresh snow", output="fox.mp4")
print(result.summary())
```

Other commands:

```bash
python -m seedance.cli models                      # known checkpoints
python -m seedance.cli inspect <repo-or-path>      # what is this checkpoint?
python -m seedance.cli arch --variant seedance-2.0 # the architecture, summarised
python -m seedance.cli bench out.mp4               # QA metrics on a finished clip
python -m seedance.cli generate ... --dry-run      # resolve the plan, sample nothing
```

## The four stages

| stage | what it does | Seedance analog |
|---|---|---|
| 1. prompt | dense-caption rewriting, storyboard expansion, subject locking | the Qwen2.5-14B PE model (SFT + DPO) |
| 2. identity + motion | VACE/Phantom references, LoRA, caption locking; depth / pose / flow / camera / trajectory guides | *none* — Seedance gets this from multi-shot training + RLHF |
| 3. physics | simulate a proxy scene, render it, condition on it | *none* — Seedance has **no physics engine** |
| 4. polish | colour stabilisation → interpolation → face restore → upscale → QA | partly the diffusion refiner; mostly not needed at Seedance's quality |

Presets: `fast`, `balanced`, `quality`, `cinematic`.

```python
from seedance.pipelines.staged import StageSettings
from seedance.stages.physics import PhysicsScene

settings = StageSettings.preset("quality")
settings.subject_lock = "A woman in a red wool coat, shoulder-length dark hair"
settings.physics = PhysicsScene.collision(fps=16)
settings.use_depth = True
```

Stage order in stage 4 is deliberate: interpolating before fixing colour drift
propagates the drift; upscaling before restoring faces amplifies artifacts.

## What is *not* reproducible from open weights

`pipe.capability_report()` returns this list at runtime, and the pipeline puts
it in every report:

* **native joint audio-video** — Seedance 1.5/2.0 generate both in one
  dual-branch model. Wan 2.1 has no audio branch. `seedance.stages.audio_stage`
  mixes and muxes authored tracks; synchronisation is placed by you, not generated.
* **the `@mention` 4-modality reference system** — documented only at product
  level; `seedance.story` implements the *surface* (budgets, roles, resolution),
  not the token injection.
* **emergent physics** — see stage 3 above.
* **the reported ~90% first-try usability rate.**

The honest ceiling for the open stack is roughly 60–70% of Seedance behaviour,
and that figure is an engineering judgment, not a measurement.

## The architecture half

```bash
python -m seedance.examples.native_architecture --variant dev-tiny
```

Runs the whole thing on CPU in seconds: VAE round trip, a training step with
gradients, multi-shot sampling. What is implemented, and where it comes from:

* **Decoupled spatial/temporal MMDiT** (`models/dit.py`) — spatial layers do
  intra-frame attention jointly over visual+text tokens with separate per-modality
  weights; temporal layers do inter-frame attention over visual tokens only,
  inside spatial windows. Q/K normalised before attention.
* **Temporally-causal 3D VAE** (`models/vae3d.py`) — (4,16,16) downsampling,
  C=48, MAGVIT-style causality, PatchGAN hybrid discriminator, and the **Thin
  decoder** that narrows the stages nearest pixel space for ~2× faster decode.
* **Multishot MM-RoPE** (`models/rope.py`) — 3D axial RoPE for visual tokens,
  1D for text, per-shot temporal offsets so cuts need no special tokens.
* **Unified task formulation** (`models/conditioning.py`) — noisy latents
  channel-concatenated with clean/zero frames plus a binary mask; one mechanism
  for t2v / i2v / flf2v / v2v / extend / inpaint.
* **Cascaded refiner** (`models/refiner.py`) — 480p base → partial-noise
  trajectory at 720p/1080p.
* **Dual-branch audio** (`models/audio.py`) — audio DiT plus a cross-modal joint
  module with an explicit ±120 ms alignment mask, zero-gated so it is a no-op
  when bolted onto a trained video model.
* **Rectified flow** (`flow/`) — SD3-style shift and logit-normal sampling,
  CFG with rescale and CFG-Zero*, UniPC/DPM++ via Wan's solvers.
* **Distillation** (`flow/distill.py`) — TSCD, score distillation, adversarial
  distillation with a preference head, and a staged plan runner.
* **RewardDance + DanceGRPO** (`reward/`) — yes-token generative reward over any
  VLM, normalised multi-reward advantages, GRPO over the denoising MDP. Weightless
  heuristic proxies let the RL loop run with no downloads.

Every config field and class carries a provenance tag: `[1.0]` for the Seedance
1.0 technical report, `[1.5/2.0]` for the later abstracts, `[inferred]` for
reconstruction. See `docs/PROVENANCE.md` for the full claim-by-claim table —
including the widely repeated "DB-DiT with a millisecond-level attention bridge"
claim, which comes from marketing copy and not from any paper.

## Tests

```bash
python -m pytest seedance/tests -q
```

Torch-free tests (config, registry, checkpoint detection, storyboards, the
physics integrator, prompt templates) run anywhere. The architecture tests skip
themselves without torch and otherwise run at `dev-tiny` scale on CPU in a few
seconds — they check causality, window round-trips, RoPE norm preservation,
mask effects, the flow parameterisation, and that a fresh DiT predicts exactly
zero (adaLN-zero + zero head).

## Layout

```
seedance/
  config.py            provenance-tagged configs, torch-free
  story.py             storyboards, references, @mention parsing, torch-free
  models/              DiT, VAE, RoPE, attention, audio, text encoder, refiner
  flow/                rectified flow, solvers, distillation
  reward/              RewardDance, DanceGRPO, heuristic proxies
  backends/            registry + detection, Wan native, diffusers, mock
  stages/              prompt, identity, motion, physics, polish, audio, multishot
  pipelines/           staged (runnable) and native (reference)
  runtime/             VRAM planning, acceleration, video/audio I/O
  bench/               Physics-IQ protocol metrics
  examples/  tests/    runnable scripts and the test suite
```

## Hugging Face Space

```bash
python -m seedance.app_gradio --model Wan-AI/Wan2.1-T2V-1.3B
```

The same file works as a Space entry point. Set `SEEDANCE_MODEL` to choose the
checkpoint and `SEEDANCE_OFFLINE_PROMPT=0` to enable the LLM prompt rewriter.
The model loads lazily on the first generate, so the Space boots immediately
instead of downloading weights at startup. Use a GPU tier — L4/A10G for the
1.3B model; CPU tiers cannot run any Wan checkpoint (`--model mock` verifies the
UI without weights).
