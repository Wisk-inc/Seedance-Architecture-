# The Seedance architecture, as implemented here

Read `PROVENANCE.md` first if you care which parts are documented and which are
reconstruction. This file explains how the pieces fit and why they are shaped
the way they are.

## Data flow

```
prompt ──► decoder-only LLM ──────────────► text tokens ─┐
                                                          │
video  ──► temporally-causal 3D VAE ──► latents ──► noise │
           (4,16,16) C=48                                 │
                                                          ▼
                    ┌──────────── decoupled MMDiT ────────────┐
                    │  spatial layer:  intra-frame,  joint    │
                    │                  visual + text          │
                    │  temporal layer: inter-frame,  visual   │
                    │                  only, windowed         │
                    │  (× N, interleaved; audio bridge every  │
                    │   4th spatial layer when audio is on)   │
                    └─────────────────────────────────────────┘
                                                          │
                                            velocity ─────┘
                                                          │
                    rectified-flow solver (UniPC / DPM++ / Euler)
                                                          │
                              latents ──► refiner (480p → 720p/1080p)
                                                          │
                                        Thin VAE decoder ──► pixels
```

## Why the decoupled split

Full 3D attention over a video latent is quadratic in `T·H·W`. At 720p/5 s
that is ~10⁵ tokens per forward, which is why most video DiTs factorise.
Seedance's split is asymmetric in an interesting way:

* **Spatial layers** are where text enters. Attention is over one frame's
  tokens plus the whole text sequence, so cross-modal grounding happens at
  full spatial resolution but costs `O(S²)` per frame rather than `O((TS)²)`.
* **Temporal layers** never see text. They attend across *all* frames inside a
  spatial window, so the temporal receptive field is global from layer one
  while the cost is `O(T²·w²)` per window.

`SeedanceDiT.flops_estimate()` prints the split for a given resolution; the
temporal half is typically the cheaper one, which is the opposite of what
people expect and is entirely due to windowing.

One design decision is ours, not the paper's: since the text stream is shared
across frames but spatial attention runs per frame, the text stream's per-frame
updates are **averaged** before the residual. Mean (not sum) keeps the update
magnitude independent of frame count, so a 5-second clip and a 15-second clip
train with the same effective text learning rate.

## Why the VAE is temporally causal

Causality buys three things at once:

1. The `(T'+1) → (T+1)` shape contract, which is what lets the first frame be a
   clean conditioning image in I2V without a separate encoder path.
2. Streaming: `CausalConv3d` keeps a rolling cache of the last `k-1` frames, so
   arbitrary-length video encodes and decodes in chunks with bit-identical
   results to a single pass. `encode(..., chunk_frames=N)` uses it.
3. Test: `test_vae_is_temporally_causal` perturbs the final frame and asserts
   every earlier latent is unchanged. If a non-causal op sneaks in — a
   bidirectional temporal attention, a global norm over time — that test fails.

The **Thin decoder** exists because profiling puts most decode latency in the
stages nearest pixel space, where feature maps are largest. Narrowing exactly
those stages and retraining with a frozen encoder keeps the latent space intact
(so the DiT does not need retraining) while roughly halving decode time.

## Multishot MM-RoPE

Visual tokens get 3D axial RoPE: the head dimension splits into `(t, h, w)`
groups, with time getting the remainder because it has the longest range.
Text tokens get 1D RoPE in a *disjoint* position range (`text_offset = 2²⁰`),
so text and visual positions cannot alias inside the joint attention op.

Multi-shot works by giving shot *k* a temporal offset with a gap:

```python
shot_time_positions([3, 2, 4], gap=8)  # [0,1,2, 11,12, 21,22,23,24]
```

A cut is then a discontinuity in rotary phase rather than a special token. This
is why the architecture needs no `<cut>` vocabulary and why shot boundaries
generalise to counts never seen in training.

## The unified task formulation

Every task is the same tensor contract:

```
model_input = concat([noisy_latents, condition_latents, binary_mask], dim=1)
```

`condition_latents` holds clean latents where a frame is an instruction and
zeros elsewhere; the mask says which is which. t2v is the all-zeros case, i2v
pins frame 0, flf2v pins frames 0 and T−1, v2v pins everything, extend pins a
leading run, inpaint uses a spatial mask. Training-task proportions are then
just sampling proportions over mask patterns — no architectural branches, which
is what makes one checkpoint serve every task.

## The audio branch

Two things make the dual-branch design work, and both are implemented in
`models/audio.py`:

**Alignment.** Audio runs at ~100 tokens/s, video at 24 fps. Unrestricted
cross-attention lets a footstep bind to a frame two seconds away. The bridge
builds an explicit mask from the two rates and permits attention only inside a
±120 ms window — wide enough for perceptual audio-visual binding, narrow enough
that causality is preserved.

**Zero gating.** Both directions of the bridge are gated by `tanh(g)` with
`g` initialised to zero, so inserting the bridge into a pretrained video-only
backbone is exactly a no-op at step 0. `test_audio_bridge_starts_as_a_no_op`
asserts this. Without it, adding audio to a trained model destroys it.

Video tokens are pooled per frame before fusion: a sound event binds to a
*moment*, not a pixel, and pooling keeps the bridge linear in frames.

## Rectified flow

Convention, used everywhere:

```
x_t      = (1 - t)·x_0 + t·ε ,  t ∈ [0, 1]
v_target = ε - x_0 = dx_t/dt
sampling: x ← x - dt·v          (integrating down from t=1)
```

Sign errors here are the classic flow-matching bug and are silent — you get
plausible-looking noise. `RectifiedFlow.self_consistency_check()` integrates an
analytic velocity field and asserts it lands on the data point; it runs as a
test.

Timestep shifting (SD3 dynamic shift keyed to sequence length), logit-normal
timestep sampling, CFG rescaling and CFG-Zero* are all `[inferred]` — sensible
defaults, not disclosed Seedance settings.

## Distillation and RL

`flow/distill.py` implements the three-stage ladder the report names. TSCD is
the interesting one: instead of forcing a single map from every `t` to `t=0`,
`[0,1]` is split into `k` segments and consistency is only required *within* a
segment, with `k` annealed 8 → 4 → 2 → 1. Each `k` is a usable few-step sampler
on the way down.

`reward/dance_grpo.py` treats denoising as an MDP. This requires a *stochastic*
sampler: a deterministic ODE has a zero-variance policy, no log-probability to
differentiate and no exploration, so `rollout()` injects noise per step and
records the induced Gaussian log-probs. Advantages are group-normalised per
prompt (no value function — unaffordable for video), and multiple reward models
are normalised *before* summation so the widest-range model cannot dominate.

Without a VLM installed, `reward/heuristics.py` supplies weightless proxies for
the same dimensions so the loop runs end to end. They correlate with the right
failure modes and are not quality metrics; the code says so in the docstring.

## Where the open-weights pipeline diverges

| Seedance does it | the staged pipeline does |
|---|---|
| one model, native multi-shot | generates shots separately and chains via I2V/FLF2V, colour-matching each seam |
| identity from multi-shot training + RLHF | VACE/Phantom references, LoRA, locked subject clauses |
| emergent physics | simulated proxy rendered as a control guide |
| joint audio-video | authored tracks mixed and muxed |
| quality from the base model | colour → interpolate → restore → upscale, then QA |

Each substitution is weaker than the thing it replaces. The pipeline reports
which one it used and warns when a requested control cannot be applied by the
loaded checkpoint, because the failure mode worth avoiding is a control signal
that is silently ignored for a whole render.
