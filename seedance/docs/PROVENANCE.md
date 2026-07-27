# Provenance

Every architectural claim this package implements, with its source and its
status. The short version: **only Seedance 1.0 (arXiv:2506.09113) documents
internals.** The 1.5 Pro (arXiv:2512.13507) and 2.0 (arXiv:2604.14148) papers
are capability and evaluation reports with no model-design section — no VAE
numbers, no text-encoder identity, no flow-matching or CFG equations, no
attention-bridge internals, no description of the reference-token system.

Tags used throughout the code:

| tag | meaning |
|---|---|
| `[1.0]` | stated in the Seedance 1.0 technical report |
| `[1.5/2.0]` | stated in the abstract or introduction of the later reports |
| `[product-level]` | documented only in API/product material (fal, Replicate, Volcano Engine) |
| `[inferred]` | our reconstruction, following SD3/rectified-flow convention — **not** a Seedance fact |

## Verified from the Seedance 1.0 report

| claim | where it is implemented |
|---|---|
| Decoupled spatial and temporal layers; spatial = intra-frame, temporal = inter-frame | `models/dit.py` |
| Window partitioning within each frame in temporal layers, for a global temporal receptive field | `models/attention.py:window_partition`, `models/dit.py:TemporalDiTBlock` |
| MMDiT-style multi-modality self-attention **only** in spatial layers; temporal layers are visual-only | `models/dit.py`, `models/attention.py:JointAttention` |
| Separate per-modality weights (adaLN, QKV, MLP) in spatial layers | `models/dit.py:SpatialMMDiTBlock` |
| Q/K normalised before attention to prevent training instability | `models/attention.py` (`qk_norm`) |
| Multishot MM-RoPE: 3D RoPE for visual + 1D for text, shots in temporal order with per-shot captions | `models/rope.py` |
| Unified task formulation: channel-concat clean/zero frames + binary mask | `models/conditioning.py` |
| Temporally-causal 3D VAE (MAGVIT lineage), downsampling (4,16,16), C=48 | `models/vae3d.py`, `config.py:VAEConfig` |
| Patchification removed on the DiT side (DC-AE) | `config.py` default `patch_size=(1,1,1)` |
| VAE losses: L1 + KL + LPIPS + adversarial, PatchGAN-style hybrid discriminator modelling appearance and motion | `models/vae3d.py` |
| Thin VAE decoder: narrow the stages closest to pixel space, retrain with a frozen encoder, ~2× decode speedup | `models/vae3d.py:Decoder(thin=True)` |
| Text encoder is a fine-tuned decoder-only LLM | `models/text_encoder.py` |
| Prompt-engineering model initialised from Qwen2.5-14B; SFT on prompt→dense-caption pairs, then DPO with LoRA | `stages/prompt_enhancer.py` |
| Cascaded HR generation: 480p base, then a refiner conditioned on the upsampled low-res video concatenated along channels | `models/refiner.py` |
| Multi-stage distillation: TSCD + score distillation + adversarial distillation with human-preference supervision | `flow/distill.py` |
| 5 s of 1080p in 41.4 s on an L20 (>10× speedup) | quoted in `flow/distill.py` |
| Data pipeline: shot-aware segmentation (~12 s clips), overlay rectification, dedup, dense captioning; pretrain → CT → SFT → RLHF | described in `docs/ARCHITECTURE.md` |
| Video RLHF with multi-dimensional reward models (motion naturalness, structural coherence, visual fidelity) | `reward/` |

## Verified from RewardDance and DanceGRPO

| claim | source | implementation |
|---|---|---|
| Reward = VLM probability of the "yes" token | arXiv:2509.08826 | `reward/reward_dance.py` |
| Scales to 26B reward models (InternVL variants); context scaling via task instruction, reference examples, CoT | arXiv:2509.08826 | `reward/reward_dance.py:RewardPrompt` |
| Seedance-1.0 T2V alignment GSB: +28.0% (1B RM) → +49.0% (26B RM) | arXiv:2509.08826 | quoted in docstrings |
| Large RMs keep reward variance high and resist reward hacking | arXiv:2509.08826 | quoted |
| GRPO adapted to diffusion **and** rectified flows; denoising as an MDP; group-normalised advantages; normalised summation across up to 5 reward models | arXiv:2505.07818 | `reward/dance_grpo.py` |
| HunyuanVideo: +56% visual quality, +181% motion quality on VideoAlign | arXiv:2505.07818 | quoted |

## Stated for 1.5 Pro / 2.0 (abstract level only)

| claim | status |
|---|---|
| Dual-branch Diffusion Transformer with a cross-modal joint module on an MMDiT / rectified-flow backbone | `[1.5/2.0]` — stated; **all internal dimensions in `config.py:AudioConfig` are inferred** |
| Multi-task pretraining on mixed-modality data (T2VA / I2VA / T2V / I2V) | `[1.5/2.0]` |
| Reward dimensions: motion quality, visual aesthetics, audio fidelity | `[1.5/2.0]` |
| ~3× RLHF training-speed improvement; distillation lineage (MeanFlow / Hyper-SD / RayFlow) | `[1.5/2.0]` |
| 4–15 s duration, native 480p/720p, binaural multi-track audio, up to 3 videos / 9 images / 3 audio references | `[product-level]` — encoded as budgets in `config.py` and `story.py` |
| The `@mention` / `[Image1]` reference control system | `[product-level]` — surface only, in `story.py` |

## Explicitly **not** supported by any primary source

* **"DB-DiT with a millisecond-level attention bridge" and separate
  waveform/spectral token branches.** This comes from third-party and marketing
  material, not from the papers. `models/audio.py` therefore makes the
  tokenizer *configurable* (`mel` / `waveform` / `hybrid`) instead of asserting
  a design, and the ±120 ms alignment window is labelled as our mechanism.
* **Any physics engine anywhere in the Seedance stack.** Physical plausibility
  is emergent from data and reward modelling. Independent testing still finds
  failures on multi-character sports and fluid dynamics. Stage 3 in this package
  is a PhysGen-style proxy — a *different* approach, not a reproduction.
* **Layer counts, widths, head counts, and the VAE's exact channel schedule.**
  Not disclosed. `config.py` marks these inferred.
* **Timestep shifting and the CFG configuration.** Not disclosed for Seedance;
  `flow/rectified_flow.py` follows SD3 convention and says so.

## Benchmark numbers to treat with care

* Seedance 2.0's SeedVideoBench 2.0 and Arena.AI results (Elo 1450 ±15 T2V /
  1449 ±11 I2V) are **vendor-reported** and were preliminary at access time.
* Physics-IQ (arXiv:2501.09038): 396 real videos, 66 scenarios, 3 perspectives.
  "Physical understanding is severely limited, and unrelated to visual realism";
  VideoPoet scored 24.1%. Physics-IQ Verified (arXiv:2606.18943) refines 57.6%
  of samples and 34.8% of prompts. `bench/physics_iq.py` implements the metric
  protocol; the dataset is not redistributed here.
* VRAM, runtime and cost figures for Wan vary widely with resolution, frame
  count, precision and offload settings. `runtime/memory.py` estimates are
  order-of-magnitude and pessimistic by design.

## Sources

| key | reference |
|---|---|
| Seedance 1.0 | arXiv:2506.09113 (Gao et al., ByteDance Seed, 10 Jun 2025) |
| Seedance 1.5 Pro | arXiv:2512.13507 |
| Seedance 2.0 | arXiv:2604.14148 |
| RewardDance | arXiv:2509.08826 (Wu, Gao et al.) |
| DanceGRPO | arXiv:2505.07818 (Xue et al., ByteDance Seed & HKU) |
| Physics-IQ | arXiv:2501.09038 (Motamed et al., INSAIT & Google DeepMind) |
| Physics-IQ Verified | arXiv:2606.18943 |
| PhysGen | arXiv:2409.18964 |
| PhysDreamer | arXiv:2404.13026 |
| PhysMaster | arXiv:2510.13809 |
| CameraCtrl | arXiv:2404.02101 |
| Tora | arXiv:2407.21705 |
| SeedVR2 | arXiv:2506.05301 |
| Wan | arXiv:2503.20314 |
