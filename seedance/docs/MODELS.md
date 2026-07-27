# Running Wan checkpoints

## Picking one

| checkpoint | params | VRAM (bf16) | good for |
|---|---|---|---|
| `Wan-AI/Wan2.1-T2V-1.3B` | 1.3B | ~8–12 GB | 480p T2V on consumer GPUs; the default |
| `Wan-AI/Wan2.2-TI2V-5B` | 5B | ~12–24 GB | 720p T2V+I2V on one 4090; high-compression VAE |
| `Wan-AI/Wan2.1-T2V-14B` | 14B | ~40–48 GB @480p, 65–80 GB @720p | best open T2V quality |
| `Wan-AI/Wan2.1-I2V-14B-480P/720P` | 14B | 40–48 GB | image-to-video |
| `Wan-AI/Wan2.1-FLF2V-14B-720P` | 14B | ~48 GB | pinned first *and* last frame |
| `Wan-AI/Wan2.1-VACE-1.3B` / `-14B` | 1.3B / 14B | ~10 / ~48 GB | **reference + control**: the only ones stages 2–3 can fully drive |
| `Wan-AI/Wan2.2-T2V-A14B` | 27B MoE (14B active) | ~48 GB | MoE quality, diffusers loader |

Numbers are order-of-magnitude and vary with resolution, frame count, precision
and offload. `python -m seedance.cli doctor` recommends one for your GPU.

**If you want stages 2 and 3 to do anything, use a VACE checkpoint.** Depth,
pose, flow, trajectory and physics guides all enter through the control path,
which plain T2V/I2V checkpoints do not have. The pipeline warns rather than
silently dropping them.

## Any repo, not just these

```bash
python -m seedance.cli inspect some-org/their-wan-finetune
```

The checkpoint is downloaded and classified from the files present, not the
name:

* `model_index.json` or a `transformer/` subfolder → diffusers loader
* `Wan2.x_VAE.pth` + `models_t5_umt5-*.pth` → native Wan loader
* a CLIP `.pth`, or `in_dim ≥ 32` in `config.json` → I2V
* `high_noise_model/` + `low_noise_model/` → Wan 2.2 MoE
* `*.gguf` → flagged with a pointer to ComfyUI-GGUF (this package cannot load them)

`seedance.backends.auto_backend()` then picks the native or diffusers backend
accordingly. A local directory works the same way: `--model /path/to/weights`.

## Memory

`seedance.runtime.memory.plan_vram()` estimates peak usage and, when it does
not fit, applies fixes in cheapest-first order: text encoder to CPU → VAE
tiling → model offload → fp8 → 480p → fewer frames → sequential offload. The
staged pipeline runs this automatically and logs the suggestions.

Manual knobs:

```python
SeedancePipeline(
    "Wan-AI/Wan2.1-T2V-14B",
    t5_cpu=True,          # native backend: keep UMT5 off the GPU
    offload_model=True,   # move the DiT off between denoise and decode
)
SeedancePipeline(
    "Wan-AI/Wan2.2-T2V-A14B",
    prefer_backend="diffusers",
    sequential_offload=True,   # slowest, smallest footprint
    vae_tiling=True,
)
```

For genuinely small GPUs (≤ 8 GB), the practical route is GGUF Q4 weights in
ComfyUI rather than this package — it is a learning setup, not a production one,
and the loader here does not read GGUF.

## Multi-GPU

The native Wan backend passes through Wan's own flags:

```python
SeedancePipeline("Wan-AI/Wan2.1-T2V-14B", dit_fsdp=True, t5_fsdp=True, use_usp=True)
```

Launch with `torchrun`, as Wan's own `generate.py` does. Sequence parallelism
(`use_usp`) needs `xfuser` installed.

## Speed

* Fewer steps: 40 is usually indistinguishable from 50 at guidance 5.
* `shift` matters more than steps at 480p — Wan's own guidance is 3.0 for the
  1.3B model, 5.0 for 14B, 16.0 for FLF2V. The pipeline defaults to the right
  one per checkpoint.
* `runtime.accel.TeaCache(threshold=0.2)` skips redundant mid-trajectory steps
  for roughly 2× at a small motion-softness cost.
* `runtime.accel.AccelConfig(compile_model=True)` for `torch.compile`; the first
  call pays a long warmup, so it only pays off across many renders.
* Stage 4 interpolation is cheaper than generating more frames: render 49
  frames at 16 fps and interpolate ×2 rather than rendering 97.
