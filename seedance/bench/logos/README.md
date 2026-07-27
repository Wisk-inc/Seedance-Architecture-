# Logo assets

Drop official vendor logos here as `<key>.svg` and the benchmark charts pick
them up automatically — they are scaled and centred into the tile, replacing
the built-in monogram.

Keys used by the shipped result files:

| key | model |
|---|---|
| `seedance` | Seedance (this package / ByteDance Seedance) |
| `wan` | Wan 2.1 / 2.2 / 2.6 |
| `sora` | Sora |
| `veo` | Veo |
| `kling` | Kling |
| `runway` | Runway |
| `pika` | Pika |
| `luma` | Luma / Dream Machine |
| `hunyuan` | HunyuanVideo |
| `cogvideo` | CogVideoX |
| `ltx` | LTX-Video |
| `mochi` | Mochi |
| `svd` | Stable Video Diffusion |
| `opensora` | Open-Sora |
| `videopoet` | VideoPoet |

Nothing is shipped here on purpose: these are third-party trademarks, and
redistributing them inside an unrelated package is not ours to do. Add the ones
you have the right to use, then re-run:

```bash
python -m seedance.bench.make_charts
```
