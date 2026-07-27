"""Audio track assembly and muxing.

Be clear about what this is: **native joint audio-video generation is not
reproducible from open Wan weights today.** Seedance 1.5 Pro / 2.0 generate
audio and video in one dual-branch model, so the footsteps land on the frame
where the foot lands. Wan 2.1 has no audio branch at all, and Wan 2.5+ has only
recently added synchronised audio.

What this stage does instead:

* assembles multi-track audio (background / ambience / voice) the way 2.0
  exposes it, from files, arrays, or the native (untrained) audio branch;
* aligns and mixes tracks with headroom-aware gain staging;
* muxes the result onto the video.

Synchronisation is therefore *authored*, not generated: you place the events.
:func:`suggest_sync_points` helps by finding motion peaks in the video, which
is where impact sounds usually belong.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = ["AudioTrack", "AudioMix", "suggest_sync_points", "attach_audio"]


@dataclass
class AudioTrack:
    """One audio layer to be mixed."""

    name: str = "ambience"
    source: Any = None  # path | 1D/2D array | tensor | None (silence)
    gain_db: float = 0.0
    start_s: float = 0.0
    loop: bool = False
    fade_in_s: float = 0.05
    fade_out_s: float = 0.15

    @property
    def gain(self) -> float:
        return 10 ** (self.gain_db / 20)


class AudioMix:
    """Mixes tracks to a single interleaved waveform."""

    def __init__(self, sample_rate: int = 48000, channels: int = 2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.tracks: List[AudioTrack] = []

    def add(self, track: AudioTrack) -> "AudioMix":
        self.tracks.append(track)
        return self

    def _load(self, source: Any, num_samples: int):
        import torch

        if source is None:
            return torch.zeros(self.channels, num_samples)
        if torch.is_tensor(source):
            wave = source.detach().float().cpu()
        elif isinstance(source, (str, os.PathLike)):
            wave = _read_wav(str(source), self.sample_rate)
        else:
            import numpy as np

            wave = torch.from_numpy(np.asarray(source, dtype="float32"))
        if wave.dim() == 1:
            wave = wave.unsqueeze(0)
        if wave.shape[0] == 1 and self.channels > 1:
            wave = wave.expand(self.channels, -1).contiguous()
        elif wave.shape[0] > self.channels:
            wave = wave[: self.channels]
        return wave

    def render(self, duration_s: float):
        """Mix everything to ``[channels, N]`` with peak normalisation."""
        import torch

        n = int(round(duration_s * self.sample_rate))
        out = torch.zeros(self.channels, n)
        for track in self.tracks:
            wave = self._load(track.source, n) * track.gain
            if track.loop and wave.shape[1] < n and wave.shape[1] > 0:
                reps = math.ceil(n / wave.shape[1])
                wave = wave.repeat(1, reps)
            start = int(round(track.start_s * self.sample_rate))
            usable = min(n - start, wave.shape[1])
            if usable <= 0:
                continue
            segment = wave[:, :usable].clone()
            segment = _fade(segment, self.sample_rate, track.fade_in_s, track.fade_out_s)
            out[:, start : start + usable] += segment
        peak = float(out.abs().max())
        if peak > 0.99:
            # leave 1 dB of headroom rather than clipping the sum
            out = out * (0.89 / peak)
        return out

    def save(self, path: str, duration_s: float) -> str:
        from ..runtime.io import save_audio

        return save_audio(self.render(duration_s), path, self.sample_rate)


def _fade(wave, sample_rate: int, fade_in_s: float, fade_out_s: float):
    import torch

    n = wave.shape[1]
    fi = min(int(fade_in_s * sample_rate), n // 2)
    fo = min(int(fade_out_s * sample_rate), n // 2)
    if fi > 0:
        wave[:, :fi] *= torch.linspace(0, 1, fi)
    if fo > 0:
        wave[:, n - fo :] *= torch.linspace(1, 0, fo)
    return wave


def _read_wav(path: str, target_rate: int):
    """Read a wav without pulling in torchaudio."""
    import torch
    import wave as wavelib

    import numpy as np

    with wavelib.open(path, "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())
    data = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
    data = data.reshape(-1, channels).T
    tensor = torch.from_numpy(data.copy())
    if rate != target_rate and tensor.shape[1] > 1:
        import torch.nn.functional as F

        new_len = int(round(tensor.shape[1] * target_rate / rate))
        tensor = F.interpolate(
            tensor.unsqueeze(0), size=new_len, mode="linear", align_corners=False
        )[0]
    return tensor


def suggest_sync_points(video, fps: int = 16, *, top_k: int = 8) -> List[float]:
    """Timestamps (seconds) of motion-acceleration peaks.

    Impact sounds belong where motion changes abruptly, not where motion is
    merely fast -- so this looks at the second difference, same reasoning as
    the motion-quality proxy.
    """
    import torch

    v = video[0] if video.dim() == 5 else video
    gray = v.mean(dim=0)  # [T, H, W]
    if gray.shape[0] < 3:
        return []
    accel = (gray[2:] - 2 * gray[1:-1] + gray[:-2]).abs().flatten(1).mean(dim=1)
    k = min(top_k, accel.shape[0])
    idx = torch.topk(accel, k).indices + 1
    return sorted(float(i) / fps for i in idx.tolist())


def attach_audio(
    video_path: str,
    *,
    tracks: Sequence[AudioTrack] = (),
    duration_s: float = 0.0,
    sample_rate: int = 48000,
    channels: int = 2,
    workdir: Optional[str] = None,
) -> Dict[str, Any]:
    """Mix ``tracks`` and mux them onto ``video_path``.

    Returns a report including the final path (which is the original path when
    there is nothing to attach or ffmpeg is missing).
    """
    from ..runtime.io import ffmpeg_available, mux_audio

    report: Dict[str, Any] = {"path": video_path, "tracks": [t.name for t in tracks]}
    if not tracks:
        report["note"] = "no audio tracks supplied; video left silent"
        return report
    if not ffmpeg_available():
        report["note"] = "ffmpeg not available; audio was mixed but not muxed"

    mix = AudioMix(sample_rate=sample_rate, channels=channels)
    for track in tracks:
        mix.add(track)
    workdir = workdir or os.path.dirname(os.path.abspath(video_path))
    audio_path = os.path.join(workdir, "seedance_audio.wav")
    mix.save(audio_path, duration_s)
    report["audio_path"] = audio_path

    if ffmpeg_available():
        muxed = mux_audio(video_path, audio_path)
        if muxed:
            report["path"] = muxed
    return report
