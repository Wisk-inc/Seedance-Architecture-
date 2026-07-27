"""Video and audio I/O.

Everything in this package passes video around as ``[C, T, H, W]`` float
tensors in ``[-1, 1]``. This module is the only place that talks to files, so
the convention is enforced in exactly one spot.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "save_video",
    "load_video",
    "save_frames",
    "load_image",
    "to_uint8_frames",
    "from_uint8_frames",
    "mux_audio",
    "save_audio",
    "ffmpeg_available",
    "concat_videos",
]


def to_uint8_frames(video) -> "list":
    """``[C, T, H, W]`` in ``[-1, 1]`` -> list of ``[H, W, 3]`` uint8 arrays."""
    import numpy as np
    import torch

    if not torch.is_tensor(video):
        video = torch.as_tensor(video)
    if video.dim() == 5:
        video = video[0]
    v = video.detach().float().cpu()
    if v.shape[0] not in (1, 3) and v.shape[-1] in (1, 3):  # T,H,W,C given
        v = v.permute(3, 0, 1, 2)
    if v.min() < -0.01:
        v = (v + 1) / 2
    v = v.clamp(0, 1)
    if v.shape[0] == 1:
        v = v.expand(3, -1, -1, -1)
    arr = (v.permute(1, 2, 3, 0).numpy() * 255).astype(np.uint8)
    return [arr[i] for i in range(arr.shape[0])]


def from_uint8_frames(frames: Sequence, device: str = "cpu"):
    """List of ``[H, W, 3]`` uint8 arrays -> ``[C, T, H, W]`` in ``[-1, 1]``."""
    import numpy as np
    import torch

    arr = np.stack([np.asarray(f)[:, :, :3] for f in frames]).astype("float32") / 255.0
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2)
    return (tensor * 2 - 1).to(device)


def save_video(
    video,
    path: str,
    *,
    fps: int = 16,
    quality: int = 8,
    codec: str = "libx264",
    audio_path: Optional[str] = None,
) -> str:
    """Write ``[C, T, H, W]`` to ``path``; returns the final path.

    If ``audio_path`` is given and ffmpeg is available, the track is muxed in
    and the muxed file is what you get back.
    """
    import imageio

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    frames = to_uint8_frames(video)
    # H.264 requires even dimensions; pad rather than fail at the last step.
    h, w = frames[0].shape[:2]
    if codec == "libx264" and (h % 2 or w % 2):
        import numpy as np

        frames = [np.pad(f, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge") for f in frames]

    writer = imageio.get_writer(path, fps=fps, codec=codec, quality=quality)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()

    if audio_path:
        muxed = mux_audio(path, audio_path)
        if muxed:
            return muxed
    return path


def load_video(path: str, *, max_frames: int = 0, device: str = "cpu", stride: int = 1):
    """Read a video file into ``[C, T, H, W]`` in ``[-1, 1]``."""
    import imageio.v3 as iio

    frames: List = []
    for i, frame in enumerate(iio.imiter(path)):
        if i % stride:
            continue
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return from_uint8_frames(frames, device=device)


def save_frames(video, directory: str, *, prefix: str = "frame", ext: str = "png") -> List[str]:
    """Write each frame as an image; returns the paths, in order."""
    import imageio

    os.makedirs(directory, exist_ok=True)
    paths = []
    for i, frame in enumerate(to_uint8_frames(video)):
        p = os.path.join(directory, f"{prefix}_{i:05d}.{ext}")
        imageio.imwrite(p, frame)
        paths.append(p)
    return paths


def load_image(path_or_image):
    from PIL import Image

    if hasattr(path_or_image, "convert"):
        return path_or_image.convert("RGB")
    return Image.open(path_or_image).convert("RGB")


def ffmpeg_available() -> bool:
    if shutil.which("ffmpeg"):
        return True
    try:  # imageio bundles a static ffmpeg
        import imageio_ffmpeg

        return bool(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return False


def _ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def mux_audio(video_path: str, audio_path: str, out_path: Optional[str] = None) -> Optional[str]:
    """Combine a silent video and an audio file. Returns the output path."""
    if not ffmpeg_available():
        logger.warning("ffmpeg not available; leaving %s without audio", video_path)
        return None
    out_path = out_path or video_path.replace(".mp4", "_audio.mp4")
    cmd = [
        _ffmpeg_exe(), "-y", "-loglevel", "error",
        "-i", video_path, "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", "-shortest", out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return out_path
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        logger.error("ffmpeg mux failed: %s", exc.stderr.decode(errors="ignore")[:400])
        return None


def save_audio(waveform, path: str, sample_rate: int = 48000) -> str:
    """Write ``[C, N]`` or ``[N]`` audio in ``[-1, 1]`` to a wav file."""
    import numpy as np
    import torch
    import wave

    if torch.is_tensor(waveform):
        data = waveform.detach().float().cpu().numpy()
    else:
        data = np.asarray(waveform, dtype="float32")
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim == 3:
        data = data[0]
    channels = data.shape[0]
    pcm = (np.clip(data, -1, 1) * 32767).astype("<i2").T.tobytes()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return path


def concat_videos(paths: Sequence[str], out_path: str) -> Optional[str]:
    """Concatenate clips losslessly with ffmpeg's concat demuxer."""
    if not ffmpeg_available():
        logger.warning("ffmpeg not available; cannot concatenate")
        return None
    listing = out_path + ".txt"
    with open(listing, "w", encoding="utf-8") as handle:
        for p in paths:
            handle.write(f"file '{os.path.abspath(p)}'\n")
    cmd = [
        _ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", listing, "-c", "copy", out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return out_path
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        logger.error("ffmpeg concat failed: %s", exc.stderr.decode(errors="ignore")[:400])
        return None
    finally:
        if os.path.exists(listing):
            os.remove(listing)
