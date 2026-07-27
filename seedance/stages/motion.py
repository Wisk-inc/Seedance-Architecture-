"""Stage 2b: motion, camera and trajectory control.

Control signals here are all *guide videos*: a per-frame image sequence that a
VACE-style checkpoint consumes as structure conditioning. Keeping every control
type in that one format is what lets depth, pose, flow, camera and physics
(Stage 3) compose -- they are all just guide frames, and they can be blended.

Optional dependencies are used when present and degrade to something usable
when absent:

* depth   -- Depth Anything V2 via transformers, else a luminance-gradient proxy.
* pose    -- DWPose/OpenPose via controlnet_aux, else skipped (no useful fallback).
* flow    -- RAFT via torchvision, else Farneback via OpenCV, else skipped.
* camera  -- Plücker embeddings, computed here with no dependencies.
* traj    -- drawn motion paths, computed here with no dependencies.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "CameraTrajectory",
    "plucker_embedding",
    "depth_guide",
    "pose_guide",
    "flow_guide",
    "trajectory_guide",
    "blend_guides",
    "MotionController",
]


# ------------------------------------------------------------------- camera

@dataclass
class CameraTrajectory:
    """A camera path as per-frame world-to-camera transforms.

    CameraCtrl (arXiv:2404.02101) parameterises camera pose as Plücker
    embeddings -- a 6-channel per-pixel ray representation -- because raw
    extrinsics are a poor conditioning signal: they are global, low-dimensional
    and not spatially aligned with the image. Plücker coordinates are
    per-pixel, so they drop straight into a ControlNet-shaped conditioning path.
    """

    positions: List[Tuple[float, float, float]] = field(default_factory=list)
    look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov_deg: float = 45.0

    @classmethod
    def orbit(
        cls, num_frames: int, radius: float = 3.0, degrees: float = 90.0, height: float = 0.0
    ) -> "CameraTrajectory":
        positions = []
        for i in range(num_frames):
            a = math.radians(degrees * i / max(num_frames - 1, 1))
            positions.append((radius * math.sin(a), height, radius * math.cos(a)))
        return cls(positions=positions)

    @classmethod
    def dolly(cls, num_frames: int, start: float = 4.0, end: float = 1.5) -> "CameraTrajectory":
        positions = [
            (0.0, 0.0, start + (end - start) * i / max(num_frames - 1, 1))
            for i in range(num_frames)
        ]
        return cls(positions=positions)

    @classmethod
    def pan(cls, num_frames: int, degrees: float = 30.0, distance: float = 3.0) -> "CameraTrajectory":
        positions = [(0.0, 0.0, distance)] * num_frames
        traj = cls(positions=positions)
        traj._pan_degrees = degrees  # type: ignore[attr-defined]
        return traj

    def extrinsics(self):
        """Per-frame 4x4 world-to-camera matrices."""
        import torch

        mats = []
        target = torch.tensor(self.look_at, dtype=torch.float32)
        up = torch.tensor(self.up, dtype=torch.float32)
        for pos in self.positions:
            eye = torch.tensor(pos, dtype=torch.float32)
            forward = target - eye
            n = forward.norm()
            forward = forward / n if float(n) > 1e-6 else torch.tensor([0.0, 0.0, -1.0])
            right = torch.cross(forward, up, dim=0)
            rn = right.norm()
            right = right / rn if float(rn) > 1e-6 else torch.tensor([1.0, 0.0, 0.0])
            true_up = torch.cross(right, forward, dim=0)
            rot = torch.stack([right, true_up, -forward], dim=0)
            mat = torch.eye(4)
            mat[:3, :3] = rot
            mat[:3, 3] = -rot @ eye
            mats.append(mat)
        return torch.stack(mats)


def plucker_embedding(
    trajectory: CameraTrajectory, height: int, width: int, fov_deg: Optional[float] = None
):
    """Per-pixel Plücker ray embedding ``[6, T, H, W]``.

    A ray through pixel ``p`` with origin ``o`` and direction ``d`` is encoded
    as ``(o x d, d)`` -- the moment plus the direction. That pair is invariant
    to where along the ray you sample, which is exactly the property that makes
    it a clean camera conditioning signal.
    """
    import torch

    extr = trajectory.extrinsics()
    t = extr.shape[0]
    fov = math.radians(fov_deg or trajectory.fov_deg)
    focal = 0.5 * width / math.tan(fov / 2)

    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    dirs_cam = torch.stack(
        [(xs - width / 2) / focal, -(ys - height / 2) / focal, -torch.ones_like(xs)], dim=-1
    )

    out = torch.zeros(6, t, height, width)
    for i in range(t):
        c2w = torch.linalg.inv(extr[i])
        rot, origin = c2w[:3, :3], c2w[:3, 3]
        dirs = dirs_cam @ rot.T
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        moment = torch.cross(origin.expand_as(dirs), dirs, dim=-1)
        out[:3, i] = moment.permute(2, 0, 1)
        out[3:, i] = dirs.permute(2, 0, 1)
    return out


# -------------------------------------------------------------------- guides

def depth_guide(video, *, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"):
    """Per-frame depth as a 3-channel guide video ``[3, T, H, W]`` in ``[-1, 1]``."""
    import torch

    from ..runtime.io import from_uint8_frames, to_uint8_frames

    frames = to_uint8_frames(video)
    try:
        from PIL import Image
        from transformers import pipeline

        estimator = pipeline("depth-estimation", model=model_id)
        maps = []
        for frame in frames:
            depth = estimator(Image.fromarray(frame))["depth"]
            maps.append(depth.convert("RGB"))
        import numpy as np

        return from_uint8_frames([np.asarray(m) for m in maps])
    except Exception as exc:
        logger.warning("Depth Anything unavailable (%s); using a gradient proxy", exc)

    # Proxy: local contrast energy, which correlates with depth edges. Enough
    # to hold structure; not a metric depth map.
    import torch.nn.functional as F

    v = video[0] if video.dim() == 5 else video
    gray = v.mean(dim=0, keepdim=True)  # [1, T, H, W]
    kernel = torch.tensor([[-1.0, -1, -1], [-1, 8, -1], [-1, -1, -1]]).view(1, 1, 3, 3)
    t = gray.shape[1]
    edges = F.conv2d(gray.permute(1, 0, 2, 3), kernel, padding=1).abs()
    edges = edges / edges.amax().clamp(min=1e-6)
    blurred = F.avg_pool2d(edges, 9, stride=1, padding=4)
    guide = (1 - blurred).permute(1, 0, 2, 3).expand(3, t, -1, -1)
    return guide * 2 - 1


def pose_guide(video, *, detector: str = "dwpose"):
    """Skeleton guide video via controlnet_aux. Returns ``None`` if unavailable.

    There is deliberately no fallback: a fabricated skeleton is worse than no
    skeleton, because the model will follow it.
    """
    try:
        import numpy as np
        from PIL import Image

        if detector == "dwpose":
            from controlnet_aux import DWposeDetector

            det = DWposeDetector()
        else:
            from controlnet_aux import OpenposeDetector

            det = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    except Exception as exc:
        logger.warning("pose detector unavailable (%s); no pose guide produced", exc)
        return None

    from ..runtime.io import from_uint8_frames, to_uint8_frames

    out = []
    for frame in to_uint8_frames(video):
        out.append(np.asarray(det(Image.fromarray(frame)).convert("RGB")))
    return from_uint8_frames(out)


def flow_guide(video, *, visualize: bool = True):
    """Optical flow between consecutive frames, as an RGB guide video."""
    import torch

    v = video[0] if video.dim() == 5 else video
    try:
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=False).eval()
        frames = ((v.permute(1, 0, 2, 3) + 1) / 2).clamp(0, 1)
        flows = []
        with torch.no_grad():
            for i in range(frames.shape[0] - 1):
                pair = weights.transforms()(frames[i : i + 1], frames[i + 1 : i + 2])
                flows.append(model(*pair)[-1][0])
        flows.append(flows[-1] if flows else torch.zeros(2, *v.shape[2:]))
        flow = torch.stack(flows, dim=1)  # [2, T, H, W]
    except Exception as exc:
        logger.warning("RAFT unavailable (%s); trying OpenCV Farneback", exc)
        flow = _farneback(v)
        if flow is None:
            return None
    return _flow_to_rgb(flow) if visualize else flow


def _farneback(v):
    try:
        import cv2
        import numpy as np
        import torch
    except ImportError:
        return None
    frames = (((v.permute(1, 2, 3, 0) + 1) / 2).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    flows = []
    for i in range(len(grays) - 1):
        f = cv2.calcOpticalFlowFarneback(grays[i], grays[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(torch.from_numpy(f).permute(2, 0, 1))
    if not flows:
        return None
    flows.append(flows[-1])
    return torch.stack(flows, dim=1)


def _flow_to_rgb(flow):
    """Standard hue=direction, value=magnitude flow visualisation."""
    import torch

    fx, fy = flow[0], flow[1]
    mag = torch.sqrt(fx**2 + fy**2)
    mag = mag / mag.amax().clamp(min=1e-6)
    ang = (torch.atan2(fy, fx) + math.pi) / (2 * math.pi)
    h = ang * 6
    i = h.floor()
    f = h - i
    p, q, t_ = torch.zeros_like(mag), mag * (1 - f), mag * f
    idx = (i % 6).long()
    r = torch.where(idx == 0, mag, torch.where(idx == 1, q, torch.where(idx == 2, p,
        torch.where(idx == 3, p, torch.where(idx == 4, t_, mag)))))
    g = torch.where(idx == 0, t_, torch.where(idx == 1, mag, torch.where(idx == 2, mag,
        torch.where(idx == 3, q, torch.where(idx == 4, p, p)))))
    b = torch.where(idx == 0, p, torch.where(idx == 1, p, torch.where(idx == 2, t_,
        torch.where(idx == 3, mag, torch.where(idx == 4, mag, q)))))
    rgb = torch.stack([r, g, b], dim=0)
    return rgb * 2 - 1


def trajectory_guide(
    paths: Sequence[Sequence[Tuple[float, float]]],
    num_frames: int,
    height: int,
    width: int,
    *,
    radius: int = 12,
    trail: int = 6,
):
    """Draw object motion paths as a guide video (Tora / DragAnything style).

    ``paths`` holds one list of normalised ``(x, y)`` points per object; points
    are resampled to ``num_frames``. Each object gets its own colour so a model
    conditioned on this can keep them apart.
    """
    import torch

    canvas = torch.full((3, num_frames, height, width), -1.0)
    colours = [(1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0), (1.0, 1.0, -1.0)]
    ys = torch.arange(height).view(-1, 1)
    xs = torch.arange(width).view(1, -1)
    for oi, path in enumerate(paths):
        if len(path) < 2:
            continue
        colour = colours[oi % len(colours)]
        pts = _resample(path, num_frames)
        for t, (nx, ny) in enumerate(pts):
            cx, cy = int(nx * (width - 1)), int(ny * (height - 1))
            for back in range(min(trail, t + 1)):
                bx, by = pts[t - back]
                px, py = int(bx * (width - 1)), int(by * (height - 1))
                rad = max(2, int(radius * (1 - back / max(trail, 1))))
                disk = ((xs - px) ** 2 + (ys - py) ** 2) <= rad**2
                for c in range(3):
                    canvas[c, t][disk] = colour[c]
    return canvas


def _resample(path: Sequence[Tuple[float, float]], n: int) -> List[Tuple[float, float]]:
    if len(path) == n:
        return list(path)
    out = []
    for i in range(n):
        pos = i * (len(path) - 1) / max(n - 1, 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(path) - 1)
        frac = pos - lo
        out.append(
            (
                path[lo][0] * (1 - frac) + path[hi][0] * frac,
                path[lo][1] * (1 - frac) + path[hi][1] * frac,
            )
        )
    return out


def blend_guides(guides: Sequence, weights: Optional[Sequence[float]] = None):
    """Weighted blend of guide videos of the same shape."""
    import torch

    guides = [g for g in guides if g is not None]
    if not guides:
        return None
    if len(guides) == 1:
        return guides[0]
    shape = guides[0].shape
    for g in guides[1:]:
        if g.shape != shape:
            raise ValueError(f"guide shape mismatch: {tuple(g.shape)} vs {tuple(shape)}")
    w = list(weights or [1.0] * len(guides))
    total = sum(w) or 1.0
    out = torch.zeros_like(guides[0])
    for weight, guide in zip(w, guides):
        out = out + (weight / total) * guide
    return out.clamp(-1, 1)


class MotionController:
    """Builds the guide video for a request from whichever signals are asked for."""

    def __init__(
        self,
        *,
        use_depth: bool = False,
        use_pose: bool = False,
        use_flow: bool = False,
        camera: Optional[CameraTrajectory] = None,
        trajectories: Optional[Sequence[Sequence[Tuple[float, float]]]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.use_depth = use_depth
        self.use_pose = use_pose
        self.use_flow = use_flow
        self.camera = camera
        self.trajectories = list(trajectories or [])
        self.weights = weights or {}

    def build(
        self, source_video=None, *, num_frames: int = 81, height: int = 480, width: int = 832
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        """Return ``(guide_video, report)``. ``guide_video`` may be ``None``."""
        guides, names = [], []
        if source_video is not None:
            if self.use_depth:
                g = depth_guide(source_video)
                if g is not None:
                    guides.append(g)
                    names.append("depth")
            if self.use_pose:
                g = pose_guide(source_video)
                if g is not None:
                    guides.append(g)
                    names.append("pose")
            if self.use_flow:
                g = flow_guide(source_video)
                if g is not None:
                    guides.append(g)
                    names.append("flow")
        if self.trajectories:
            guides.append(trajectory_guide(self.trajectories, num_frames, height, width))
            names.append("trajectory")

        report: Dict[str, Any] = {"guides": names}
        if self.camera is not None:
            # Plücker maps are 6-channel and are not a drop-in RGB guide; they
            # are returned separately for a CameraCtrl-style adapter.
            report["plucker_shape"] = tuple(
                plucker_embedding(self.camera, height, width).shape
            )
        if not guides:
            return None, report
        blended = blend_guides(guides, [self.weights.get(n, 1.0) for n in names])
        return blended, report
