"""Stage 1-4 building blocks used by the staged pipeline."""

from .audio_stage import AudioMix, AudioTrack, attach_audio, suggest_sync_points
from .identity import ADAPTER_NOTES, IdentityConfig, IdentityController, identity_drift
from .motion import (
    CameraTrajectory,
    MotionController,
    blend_guides,
    depth_guide,
    flow_guide,
    plucker_embedding,
    pose_guide,
    trajectory_guide,
)
from .multishot import ShotChainer, ShotResult, concat_shots, crossfade
from .physics import Body, PhysicsProxyRenderer, PhysicsScene, available_engines
from .polish import (
    PolishConfig,
    adain_normalize,
    interpolate_frames,
    match_colors,
    polish,
    restore_faces,
    temporal_qa,
    upscale,
)
from .prompt_enhancer import DenseCaption, PromptEnhancer

__all__ = [
    "PromptEnhancer",
    "DenseCaption",
    "IdentityConfig",
    "IdentityController",
    "identity_drift",
    "ADAPTER_NOTES",
    "MotionController",
    "CameraTrajectory",
    "plucker_embedding",
    "depth_guide",
    "pose_guide",
    "flow_guide",
    "trajectory_guide",
    "blend_guides",
    "PhysicsScene",
    "Body",
    "PhysicsProxyRenderer",
    "available_engines",
    "PolishConfig",
    "polish",
    "match_colors",
    "adain_normalize",
    "interpolate_frames",
    "restore_faces",
    "upscale",
    "temporal_qa",
    "AudioTrack",
    "AudioMix",
    "attach_audio",
    "suggest_sync_points",
    "ShotChainer",
    "ShotResult",
    "concat_shots",
    "crossfade",
]
