"""The staged Seedance pipeline over an open-weights backend.

This is the thing you actually run. It implements the four-stage plan from the
engineering brief on top of any Wan checkpoint:

* **Stage 1 -- prompt.** Dense-caption rewriting, mirroring Seedance's Qwen PE
  stage, plus storyboard expansion for multi-shot requests.
* **Stage 2 -- identity and motion.** Reference conditioning (VACE/Phantom),
  caption locking, LoRA, and depth/pose/flow/camera/trajectory guides.
* **Stage 3 -- physics.** A simulated proxy rendered as a guide video, the
  PhysGen pattern, for scenes dominated by collisions and gravity.
* **Stage 4 -- polish.** Colour stabilisation, interpolation, face restore,
  upscaling, then temporal QA on the finished clip.

What it does not do, and will say so in the report: native joint audio-video,
the @mention 4-modality reference system, and Seedance's emergent physics. Those
are not reproducible from open weights.
"""

from __future__ import annotations

import logging
import os
import random
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..backends import GenerationRequest, VideoBackend, auto_backend, parse_size
from ..story import Shot, Storyboard
from ..stages.audio_stage import AudioTrack, attach_audio
from ..stages.identity import IdentityConfig, IdentityController, identity_drift
from ..stages.motion import CameraTrajectory, MotionController, blend_guides
from ..stages.multishot import ShotChainer
from ..stages.physics import PhysicsProxyRenderer, PhysicsScene
from ..stages.polish import PolishConfig, polish
from ..stages.prompt_enhancer import PromptEnhancer
from .base import PipelineResult, StageTimer

logger = logging.getLogger(__name__)

__all__ = ["StageSettings", "SeedancePipeline"]


@dataclass
class StageSettings:
    """Which stages run, and how."""

    # Stage 1
    enhance_prompt: bool = True
    enhancer_model: str = "Qwen/Qwen2.5-7B-Instruct"
    enhancer_offline_only: bool = False
    subject_lock: str = ""

    # Stage 2
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    use_depth: bool = False
    use_pose: bool = False
    use_flow: bool = False
    camera: Optional[CameraTrajectory] = None
    trajectories: List[Sequence[Tuple[float, float]]] = field(default_factory=list)
    guide_weights: Dict[str, float] = field(default_factory=dict)

    # Stage 3
    physics: Optional[PhysicsScene] = None
    physics_engine: str = "auto"
    physics_render_mode: str = "shaded"
    physics_weight: float = 1.0

    # Stage 4
    polish: PolishConfig = field(default_factory=PolishConfig)

    # audio
    audio_tracks: List[AudioTrack] = field(default_factory=list)

    # multishot
    transition_frames: int = 0
    chain_mode: str = "auto"

    # bookkeeping
    measure_identity_drift: bool = False
    save_guide_video: bool = False

    @classmethod
    def preset(cls, name: str) -> "StageSettings":
        """Named presets matching the brief's Stage 1-4 recommendations."""
        if name == "fast":
            return cls(
                enhance_prompt=True,
                enhancer_offline_only=True,
                polish=PolishConfig(color_stabilize=True, interpolate=1, upscale_factor=1),
            )
        if name == "balanced":
            return cls(
                polish=PolishConfig(color_stabilize=True, interpolate=2, upscale_factor=1)
            )
        if name == "quality":
            return cls(
                polish=PolishConfig(
                    color_stabilize=True, interpolate=2, upscale_factor=2, face_restore=True
                ),
                measure_identity_drift=True,
            )
        if name == "cinematic":
            return cls(
                use_depth=True,
                polish=PolishConfig(
                    color_stabilize=True,
                    color_method="adain",
                    interpolate=2,
                    upscale_factor=2,
                    face_restore=True,
                    face_strength=0.3,
                ),
                transition_frames=4,
                measure_identity_drift=True,
            )
        raise KeyError(f"unknown preset {name!r}; try fast|balanced|quality|cinematic")


class SeedancePipeline:
    """Seedance-shaped orchestration over an open-weights video backend."""

    def __init__(
        self,
        model: str = "Wan-AI/Wan2.1-T2V-1.3B",
        *,
        backend: Optional[VideoBackend] = None,
        settings: Optional[StageSettings] = None,
        device_id: int = 0,
        cache_dir: Optional[str] = None,
        prefer_backend: str = "auto",
        plan_memory: bool = True,
        **backend_kwargs: Any,
    ):
        self.model = model
        self.settings = settings or StageSettings()
        self.device_id = device_id
        self.plan_memory = plan_memory
        self._backend = backend
        self._backend_kwargs = dict(backend_kwargs)
        self._prefer = prefer_backend
        self._cache_dir = cache_dir
        self._enhancer: Optional[PromptEnhancer] = None

    # -------------------------------------------------------------- backend
    @property
    def backend(self) -> VideoBackend:
        if self._backend is None:
            self._backend = auto_backend(
                self.model,
                prefer=self._prefer,
                device_id=self.device_id,
                cache_dir=self._cache_dir,
                **self._backend_kwargs,
            )
        return self._backend

    def load(self) -> "SeedancePipeline":
        self.backend.load()
        return self

    def unload(self) -> None:
        if self._backend is not None:
            self._backend.unload()

    @property
    def enhancer(self) -> PromptEnhancer:
        if self._enhancer is None:
            self._enhancer = PromptEnhancer(
                self.settings.enhancer_model,
                offline_only=self.settings.enhancer_offline_only,
                enabled=self.settings.enhance_prompt,
            )
        return self._enhancer

    def defaults(self) -> Dict[str, Any]:
        getter = getattr(self.backend, "defaults", None)
        return getter() if callable(getter) else {}

    # ------------------------------------------------------------- generate
    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        size: Optional[str] = None,
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        shift: Optional[float] = None,
        seed: int = -1,
        solver: str = "unipc",
        task: Optional[str] = None,
        image: Any = None,
        last_image: Any = None,
        source_video: Optional[str] = None,
        storyboard: Optional[Storyboard] = None,
        output: Optional[str] = None,
        progress: Optional[Callable[[str, str], None]] = None,
    ) -> PipelineResult:
        """Run the full staged pipeline for one request."""
        timer = StageTimer()
        report: Dict[str, Any] = {"model": self.model, "stages": []}
        warnings: List[str] = []
        emit = progress or (lambda stage, msg: logger.info("[%s] %s", stage, msg))

        defaults = self.defaults()
        size = size or defaults.get("size", "832*480")
        fps = fps or defaults.get("fps", 16)
        num_frames = num_frames or defaults.get("num_frames", 81)
        steps = steps if steps is not None else defaults.get("steps", 50)
        guidance_scale = (
            guidance_scale if guidance_scale is not None else defaults.get("guidance_scale", 5.0)
        )
        shift = shift if shift is not None else defaults.get("shift", 5.0)
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)
        width, height = parse_size(size)

        caps = self.backend.capabilities()
        if task is None:
            task = "i2v" if image is not None and caps.supports("i2v") else "t2v"

        # ---------------------------------------------------------- stage 1
        with timer("stage1_prompt"):
            emit("stage1", "rewriting the prompt")
            if storyboard is None:
                caption = self.enhancer.enhance(prompt, task=task, image=image, seed=seed)
                enhanced = caption.text
                report["prompt_source"] = caption.source
                warnings.extend(caption.warnings)
                board = Storyboard.single(enhanced, num_frames)
            else:
                captions = self.enhancer.enhance_storyboard(
                    storyboard.captions(), task=task, subject_lock=self.settings.subject_lock
                )
                board = Storyboard(prompt=prompt)
                for shot, caption in zip(storyboard.shots, captions):
                    board.add(Shot(caption=caption.text, num_frames=shot.num_frames,
                                   camera=shot.camera, references=shot.references))
                enhanced = " | ".join(c.text for c in captions)
                report["prompt_source"] = captions[0].source if captions else "passthrough"
            report["stages"].append("prompt")

        # ---------------------------------------------------------- stage 2
        identity = IdentityController(self.settings.identity)
        with timer("stage2_control"):
            guides = []
            guide_names: List[str] = []
            motion = MotionController(
                use_depth=self.settings.use_depth,
                use_pose=self.settings.use_pose,
                use_flow=self.settings.use_flow,
                camera=self.settings.camera,
                trajectories=self.settings.trajectories,
                weights=self.settings.guide_weights,
            )
            source_tensor = None
            if source_video and (self.settings.use_depth or self.settings.use_pose or self.settings.use_flow):
                from ..runtime.io import load_video

                emit("stage2", f"reading control source {source_video}")
                source_tensor = load_video(source_video, max_frames=num_frames)
            guide, motion_report = motion.build(
                source_tensor, num_frames=num_frames, height=height, width=width
            )
            if guide is not None:
                guides.append(guide)
                guide_names.extend(motion_report.get("guides", []))
            report["motion"] = motion_report
            if guide_names:
                report["stages"].append("motion")

        # ---------------------------------------------------------- stage 3
        physics_report: Dict[str, Any] = {}
        if self.settings.physics is not None:
            with timer("stage3_physics"):
                emit("stage3", "simulating the physics proxy")
                renderer = PhysicsProxyRenderer(
                    self.settings.physics_engine,
                    height=height,
                    width=width,
                    render_mode=self.settings.physics_render_mode,
                )
                scene = self.settings.physics
                scene.fps = fps
                proxy = renderer.guide_for_request(
                    scene, num_frames=num_frames, height=height, width=width
                )
                guides.append(proxy)
                guide_names.append("physics")
                physics_report = {
                    "engine": renderer.engine,
                    "bodies": [b.name for b in scene.bodies],
                    "note": (
                        "Seedance itself has no physics engine -- this is a "
                        "PhysGen-style proxy, not a reproduction of its behaviour"
                    ),
                }
                report["physics"] = physics_report
                report["stages"].append("physics")

        guide_path = None
        if guides:
            combined = blend_guides(
                guides,
                [
                    self.settings.physics_weight if n == "physics"
                    else self.settings.guide_weights.get(n, 1.0)
                    for n in guide_names
                ],
            )
            if caps.control_video:
                from ..runtime.io import save_video

                workdir = tempfile.mkdtemp(prefix="seedance_guide_")
                guide_path = save_video(combined, os.path.join(workdir, "guide.mp4"), fps=fps)
                report["guide_video"] = guide_path
            else:
                warnings.append(
                    "control guides were built but this checkpoint has no control "
                    "path; use a VACE checkpoint to apply them"
                )

        # ------------------------------------------------------------ sample
        request = GenerationRequest(
            prompt=board.shots[0].caption,
            negative_prompt=negative_prompt,
            size=size,
            num_frames=num_frames,
            fps=fps,
            steps=steps,
            guidance_scale=guidance_scale,
            shift=shift,
            seed=seed,
            solver=solver,
            task=task,
            image=image,
            last_image=last_image,
            source_video=source_video,
            control_video=guide_path,
        )
        identity_info = identity.apply(request, caps)
        report["identity"] = identity_info
        if identity_info.get("method") not in (None, "caption_only"):
            report["stages"].append("identity")

        if self.plan_memory:
            report["memory"] = self._plan_memory(width, height, num_frames)

        with timer("sample"):
            if len(board.shots) > 1:
                emit("sample", f"generating {len(board.shots)} shots")
                chainer = ShotChainer(
                    lambda caption, **kw: self._generate_one(request, caption, **kw),
                    mode=self.settings.chain_mode,
                    transition_frames=self.settings.transition_frames,
                )
                chained = chainer.run(
                    board,
                    capabilities=caps,
                    fps=fps,
                    progress=lambda i, n, c: emit("sample", f"shot {i}/{n}"),
                )
                video = chained["video"]
                report["shots"] = [
                    {"index": s.index, "mode": s.mode, "seconds": s.seconds}
                    for s in chained["shots"]
                ]
            else:
                emit("sample", f"{steps} steps at {size}, {num_frames} frames")
                video = self.backend.generate(request)
            for problem in self.backend.validate(request):
                warnings.append(problem)

        # ---------------------------------------------------------- stage 4
        with timer("stage4_polish"):
            emit("stage4", "post-processing")
            video, polish_report = polish(video, self.settings.polish)
            report["polish"] = polish_report
            report["stages"].append("polish")
            out_fps = fps * max(1, self.settings.polish.interpolate)

        if self.settings.measure_identity_drift:
            with timer("identity_drift"):
                report["identity_drift"] = identity_drift(video)

        # ------------------------------------------------------------ output
        result = PipelineResult(
            video=video,
            fps=out_fps,
            prompt=prompt,
            enhanced_prompt=enhanced,
            seed=seed,
            report=report,
            timings=timer.timings,
            warnings=warnings,
        )
        if output:
            with timer("save"):
                from ..runtime.io import save_video

                path = save_video(video, output, fps=out_fps)
                result.path = path
                if self.settings.audio_tracks:
                    audio_report = attach_audio(
                        path,
                        tracks=self.settings.audio_tracks,
                        duration_s=result.duration_s,
                    )
                    result.path = audio_report.get("path", path)
                    result.audio_path = audio_report.get("audio_path")
                    report["audio"] = audio_report
        if self.settings.save_guide_video and guide_path:
            result.report["guide_video"] = guide_path
        return result

    # --------------------------------------------------------------- helpers
    def _generate_one(self, template: GenerationRequest, caption: str, **kwargs: Any):
        """Per-shot generation used by the chainer."""
        shot_index = kwargs.pop("shot_index", 0)
        request = template.with_(
            prompt=caption,
            num_frames=kwargs.pop("num_frames", template.num_frames),
            task=kwargs.pop("task", template.task),
            image=kwargs.pop("image", None),
            last_image=kwargs.pop("last_image", None),
            # vary the seed per shot so shots are not near-duplicates
            seed=(template.seed + shot_index) if template.seed >= 0 else -1,
        )
        return self.backend.generate(request)

    def _plan_memory(self, width: int, height: int, num_frames: int) -> Dict[str, Any]:
        from ..runtime.memory import plan_vram

        info = getattr(self.backend, "describe", lambda: {})()
        params = "1.3B" if (info.get("dim") or 5120) <= 2048 else "14B"
        plan = plan_vram(
            params=params, width=width, height=height, num_frames=num_frames,
            device_id=self.device_id,
        )
        if not plan.fits:
            logger.warning("VRAM plan: %s", plan.summary())
            for note in plan.notes:
                logger.warning("  suggestion: %s", note)
        return {
            "fits": plan.fits,
            "estimated_gb": plan.estimated_gb,
            "available_gb": plan.available_gb,
            "suggestions": plan.notes,
        }

    # ------------------------------------------------------------ capability
    def capability_report(self) -> Dict[str, Any]:
        """What this model+pipeline combination can and cannot do."""
        caps = self.backend.capabilities()
        return {
            "model": self.model,
            "backend": self.backend.name,
            "tasks": list(caps.tasks),
            "reference_images": caps.reference_images,
            "control_video": caps.control_video,
            "native_audio": caps.audio,
            "not_reproducible": [
                "native joint audio-video generation (dual-branch DiT)",
                "the @mention 4-modality reference system",
                "Seedance's emergent physics (this pipeline approximates it with a simulated proxy)",
                "the reported ~90% first-try usability rate",
            ],
        }
