"""Stage 3: physics-engine proxy conditioning.

**Seedance has no physics engine.** Both the 1.0 report and the 1.5/2.0 reports
attribute motion realism to data curation and reward modelling; physical
plausibility is emergent, and independent testing still finds it failing on
complex multi-character sports and fluid dynamics. The literature agrees on the
ceiling: on Physics-IQ (arXiv:2501.09038, 396 real videos / 66 scenarios),
"physical understanding is severely limited, and unrelated to visual realism"
across Sora, Runway, Pika, Lumiere, SVD and VideoPoet -- VideoPoet scored 24.1%.

So this stage does not reproduce Seedance. It closes part of the gap a
different way, the PhysGen (arXiv:2409.18964) way: simulate a coarse proxy of
the scene, render it, and feed it to the diffusion model as structure
conditioning. The simulation supplies the causality the model lacks; the model
supplies the appearance the simulation lacks.

Backends, in preference order: PyBullet, MuJoCo, then a built-in analytic
integrator that needs nothing installed. The built-in one handles gravity,
drag, ground and wall collisions with restitution, and sphere-sphere impacts --
enough for falling, bouncing, rolling and collision shots, which is where
diffusion models fail most visibly.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = ["Body", "PhysicsScene", "PhysicsProxyRenderer", "available_engines"]


def available_engines() -> Tuple[str, ...]:
    engines = ["analytic"]
    for name in ("pybullet", "mujoco"):
        try:
            __import__(name)
            engines.append(name)
        except Exception:
            pass
    return tuple(engines)


@dataclass
class Body:
    """A rigid body in the proxy scene. Units are metres, kilograms, seconds."""

    name: str = "body"
    shape: str = "sphere"  # "sphere" | "box"
    radius: float = 0.15
    size: Tuple[float, float, float] = (0.3, 0.3, 0.3)
    mass: float = 1.0
    position: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    restitution: float = 0.6
    friction: float = 0.4
    colour: Tuple[float, float, float] = (0.9, 0.35, 0.25)
    static: bool = False


@dataclass
class PhysicsScene:
    """Scene description shared by every backend."""

    bodies: List[Body] = field(default_factory=list)
    gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)
    ground_y: float = 0.0
    walls: Tuple[float, float] = (-2.5, 2.5)  # x bounds
    depth_bounds: Tuple[float, float] = (-2.5, 2.5)  # z bounds
    drag: float = 0.02
    fps: int = 24
    duration_s: float = 3.0
    camera_distance: float = 5.0
    camera_height: float = 1.2
    fov_deg: float = 45.0

    @property
    def num_frames(self) -> int:
        return max(2, int(round(self.duration_s * self.fps)))

    def add(self, body: Body) -> "PhysicsScene":
        self.bodies.append(body)
        return self

    @classmethod
    def bouncing_ball(cls, height: float = 2.0, fps: int = 24, duration_s: float = 3.0) -> "PhysicsScene":
        return cls(
            bodies=[Body(name="ball", position=(0.0, height, 0.0), velocity=(0.6, 0.0, 0.0))],
            fps=fps,
            duration_s=duration_s,
        )

    @classmethod
    def collision(cls, fps: int = 24, duration_s: float = 3.0) -> "PhysicsScene":
        return cls(
            bodies=[
                Body(name="a", position=(-1.2, 0.15, 0.0), velocity=(2.0, 0.0, 0.0),
                     colour=(0.9, 0.3, 0.2)),
                Body(name="b", position=(1.2, 0.15, 0.0), velocity=(-1.0, 0.0, 0.0),
                     colour=(0.2, 0.5, 0.9)),
            ],
            fps=fps,
            duration_s=duration_s,
        )


class PhysicsProxyRenderer:
    """Simulates a :class:`PhysicsScene` and renders it as a guide video."""

    def __init__(
        self,
        engine: str = "auto",
        *,
        height: int = 480,
        width: int = 832,
        substeps: int = 4,
        render_mode: str = "shaded",  # "shaded" | "depth" | "mask"
    ):
        self.engine = self._pick_engine(engine)
        self.height = height
        self.width = width
        self.substeps = substeps
        self.render_mode = render_mode

    @staticmethod
    def _pick_engine(engine: str) -> str:
        if engine != "auto":
            if engine not in available_engines():
                logger.warning("engine %s unavailable; using the analytic integrator", engine)
                return "analytic"
            return engine
        engines = available_engines()
        for preferred in ("pybullet", "mujoco"):
            if preferred in engines:
                return preferred
        return "analytic"

    # ------------------------------------------------------------- simulate
    def simulate(self, scene: PhysicsScene) -> List[List[Dict[str, Any]]]:
        """Return per-frame body states: ``[frame][body] -> {position, ...}``."""
        if self.engine == "pybullet":
            try:
                return self._simulate_pybullet(scene)
            except Exception as exc:
                logger.warning("pybullet simulation failed (%s); falling back", exc)
        elif self.engine == "mujoco":
            try:
                return self._simulate_mujoco(scene)
            except Exception as exc:
                logger.warning("mujoco simulation failed (%s); falling back", exc)
        return self._simulate_analytic(scene)

    def _simulate_analytic(self, scene: PhysicsScene) -> List[List[Dict[str, Any]]]:
        """Semi-implicit Euler with restitution and sphere-sphere impulses.

        Semi-implicit (velocity updated before position) rather than explicit
        Euler: it conserves energy far better at these step sizes, which is the
        difference between a ball that settles and one that gains height.
        """
        dt = 1.0 / (scene.fps * self.substeps)
        state = [
            {
                "name": b.name,
                "p": list(b.position),
                "v": list(b.velocity),
                "body": b,
            }
            for b in scene.bodies
        ]
        frames: List[List[Dict[str, Any]]] = []
        for _ in range(scene.num_frames):
            for _ in range(self.substeps):
                for s in state:
                    b: Body = s["body"]
                    if b.static:
                        continue
                    for axis in range(3):
                        s["v"][axis] += scene.gravity[axis] * dt
                        s["v"][axis] *= 1.0 - scene.drag * dt
                        s["p"][axis] += s["v"][axis] * dt
                    r = b.radius if b.shape == "sphere" else b.size[1] / 2
                    if s["p"][1] - r <= scene.ground_y:
                        s["p"][1] = scene.ground_y + r
                        if s["v"][1] < 0:
                            impact = -s["v"][1]
                            s["v"][1] = impact * b.restitution
                            if s["v"][1] < 0.15:
                                s["v"][1] = 0.0  # settle instead of buzzing
                            if impact > 0.15:
                                # Coulomb-ish tangential loss, only on a real impact
                                damp = min(0.9, b.friction * impact)
                                s["v"][0] *= 1.0 - damp
                                s["v"][2] *= 1.0 - damp
                        # rolling resistance while in contact, integrated over dt
                        roll = max(0.0, 1.0 - b.friction * dt)
                        s["v"][0] *= roll
                        s["v"][2] *= roll
                    for axis, bounds in ((0, scene.walls), (2, scene.depth_bounds)):
                        lo, hi = bounds
                        if s["p"][axis] - r < lo:
                            s["p"][axis] = lo + r
                            s["v"][axis] = abs(s["v"][axis]) * b.restitution
                        elif s["p"][axis] + r > hi:
                            s["p"][axis] = hi - r
                            s["v"][axis] = -abs(s["v"][axis]) * b.restitution
                self._resolve_contacts(state)
            frames.append(
                [
                    {"name": s["name"], "position": tuple(s["p"]), "velocity": tuple(s["v"]),
                     "body": s["body"]}
                    for s in state
                ]
            )
        return frames

    @staticmethod
    def _resolve_contacts(state: List[Dict[str, Any]]) -> None:
        """Elastic-with-restitution impulses between sphere pairs."""
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                a, b = state[i], state[j]
                ba, bb = a["body"], b["body"]
                if ba.shape != "sphere" or bb.shape != "sphere":
                    continue
                delta = [b["p"][k] - a["p"][k] for k in range(3)]
                dist = math.sqrt(sum(d * d for d in delta)) or 1e-6
                overlap = ba.radius + bb.radius - dist
                if overlap <= 0:
                    continue
                n = [d / dist for d in delta]
                # positional correction, split by inverse mass
                inv_a = 0.0 if ba.static else 1.0 / max(ba.mass, 1e-6)
                inv_b = 0.0 if bb.static else 1.0 / max(bb.mass, 1e-6)
                inv_sum = inv_a + inv_b or 1.0
                for k in range(3):
                    a["p"][k] -= n[k] * overlap * (inv_a / inv_sum)
                    b["p"][k] += n[k] * overlap * (inv_b / inv_sum)
                rel = sum((b["v"][k] - a["v"][k]) * n[k] for k in range(3))
                if rel >= 0:
                    continue
                e = min(ba.restitution, bb.restitution)
                impulse = -(1 + e) * rel / inv_sum
                for k in range(3):
                    a["v"][k] -= impulse * n[k] * inv_a
                    b["v"][k] += impulse * n[k] * inv_b

    def _simulate_pybullet(self, scene: PhysicsScene) -> List[List[Dict[str, Any]]]:
        import pybullet as p
        import pybullet_data

        client = p.connect(p.DIRECT)
        try:
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
            p.setGravity(*scene.gravity, physicsClientId=client)
            p.setTimeStep(1.0 / (scene.fps * self.substeps), physicsClientId=client)
            p.loadURDF("plane.urdf", [0, 0, 0], physicsClientId=client)
            ids = []
            for body in scene.bodies:
                if body.shape == "sphere":
                    col = p.createCollisionShape(p.GEOM_SPHERE, radius=body.radius,
                                                 physicsClientId=client)
                else:
                    col = p.createCollisionShape(
                        p.GEOM_BOX, halfExtents=[s / 2 for s in body.size], physicsClientId=client
                    )
                # PyBullet is z-up; the scene is y-up.
                pos = (body.position[0], body.position[2], body.position[1])
                uid = p.createMultiBody(
                    baseMass=0 if body.static else body.mass,
                    baseCollisionShapeIndex=col,
                    basePosition=pos,
                    physicsClientId=client,
                )
                p.changeDynamics(uid, -1, restitution=body.restitution,
                                 lateralFriction=body.friction, physicsClientId=client)
                p.resetBaseVelocity(
                    uid,
                    linearVelocity=(body.velocity[0], body.velocity[2], body.velocity[1]),
                    physicsClientId=client,
                )
                ids.append(uid)

            frames = []
            for _ in range(scene.num_frames):
                for _ in range(self.substeps):
                    p.stepSimulation(physicsClientId=client)
                row = []
                for uid, body in zip(ids, scene.bodies):
                    (x, z, y), _ = p.getBasePositionAndOrientation(uid, physicsClientId=client)
                    (vx, vz, vy), _ = p.getBaseVelocity(uid, physicsClientId=client)
                    row.append(
                        {"name": body.name, "position": (x, y, z), "velocity": (vx, vy, vz),
                         "body": body}
                    )
                frames.append(row)
            return frames
        finally:
            p.disconnect(physicsClientId=client)

    def _simulate_mujoco(self, scene: PhysicsScene) -> List[List[Dict[str, Any]]]:
        import mujoco

        model = mujoco.MjModel.from_xml_string(self._mjcf(scene))
        data = mujoco.MjData(model)
        for i, body in enumerate(scene.bodies):
            adr = model.jnt_qposadr[i]
            data.qpos[adr : adr + 3] = [body.position[0], body.position[2], body.position[1]]
            vadr = model.jnt_dofadr[i]
            data.qvel[vadr : vadr + 3] = [body.velocity[0], body.velocity[2], body.velocity[1]]
        steps = max(1, int(round(1.0 / (scene.fps * model.opt.timestep))))
        frames = []
        for _ in range(scene.num_frames):
            for _ in range(steps):
                mujoco.mj_step(model, data)
            row = []
            for i, body in enumerate(scene.bodies):
                adr = model.jnt_qposadr[i]
                x, z, y = data.qpos[adr : adr + 3]
                row.append({"name": body.name, "position": (float(x), float(y), float(z)),
                            "velocity": (0.0, 0.0, 0.0), "body": body})
            frames.append(row)
        return frames

    @staticmethod
    def _mjcf(scene: PhysicsScene) -> str:
        bodies = []
        for i, b in enumerate(scene.bodies):
            geom = (
                f'<geom type="sphere" size="{b.radius}" mass="{b.mass}" '
                f'friction="{b.friction} 0.005 0.0001"/>'
                if b.shape == "sphere"
                else f'<geom type="box" size="{b.size[0]/2} {b.size[2]/2} {b.size[1]/2}" mass="{b.mass}"/>'
            )
            bodies.append(
                f'<body name="{b.name or i}" pos="{b.position[0]} {b.position[2]} {b.position[1]}">'
                f'<freejoint/>{geom}</body>'
            )
        return f"""<mujoco>
  <option gravity="0 0 {scene.gravity[1]}" timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 {scene.ground_y}"/>
    {''.join(bodies)}
  </worldbody>
</mujoco>"""

    # --------------------------------------------------------------- render
    def render(self, scene: PhysicsScene, frames: Optional[List[List[Dict[str, Any]]]] = None):
        """Render the simulation to a guide video ``[3, T, H, W]`` in ``[-1, 1]``.

        A painter's-algorithm rasteriser: spheres and boxes projected with a
        pinhole camera, sorted back to front, with Lambertian shading and a
        ground plane. Crude on purpose -- the diffusion model only needs
        position, scale, occlusion order and contact timing from this.
        """
        import torch

        frames = frames or self.simulate(scene)
        t = len(frames)
        h, w = self.height, self.width
        focal = 0.5 * w / math.tan(math.radians(scene.fov_deg) / 2)
        cam = (0.0, scene.camera_height, scene.camera_distance)

        ys = torch.arange(h, dtype=torch.float32).view(-1, 1)
        xs = torch.arange(w, dtype=torch.float32).view(1, -1)
        video = torch.zeros(3, t, h, w)

        for fi, row in enumerate(frames):
            canvas = self._background(h, w, scene, cam, focal, ys, xs)
            drawn = []
            for item in row:
                body: Body = item["body"]
                px, py, pz = item["position"]
                depth = cam[2] - pz
                if depth <= 0.05:
                    continue
                sx = (px - cam[0]) * focal / depth + w / 2
                sy = -(py - cam[1]) * focal / depth + h / 2
                radius = (body.radius if body.shape == "sphere" else max(body.size) / 2)
                sr = radius * focal / depth
                drawn.append((depth, sx, sy, sr, body))
            for depth, sx, sy, sr, body in sorted(drawn, key=lambda d: -d[0]):
                dist2 = (xs - sx) ** 2 + (ys - sy) ** 2
                mask = dist2 <= sr**2
                if not bool(mask.any()):
                    continue
                if self.render_mode == "mask":
                    for c in range(3):
                        canvas[c][mask] = 1.0
                    continue
                # Lambertian shading from a fixed key light, so the guide has
                # enough form for a depth/structure-conditioned model to read.
                normal_z = torch.sqrt((sr**2 - dist2).clamp(min=0)) / max(sr, 1e-6)
                nx = (xs - sx) / max(sr, 1e-6)
                ny = -(ys - sy) / max(sr, 1e-6)
                shade = (0.35 + 0.65 * (0.4 * nx + 0.5 * ny + 0.75 * normal_z).clamp(0, 1))
                if self.render_mode == "depth":
                    value = 1.0 - min(1.0, max(0.0, depth / (scene.camera_distance * 2)))
                    for c in range(3):
                        canvas[c][mask] = value
                else:
                    for c in range(3):
                        canvas[c][mask] = (body.colour[c] * shade)[mask]
            video[:, fi] = canvas
        return (video.clamp(0, 1) * 2 - 1)

    @staticmethod
    def _background(h: int, w: int, scene: PhysicsScene, cam, focal: float, ys, xs):
        """Ground plane with a horizon, shaded by distance."""
        import torch

        canvas = torch.full((3, h, w), 0.08)
        # y_screen for the ground plane at depth d:  sy = (cam_y - ground) * f / d + h/2
        horizon = h / 2
        below = ys.expand(h, w) > horizon
        depth = (cam[1] - scene.ground_y) * focal / (ys.expand(h, w) - horizon).clamp(min=1e-3)
        fade = (1.0 - (depth / (scene.camera_distance * 3)).clamp(0, 1)) * 0.45 + 0.1
        for c, tint in enumerate((1.0, 1.02, 1.05)):
            channel = canvas[c]
            channel[below] = (fade * tint)[below]
        return canvas

    def guide_for_request(
        self, scene: PhysicsScene, *, num_frames: int, height: int, width: int
    ):
        """Simulate and render at exactly the size the generator wants."""
        self.height, self.width = height, width
        scene.duration_s = num_frames / max(scene.fps, 1)
        video = self.render(scene)
        if video.shape[1] != num_frames:
            import torch.nn.functional as F

            video = F.interpolate(
                video.unsqueeze(0), size=(num_frames, height, width), mode="trilinear",
                align_corners=False,
            )[0]
        return video
