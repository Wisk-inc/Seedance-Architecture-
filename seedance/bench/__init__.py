"""Evaluation harnesses and benchmark-chart rendering."""

from .ablation import DEFAULT_STEPS, AblationStep, run_ablation
from .leaderboard import (
    Benchmark,
    Entry,
    Theme,
    load_benchmark,
    render_grid,
    render_png,
    render_svg,
    save_benchmark,
)
from .logos import LOGOS, logo_tile_svg
from .physics_iq import (
    PhysicsIQScenario,
    action_mask,
    frame_mse,
    run_benchmark,
    score_pair,
    spatial_iou,
    spatiotemporal_iou,
    synthetic_scenarios,
    weighted_spatial_iou,
)

__all__ = [
    "Benchmark",
    "Entry",
    "Theme",
    "render_svg",
    "render_png",
    "render_grid",
    "load_benchmark",
    "save_benchmark",
    "LOGOS",
    "logo_tile_svg",
    "AblationStep",
    "DEFAULT_STEPS",
    "run_ablation",
    "PhysicsIQScenario",
    "run_benchmark",
    "score_pair",
    "action_mask",
    "spatial_iou",
    "spatiotemporal_iou",
    "weighted_spatial_iou",
    "frame_mse",
    "synthetic_scenarios",
]
