"""Stage tests that need no ML stack: prompt rewriting and the physics proxy."""

from __future__ import annotations

import os

import pytest

from seedance.stages.physics import Body, PhysicsProxyRenderer, PhysicsScene
from seedance.stages.prompt_enhancer import PromptEnhancer


# ------------------------------------------------ torch-free: prompts, physics

def test_template_expansion_adds_caption_slots_without_inventing_facts():
    enhancer = PromptEnhancer(offline_only=True)
    caption = enhancer.template_expand("a fox in the snow")
    assert "fox" in caption.text
    assert caption.source == "template"
    assert len(caption.text) > len("a fox in the snow")
    assert "camera" in caption.text.lower()


def test_existing_camera_direction_is_not_duplicated():
    enhancer = PromptEnhancer(offline_only=True)
    caption = enhancer.template_expand("slow dolly toward a lighthouse")
    assert caption.camera == "dolly"
    assert caption.text.lower().count("camera") <= 1


def test_storyboard_rewrites_share_one_subject_clause():
    enhancer = PromptEnhancer(offline_only=True)
    captions = enhancer.enhance_storyboard(
        ["she opens the door", "she steps outside"], subject_lock="A woman in a red coat"
    )
    assert all("red coat" in c.text for c in captions)


def test_analytic_physics_conserves_plausibility():
    """A dropped ball must fall, bounce lower each time, and settle."""
    scene = PhysicsScene.bouncing_ball(height=2.0, fps=24, duration_s=3.0)
    renderer = PhysicsProxyRenderer("analytic", height=64, width=64)
    frames = renderer.simulate(scene)
    heights = [f[0]["position"][1] for f in frames]
    assert heights[0] > heights[5], "the ball should fall"
    assert min(heights) >= -1e-6, "the ball must not pass through the floor"
    first_peak = max(heights[:20])
    last_peak = max(heights[-20:])
    assert last_peak < first_peak, "bounces must lose energy"


def test_collision_exchanges_momentum():
    scene = PhysicsScene.collision(fps=24, duration_s=2.0)
    frames = PhysicsProxyRenderer("analytic").simulate(scene)
    start = frames[0]
    end = frames[-1]
    # they approach, collide, and separate: the gap shrinks then grows
    gaps = [abs(f[1]["position"][0] - f[0]["position"][0]) for f in frames]
    assert min(gaps) < abs(start[1]["position"][0] - start[0]["position"][0])
    assert gaps[-1] > min(gaps)


def test_physics_engine_selection_falls_back_cleanly():
    renderer = PhysicsProxyRenderer("definitely-not-an-engine")
    assert renderer.engine == "analytic"
