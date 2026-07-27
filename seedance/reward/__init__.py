"""Reward modelling and RL: RewardDance scoring, DanceGRPO optimisation."""

from .dance_grpo import DanceGRPO, GRPOConfig, Trajectory
from .heuristics import HeuristicRewardModel
from .reward_dance import (
    DIMENSION_PROMPTS,
    RewardDance,
    RewardEnsemble,
    RewardPrompt,
    build_reward_model,
)

__all__ = [
    "DanceGRPO",
    "GRPOConfig",
    "Trajectory",
    "HeuristicRewardModel",
    "RewardDance",
    "RewardEnsemble",
    "RewardPrompt",
    "DIMENSION_PROMPTS",
    "build_reward_model",
]
