"""Residual RL scaffolding for LeIsaac policies.

This package provides generic, model-agnostic building blocks for
ResFiT-style residual off-policy RL:

- config dataclasses
- actor/critic network definitions
- replay buffer
- learner/update loop

Task- and policy-specific logic should live in the training script.
"""

from .config import ResidualRLConfig
from .learner import ResidualLearner, ResidualUpdateStats
from .networks import ResidualActor, TwinCritic
from .replay import ReplayBatch, ReplayBuffer, TransitionBatch

__all__ = [
    "ResidualActor",
    "ResidualLearner",
    "ResidualRLConfig",
    "ResidualUpdateStats",
    "ReplayBatch",
    "ReplayBuffer",
    "TransitionBatch",
    "TwinCritic",
]
