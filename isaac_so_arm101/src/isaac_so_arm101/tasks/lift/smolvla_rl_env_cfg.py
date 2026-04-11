"""Environment config for SmolVLA RWR (Reward-Weighted Regression) finetuning.

Inherits the fixed-layout trajectory-guided sparse env and enables cameras so
SmolVLA can observe the scene.  The frozen-backbone observation term from
``smolvla_env_cfg.py`` is intentionally OMITTED here: SmolVLA inference is run
separately inside the rollout loop, so we avoid running the backbone twice per step.

Registered gym IDs:
    Isaac-SO-ARM101-SmolVLA-RL-v0
    Isaac-SO-ARM101-SmolVLA-RL-Play-v0
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .fixed_layout_env_cfg import SoArm101FixedLayoutGuidedLiftCubeSparseEnvCfg


@configclass
class SoArm101SmolVLARLEnvCfg(SoArm101FixedLayoutGuidedLiftCubeSparseEnvCfg):
    """Lift-cube env for SmolVLA RWR training.

    Cameras are enabled for SmolVLA visual input.
    Trajectory guidance rewards are active (inherited).
    No frozen-backbone obs term (SmolVLA runs in the rollout loop instead).
    """

    disable_task_cameras: bool = False

    def __post_init__(self):
        super().__post_init__()
        # Use a reasonable episode length for RL (5 s at 50 Hz = 250 steps).
        self.episode_length_s = 5.0
        # Reduce num_envs default — SmolVLA needs GPU memory.
        self.scene.num_envs = 8


@configclass
class SoArm101SmolVLARLEnvCfg_PLAY(SoArm101SmolVLARLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
