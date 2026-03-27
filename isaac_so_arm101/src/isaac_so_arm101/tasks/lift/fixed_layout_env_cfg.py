# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Fixed cube spawn + fixed goal pose for controlled reward ablations and teacher/student alignment.

Use the same ``--task`` for (1) training a dense teacher, (2) ``collect_trajectories``, and
(3) guided sparse training so scene randomization matches. Trajectory row is chosen by cube/layout
(``object_only``) or ``trajectory_guidance_fixed_traj_index``; at reset, EE is aligned to the teacher
polyline start via IK (not by replaying recorded joint vectors).
"""

from __future__ import annotations

import os

import isaac_so_arm101.tasks.lift.mdp as lift_mdp
from isaaclab.utils import configclass

from .guided_env_cfg import GuidedEEAlignEventCfg, GuidedObservationsCfg, GuidedRewardsCfg
from .joint_pos_env_cfg import SoArm101LiftCubeEnvCfg, SoArm101LiftCubeSparseEnvCfg

_DEFAULT_FIXED_LAYOUT_TRAJECTORY_FILE = os.environ.get(
    "ISAAC_SO_ARM101_TRAJECTORY_FILE",
    os.path.join(
        "isaac_so_arm101", "logs", "rsl_rl", "teacher_trajectories", "so101_fixed_layout_teacher.pt"
    ),
)


def apply_so101_fixed_layout_to_cfg(env_cfg) -> None:
    """Pin goal command to one pose and remove cube XY reset jitter (see ``lift_env_cfg.CommandsCfg``).

    Goal is the **midpoint** of the default UniformPoseCommand training ranges:
    ``pos_x ∈ [-0.1,0.1] → 0``, ``pos_y ∈ [-0.3,-0.1] → -0.2``, ``pos_z ∈ [0.2,0.35] → 0.275``.
    Resampling interval is huge so the goal does not change during an experiment run.
    """
    env_cfg.commands.object_pose.ranges = lift_mdp.UniformPoseCommandCfg.Ranges(
        pos_x=(0.0, 0.0),
        pos_y=(-0.2, -0.2),
        pos_z=(0.275, 0.275),
        roll=(0.0, 0.0),
        pitch=(0.0, 0.0),
        yaw=(0.0, 0.0),
    )
    env_cfg.commands.object_pose.resampling_time_range = (1.0e6, 1.0e6)
    env_cfg.events.reset_object_position.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
    }


@configclass
class SoArm101FixedLayoutLiftCubeEnvCfg(SoArm101LiftCubeEnvCfg):
    """Dense lift-cube with fixed cube start (no reset jitter) and fixed goal pose."""

    def __post_init__(self):
        super().__post_init__()
        apply_so101_fixed_layout_to_cfg(self)


@configclass
class SoArm101FixedLayoutLiftCubeEnvCfg_PLAY(SoArm101FixedLayoutLiftCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class SoArm101FixedLayoutLiftCubeSparseEnvCfg(SoArm101LiftCubeSparseEnvCfg):
    """Sparse lift-cube on the same fixed layout (for guided student training)."""

    def __post_init__(self):
        super().__post_init__()
        apply_so101_fixed_layout_to_cfg(self)


@configclass
class SoArm101FixedLayoutLiftCubeSparseEnvCfg_PLAY(SoArm101FixedLayoutLiftCubeSparseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


def _patch_trajectory_file(env_cfg, trajectory_file: str) -> None:
    """Override ``trajectory_file`` in every reward/event term that references it."""
    for term in (
        env_cfg.rewards.trajectory_guidance,
        env_cfg.rewards.teacher_gripper_alignment,
    ):
        if hasattr(term, "params") and "trajectory_file" in term.params:
            term.params["trajectory_file"] = trajectory_file
    if hasattr(env_cfg.rewards, "discriminator_guidance"):
        # discriminator uses discriminator_file, not trajectory_file — skip
        pass
    for term_name in ("align_ee_to_teacher_trajectory_start", "visualize_teacher_trajectory"):
        term = getattr(env_cfg.events, term_name, None)
        if term is not None and hasattr(term, "params") and "trajectory_file" in term.params:
            term.params["trajectory_file"] = trajectory_file


@configclass
class SoArm101FixedLayoutGuidedLiftCubeSparseEnvCfg(SoArm101FixedLayoutLiftCubeSparseEnvCfg):
    """Sparse lift + trajectory guidance on fixed layout (no dense task shaping)."""

    # Use dataset row 0 (typical single-layout ``collect_trajectories``) and skip ``cdist`` matching each reset.
    trajectory_guidance_fixed_traj_index: int | None = 0

    rewards: GuidedRewardsCfg = GuidedRewardsCfg()
    observations: GuidedObservationsCfg = GuidedObservationsCfg()
    events: GuidedEEAlignEventCfg = GuidedEEAlignEventCfg()

    def __post_init__(self):
        super().__post_init__()
        _patch_trajectory_file(self, _DEFAULT_FIXED_LAYOUT_TRAJECTORY_FILE)
        # Dense task rewards stay OFF (sparse base).
        self.rewards.lifting_object.weight = 100.0
        self.rewards.lifting_object.params["minimal_height"] = 0.025
        self.rewards.trajectory_guidance.weight = 5.0
        self.rewards.teacher_gripper_alignment.weight = 1.0
        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None


@configclass
class SoArm101FixedLayoutGuidedLiftCubeSparseEnvCfg_PLAY(SoArm101FixedLayoutGuidedLiftCubeSparseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
