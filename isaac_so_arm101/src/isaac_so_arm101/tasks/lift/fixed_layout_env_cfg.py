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

import isaac_so_arm101.tasks.lift.mdp as lift_mdp
from isaaclab.utils import configclass

from .guided_env_cfg import GuidedEEAlignEventCfg, GuidedObservationsCfg, GuidedRewardsCfg
from .joint_pos_env_cfg import SoArm101LiftCubeEnvCfg, SoArm101LiftCubeSparseEnvCfg


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


@configclass
class SoArm101FixedLayoutGuidedLiftCubeSparseEnvCfg(SoArm101FixedLayoutLiftCubeSparseEnvCfg):
    """Guided sparse lift on fixed layout — match teacher trajectories collected on the same task."""

    # Use dataset row 0 (typical single-layout ``collect_trajectories``) and skip ``cdist`` matching each reset.
    trajectory_guidance_fixed_traj_index: int | None = 0

    rewards: GuidedRewardsCfg = GuidedRewardsCfg()
    observations: GuidedObservationsCfg = GuidedObservationsCfg()
    events: GuidedEEAlignEventCfg = GuidedEEAlignEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.reaching_object.weight = 0.0
        self.rewards.object_goal_tracking.weight = 0.0
        self.rewards.object_goal_tracking_fine_grained.weight = 0.0
        self.rewards.lifting_object.weight = 1.0
        self.rewards.lifting_object.params["minimal_height"] = 0.025
        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None


@configclass
class SoArm101FixedLayoutGuidedLiftCubeSparseEnvCfg_PLAY(SoArm101FixedLayoutGuidedLiftCubeSparseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
