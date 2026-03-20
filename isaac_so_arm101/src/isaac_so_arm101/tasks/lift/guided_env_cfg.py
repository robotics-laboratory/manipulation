from __future__ import annotations

import os

import isaac_so_arm101.tasks.lift.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .joint_pos_env_cfg import SoArm101LiftCubeEnvCfg, SoArm101LiftCubeSparseEnvCfg
from .lift_env_cfg import EventCfg, RewardsCfg

_DEFAULT_TRAJECTORY_FILE = os.environ.get(
    "ISAAC_SO_ARM101_TRAJECTORY_FILE",
    os.path.join(
        "isaac_so_arm101", "logs", "rsl_rl", "teacher_trajectories", "so101_lift_cube_teacher.pt"
    ),
)

_DEFAULT_DISCRIMINATOR_FILE = os.environ.get(
    "ISAAC_SO_ARM101_TRAJECTORY_DISCRIMINATOR_FILE",
    os.path.join(
        "isaac_so_arm101", "logs", "rsl_rl", "teacher_trajectories", "trajectory_discriminator_lift_cube.pt"
    ),
)

# Match LiftEnvCfg: decimation * sim.dt (default 2 * 0.01 s).
_DISCRIMINATOR_METRICS_INTERVAL_S = 0.02


@configclass
class DiscriminatorDiagnosticsEventCfg(EventCfg):
    """Adds a per-step flush of discriminator diagnostics into ``extras['log']``."""

    log_discriminator_metrics = EventTerm(
        func=mdp.flush_discriminator_metrics_to_log,
        mode="interval",
        interval_range_s=(_DISCRIMINATOR_METRICS_INTERVAL_S, _DISCRIMINATOR_METRICS_INTERVAL_S),
        is_global_time=True,
    )


@configclass
class GuidedRewardsCfg(RewardsCfg):
    """Trajectory-distance guided reward (teacher EE matching).

    Intended for ablations where the student reward is shaped by matching the
    student's EE trajectory to a stored teacher EE trajectory.
    """
    trajectory_guidance = RewTerm(
        func=mdp.trajectory_guidance_reward,
        params={
            "trajectory_file": _DEFAULT_TRAJECTORY_FILE,
            "std": 0.10,
            "command_name": "object_pose",
            "match_mode": "object_only",
            "exact_match_tol": 5.0e-3,
        },
        weight=5.0,
    )

    # Debug-only term: logs distance between student EE and matched teacher EE.
    # Shows up as `Episode_Reward/trajectory_guidance_debug_distance_over_std`.
    # Weight is tiny so it should not affect learning meaningfully.
    trajectory_guidance_debug_distance_over_std = RewTerm(
        func=mdp.trajectory_guidance_debug_distance_over_std,
        params={
            "trajectory_file": _DEFAULT_TRAJECTORY_FILE,
            "std": 0.10,
            "command_name": "object_pose",
            "match_mode": "object_only",
            "exact_match_tol": 5.0e-3,
        },
        weight=1.0e-3,
    )

    discriminator_guidance = RewTerm(
        func=mdp.discriminator_guidance_reward,
        params={
            "discriminator_file": _DEFAULT_DISCRIMINATOR_FILE,
            "command_name": "object_pose",
            "g_mode": "object_only",
            "eps": 1e-6,
        },
        # Disabled in the trajectory-guided config (weight=0 avoids loading the file).
        weight=0.0,
    )


@configclass
class GuidedDiscriminatorRewardsCfg(RewardsCfg):
    """Discriminator-guided reward (learned log D over EE transitions)."""

    trajectory_guidance = RewTerm(
        func=mdp.trajectory_guidance_reward,
        params={
            "trajectory_file": _DEFAULT_TRAJECTORY_FILE,
            "std": 0.10,
            "command_name": "object_pose",
            "match_mode": "object_only",
            "exact_match_tol": 5.0e-3,
        },
        weight=0.0,
    )

    trajectory_guidance_debug_distance_over_std = RewTerm(
        func=mdp.trajectory_guidance_debug_distance_over_std,
        params={
            "trajectory_file": _DEFAULT_TRAJECTORY_FILE,
            "std": 0.10,
            "command_name": "object_pose",
            "match_mode": "object_only",
            "exact_match_tol": 5.0e-3,
        },
        weight=0.0,
    )

    discriminator_guidance = RewTerm(
        func=mdp.discriminator_guidance_reward,
        params={
            "discriminator_file": _DEFAULT_DISCRIMINATOR_FILE,
            "command_name": "object_pose",
            "g_mode": "object_only",
            "eps": 1e-6,
            # Saturated D: raw log D + batch center + symmetric clamp (see reward docstring).
            "logit_temperature": 1.5,
            "center_per_env_batch": True,
            "center_min_std": 1.0e-4,
            "log_d_min": -1.5,
            "log_d_max": 1.5,
            "log_metrics": True,
        },
        weight=0.002,
    )


@configclass
class SoArm101GuidedLiftCubeEnvCfg(SoArm101LiftCubeEnvCfg):
    """Lift-cube SO-101 task with an additional teacher-trajectory guidance reward."""

    rewards: GuidedRewardsCfg = GuidedRewardsCfg()


@configclass
class SoArm101GuidedLiftCubeEnvCfg_PLAY(SoArm101GuidedLiftCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class SoArm101GuidedLiftCubeSparseEnvCfg(SoArm101LiftCubeSparseEnvCfg):
    """Sparse Lift-cube with additional teacher-trajectory guidance reward."""

    rewards: GuidedRewardsCfg = GuidedRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Keep task sparse while adding dense trajectory guidance.
        self.rewards.reaching_object.weight = 0.0
        self.rewards.object_goal_tracking.weight = 0.0
        self.rewards.object_goal_tracking_fine_grained.weight = 0.0
        self.rewards.lifting_object.weight = 1.0
        self.rewards.lifting_object.params["minimal_height"] = 0.025
        # Remove regularization penalties so dense signal comes from trajectory guidance.
        self.rewards.action_rate = None
        self.rewards.joint_vel = None
        # Disable curricula that would otherwise re-introduce stronger negative penalties.
        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None


@configclass
class SoArm101GuidedLiftCubeSparseEnvCfg_PLAY(SoArm101GuidedLiftCubeSparseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class SoArm101GuidedLiftCubeSparseDiscriminatorEnvCfg(SoArm101LiftCubeSparseEnvCfg):
    """Sparse Lift-cube with discriminator-guided exploration reward."""

    rewards: GuidedDiscriminatorRewardsCfg = GuidedDiscriminatorRewardsCfg()
    events: DiscriminatorDiagnosticsEventCfg = DiscriminatorDiagnosticsEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # Keep task sparse while adding dense discriminator guidance.
        self.rewards.reaching_object.weight = 0.0
        self.rewards.object_goal_tracking.weight = 0.0
        self.rewards.object_goal_tracking_fine_grained.weight = 0.0
        self.rewards.lifting_object.weight = 1.0
        self.rewards.lifting_object.params["minimal_height"] = 0.025

        # Remove regularization penalties so learning signal is mainly from
        # task success + discriminator guidance.
        self.rewards.action_rate = None
        self.rewards.joint_vel = None
        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None


@configclass
class SoArm101GuidedLiftCubeSparseDiscriminatorEnvCfg_PLAY(SoArm101GuidedLiftCubeSparseDiscriminatorEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
