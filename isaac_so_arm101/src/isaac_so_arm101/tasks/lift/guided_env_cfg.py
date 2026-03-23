from __future__ import annotations

import os

import isaac_so_arm101.tasks.lift.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .joint_pos_env_cfg import SoArm101LiftCubeEnvCfg, SoArm101LiftCubeSparseEnvCfg
from .lift_env_cfg import EventCfg, ObservationsCfg, RewardsCfg

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
class GuidedObservationsCfg(ObservationsCfg):
    """Adds teacher gripper targets aligned with path progress (requires ``gripper_trajectories`` in dataset)."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        # Single term (shape N×2): path-aligned cmd + close hint; shares one polyline projection / step with rewards.
        teacher_gripper_cmd_and_close_hint = ObsTerm(
            func=mdp.teacher_gripper_cmd_and_close_hint,
            params={
                "trajectory_file": _DEFAULT_TRAJECTORY_FILE,
                "command_name": "object_pose",
                "match_mode": "object_only",
                "exact_match_tol": 5.0e-3,
                "close_threshold": 0.15,
            },
        )

    policy: PolicyCfg = PolicyCfg()


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
class GuidedEEAlignEventCfg(EventCfg):
    """After default + object reset: choose a teacher row (not by EE), IK arm to first EE waypoint."""

    align_ee_to_teacher_trajectory_start = EventTerm(
        func=mdp.align_ee_to_teacher_trajectory_start,
        # After command_manager.reset so goal/command matches TrajectoryStore.match (see ManagerBasedRLEnv).
        mode="post_command_reset",
        params={
            "trajectory_file": _DEFAULT_TRAJECTORY_FILE,
            "command_name": "object_pose",
            "match_mode": "object_only",
            "exact_match_tol": 5.0e-3,
            # If False: pick dataset row by cube (+ goal) match. If True: random row (still not EE-based).
            "sample_traj_index": False,
        },
    )


@configclass
class GuidedDiscriminatorEEAlignEventCfg(DiscriminatorDiagnosticsEventCfg):
    """Discriminator sparse + EE alignment at reset."""

    align_ee_to_teacher_trajectory_start = EventTerm(
        func=mdp.align_ee_to_teacher_trajectory_start,
        mode="post_command_reset",
        params={
            "trajectory_file": _DEFAULT_TRAJECTORY_FILE,
            "command_name": "object_pose",
            "match_mode": "object_only",
            "exact_match_tol": 5.0e-3,
            "sample_traj_index": False,
        },
    )


@configclass
class GuidedRewardsCfg(RewardsCfg):
    """Trajectory guided reward: **path progress** along the teacher EE polyline (default).

    ``guidance_mode="path_progress"`` rewards forward motion along the matched teacher path
    (arc-length progress), which still yields signal when the student is slower than the
    recording. Use ``guidance_mode="time_sync"`` for the legacy per-step teacher EE target.
    """
    trajectory_guidance = RewTerm(
        func=mdp.trajectory_guidance_reward,
        params={
            "trajectory_file": _DEFAULT_TRAJECTORY_FILE,
            "std": 0.10,
            "command_name": "object_pose",
            "match_mode": "object_only",
            "exact_match_tol": 5.0e-3,
            "guidance_mode": "path_progress",
            # Smaller sigma => tanh reacts more to typical Δs per RL step; scale boosts raw term.
            "path_progress_delta_std": 0.02,
            "path_progress_scale": 2.0,
            "lateral_penalty_weight": 0.0,
            "lateral_std": 0.1,
            "only_forward_progress": True,
            "progress_lateral_gate": None,
            "num_path_milestones": 8,
            "max_milestone_jump": 1,
            "milestone_reward_scale": 1.0,
            "milestone_lateral_gate": 0.08,
        },
        weight=12.0,
    )

    teacher_gripper_alignment = RewTerm(
        func=mdp.teacher_gripper_alignment_reward,
        params={
            "trajectory_file": _DEFAULT_TRAJECTORY_FILE,
            "std": 0.15,
            "command_name": "object_pose",
            "match_mode": "object_only",
            "exact_match_tol": 5.0e-3,
        },
        weight=1.0,
    )

    # Debug: time_sync -> norm(student - teacher(t))/std; path_progress -> lateral distance to polyline / lateral_std.
    # Shows up as `Episode_Reward/trajectory_guidance_debug_distance_over_std`.
    trajectory_guidance_debug_distance_over_std = RewTerm(
        func=mdp.trajectory_guidance_debug_distance_over_std,
        params={
            "trajectory_file": _DEFAULT_TRAJECTORY_FILE,
            "std": 0.10,
            "command_name": "object_pose",
            "match_mode": "object_only",
            "exact_match_tol": 5.0e-3,
            "guidance_mode": "path_progress",
            "lateral_std": 0.1,
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
            "guidance_mode": "path_progress",
            "path_progress_delta_std": 0.02,
            "path_progress_scale": 2.0,
            "lateral_penalty_weight": 0.0,
            "lateral_std": 0.1,
            "only_forward_progress": True,
            "progress_lateral_gate": None,
            "num_path_milestones": 8,
            "max_milestone_jump": 1,
            "milestone_reward_scale": 1.0,
            "milestone_lateral_gate": 0.08,
        },
        weight=0.0,
    )

    teacher_gripper_alignment = RewTerm(
        func=mdp.teacher_gripper_alignment_reward,
        params={
            "trajectory_file": _DEFAULT_TRAJECTORY_FILE,
            "std": 0.15,
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
            "guidance_mode": "path_progress",
            "lateral_std": 0.1,
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
    observations: GuidedObservationsCfg = GuidedObservationsCfg()
    events: GuidedEEAlignEventCfg = GuidedEEAlignEventCfg()


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
    observations: GuidedObservationsCfg = GuidedObservationsCfg()
    events: GuidedEEAlignEventCfg = GuidedEEAlignEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # Keep task sparse while adding dense trajectory guidance.
        self.rewards.reaching_object.weight = 0.0
        self.rewards.object_goal_tracking.weight = 0.0
        self.rewards.object_goal_tracking_fine_grained.weight = 0.0
        self.rewards.lifting_object.weight = 1.0
        self.rewards.lifting_object.params["minimal_height"] = 0.025
        # Keep light L2 penalties (``-1e-4``) but disable curriculum ramp to ``-0.1`` — that dominates
        # the path-progress signal and slows motion along the teacher polyline.
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
    observations: GuidedObservationsCfg = GuidedObservationsCfg()
    events: GuidedDiscriminatorEEAlignEventCfg = GuidedDiscriminatorEEAlignEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # Keep task sparse while adding dense discriminator guidance.
        self.rewards.reaching_object.weight = 0.0
        self.rewards.object_goal_tracking.weight = 0.0
        self.rewards.object_goal_tracking_fine_grained.weight = 0.0
        self.rewards.lifting_object.weight = 1.0
        self.rewards.lifting_object.params["minimal_height"] = 0.025
        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None


@configclass
class SoArm101GuidedLiftCubeSparseDiscriminatorEnvCfg_PLAY(SoArm101GuidedLiftCubeSparseDiscriminatorEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
