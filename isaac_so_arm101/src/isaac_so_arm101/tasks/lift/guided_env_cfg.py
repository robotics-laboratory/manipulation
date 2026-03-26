from __future__ import annotations

import os

import isaac_so_arm101.tasks.lift.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
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
    """Same observations as the base task — no teacher hints in the policy vector.

    Trajectory guidance acts purely as reward shaping so the learned policy
    is deployable without any teacher artifacts (clean obs for VLA transfer).
    """

    pass


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
    """After default + object reset: choose a teacher row (not by EE), IK arm to first EE waypoint.

    ``reset_object_from_dataset=True`` snaps the cube to the matched trajectory's initial object
    position so the teacher polyline is geometrically consistent with the actual cube location.
    """

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
            "reset_object_from_dataset": True,
        },
    )

    visualize_teacher_trajectory = EventTerm(
        func=mdp.visualize_teacher_trajectory,
        mode="interval",
        interval_range_s=(_DISCRIMINATOR_METRICS_INTERVAL_S, _DISCRIMINATOR_METRICS_INTERVAL_S),
        is_global_time=True,
        params={
            "trajectory_file": _DEFAULT_TRAJECTORY_FILE,
            "max_envs_to_draw": 1,
            "marker_radius": 0.005,
            "subsample_step": 2,
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
            "reset_object_from_dataset": True,
        },
    )

    visualize_teacher_trajectory = EventTerm(
        func=mdp.visualize_teacher_trajectory,
        mode="interval",
        interval_range_s=(_DISCRIMINATOR_METRICS_INTERVAL_S, _DISCRIMINATOR_METRICS_INTERVAL_S),
        is_global_time=True,
        params={
            "trajectory_file": _DEFAULT_TRAJECTORY_FILE,
            "max_envs_to_draw": 1,
            "marker_radius": 0.005,
            "subsample_step": 2,
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
        weight=5.0,
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
    disable_task_cameras: bool = True


@configclass
class SoArm101GuidedLiftCubeEnvCfg_PLAY(SoArm101GuidedLiftCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class SoArm101GuidedLiftCubeSparseEnvCfg(SoArm101LiftCubeSparseEnvCfg):
    """Sparse lift + trajectory guidance as the only dense exploration signal.

    Task rewards stay sparse (binary ``lifting_object``).  Trajectory guidance
    and gripper alignment are the *only* dense shaping — this is the clean
    VLA-transfer setup where the teacher path replaces hand-crafted dense rewards.
    """

    rewards: GuidedRewardsCfg = GuidedRewardsCfg()
    observations: GuidedObservationsCfg = GuidedObservationsCfg()
    events: GuidedEEAlignEventCfg = GuidedEEAlignEventCfg()
    disable_task_cameras: bool = True

    def __post_init__(self):
        super().__post_init__()
        # Dense task rewards stay OFF (inherited from SoArm101LiftCubeSparseEnvCfg):
        #   reaching_object = 0, object_goal_tracking = 0, fine_grained = 0.
        # Sparse task signal (main success objective — scale vs dense shaping):
        self.rewards.lifting_object.weight = 100.0
        self.rewards.lifting_object.params["minimal_height"] = 0.025
        # Weak teacher dense shaping (no post-lift suppression — lift stays primary via weighting).
        self.rewards.trajectory_guidance.weight = 0.35
        self.rewards.teacher_gripper_alignment.weight = 0.35
        # Disable curriculum ramp — it dominates the path-progress signal.
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
    """Sparse lift + discriminator-guided exploration reward (no dense task shaping)."""

    rewards: GuidedDiscriminatorRewardsCfg = GuidedDiscriminatorRewardsCfg()
    observations: GuidedObservationsCfg = GuidedObservationsCfg()
    events: GuidedDiscriminatorEEAlignEventCfg = GuidedDiscriminatorEEAlignEventCfg()
    disable_task_cameras: bool = True

    def __post_init__(self):
        super().__post_init__()
        # Dense task rewards stay OFF (sparse base).
        self.rewards.lifting_object.weight = 100.0
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
