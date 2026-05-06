import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import (
    ActionStateRecorderManagerCfg,
    InitialStateRecorderCfg,
    PostStepProcessedActionsRecorderCfg,
    PostStepStatesRecorderCfg,
    PreStepActionsRecorderCfg,
)
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from leisaac.assets.scenes.simple import TABLE_WITH_CUBE_CFG, TABLE_WITH_CUBE_USD_PATH
from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_camera_uniform,
    randomize_object_uniform,
)
from leisaac.utils.general_assets import parse_usd_and_create_subassets
from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot

from ..template import (
    SingleArmActionsCfg,
    SingleArmObservationsCfg,
    SingleArmTaskEnvCfg,
    SingleArmTaskSceneCfg,
    SingleArmTerminationsCfg,
)
from . import mdp


@configclass
class LiftCubeSceneCfg(SingleArmTaskSceneCfg):
    """Scene configuration for the lift cube task.

    Keep both template SO-101 cameras enabled so teleop/LeRobot recording exports
    observation.images.front and observation.images.wrist.
    """

    scene: AssetBaseCfg = TABLE_WITH_CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")
    front: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.0, -0.71935, 0.69892),
            rot=(0.92788, 0.37234, -0.00185, -0.00461),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )
    wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.02174),
            rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )


@configclass
class LiftCubeVisionSceneCfg(LiftCubeSceneCfg):
    """Lower-resolution camera scene for visual-feature PPO training."""

    front: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.0, -0.71935, 0.69892),
            rot=(0.92788, 0.37234, -0.00185, -0.00461),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.7,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=224,
        height=224,
        update_period=1 / 30.0,
    )
    wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.02174),
            rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=224,
        height=224,
        update_period=1 / 30.0,
    )


@configclass
class ObservationsCfg(SingleArmObservationsCfg):

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        pick_cube = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg(SingleArmTerminationsCfg):

    success = DoneTerm(
        func=mdp.cube_height_above_base,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "robot_base_name": "base",
            "height_threshold": 0.20,
        },
    )


@configclass
class LiftCubeMlpObservationsCfg:
    """State-only observations for dense reward PPO."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cube_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("cube")},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class LiftCubeVisionObservationsCfg:
    """Vision-based PPO actor observations with privileged critic state."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        front_features = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("front"),
                "data_type": "rgb",
                "model_name": "resnet18",
            },
        )
        wrist_features = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("wrist"),
                "data_type": "rgb",
                "model_name": "resnet18",
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cube_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("cube")},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class LiftCubeMlpCollectObservationsCfg(LiftCubeMlpObservationsCfg):
    """MLP policy observations plus recorder-only terms for LeRobot export."""

    @configclass
    class RecordCfg(ObsGroup):
        joint_pos_abs = ObsTerm(func=mdp.joint_pos)
        front = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("front"), "data_type": "rgb", "normalize": False}
        )
        wrist = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist"), "data_type": "rgb", "normalize": False}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    record: RecordCfg = RecordCfg()


@configclass
class LiftCubeVisionCollectObservationsCfg(LiftCubeVisionObservationsCfg):
    """Vision PPO observations plus recorder-only raw frames for LeRobot export."""

    @configclass
    class RecordCfg(ObsGroup):
        joint_pos_abs = ObsTerm(func=mdp.joint_pos)
        front = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("front"), "data_type": "rgb", "normalize": False}
        )
        wrist = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist"), "data_type": "rgb", "normalize": False}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    record: RecordCfg = RecordCfg()


class PreStepRecordObservationsRecorder(RecorderTerm):
    """Recorder term that captures collection-only observations without changing the MLP policy input."""

    def record_pre_step(self):
        if "record" in self._env.obs_buf:
            return "obs", self._env.obs_buf["record"]
        return "obs", self._env.observation_manager.compute_group("record")


@configclass
class PreStepRecordObservationsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepRecordObservationsRecorder


@configclass
class LeRobotRslRlRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Recorder config for RSL-RL rollouts exported as LeRobot episodes."""

    record_initial_state = InitialStateRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()
    record_pre_step_actions = PreStepActionsRecorderCfg()
    record_pre_step_flat_policy_observations = None
    record_pre_step_record_observations = PreStepRecordObservationsRecorderCfg()
    record_post_step_processed_actions = PostStepProcessedActionsRecorderCfg()


def false_success_termination(env) -> torch.Tensor:
    """Keep the success term present without allowing automatic success resets."""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def reset_so101_awkward_joints(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Sample conservative SO-101 arm starts that resemble common failed VLA approach poses."""
    robot = env.scene[asset_cfg.name]
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    offset_ranges_deg = {
        "shoulder_pan": (-12.0, 12.0),
        "shoulder_lift": (-10.0, 18.0),
        "elbow_flex": (-25.0, 15.0),
        "wrist_flex": (-25.0, 25.0),
        "wrist_roll": (-20.0, 20.0),
    }
    for joint_name, (low_deg, high_deg) in offset_ranges_deg.items():
        joint_idx = robot.data.joint_names.index(joint_name)
        low = low_deg * torch.pi / 180.0
        high = high_deg * torch.pi / 180.0
        joint_pos[:, joint_idx] += torch.empty((len(env_ids),), device=env.device).uniform_(low, high)

    # Keep the gripper open so the teacher still has to solve approach, grasp, and lift.
    gripper_idx = robot.data.joint_names.index("gripper")
    joint_pos[:, gripper_idx] = 1.0

    if robot.data.soft_joint_pos_limits is not None:
        joint_limits = robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clamp(joint_pos, joint_limits[..., 0], joint_limits[..., 1])

    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def enable_so101_awkward_resets(env_cfg) -> None:
    """Attach the awkward-reset event to a LiftCube config."""
    env_cfg.events.awkward_robot_joints = EventTerm(
        func=reset_so101_awkward_joints,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class LiftCubeCommandsCfg:
    """Command terms used by the dense MLP teacher observation."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="gripper",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.55),
            pos_y=(-0.20, 0.20),
            pos_z=(0.20, 0.40),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class LiftCubeRewardDenseRewardsCfg:
    """Dense reward shaping for lift-cube PPO."""

    reach_cube = RewTerm(
        func=mdp.reach_object_dense,
        params={"std": 0.08, "object_cfg": SceneEntityCfg("cube"), "ee_frame_cfg": SceneEntityCfg("ee_frame")},
        weight=2.0,
    )
    grasp_cube = RewTerm(
        func=mdp.grasp_closure_dense,
        params={
            "std": 0.08,
            "close_joint_threshold": 0.7,
            "object_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=4.0,
    )
    gripper_action_rate = RewTerm(func=mdp.gripper_action_rate_l2, weight=-5.0e-3)
    lift_cube = RewTerm(
        func=mdp.lift_progress_dense,
        params={
            "target_height_delta": 0.20,
            "object_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "robot_base_name": "base",
        },
        weight=8.0,
    )
    # goal_tracking = RewTerm(
    #     func=mdp.goal_tracking_dense,
    #     params={
    #         "std": 0.30,
    #         "command_name": "object_pose",
    #         "lifted_height_delta": 0.025,
    #         "robot_cfg": SceneEntityCfg("robot"),
    #         "object_cfg": SceneEntityCfg("cube"),
    #     },
    #     weight=0.0,
    # )
    # goal_tracking_fine = RewTerm(
    #     func=mdp.goal_tracking_dense,
    #     params={
    #         "std": 0.05,
    #         "command_name": "object_pose",
    #         "lifted_height_delta": 0.025,
    #         "robot_cfg": SceneEntityCfg("robot"),
    #         "object_cfg": SceneEntityCfg("cube"),
    #     },
    #     weight=0.0,
    # )
    # goal_success_bonus = RewTerm(
    #     func=mdp.goal_success_bonus,
    #     params={
    #         "command_name": "object_pose",
    #         "position_tolerance": 0.05,
    #         "lifted_height_delta": 0.025,
    #         "robot_cfg": SceneEntityCfg("robot"),
    #         "object_cfg": SceneEntityCfg("cube"),
    #     },
    #     weight=0.0,
    # )
    lifted_stillness = RewTerm(
        func=mdp.lifted_stillness_dense,
        params={
            "lifted_height_delta": 0.15,
            "velocity_std": 0.08,
            "object_cfg": SceneEntityCfg("cube"),
        },
        weight=4.0,
    )
    lifted_angular_stillness = RewTerm(
        func=mdp.lifted_angular_stillness_dense,
        params={
            "lifted_height_delta": 0.025,
            "angular_velocity_std": 1.0,
            "object_cfg": SceneEntityCfg("cube"),
        },
        weight=4.0,
    )
    wrist_flip = RewTerm(
        func=mdp.wrist_flip_penalty,
        params={
            "max_abs_wrist_flex": 1.05,
            "std": 0.25,
            "lifted_height_delta": 0.025,
            "object_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "wrist_joint_name": "wrist_flex",
        },
        weight=-4.0,
    )
    human_lift_posture = RewTerm(
        func=mdp.human_lift_posture_dense,
        params={
            "target_motor_positions": {
                "elbow_flex": -25.0,
                "wrist_flex": 88.0,
                "wrist_roll": 3.0,
            },
            "std": 0.75,
            "lifted_height_delta": 0.025,
            "object_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
        weight=1.0,
    )
    xy_position_stability = RewTerm(
        func=mdp.xy_position_stability_dense,
        params={
            "std": 0.08,
            "lifted_height_delta": 0.025,
            "object_cfg": SceneEntityCfg("cube"),
        },
        weight=1.5,
    )
    success_bonus = RewTerm(
        func=mdp.cube_height_above_base_bonus,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "robot_base_name": "base",
            "height_threshold": 0.20,
            "ramp_width": 0.05,
        },
        weight=10.0,
    )
    excessive_lift = RewTerm(
        func=mdp.excessive_lift_penalty,
        params={
            "max_height": 0.25,
            "std": 0.05,
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "robot_base_name": "base",
        },
        weight=-2.0,
    )
    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class LiftCubeRewardDenseCurriculumCfg:
    """Curriculum schedule for regularization terms."""

    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "action_rate", "weight": -1e-4, "num_steps": 30000},
    # )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-4, "num_steps": 30000},
    )


@configclass
class LiftCubeEnvCfg(SingleArmTaskEnvCfg):
    """Default teleop/VLA-oriented lift-cube configuration."""

    scene: LiftCubeSceneCfg = LiftCubeSceneCfg(env_spacing=8.0)
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    task_description: str = "Lift the red cube up."

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (-0.4, -0.6, 0.5)
        self.viewer.lookat = (0.9, 0.0, -0.3)
        self.scene.robot.init_state.pos = (0.35, -0.64, 0.01)

        parse_usd_and_create_subassets(TABLE_WITH_CUBE_USD_PATH, self)

        domain_randomization(
            self,
            random_options=[
                randomize_object_uniform(
                    "cube",
                    pose_range={
                        "x": (-0.075, 0.075),
                        "y": (-0.075, 0.075),
                        "z": (0.0, 0.0),
                        "yaw": (-30 * torch.pi / 180, 30 * torch.pi / 180),
                    },
                ),
                randomize_camera_uniform(
                    "front",
                    pose_range={
                        "x": (-0.005, 0.005),
                        "y": (-0.005, 0.005),
                        "z": (-0.005, 0.005),
                        "roll": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                        "pitch": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                        "yaw": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                    },
                    convention="opengl",
                ),
            ],
        )


@configclass
class LiftCubeRewardDenseEnvCfg(LiftCubeEnvCfg):
    """Fast state-only dense reward configuration for MLP PPO."""

    scene: LiftCubeSceneCfg = LiftCubeSceneCfg(num_envs=64, env_spacing=8.0)
    observations: LiftCubeMlpObservationsCfg = LiftCubeMlpObservationsCfg()
    rewards: LiftCubeRewardDenseRewardsCfg = LiftCubeRewardDenseRewardsCfg()
    curriculum: LiftCubeRewardDenseCurriculumCfg = LiftCubeRewardDenseCurriculumCfg()
    actions: SingleArmActionsCfg = SingleArmActionsCfg(
        arm_action=mdp.RateLimitedJointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=0.25,
            max_delta=0.020,
        ),
        gripper_action=mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            open_command_expr={"gripper": 1.0},
            close_command_expr={"gripper": 0.2},
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        # Disable vision sensors for fast state-only PPO training/evaluation.
        self.scene.wrist = None
        self.scene.front = None

        for event_name, event_term in vars(self.events).items():
            if event_name.startswith("_") or event_term is None:
                continue
            asset_cfg = event_term.params.get("asset_cfg")
            if asset_cfg is not None and getattr(asset_cfg, "name", None) in {"front", "wrist"}:
                setattr(self.events, event_name, None)

        self.recorders = None
        self.decimation = 1
        self.episode_length_s = 5.0


@configclass
class LiftCubeRewardDenseTrainEnvCfg(LiftCubeRewardDenseEnvCfg):
    """Training variant that avoids resetting immediately at success."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.terminations.success = None


@configclass
class LiftCubeRewardDenseAwkwardResetEnvCfg(LiftCubeRewardDenseEnvCfg):
    """Dense reward configuration with conservative randomized arm reset poses."""

    def __post_init__(self) -> None:
        super().__post_init__()
        enable_so101_awkward_resets(self)


@configclass
class LiftCubeRewardDenseAwkwardResetTrainEnvCfg(LiftCubeRewardDenseAwkwardResetEnvCfg):
    """Awkward-reset training variant that avoids resetting immediately at success."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.terminations.success = None


@configclass
class LiftCubeRewardDenseVisionEnvCfg(LiftCubeEnvCfg):
    """Vision-feature dense reward configuration for PPO."""

    scene: LiftCubeVisionSceneCfg = LiftCubeVisionSceneCfg(num_envs=8, env_spacing=8.0)
    observations: LiftCubeVisionObservationsCfg = LiftCubeVisionObservationsCfg()
    rewards: LiftCubeRewardDenseRewardsCfg = LiftCubeRewardDenseRewardsCfg()
    curriculum: LiftCubeRewardDenseCurriculumCfg = LiftCubeRewardDenseCurriculumCfg()
    actions: SingleArmActionsCfg = SingleArmActionsCfg(
        arm_action=mdp.RateLimitedJointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=0.25,
            max_delta=0.005,
        ),
        gripper_action=mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            open_command_expr={"gripper": 1.0},
            close_command_expr={"gripper": 0.4},
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        self.recorders = None
        self.decimation = 1
        self.episode_length_s = 5.0


@configclass
class LiftCubeRewardDenseVisionTrainEnvCfg(LiftCubeRewardDenseVisionEnvCfg):
    """Vision-feature training variant that avoids resetting immediately at success."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.terminations.success = None


@configclass
class LiftCubeRewardDenseCollectEnvCfg(LiftCubeEnvCfg):
    """Camera-enabled collection variant."""

    scene: LiftCubeSceneCfg = LiftCubeSceneCfg(num_envs=1, env_spacing=8.0)
    observations: LiftCubeMlpCollectObservationsCfg = LiftCubeMlpCollectObservationsCfg()
    recorders: LeRobotRslRlRecorderManagerCfg = LeRobotRslRlRecorderManagerCfg()
    actions: SingleArmActionsCfg = SingleArmActionsCfg(
        arm_action=mdp.RateLimitedJointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=0.25,
            max_delta=0.020,
        ),
        gripper_action=mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            open_command_expr={"gripper": 1.0},
            close_command_expr={"gripper": 0.2},
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.recorders = LeRobotRslRlRecorderManagerCfg()
        self.terminations.time_out = None
        self.terminations.success = DoneTerm(func=false_success_termination)

        # Match the teleop loop pace as closely as possible during dataset recording.
        self.decimation = 1
        self.episode_length_s = 5.0
        self.num_rerenders_on_reset = 3

    def build_lerobot_frame(self, episode_data, dataset_cfg) -> dict:
        obs_data = episode_data._data["obs"]
        action = episode_data._data["processed_actions"][-1]
        if dataset_cfg.action_align:
            processed_action = convert_leisaac_action_to_lerobot(action.unsqueeze(0)).squeeze(0)
        else:
            processed_action = action.cpu().numpy()
        frame = {
            "action": processed_action,
            "observation.state": convert_leisaac_action_to_lerobot(
                obs_data["joint_pos_abs"][-1].unsqueeze(0)
            ).squeeze(0),
            "task": self.task_description,
        }
        for frame_key in dataset_cfg.features.keys():
            if not frame_key.startswith("observation.images"):
                continue
            camera_key = frame_key.split(".")[-1]
            frame[frame_key] = obs_data[camera_key][-1].cpu().numpy()

        return frame


@configclass
class LiftCubeRewardDenseAwkwardResetCollectEnvCfg(LiftCubeRewardDenseCollectEnvCfg):
    """Camera-enabled collection variant with conservative randomized arm reset poses."""

    def __post_init__(self) -> None:
        super().__post_init__()
        enable_so101_awkward_resets(self)


@configclass
class LiftCubeRewardDenseVisionCollectEnvCfg(LiftCubeRewardDenseVisionEnvCfg):
    """Vision-feature collection variant that exports LeRobot episodes."""

    scene: LiftCubeVisionSceneCfg = LiftCubeVisionSceneCfg(num_envs=1, env_spacing=8.0)
    observations: LiftCubeVisionCollectObservationsCfg = LiftCubeVisionCollectObservationsCfg()
    recorders: LeRobotRslRlRecorderManagerCfg = LeRobotRslRlRecorderManagerCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.recorders = LeRobotRslRlRecorderManagerCfg()
        self.terminations.time_out = None
        self.terminations.success = DoneTerm(func=false_success_termination)
        self.num_rerenders_on_reset = 3

    def build_lerobot_frame(self, episode_data, dataset_cfg) -> dict:
        obs_data = episode_data._data["obs"]
        action = episode_data._data["processed_actions"][-1]
        if dataset_cfg.action_align:
            processed_action = convert_leisaac_action_to_lerobot(action.unsqueeze(0)).squeeze(0)
        else:
            processed_action = action.cpu().numpy()
        frame = {
            "action": processed_action,
            "observation.state": convert_leisaac_action_to_lerobot(
                obs_data["joint_pos_abs"][-1].unsqueeze(0)
            ).squeeze(0),
            "task": self.task_description,
        }
        for frame_key in dataset_cfg.features.keys():
            if not frame_key.startswith("observation.images"):
                continue
            camera_key = frame_key.split(".")[-1]
            frame[frame_key] = obs_data[camera_key][-1].cpu().numpy()

        return frame
