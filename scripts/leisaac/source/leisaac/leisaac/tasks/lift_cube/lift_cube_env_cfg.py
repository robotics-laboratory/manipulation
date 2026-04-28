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
            pos=(0.0, -0.60114, 0.66027),
            rot=(0.94246, 0.33432, 0.00055, 0.00156),
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
        target_cube_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


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


class PreStepRecordObservationsRecorder(RecorderTerm):
    """Recorder term that captures collection-only observations without changing the MLP policy input."""

    def record_pre_step(self):
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
    success_bonus = RewTerm(
        func=mdp.cube_height_above_base_bonus,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "robot_base_name": "base",
            "height_threshold": 0.20,
            "ramp_width": 0.05,
        },
        weight=20.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class LiftCubeRewardDenseCurriculumCfg:
    """Curriculum schedule for regularization terms."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-4, "num_steps": 30000},
    )
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
    commands: LiftCubeCommandsCfg = LiftCubeCommandsCfg()
    rewards: LiftCubeRewardDenseRewardsCfg = LiftCubeRewardDenseRewardsCfg()
    curriculum: LiftCubeRewardDenseCurriculumCfg = LiftCubeRewardDenseCurriculumCfg()
    actions: SingleArmActionsCfg = SingleArmActionsCfg(
        arm_action=mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=0.35,
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
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation


@configclass
class LiftCubeRewardDenseTrainEnvCfg(LiftCubeRewardDenseEnvCfg):
    """Training variant that avoids resetting immediately at success."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.terminations.success = None


@configclass
class LiftCubeRewardDenseCollectEnvCfg(LiftCubeEnvCfg):
    """Camera-enabled collection variant."""

    scene: LiftCubeSceneCfg = LiftCubeSceneCfg(num_envs=1, env_spacing=8.0)
    observations: LiftCubeMlpCollectObservationsCfg = LiftCubeMlpCollectObservationsCfg()
    commands: LiftCubeCommandsCfg = LiftCubeCommandsCfg()
    recorders: LeRobotRslRlRecorderManagerCfg = LeRobotRslRlRecorderManagerCfg()
    actions: SingleArmActionsCfg = SingleArmActionsCfg(
        arm_action=mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=0.35,
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
        self.recorders = LeRobotRslRlRecorderManagerCfg()
        self.terminations.success = DoneTerm(func=false_success_termination)

        # Match the dense MLP control loop used by the teacher policy.
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.num_rerenders_on_reset = 3

    def build_lerobot_frame(self, episode_data, dataset_cfg) -> dict:
        obs_data = episode_data._data["obs"]
        action = episode_data._data["processed_actions"][-1]
        frame = {
            "action": convert_leisaac_action_to_lerobot(action.unsqueeze(0)).squeeze(0),
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

