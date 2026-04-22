# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaac_so_arm101.tasks.lift.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObjectCfg,
    RigidObjectCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Scene definition
##

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FOCAL_LENGTH = 18.15
CAMERA_HORIZONTAL_APERTURE = 20.955


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.

    The exact scene is defined in the derived classes which set the target object, robot and EE frames.
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Cameras: top + wrist (+ optional side). Many SmolVLA finetunes use top + wrist only;
    # use --rename_map if your policy expects side/up instead of wrist for the 2nd view.
    camera_top = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraTop",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.06488, 0.00395, 1.06841),
            rot=(0.700464905, -0.087983463, -0.008909328, -0.708186734),
            convention="opengl",
        ),
        data_types=["rgb"],
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAMERA_FOCAL_LENGTH,
            focus_distance=400.0,
            horizontal_aperture=CAMERA_HORIZONTAL_APERTURE,
            clipping_range=(0.1, 1.0e5),
        ),
    )
    camera_wrist = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_link/CameraWrist",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.00165, 0.10846, -0.02989),
            rot=(0.902356149, -0.419715014, -0.068417350, -0.070083908),
            convention="opengl",
        ),
        data_types=["rgb"],
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAMERA_FOCAL_LENGTH,
            focus_distance=400.0,
            horizontal_aperture=CAMERA_HORIZONTAL_APERTURE,
            clipping_range=(0.1, 1.0e5),
        ),
    )
    camera_side = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraSide",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.52, -0.64128, 0.35135),
            rot=(0.809384006, 0.523186174, 0.192846850, 0.184347092),
            convention="opengl",
        ),
        data_types=["rgb"],
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAMERA_FOCAL_LENGTH,
            focus_distance=400.0,
            horizontal_aperture=CAMERA_HORIZONTAL_APERTURE,
            clipping_range=(0.1, 1.0e5),
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.1, 0.1),
            pos_y=(-0.3, -0.1),
            pos_z=(0.2, 0.35),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ImagesCfg(ObsGroup):
        """Camera images for vision policies. Use with --enable_cameras."""

        images_top = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera_top"), "data_type": "rgb", "normalize": False},
        )
        images_wrist = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera_wrist"), "data_type": "rgb", "normalize": False},
        )
        images_side = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera_side"), "data_type": "rgb", "normalize": False},
        )
        images_up = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera_wrist"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    observation: ImagesCfg = ImagesCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.2, 0.2), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )

    # Optional: enabled from ``collect_lerobot_dataset.py`` when using fall-restore (see docs).
    fall_restore_reset: EventTerm | None = None
    fall_restore_interval: EventTerm | None = None


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.05}, weight=1.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.025}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.025, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.025, "command_name": "object_pose"},
        weight=5.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment configuration
##


def apply_disable_task_cameras_if_set(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Strip TiledCamera sensors and image obs when ``disable_task_cameras`` is True.

    Call this when the flag is set **after** ``LiftEnvCfg.__post_init__`` (e.g. RSL-RL
    ``train.py`` / ``play.py`` CLI overrides). Otherwise cameras remain in the scene
    while ``AppLauncher`` has ``enable_cameras=False``, which triggers a runtime error.
    """
    if not getattr(env_cfg, "disable_task_cameras", False):
        return
    scene = getattr(env_cfg, "scene", None)
    if scene is None:
        return
    for name in ("camera_top", "camera_wrist", "camera_side"):
        if hasattr(scene, name):
            setattr(scene, name, None)
    if hasattr(env_cfg, "image_obs_list"):
        env_cfg.image_obs_list = []
    obs_grp = getattr(getattr(env_cfg, "observations", None), "observation", None)
    if obs_grp is not None:
        for term in ("images_top", "images_wrist", "images_side", "images_up"):
            if hasattr(obs_grp, term):
                setattr(obs_grp, term, None)


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    disable_task_cameras: bool = False
    trajectory_guidance_fixed_traj_index: int | None = None
    suppress_dense_teacher_rewards_after_lift: bool = False
    lift_suppression_min_height: float | None = None
    suppress_dense_teacher_after_lift_scope: str = "episode"

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        # 60 Hz physics × decimation 2 → 30 Hz control (matches LeRobot ``--fps 30`` metadata).
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.image_obs_list = ["camera_top", "camera_wrist", "camera_side"]

        apply_disable_task_cameras_if_set(self)
