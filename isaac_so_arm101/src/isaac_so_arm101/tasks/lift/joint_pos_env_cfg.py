# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path

import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

# from isaaclab.managers NotImplementedError
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaac_so_arm101.robots import SO_ARM100_CFG, SO_ARM101_CFG  # noqa: F401
from isaac_so_arm101.tasks.lift.lift_env_cfg import LiftEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


def _resolve_leisaac_orange_usd_path() -> str:
    rel_path = Path("scenes") / "kitchen_with_orange" / "objects" / "Orange001" / "Orange001.usd"
    candidates: list[Path] = []
    env_root = os.environ.get("LEISAAC_ASSETS_ROOT")
    if env_root:
        candidates.append(Path(env_root) / rel_path)
    candidates.append(Path.home() / "leisaac" / "assets" / rel_path)
    candidates.append(Path("/home/robotics/leisaac/assets") / rel_path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    # Keep a deterministic fallback path for error messages / offline config dumps.
    return str(candidates[0])


@configclass
class SoArm100LiftCubeEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set so arm as robot
        self.scene.robot = SO_ARM100_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            open_command_expr={"gripper": 0.5},
            close_command_expr={"gripper": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = ["gripper"]

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.0, 0.015], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.5, 0.5, 0.5),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, -0.09, 0.01],
                    ),
                ),
            ],
        )


@configclass
class SoArm100LiftCubeEnvCfg_PLAY(SoArm100LiftCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class SoArm100TargetCubeEnvCfg(SoArm100LiftCubeEnvCfg):
    """Variant of lift where target-point tracking is prioritized over pure lifting."""

    def __post_init__(self):
        super().__post_init__()
        # Keep a tiny lift incentive to pick the cube, but prioritize target tracking.
        self.rewards.lifting_object.weight = 2.0
        self.rewards.object_goal_tracking.weight = 28.0
        self.rewards.object_goal_tracking_fine_grained.weight = 14.0
        # Lower gate so goal-tracking reward activates earlier once grasped.
        self.rewards.object_goal_tracking.params["minimal_height"] = 0.01
        self.rewards.object_goal_tracking_fine_grained.params["minimal_height"] = 0.01


@configclass
class SoArm100TargetCubeEnvCfg_PLAY(SoArm100TargetCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class SoArm101LiftCubeEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set so arm as robot
        self.scene.robot = SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            open_command_expr={"gripper": 0.5},
            close_command_expr={"gripper": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = ["gripper_link"]

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.0, 0.015], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.5, 0.5, 0.5),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.01, 0.0, -0.09],
                    ),
                ),
            ],
        )


@configclass
class SoArm101LiftCubeEnvCfg_PLAY(SoArm101LiftCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class SoArm101LiftOrangeEnvCfg(SoArm101LiftCubeEnvCfg):
    """SO-101 lift task variant with an orange sphere object."""

    def __post_init__(self):
        super().__post_init__()
        orange_usd_path = _resolve_leisaac_orange_usd_path()
        if not Path(orange_usd_path).exists():
            raise FileNotFoundError(
                "LeIsaac orange asset not found. Set LEISAAC_ASSETS_ROOT to your leisaac assets directory "
                f"(expected: {orange_usd_path})."
            )
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.0, 0.025], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=orange_usd_path,
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
        )


@configclass
class SoArm101LiftOrangeEnvCfg_PLAY(SoArm101LiftOrangeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class SoArm101LiftCubeSparseEnvCfg(SoArm101LiftCubeEnvCfg):
    """Sparse-reward lift setup for VLA-like conditions.

    Keeps only binary lifting success and tiny action regularization.
    """

    def __post_init__(self):
        super().__post_init__()
        # Disable dense shaping terms.
        self.rewards.reaching_object.weight = 0.0
        self.rewards.object_goal_tracking.weight = 0.0
        self.rewards.object_goal_tracking_fine_grained.weight = 0.0
        # Sparse success signal.
        self.rewards.lifting_object.weight = 1.0
        self.rewards.lifting_object.params["minimal_height"] = 0.025
        # Keep light regularization.
        self.rewards.action_rate.weight = -1e-4
        self.rewards.joint_vel.weight = -1e-4


@configclass
class SoArm101LiftCubeSparseEnvCfg_PLAY(SoArm101LiftCubeSparseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class SoArm101TargetCubeEnvCfg(SoArm101LiftCubeEnvCfg):
    """Variant of lift where target-point tracking is prioritized over pure lifting."""

    def __post_init__(self):
        super().__post_init__()
        # Keep a tiny lift incentive to pick the cube, but prioritize target tracking.
        self.rewards.lifting_object.weight = 2.0
        self.rewards.object_goal_tracking.weight = 28.0
        self.rewards.object_goal_tracking_fine_grained.weight = 14.0
        # Lower gate so goal-tracking reward activates earlier once grasped.
        self.rewards.object_goal_tracking.params["minimal_height"] = 0.01
        self.rewards.object_goal_tracking_fine_grained.params["minimal_height"] = 0.01


@configclass
class SoArm101TargetCubeEnvCfg_PLAY(SoArm101TargetCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
