import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from leisaac.assets.scenes.kitchen import (
    KITCHEN_WITH_ORANGE_CFG,
    KITCHEN_WITH_ORANGE_USD_PATH,
)
from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_camera_uniform,
    randomize_object_uniform,
)
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ..template import (
    SingleArmActionsCfg,
    SingleArmObservationsCfg,
    SingleArmTaskEnvCfg,
    SingleArmTaskSceneCfg,
    SingleArmTerminationsCfg,
)
from . import mdp


@configclass
class PickOrangeSceneCfg(SingleArmTaskSceneCfg):
    """Scene configuration for the pick orange task."""

    scene: AssetBaseCfg = KITCHEN_WITH_ORANGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")


@configclass
class ObservationsCfg(SingleArmObservationsCfg):

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        pick_orange001 = ObsTerm(func=mdp.orange_grasped, params={"object_cfg": SceneEntityCfg("Orange001")})
        put_orange001_to_plate = ObsTerm(
            func=mdp.put_orange_to_plate,
            params={"object_cfg": SceneEntityCfg("Orange001"), "plate_cfg": SceneEntityCfg("Plate")},
        )
        pick_orange002 = ObsTerm(func=mdp.orange_grasped, params={"object_cfg": SceneEntityCfg("Orange002")})
        put_orange002_to_plate = ObsTerm(
            func=mdp.put_orange_to_plate,
            params={"object_cfg": SceneEntityCfg("Orange002"), "plate_cfg": SceneEntityCfg("Plate")},
        )
        pick_orange003 = ObsTerm(func=mdp.orange_grasped, params={"object_cfg": SceneEntityCfg("Orange003")})
        put_orange003_to_plate = ObsTerm(
            func=mdp.put_orange_to_plate,
            params={"object_cfg": SceneEntityCfg("Orange003"), "plate_cfg": SceneEntityCfg("Plate")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg(SingleArmTerminationsCfg):

    success = DoneTerm(
        func=mdp.task_done,
        params={
            "oranges_cfg": [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")],
            "plate_cfg": SceneEntityCfg("Plate"),
        },
    )


@configclass
class PickOrangeMlpObservationsCfg:
    """MLP-friendly observations for dense reward training."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        active_orange_position = ObsTerm(
            func=mdp.active_unplaced_orange_position_in_robot_root_frame,
            params={
                "oranges_cfg": [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")],
                "plate_cfg": SceneEntityCfg("Plate"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )
        orange_positions = ObsTerm(
            func=mdp.objects_positions_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "objects_cfg": [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")],
            },
        )
        plate_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("Plate")},
        )
        placed_oranges = ObsTerm(
            func=mdp.placed_oranges_flags,
            params={
                "oranges_cfg": [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")],
                "plate_cfg": SceneEntityCfg("Plate"),
            },
        )
        all_oranges_placed = ObsTerm(
            func=mdp.all_oranges_placed,
            params={
                "oranges_cfg": [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")],
                "plate_cfg": SceneEntityCfg("Plate"),
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class PickOrangeRewardDenseRewardsCfg:
    """Dense reward shaping for full three-orange pick-and-place."""

    reaching_active_orange = RewTerm(
        func=mdp.reach_unplaced_oranges_dense,
        params={
            "std": 0.04,
            "oranges_cfg": [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")],
            "plate_cfg": SceneEntityCfg("Plate"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=1.0,
    )
    grasp_active_orange = RewTerm(
        func=mdp.grasp_unplaced_oranges_dense,
        params={
            "std": 0.08,
            "close_joint_threshold": 0.7,
            "oranges_cfg": [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")],
            "plate_cfg": SceneEntityCfg("Plate"),
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "lift_progress_height": 0.05,
            "lift_progress_floor": 1.0,
        },
        weight=2.0,
    )
    lift_active_orange = RewTerm(
        func=mdp.lift_unplaced_oranges_dense,
        params={
            "target_height_delta": 0.05,
            "oranges_cfg": [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")],
            "plate_cfg": SceneEntityCfg("Plate"),
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "grasp_distance": 0.08,
            "close_joint_threshold": 0.7,
        },
        weight=8.0,
    )
    excessive_lift_height = RewTerm(
        func=mdp.active_orange_over_lift_penalty,
        params={
            "oranges_cfg": [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")],
            "plate_cfg": SceneEntityCfg("Plate"),
            "max_height_delta": 0.12,
        },
        weight=-20.0,
    )
    move_active_orange_to_plate = RewTerm(
        func=mdp.move_unplaced_oranges_to_plate_dense,
        params={
            "std": 0.40,
            "lifted_height_delta": 0.04,
            "preplace_height_above_plate": 0.10,
            "oranges_cfg": [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")],
            "plate_cfg": SceneEntityCfg("Plate"),
        },
        weight=8.0,
    )
    lower_active_orange_to_plate = RewTerm(
        func=mdp.lower_unplaced_oranges_to_plate_dense,
        params={
            "std": 0.04,
            "lifted_height_delta": 0.04,
            "target_height_above_plate": 0.05,
            "near_plate_xy": 0.12,
            "oranges_cfg": [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")],
            "plate_cfg": SceneEntityCfg("Plate"),
        },
        weight=6.0,
    )
    place_oranges_on_plate = RewTerm(
        func=mdp.oranges_on_plate_fraction,
        params={
            "oranges_cfg": [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")],
            "plate_cfg": SceneEntityCfg("Plate"),
            "lifted_height_delta": 0.05,
        },
        weight=24.0,
    )
    rest_pose_after_placing = RewTerm(
        func=mdp.rest_pose_after_all_placed,
        params={
            "oranges_cfg": [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")],
            "plate_cfg": SceneEntityCfg("Plate"),
        },
        weight=8.0,
    )
    future_oranges_displacement = RewTerm(
        func=mdp.non_active_orange_displacement_penalty,
        params={
            "oranges_cfg": [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")],
            "plate_cfg": SceneEntityCfg("Plate"),
            "displacement_tolerance": 0.05,
        },
        weight=-12.0,
    )
    # active_orange_pre_lift_displacement = RewTerm(
    #     func=mdp.active_orange_pre_lift_displacement_penalty,
    #     params={
    #         "oranges_cfg": [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")],
    #         "plate_cfg": SceneEntityCfg("Plate"),
    #         "displacement_tolerance": 0.05,
    #         "lifted_height_delta": 0.04,
    #     },
    #     weight=-3.0,
    # )
    # success_bonus = RewTerm(
    #     func=mdp.pick_orange_success_bonus,
    #     params={
    #         "oranges_cfg": [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")],
    #         "plate_cfg": SceneEntityCfg("Plate"),
    #     },
    #     weight=50.0,
    # )
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class PickOrangeRewardDenseCurriculumCfg:
    """Curriculum schedule for regularization terms."""

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-4, "num_steps": 30000},
    )


@configclass
class PickOrangeEnvCfg(SingleArmTaskEnvCfg):
    """Configuration for the pick orange environment."""

    scene: PickOrangeSceneCfg = PickOrangeSceneCfg(env_spacing=8.0)

    observations: ObservationsCfg = ObservationsCfg()

    terminations: TerminationsCfg = TerminationsCfg()

    task_description: str = "Pick three oranges and put them into the plate, then reset the arm to rest state."

    def __post_init__(self) -> None:
        super().__post_init__()

        parse_usd_and_create_subassets(
            KITCHEN_WITH_ORANGE_USD_PATH, self, specific_name_list=["Orange001", "Orange002", "Orange003", "Plate"]
        )

        domain_randomization(
            self,
            random_options=[
                randomize_object_uniform(
                    "Orange001", pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)}
                ),
                randomize_object_uniform(
                    "Orange002", pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)}
                ),
                randomize_object_uniform(
                    "Orange003", pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)}
                ),
                randomize_object_uniform("Plate", pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)}),
                randomize_camera_uniform(
                    "front",
                    pose_range={
                        "x": (-0.025, 0.025),
                        "y": (-0.025, 0.025),
                        "z": (-0.025, 0.025),
                        "roll": (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180),
                        "pitch": (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180),
                        "yaw": (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180),
                    },
                    convention="ros",
                ),
            ],
        )


@configclass
class PickOrangeRewardDenseEnvCfg(PickOrangeEnvCfg):
    """Dense reward, state-only configuration for RSL-RL."""

    scene: PickOrangeSceneCfg = PickOrangeSceneCfg(num_envs=64, env_spacing=8.0)
    observations: PickOrangeMlpObservationsCfg = PickOrangeMlpObservationsCfg()
    rewards: PickOrangeRewardDenseRewardsCfg = PickOrangeRewardDenseRewardsCfg()
    curriculum: PickOrangeRewardDenseCurriculumCfg = PickOrangeRewardDenseCurriculumCfg()
    actions: SingleArmActionsCfg = SingleArmActionsCfg(
        arm_action=mdp.RateLimitedJointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=0.25,
            max_delta=0.025,
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

        # Remove camera randomization events that reference disabled sensors.
        for event_name, event_term in vars(self.events).items():
            if event_name.startswith("_") or event_term is None:
                continue
            asset_cfg = event_term.params.get("asset_cfg")
            if asset_cfg is not None and getattr(asset_cfg, "name", None) in {"front", "wrist"}:
                setattr(self.events, event_name, None)

        # Disable recorder overhead for PPO runs.
        self.recorders = None

        # 50 Hz control loop.
        self.decimation = 2
        self.episode_length_s = 25.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

        # Strip PhysX collisions from kitchen clutter (door/drawer/handle internals on
        # cabinets, fridges, dishwashers, stoves, etc.) that the arm cannot reach. This
        # dramatically reduces broad-phase pair counts and lets us scale ``num_envs`` up.
        #
        # The kitchen USD's defaultPrim is itself named ``Scene``, so when loaded under
        # ``{ENV_REGEX_NS}/Scene`` the live hierarchy is ``/World/envs/env_X/Scene/Scene/...``.
        # We descend through the wrapper with ``scene_attr_name="Scene/Scene"`` so the
        # top-level pass sees the actual fixture groups (``stack_*_main_group_*``) rather
        # than just one opaque ``Scene`` child. All ``stack_*`` groups are kept (they hold
        # the countertop the oranges sit on); we only deep-disable specific named internals
        # under each cabinet that the arm cannot physically interact with.
        from isaaclab.managers import EventTermCfg as _EventTerm
        self.events.disable_kitchen_clutter_collisions = _EventTerm(
            func=mdp.disable_scene_clutter_colliders,
            mode="startup",
            params={
                "scene_attr_name": "Scene/Scene",
                "keep_name_patterns": [
                    r"^Orange.*$",
                    r"^Plate.*$",
                    r"^stack_.*$",
                    r"(?i).*table.*",
                    r"(?i).*counter.*",
                    r"(?i).*floor.*",
                    r"(?i).*ground.*",
                    r"(?i).*wall.*",
                    r"(?i).*robot.*",
                ],
                "deep_disable_path_patterns": [
                    r"stack_.*_main_group_.*/drawer_.*",
                    r"stack_.*_main_group_.*/door_.*",
                    r"stack_.*_main_group_.*/handle_.*",
                    r"stack_.*_main_group_.*/.*_handle",
                ],
            },
        )


@configclass
class PickOrangeRewardDenseTrainEnvCfg(PickOrangeRewardDenseEnvCfg):
    """Training variant that disables terminal success to prevent reward-hacking near completion."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.terminations.success = None
