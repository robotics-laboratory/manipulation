import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from leisaac.assets.scenes.kitchen import KITCHEN_WITH_ORANGE_USD_PATH
from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_camera_uniform,
    randomize_object_uniform,
)
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ...template import SingleArmTaskDirectEnv, SingleArmTaskDirectEnvCfg
from .. import mdp
from ..pick_orange_env_cfg import PickOrangeSceneCfg


ORANGES_CFG = [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")]
PLATE_CFG = SceneEntityCfg("Plate")
ROBOT_CFG = SceneEntityCfg("robot")
EE_FRAME_CFG = SceneEntityCfg("ee_frame")


@configclass
class PickOrangeEnvCfg(SingleArmTaskDirectEnvCfg):
    """Direct env configuration for the pick orange task."""

    scene: PickOrangeSceneCfg = PickOrangeSceneCfg(env_spacing=8.0)

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
class PickOrangeEurekaEnvCfg(PickOrangeEnvCfg):
    """State-only direct PickOrange configuration for IsaacLabEureka reward search."""

    scene: PickOrangeSceneCfg = PickOrangeSceneCfg(num_envs=64, env_spacing=8.0)

    def __post_init__(self) -> None:
        super().__post_init__()

        # Eureka/RSL-RL reward search should use low-dimensional state, not image tensors.
        self.scene.wrist = None
        self.scene.front = None
        self.cameras = []
        self.observation_space.pop("wrist", None)
        self.observation_space.pop("front", None)
        self.state_space.pop("wrist", None)
        self.state_space.pop("front", None)

        for event_name, event_term in vars(self.events).items():
            if event_name.startswith("_") or event_term is None:
                continue
            asset_cfg = event_term.params.get("asset_cfg")
            if asset_cfg is not None and getattr(asset_cfg, "name", None) in {"front", "wrist"}:
                setattr(self.events, event_name, None)

        self.recorders = None
        self.decimation = 2
        self.episode_length_s = 25.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

        self.events.disable_kitchen_clutter_collisions = EventTerm(
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


class PickOrangeEnv(SingleArmTaskDirectEnv):
    """Direct env for the pick orange task."""

    cfg: PickOrangeEnvCfg

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        # add subtask observation
        obs["subtask_terms"] = {
            "pick_orange001": mdp.orange_grasped(self, object_cfg=SceneEntityCfg("Orange001")),
            "put_orange001_to_plate": mdp.put_orange_to_plate(
                self, object_cfg=SceneEntityCfg("Orange001"), plate_cfg=PLATE_CFG
            ),
            "pick_orange002": mdp.orange_grasped(self, object_cfg=SceneEntityCfg("Orange002")),
            "put_orange002_to_plate": mdp.put_orange_to_plate(
                self, object_cfg=SceneEntityCfg("Orange002"), plate_cfg=PLATE_CFG
            ),
            "pick_orange003": mdp.orange_grasped(self, object_cfg=SceneEntityCfg("Orange003")),
            "put_orange003_to_plate": mdp.put_orange_to_plate(
                self, object_cfg=SceneEntityCfg("Orange003"), plate_cfg=PLATE_CFG
            ),
        }
        return obs

    def _get_rewards(self) -> torch.Tensor:
        """Oracle dense reward used by Eureka for reward-correlation feedback."""
        reward_terms = self._get_oracle_reward_terms()
        self._pick_orange_oracle_reward_terms = reward_terms
        return sum(reward_terms.values())

    def _get_oracle_reward_terms(self) -> dict[str, torch.Tensor]:
        return {
            "reaching_active_orange": 1.0
            * mdp.reach_unplaced_oranges_dense(
                self,
                std=0.04,
                oranges_cfg=ORANGES_CFG,
                plate_cfg=PLATE_CFG,
                ee_frame_cfg=EE_FRAME_CFG,
            ),
            "grasp_active_orange": 4.0
            * mdp.grasp_unplaced_oranges_dense(
                self,
                std=0.08,
                close_joint_threshold=0.7,
                oranges_cfg=ORANGES_CFG,
                plate_cfg=PLATE_CFG,
                robot_cfg=ROBOT_CFG,
                ee_frame_cfg=EE_FRAME_CFG,
                lift_progress_height=0.05,
                lift_progress_floor=1.0,
            ),
            "lift_active_orange": 8.0
            * mdp.lift_unplaced_oranges_dense(
                self,
                target_height_delta=0.05,
                oranges_cfg=ORANGES_CFG,
                plate_cfg=PLATE_CFG,
                robot_cfg=ROBOT_CFG,
                ee_frame_cfg=EE_FRAME_CFG,
                grasp_distance=0.08,
                close_joint_threshold=0.7,
            ),
            "excessive_lift_height": -20.0
            * mdp.active_orange_over_lift_penalty(
                self,
                oranges_cfg=ORANGES_CFG,
                plate_cfg=PLATE_CFG,
                max_height_delta=0.12,
            ),
            "move_active_orange_to_plate": 8.0
            * mdp.move_unplaced_oranges_to_plate_dense(
                self,
                std=0.40,
                lifted_height_delta=0.04,
                preplace_height_above_plate=0.10,
                oranges_cfg=ORANGES_CFG,
                plate_cfg=PLATE_CFG,
            ),
            "lower_active_orange_to_plate": 6.0
            * mdp.lower_unplaced_oranges_to_plate_dense(
                self,
                std=0.04,
                lifted_height_delta=0.04,
                target_height_above_plate=0.05,
                near_plate_xy=0.12,
                oranges_cfg=ORANGES_CFG,
                plate_cfg=PLATE_CFG,
            ),
            "release_active_orange_on_plate": 12.0
            * mdp.release_unplaced_oranges_on_plate_dense(
                self,
                height_std=0.04,
                lifted_height_delta=0.04,
                target_height_above_plate=0.05,
                near_plate_xy=0.12,
                oranges_cfg=ORANGES_CFG,
                plate_cfg=PLATE_CFG,
                robot_cfg=ROBOT_CFG,
                open_joint_threshold=0.7,
            ),
            "closed_gripper_near_plate": -6.0
            * mdp.closed_gripper_near_plate_penalty(
                self,
                height_std=0.04,
                lifted_height_delta=0.04,
                target_height_above_plate=0.05,
                near_plate_xy=0.12,
                oranges_cfg=ORANGES_CFG,
                plate_cfg=PLATE_CFG,
                robot_cfg=ROBOT_CFG,
                close_joint_threshold=0.7,
            ),
            "place_oranges_on_plate": 24.0
            * mdp.oranges_on_plate_fraction(
                self,
                oranges_cfg=ORANGES_CFG,
                plate_cfg=PLATE_CFG,
                lifted_height_delta=0.05,
            ),
            "rest_pose_after_placing": 8.0
            * mdp.rest_pose_after_all_placed(
                self,
                oranges_cfg=ORANGES_CFG,
                plate_cfg=PLATE_CFG,
            ),
            "future_oranges_displacement": -6.0
            * mdp.non_active_orange_displacement_penalty(
                self,
                oranges_cfg=ORANGES_CFG,
                plate_cfg=PLATE_CFG,
                displacement_tolerance=0.05,
            ),
            "joint_vel": -1.0e-4 * mdp.joint_vel_l2(self, asset_cfg=ROBOT_CFG),
        }

    def _eureka_success_metric(self, env_ids) -> torch.Tensor:
        """Scalar score in [0, 1] for IsaacLabEureka's reset-time success metric."""
        placed_fraction = mdp.oranges_on_plate_fraction(
            self,
            oranges_cfg=ORANGES_CFG,
            plate_cfg=PLATE_CFG,
            lifted_height_delta=0.05,
        )
        task_success = self._check_success().float()
        score = 0.75 * placed_fraction + 0.25 * task_success
        return torch.mean(score[env_ids])

    def _check_success(self) -> torch.Tensor:
        return mdp.task_done(
            env=self,
            oranges_cfg=ORANGES_CFG,
            plate_cfg=PLATE_CFG,
        )
