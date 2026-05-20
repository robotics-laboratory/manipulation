import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from leisaac.assets.scenes.simple import TABLE_WITH_CUBE_USD_PATH
from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_camera_uniform,
    randomize_object_uniform,
)
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ...template import SingleArmTaskDirectEnv, SingleArmTaskDirectEnvCfg
from .. import mdp
from ..lift_cube_env_cfg import LiftCubeSceneCfg


@configclass
class LiftCubeEnvCfg(SingleArmTaskDirectEnvCfg):
    """Direct env configuration for the lift cube task."""

    scene: LiftCubeSceneCfg = LiftCubeSceneCfg(env_spacing=8.0)

    task_description: str = "Lift the red cube up."

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot.init_state.pos = (0.35, -0.64, 0.01)

        self.viewer.eye = (-0.4, -0.6, 0.5)
        self.viewer.lookat = (0.9, 0.0, -0.3)

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
class LiftCubeEurekaEnvCfg(LiftCubeEnvCfg):
    """State-only direct LiftCube configuration for IsaacLabEureka reward search."""

    scene: LiftCubeSceneCfg = LiftCubeSceneCfg(num_envs=64, env_spacing=8.0)

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
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation


class LiftCubeEnv(SingleArmTaskDirectEnv):
    """Direct env for the lift cube task."""

    cfg: LiftCubeEnvCfg

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        # add subtask observation
        obs["subtask_terms"] = {
            "pick_cube": mdp.object_grasped(
                self,
                robot_cfg=SceneEntityCfg("robot"),
                ee_frame_cfg=SceneEntityCfg("ee_frame"),
                object_cfg=SceneEntityCfg("cube"),
            )
        }
        return obs

    def _get_rewards(self) -> torch.Tensor:
        """Oracle dense reward used by Eureka for reward-correlation feedback."""
        reward_terms = self._get_oracle_reward_terms()
        self._lift_cube_oracle_reward_terms = reward_terms
        return sum(reward_terms.values())

    def _get_oracle_reward_terms(self) -> dict[str, torch.Tensor]:
        return {
            "reach_cube": 2.0
            * mdp.reach_object_dense(
                self,
                std=0.08,
                object_cfg=SceneEntityCfg("cube"),
                ee_frame_cfg=SceneEntityCfg("ee_frame"),
            ),
            "grasp_cube": 4.0
            * mdp.grasp_closure_dense(
                self,
                std=0.08,
                close_joint_threshold=0.7,
                object_cfg=SceneEntityCfg("cube"),
                robot_cfg=SceneEntityCfg("robot"),
                ee_frame_cfg=SceneEntityCfg("ee_frame"),
            ),
            "gripper_action_rate": -5.0e-3 * mdp.gripper_action_rate_l2(self),
            "lift_cube": 8.0
            * mdp.lift_progress_dense(
                self,
                target_height_delta=0.20,
                object_cfg=SceneEntityCfg("cube"),
                robot_cfg=SceneEntityCfg("robot"),
                robot_base_name="base",
            ),
            "lifted_stillness": 4.0
            * mdp.lifted_stillness_dense(
                self,
                lifted_height_delta=0.15,
                velocity_std=0.08,
                object_cfg=SceneEntityCfg("cube"),
            ),
            "lifted_angular_stillness": 4.0
            * mdp.lifted_angular_stillness_dense(
                self,
                lifted_height_delta=0.025,
                angular_velocity_std=1.0,
                object_cfg=SceneEntityCfg("cube"),
            ),
            "wrist_flip": -4.0
            * mdp.wrist_flip_penalty(
                self,
                max_abs_wrist_flex=1.05,
                std=0.25,
                lifted_height_delta=0.025,
                object_cfg=SceneEntityCfg("cube"),
                robot_cfg=SceneEntityCfg("robot"),
                wrist_joint_name="wrist_flex",
            ),
            "human_lift_posture": 1.0
            * mdp.human_lift_posture_dense(
                self,
                target_motor_positions={
                    "elbow_flex": -25.0,
                    "wrist_flex": 88.0,
                    "wrist_roll": 3.0,
                },
                std=0.75,
                lifted_height_delta=0.025,
                object_cfg=SceneEntityCfg("cube"),
                robot_cfg=SceneEntityCfg("robot"),
            ),
            "xy_position_stability": 1.5
            * mdp.xy_position_stability_dense(
                self,
                std=0.08,
                lifted_height_delta=0.025,
                object_cfg=SceneEntityCfg("cube"),
            ),
            "success_bonus": 10.0
            * mdp.cube_height_above_base_bonus(
                self,
                height_threshold=0.20,
                ramp_width=0.05,
                cube_cfg=SceneEntityCfg("cube"),
                robot_cfg=SceneEntityCfg("robot"),
                robot_base_name="base",
            ),
            "excessive_lift": -2.0
            * mdp.excessive_lift_penalty(
                self,
                max_height=0.25,
                std=0.05,
                cube_cfg=SceneEntityCfg("cube"),
                robot_cfg=SceneEntityCfg("robot"),
                robot_base_name="base",
            ),
            "joint_vel": -1.0e-4 * mdp.joint_vel_l2(self, asset_cfg=SceneEntityCfg("robot")),
        }

    def _eureka_success_metric(self, env_ids) -> torch.Tensor:
        """Scalar score in [0, 1] for IsaacLabEureka's reset-time success metric."""
        lift_progress = mdp.lift_progress_dense(
            self,
            target_height_delta=0.20,
            object_cfg=SceneEntityCfg("cube"),
            robot_cfg=SceneEntityCfg("robot"),
            robot_base_name="base",
        )
        task_success = self._check_success().float()
        score = 0.7 * lift_progress + 0.3 * task_success
        return torch.mean(score[env_ids])

    def _check_success(self) -> torch.Tensor:
        return mdp.cube_height_above_base(
            env=self,
            cube_cfg=SceneEntityCfg("cube"),
            robot_cfg=SceneEntityCfg("robot"),
            robot_base_name="base",
            height_threshold=0.20,
        )
