"""Environment configs for SmolVLA-backed PPO training.

Inherits the trajectory-guided fixed-layout sparse env and adds:
* Cameras kept **enabled** (``disable_task_cameras = False``).
* A ``smolvla_features`` observation group whose single term runs the frozen
  SmolVLA VLM backbone on **top + side** camera images every step (dataset-style).
* Reduced ``num_envs`` default (8) to fit VLM inference in GPU memory.
"""

from __future__ import annotations

import os

import isaac_so_arm101.tasks.lift.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .fixed_layout_env_cfg import SoArm101FixedLayoutGuidedLiftCubeSparseEnvCfg
from .guided_env_cfg import GuidedObservationsCfg

_SMOLVLA_MODEL_PATH = os.environ.get("SMOLVLA_MODEL_PATH", "lerobot/smolvla_base")
_SMOLVLA_LANGUAGE_INSTRUCTION = os.environ.get("SMOLVLA_LANGUAGE_INSTRUCTION", "Pick the cube.")


@configclass
class SmolVLAFeaturesCfg(ObsGroup):
    """Flat visual-language features extracted by the frozen SmolVLA backbone."""

    features = ObsTerm(
        func=mdp.smolvla_visual_features,
        params={
            "model_path": _SMOLVLA_MODEL_PATH,
            "language_instruction": _SMOLVLA_LANGUAGE_INSTRUCTION,
            "camera_top_cfg": SceneEntityCfg("camera_top"),
            "camera_side_cfg": SceneEntityCfg("camera_side"),
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class SmolVLAObservationsCfg(GuidedObservationsCfg):
    """Observations for SmolVLA training: state policy vector + VLM features."""

    smolvla_features: SmolVLAFeaturesCfg = SmolVLAFeaturesCfg()


@configclass
class SoArm101SmolVLAGuidedLiftCubeSparseEnvCfg(SoArm101FixedLayoutGuidedLiftCubeSparseEnvCfg):
    """SmolVLA PPO: fixed-layout guided sparse lift with VLM feature observations."""

    observations: SmolVLAObservationsCfg = SmolVLAObservationsCfg()

    disable_task_cameras: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.episode_length_s = 5.0
        self.observations.observation.images_top = None
        self.observations.observation.images_wrist = None
        self.observations.observation.images_side = None
        self.observations.observation.images_up = None


@configclass
class SoArm101SmolVLAGuidedLiftCubeSparseEnvCfg_PLAY(SoArm101SmolVLAGuidedLiftCubeSparseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
