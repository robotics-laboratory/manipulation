"""OpenPI config template for pi0-FAST on LeIsaac SO-101 LiftCube.

Copy this into an OpenPI checkout, for example:

    src/openpi/training/so101_config.py

Then import ``get_so101_configs`` from ``src/openpi/training/config.py`` and
append ``*so101_config.get_so101_configs()`` to OpenPI's ``_CONFIGS`` list.

The LeIsaac rollout runner sends upstream-style OpenPI keys by default:

    observation/image
    observation/wrist_image
    observation/state
    prompt

Actions are 6D SO-101 motor-unit targets.  The LeIsaac runner converts these
motor units to LeIsaac joint radians before stepping the simulator.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Sequence

import einops
import numpy as np
from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_fast as pi0_fast
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
from openpi.training.config import (
    AssetsConfig,
    DataConfig,
    DataConfigFactory,
    ModelTransformFactory,
    TrainConfig,
)


def so101_norm_stats() -> dict[str, _transforms.NormStats]:
    """Simple SO-101 motor-unit normalization for state and action."""

    low = np.asarray([-100.0, -100.0, -100.0, -100.0, -100.0, 0.0], dtype=np.float32)
    high = np.asarray([100.0, 100.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float32)
    mean = (low + high) / 2.0
    std = np.maximum((high - low) / 4.0, 1e-6).astype(np.float32)
    stats = _transforms.NormStats(mean=mean, std=std, q01=low, q99=high)
    return {"state": stats, "actions": stats}


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = einops.rearrange(image, "c h w -> h w c")
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    return image


@dataclasses.dataclass(frozen=True)
class SO101LiftCubeInputs(_transforms.DataTransformFn):
    """Convert LeIsaac SO-101 observations into OpenPI model inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        if self.model_type == _model.ModelType.PI0_FAST:
            image_names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            images = (base_image, np.zeros_like(base_image), wrist_image)
            image_masks = (np.True_, np.True_, np.True_)
        else:
            image_names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            images = (base_image, wrist_image, np.zeros_like(base_image))
            image_masks = (np.True_, np.True_, np.False_)

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": dict(zip(image_names, images, strict=True)),
            "image_mask": dict(zip(image_names, image_masks, strict=True)),
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class SO101LiftCubeOutputs(_transforms.DataTransformFn):
    """Keep OpenPI outputs as 6D SO-101 motor-unit actions."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)[..., :6]
        low = np.asarray([-100.0, -100.0, -100.0, -100.0, -100.0, 0.0], dtype=np.float32)
        high = np.asarray([100.0, 100.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float32)
        return {**data, "actions": np.clip(actions, low, high)}


@dataclasses.dataclass(frozen=True)
class LeRobotSO101LiftCubeDataConfig(DataConfigFactory):
    """OpenPI data config for LeIsaac SO-101 LiftCube."""

    default_prompt: str | None = "Lift the red cube up."
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(
        self,
        assets_dirs: pathlib.Path,
        model_config: _model.BaseModelConfig,
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.front",
                        "observation/wrist_image": "observation.images.wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "task",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[SO101LiftCubeInputs(model_type=model_config.model_type)],
            outputs=[SO101LiftCubeOutputs()],
        )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )
        base = self.create_base_config(assets_dirs, model_config)
        return dataclasses.replace(
            base,
            norm_stats=base.norm_stats or so101_norm_stats(),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


def get_so101_configs() -> list[TrainConfig]:
    lora_model = pi0_fast.Pi0FASTConfig(
        action_dim=6,
        action_horizon=10,
        max_token_len=180,
        paligemma_variant="gemma_2b_lora",
    )
    return [
        TrainConfig(
            name="pi0_fast_so101_lift_cube",
            model=lora_model,
            data=LeRobotSO101LiftCubeDataConfig(
                repo_id="so101_lift_cube_rl",
                base_config=DataConfig(prompt_from_task=False),
                assets=AssetsConfig(asset_id="so101_lift_cube_rl"),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi0_fast_base/params"
            ),
            num_train_steps=30_000,
            batch_size=32,
            freeze_filter=lora_model.get_freeze_filter(),
            ema_decay=None,
        )
    ]
