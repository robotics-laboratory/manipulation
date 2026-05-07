# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# openpi model configs

import os

import numpy as np
import torch
from omegaconf import DictConfig


def _filter_mismatched_tensors(model, state_dict):
    model_state_dict = model.state_dict()
    return {
        key: value
        for key, value in state_dict.items()
        if key in model_state_dict and model_state_dict[key].shape == value.shape
    }


def _so101_norm_stats(transforms):
    arm_low = np.array([-100.0, -100.0, -100.0, -100.0, -100.0], dtype=np.float32)
    arm_high = np.array([100.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float32)
    grip_low = np.array([0.0], dtype=np.float32)
    grip_high = np.array([100.0], dtype=np.float32)
    low = np.concatenate([arm_low, grip_low])
    high = np.concatenate([arm_high, grip_high])
    mean = (low + high) / 2.0
    std = np.maximum((high - low) / 4.0, 1e-6).astype(np.float32)
    stats = transforms.NormStats(mean=mean, std=std, q01=low, q99=high)
    return {
        "state": stats,
        "actions": stats,
    }


def get_model(cfg: DictConfig, torch_dtype=None):
    import glob

    import openpi.shared.download as download
    import openpi.transforms as transforms
    import safetensors
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0Config,
        OpenPi0ForRLActionPrediction,
    )

    # config
    config_name = getattr(cfg.openpi, "config_name", None)
    data_kwargs = getattr(cfg, "openpi_data", None)
    actor_train_config = get_openpi_config(
        config_name, model_path=cfg.model_path, data_kwargs=data_kwargs
    )

    actor_model_config = actor_train_config.model
    actor_model_config = OpenPi0Config(**actor_model_config.__dict__)
    override_model_config_kwargs = cfg.openpi
    if override_model_config_kwargs is not None:
        for key, val in override_model_config_kwargs.items():
            actor_model_config.__dict__[key] = val

    # load model
    checkpoint_dir = download.maybe_download(str(cfg.model_path))

    # Check if this is a checkpoint directory (saved by FSDP)
    # Check for model_state_dict/full_weights.pt (direct checkpoint) or actor/model_state_dict/full_weights.pt (from runner)
    full_weights_path = os.path.join(
        checkpoint_dir, "model_state_dict", "full_weights.pt"
    )
    actor_full_weights_path = os.path.join(
        checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"
    )

    model: OpenPi0ForRLActionPrediction = OpenPi0ForRLActionPrediction(
        actor_model_config
    )
    # train expert only
    if actor_model_config.train_expert_only:
        model.freeze_vlm()

    # Load weights from checkpoint if it's a checkpoint directory, otherwise load from safetensors
    ignore_mismatched_sizes = bool(
        getattr(actor_model_config, "ignore_mismatched_sizes", False)
    )
    if os.path.exists(full_weights_path):
        # Direct checkpoint directory
        model_state_dict = torch.load(full_weights_path, map_location="cpu")
        if ignore_mismatched_sizes:
            model_state_dict = _filter_mismatched_tensors(model, model_state_dict)
        model.load_state_dict(model_state_dict, strict=False)
    elif os.path.exists(actor_full_weights_path):
        # Checkpoint directory from runner
        model_state_dict = torch.load(actor_full_weights_path, map_location="cpu")
        if ignore_mismatched_sizes:
            model_state_dict = _filter_mismatched_tensors(model, model_state_dict)
        model.load_state_dict(model_state_dict, strict=False)
    else:
        # Original model directory with safetensors files
        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not weight_paths:
            weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]
        all_state_dict = {}
        for weight_path in weight_paths:
            state_dict = safetensors.torch.load_file(weight_path, device="cpu")
            all_state_dict.update(state_dict)
        if ignore_mismatched_sizes:
            all_state_dict = _filter_mismatched_tensors(model, all_state_dict)
        model.load_state_dict(all_state_dict, strict=False)

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    # fsdp replace
    # model.paligemma_with_expert.replace_gemma_decoder_layers()
    # load data stats
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, actor_model_config
    )
    norm_stats = None
    if config_name == "pi05_so101_lift_cube":
        norm_stats = _so101_norm_stats(transforms)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)
    # wrappers
    repack_transforms = transforms.Group()
    default_prompt = None
    model.setup_wrappers(
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
    )

    return model
