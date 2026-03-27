# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for SO-101 lift tasks (extends Isaac Lab lift defaults + SmolVLA features)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.manager_based.manipulation.lift.mdp.observations import *  # noqa: F403

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# SmolVLA visual-language feature extraction (frozen backbone)
# ---------------------------------------------------------------------------


def smolvla_visual_features(
    env: "ManagerBasedRLEnv",
    model_path: str = "lerobot/smolvla_base",
    language_instruction: str = "Pick the cube.",
    camera_top_cfg: SceneEntityCfg = SceneEntityCfg("camera_top"),
    camera_side_cfg: SceneEntityCfg = SceneEntityCfg("camera_side"),
) -> torch.Tensor:
    """Extract visual-language features from SmolVLA's frozen VLM backbone.

    On the first call the model is loaded and cached on the environment instance.
    Subsequent calls reuse the cached model and only run inference.

    Returns a flat feature tensor of shape ``(num_envs, feature_dim)``.
    """
    state = _get_or_init_smolvla_state(env, model_path, language_instruction)
    model = state["model"]
    B = env.num_envs
    lang_tokens = state["lang_tokens"].expand(B, -1)
    lang_masks = state["lang_masks"].expand(B, -1)

    images_list, img_masks_list = _gather_camera_images(
        env,
        state,
        camera_top_cfg,
        camera_side_cfg,
    )

    robot_state = _gather_robot_state(env)

    with torch.no_grad():
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images_list, img_masks_list, lang_tokens, lang_masks, state=robot_state,
        )
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        outputs, _ = model.vlm_with_expert.forward(
            attention_mask=att_2d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            fill_kv_cache=True,
        )
        prefix_out = outputs[0]
        mask_f = prefix_pad_masks.unsqueeze(-1).to(dtype=prefix_out.dtype)
        pooled = (prefix_out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        pooled = pooled.to(dtype=torch.float32)

    return pooled


def _get_or_init_smolvla_state(
    env: "ManagerBasedRLEnv", model_path: str, language_instruction: str,
) -> dict:
    """Load the SmolVLA model once and cache on the env."""
    cache = getattr(env, "_smolvla_feature_state", None)
    if cache is not None and cache.get("model_path") == model_path:
        return cache

    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
    except ImportError as e:
        raise ImportError(
            "SmolVLA visual features require lerobot[smolvla]. "
            "Install with: pip install 'lerobot[smolvla]'"
        ) from e

    print(f"[SmolVLA] Loading backbone from '{model_path}' ...")
    policy = SmolVLAPolicy.from_pretrained(model_path)
    policy = policy.to(env.device).eval()
    flow_model: VLAFlowMatching = policy.model

    for p in flow_model.parameters():
        p.requires_grad = False

    tokenizer = flow_model.vlm_with_expert.processor.tokenizer
    tok_out = tokenizer(
        language_instruction,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=policy.config.tokenizer_max_length,
    )
    lang_tokens = tok_out["input_ids"].to(env.device)
    lang_masks = tok_out["attention_mask"].to(env.device).bool()

    hidden_size = flow_model.vlm_with_expert.config.text_config.hidden_size
    max_state_dim = policy.config.max_state_dim

    cache = {
        "model_path": model_path,
        "model": flow_model,
        "policy_config": policy.config,
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
        "hidden_size": hidden_size,
        "max_state_dim": max_state_dim,
    }
    env._smolvla_feature_state = cache
    print(f"[SmolVLA] Backbone ready. VLM hidden_size={hidden_size}")
    return cache


def _gather_camera_images(
    env: "ManagerBasedRLEnv",
    state: dict,
    camera_top_cfg: SceneEntityCfg,
    camera_side_cfg: SceneEntityCfg,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Collect camera images from scene sensors and prepare for SmolVLA."""
    from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad

    policy_config = state["policy_config"]
    B = env.num_envs
    device = env.device
    images_list = []
    img_masks_list = []

    for sensor_cfg in (camera_top_cfg, camera_side_cfg):
        sensor_name = sensor_cfg.name
        sensor = env.scene.sensors.get(sensor_name)
        if sensor is None:
            img = torch.zeros((B, 3, 512, 512), dtype=torch.float32, device=device)
            mask = torch.zeros(B, dtype=torch.bool, device=device)
        else:
            rgb = sensor.data.output["rgb"]
            if rgb.ndim == 4 and rgb.shape[-1] in (3, 4):
                img = rgb[..., :3].permute(0, 3, 1, 2).to(dtype=torch.float32)
            elif rgb.ndim == 4 and rgb.shape[1] in (3, 4):
                img = rgb[:, :3].to(dtype=torch.float32)
            else:
                img = rgb.to(dtype=torch.float32)

            if img.max() > 1.5:
                img = img / 255.0

            if policy_config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *policy_config.resize_imgs_with_padding, pad_value=0)
            img = img * 2.0 - 1.0
            mask = torch.ones(B, dtype=torch.bool, device=device)

        images_list.append(img)
        img_masks_list.append(mask)

    return images_list, img_masks_list


def _gather_robot_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Extract joint-state vector padded to SmolVLA's max_state_dim."""
    from lerobot.policies.smolvla.modeling_smolvla import pad_vector

    cache = env._smolvla_feature_state
    max_dim = cache["max_state_dim"]
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos
    return pad_vector(joint_pos.to(dtype=torch.float32), max_dim)
