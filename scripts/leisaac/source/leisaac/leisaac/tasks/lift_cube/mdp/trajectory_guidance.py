from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg


def reset_cube_pose_from_reference(
    env,
    env_ids: torch.Tensor,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    pose_attribute_name: str = "_trajectory_guidance_initial_cube_pose_w",
) -> None:
    """Override cube reset pose from externally provided trajectory reference pose(s).

    The rollout collector may set ``env.<pose_attribute_name>`` to either:
    - shape ``(7,)``: one world pose (xyz + xyzw quat) shared by all envs
    - shape ``(num_envs, 7)``: per-env world poses

    If the attribute is missing, this event is a no-op and default randomization remains active.
    """
    if env_ids.numel() == 0:
        return

    if not hasattr(env, pose_attribute_name):
        return

    target_pose_w = getattr(env, pose_attribute_name)
    if target_pose_w is None:
        return

    cube = env.scene[cube_cfg.name]
    pose_w = torch.as_tensor(target_pose_w, dtype=torch.float32, device=env.device)
    if pose_w.ndim == 1:
        pose_w = pose_w.unsqueeze(0).repeat(len(env_ids), 1)
    else:
        pose_w = pose_w[env_ids]

    cube.write_root_pose_to_sim(pose_w, env_ids=env_ids)
    cube.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)
