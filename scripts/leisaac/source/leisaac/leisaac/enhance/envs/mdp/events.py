from typing import Literal

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera


def randomize_camera_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, float],
    convention: Literal["opengl", "ros", "world"] = "ros",
):
    """Reset the camera to a random position and rotation uniformly within the given ranges.

    * It samples the camera position and rotation from the given ranges and adds them to the
      default camera position and rotation, before setting them into the physics simulation.

    The function takes a dictionary of pose ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or rotation is set to zero for that axis.
    """
    asset: Camera = env.scene[asset_cfg.name]

    ori_pos_w = asset.data.pos_w[env_ids]
    if convention == "ros":
        ori_quat_w = asset.data.quat_w_ros[env_ids]
    elif convention == "opengl":
        ori_quat_w = asset.data.quat_w_opengl[env_ids]
    elif convention == "world":
        ori_quat_w = asset.data.quat_w_world[env_ids]

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    # camera usually spawn with robot, so no need to add env_origins
    positions = ori_pos_w[:, 0:3] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(ori_quat_w, orientations_delta)

    asset.set_world_poses(positions, orientations, env_ids, convention)


def randomize_particle_object_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, float],
):
    """Reset the particle object to a random position and rotation uniformly within the given ranges.

    * It samples the particle object position and rotation from the given ranges and adds them to the
      default particle object position and rotation, before setting them into the physics simulation.

    The function takes a dictionary of pose ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or rotation is set to zero for that axis.
    """
    particle_object = env.scene.particle_objects[asset_cfg.name]
    ori_world_pos, ori_world_quat = particle_object.get_world_poses()

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)

    positions = ori_world_pos + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(ori_world_quat, orientations_delta)

    particle_object.set_world_poses(positions, orientations)


def disable_rigid_body_gravity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
):
    """Disable gravity for specific bodies in an articulation.

    This function disables gravity for bodies specified in the asset_cfg.body_names.
    It uses modify_rigid_body_properties to set disable_gravity=True for the specified bodies.

    Args:
        env: The environment instance.
        env_ids: The environment IDs to apply the change to.
        asset_cfg: Configuration specifying the asset and body names to disable gravity for.
                   Use body_names to specify which bodies to disable gravity (e.g., ".*arm.*" or ["shoulder", "elbow"]).
    """
    # Get the asset
    asset: Articulation = env.scene[asset_cfg.name]

    # Resolve body indices from body_names (already resolved by SceneEntityCfg)
    if asset_cfg.body_ids == slice(None):
        body_ids = list(range(asset.num_bodies))
    else:
        body_ids = asset_cfg.body_ids if isinstance(asset_cfg.body_ids, list) else [asset_cfg.body_ids]

    # Get link paths from the first environment (they follow the same pattern for all environments)
    link_paths = asset.root_physx_view.link_paths[0]

    # Disable gravity for each specified body
    for body_id in body_ids:
        if body_id >= len(link_paths):
            continue

        # Get the link path from first environment
        first_env_link_path = link_paths[body_id]

        # Convert to regex expression by replacing env_0 with env_.*
        link_path_expr = first_env_link_path.replace("/env_0/", "/env_.*/")

        # Resolve all matching prim paths and apply
        prim_paths = sim_utils.find_matching_prim_paths(link_path_expr)
        for prim_path in prim_paths:
            sim_utils.modify_rigid_body_properties(
                prim_path,
                sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            )


def disable_scene_clutter_colliders(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    scene_attr_name: str = "Scene/Scene",
    keep_name_patterns: list[str] | None = None,
    extra_disable_name_patterns: list[str] | None = None,
    deep_disable_path_patterns: list[str] | None = None,
    log: bool = True,
):
    """Disable PhysX collisions on prims under the per-env Scene root that are pure
    visual clutter, dramatically reducing collision pair counts on heavy USD scenes.

    Two passes:

    1. **Top-level pass** — enumerates immediate children of
       ``/World/envs/env_0/<scene_attr_name>``. For each child whose name does NOT match
       any pattern in ``keep_name_patterns`` (or matches ``extra_disable_name_patterns``),
       collisions are disabled recursively across all envs.
    2. **Deep pass** — for each pattern in ``deep_disable_path_patterns`` (regex relative
       to the scene root, e.g. ``"stack_.*_main_group_.*/drawer_.*"``), all matching prims
       across all envs are disabled. This is where most of the savings come from on
       Robocasa-style kitchens: door / drawer / handle internals can be culled while
       keeping the cabinet *corpus* (which holds the countertop) collidable.

    Collision modification cascades via ``modify_collision_properties``'s ``@apply_nested``
    decorator. Use as a ``"startup"`` mode event so it runs once after scene cloning but
    before the first physics step.

    Args:
        env: The environment instance.
        env_ids: Required by event manager; unused (operation is global, not per-env).
        scene_attr_name: Path (relative to ``/World/envs/env_0``) of the prim whose
            top-level children are inspected. Defaults to ``"Scene/Scene"`` because most
            UsdFileCfg-loaded kitchens introduce one extra wrapper prim (the file's
            defaultPrim, also named ``Scene``) under the env's scene attribute.
        keep_name_patterns: Regex list. Top-level children whose name matches any pattern
            keep their collisions. Defaults to a permissive list covering oranges, plate,
            cabinet/fixture stacks (``stack_*``), tables, counters, floor, walls, and the
            robot itself. Always pass an explicit list if your scene uses non-standard
            naming.
        extra_disable_name_patterns: Optional regex list. Top-level children matching any
            of these are *also* disabled even if they would have been kept by
            ``keep_name_patterns``. Defaults to ``[]``.
        deep_disable_path_patterns: Optional list of regex *paths* (slash-separated) under
            ``scene_attr_name`` to recursively disable. Each pattern is appended to
            ``/World/envs/env_.*/<scene_attr_name>/`` and resolved via
            ``find_matching_prim_paths``. Use this to nuke specific named clutter that
            lives inside a kept group (e.g. drawer/door internals).
        log: If True, print the lists of kept vs disabled top-level prim names and the
            deep-disable counts.
    """
    import re

    if keep_name_patterns is None:
        keep_name_patterns = [
            r"^Orange.*$",
            r"^Plate.*$",
            r"^stack_.*$",
            r"(?i).*table.*",
            r"(?i).*counter.*",
            r"(?i).*floor.*",
            r"(?i).*ground.*",
            r"(?i).*wall.*",
            r"(?i).*robot.*",
        ]
    if extra_disable_name_patterns is None:
        extra_disable_name_patterns = []
    if deep_disable_path_patterns is None:
        deep_disable_path_patterns = []

    keep_re = [re.compile(p) for p in keep_name_patterns]
    force_disable_re = [re.compile(p) for p in extra_disable_name_patterns]

    ref_scene_path = f"/World/envs/env_0/{scene_attr_name}"
    ref_scenes = sim_utils.find_matching_prims(ref_scene_path)
    if not ref_scenes:
        if log:
            print(f"[disable_scene_clutter_colliders] No prim at {ref_scene_path}; nothing to do.")
        return
    ref_scene = ref_scenes[0]

    keep_names: list[str] = []
    disable_names: list[str] = []
    for child in ref_scene.GetAllChildren():
        name = child.GetName()
        is_force_disabled = any(r.match(name) for r in force_disable_re)
        is_kept = any(r.match(name) for r in keep_re) and not is_force_disabled
        (keep_names if is_kept else disable_names).append(name)

    if log:
        print(
            f"[disable_scene_clutter_colliders] Keeping colliders on top-level "
            f"{ref_scene_path} children: {sorted(keep_names)}"
        )
        print(
            f"[disable_scene_clutter_colliders] Disabling colliders on top-level "
            f"{ref_scene_path} children: {sorted(disable_names)}"
        )

    disabled_prim_count = 0
    for name in disable_names:
        prim_paths = sim_utils.find_matching_prim_paths(
            f"/World/envs/env_.*/{scene_attr_name}/{name}"
        )
        for prim_path in prim_paths:
            sim_utils.modify_collision_properties(
                prim_path,
                sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            )
            disabled_prim_count += 1

    n_envs = len(sim_utils.find_matching_prim_paths(f"/World/envs/env_.*/{scene_attr_name}"))
    if log:
        print(
            f"[disable_scene_clutter_colliders] Disabled colliders on {disabled_prim_count} "
            f"top-level prim(s) across {n_envs} env(s). Sub-prims cascaded via apply_nested."
        )

    deep_disabled_total = 0
    for pattern in deep_disable_path_patterns:
        full_pattern = f"/World/envs/env_.*/{scene_attr_name}/{pattern}"
        matched_paths = sim_utils.find_matching_prim_paths(full_pattern)
        for prim_path in matched_paths:
            sim_utils.modify_collision_properties(
                prim_path,
                sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            )
        if log:
            print(
                f"[disable_scene_clutter_colliders] Deep-disabled {len(matched_paths)} "
                f"prim(s) matching '{pattern}' (across {n_envs} env(s))."
            )
        deep_disabled_total += len(matched_paths)

    if log and deep_disable_path_patterns:
        print(
            f"[disable_scene_clutter_colliders] Deep-disable total: {deep_disabled_total} "
            f"prim(s) across {len(deep_disable_path_patterns)} pattern(s)."
        )
