# Fixed-layout lift experiments (SO-101)

Use this when you want a **reproducible** setup for reward tuning: same cube spawn (no reset XY jitter), same goal pose, and teacher trajectories collected **on that identical MDP**.

## Registered tasks

| Gym ID | Role |
|--------|------|
| `Isaac-SO-ARM101-FixedLayout-Lift-Cube-v0` | Dense rewards — train the **teacher** here |
| `Isaac-SO-ARM101-FixedLayout-Lift-Cube-Sparse-v0` | Sparse baseline (same layout, no guidance) |
| `Isaac-SO-ARM101-FixedLayout-Guided-Lift-Cube-Sparse-v0` | Guided sparse student |
| `…-Play-v0` | Few envs, no obs corruption |

Implementation: `tasks/lift/fixed_layout_env_cfg.py` (`apply_so101_fixed_layout_to_cfg`).

**Pinned goal** is the midpoint of the default `UniformPoseCommand` ranges from `lift_env_cfg.py`  
(`pos_x=0`, `pos_y=-0.2`, `pos_z=0.275` in command space). **Cube reset jitter** is set to zero (same nominal spawn as default USD init).

To use a different fixed pose, edit `apply_so101_fixed_layout_to_cfg` (or duplicate the env class).

### Performance (Guided fixed-layout)

- `trajectory_guidance_fixed_traj_index` on the env config (defaults to **0** on `SoArm101FixedLayoutGuidedLiftCubeSparseEnvCfg`) skips `TrajectoryStore.match` / `cdist` on reset and pins the teacher row.
- When all envs share the same polyline, `TrajectoryStore` uses a **batched** GPU projection (`project_points_to_polyline_detailed_batched`) instead of a Python loop over envs.

Ensure your `.pt` has the intended teacher at **index 0** after success filtering (or change `trajectory_guidance_fixed_traj_index` / dataset order).

### Logs / checkpoints

Fixed-layout tasks use dedicated RSL-RL experiment folders (not mixed with randomized lift):

| Task kind | Under `isaac_so_arm101/logs/rsl_rl/` |
|-----------|----------------------------------------|
| FixedLayout dense / sparse (teacher) | `lift_fixed_layout/<timestamp>/` |
| FixedLayout guided student | `guided_lift_cube_fixed_layout/<timestamp>/` |

## Pipeline

### 1. Train dense teacher (fixed layout)

```bash
isaaclab -p -m isaac_so_arm101.scripts.rsl_rl.train \
  --task Isaac-SO-ARM101-FixedLayout-Lift-Cube-v0 \
  --disable_task_cameras \
  --headless
```

### 2. Collect trajectories (same task = matching `initial_object_pos` / `goal_pos`)

```bash
isaaclab -p -m isaac_so_arm101.scripts.rsl_rl.collect_trajectories \
  --task Isaac-SO-ARM101-FixedLayout-Lift-Cube-v0 \
  --checkpoint /path/to/teacher_model.pt \
  --output isaac_so_arm101/logs/rsl_rl/teacher_trajectories/so101_fixed_layout_teacher.pt \
  --num_episodes 500 \
  --disable_task_cameras \
  --headless
```

### 3. Guided student

```bash
export ISAAC_SO_ARM101_TRAJECTORY_FILE="$(pwd)/isaac_so_arm101/logs/rsl_rl/teacher_trajectories/so101_fixed_layout_teacher.pt"

isaaclab -p -m isaac_so_arm101.scripts.rsl_rl.train \
  --task Isaac-SO-ARM101-FixedLayout-Guided-Lift-Cube-Sparse-v0 \
  --disable_task_cameras \
  --headless
```

## Iterating on rewards

- Change weights / params in `guided_env_cfg.py` (`GuidedRewardsCfg`) or Hydra overrides, **keep the same** `--task` and trajectory file so comparisons stay meaningful.
- Optional: shorten runs with `--max_iterations` while tuning.

## Note on trajectory matching

Guided configs still use `match_mode="object_only"` by default; with a truly fixed layout, all envs share the same object pose key — trajectory matching remains consistent. If you add goal to matching later, set `match_mode="object_goal"` and keep collection/student tasks aligned.
