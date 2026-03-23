# Teacher trajectory datasets (`*.pt`)

## Building / rebuilding

Use `scripts/rsl_rl/collect_trajectories.py` with the **same task** (and layout) you will use for guided training:

```bash
# Example: teacher checkpoint + task matching training
python scripts/rsl_rl/collect_trajectories.py \
  --task Isaac-SO-ARM101-Lift-Cube-v0 \
  --checkpoint path/to/teacher/model.pt \
  --num_episodes 500 \
  --output path/to/so101_lift_cube_teacher.pt
```

Set `ISAAC_SO_ARM101_TRAJECTORY_FILE` (or reward/event `trajectory_file` params) to that path.

### Required / optional keys

| Key | Required | Notes |
|-----|----------|--------|
| `initial_object_pos`, `goal_pos` | yes | World frame at episode start / goal |
| `ee_trajectories`, `trajectory_lengths` | yes | Padded EE polylines |
| `initial_object_pos_local`, `goal_pos_local` | recommended | Env-origin–invariant matching |
| `initial_ee_pos_local` | recommended | First EE sample (env-local) for IK align |
| `gripper_trajectories` | optional | For gripper alignment obs/reward |
| `success` | optional | Filters to successful rows when present |

Rebuild the dataset after changing **command ranges**, **cube spawn**, or **teacher checkpoint** so `TrajectoryStore.match` stays consistent.

## Reset order (guided RL)

`align_ee_to_teacher_trajectory_start` is registered with event mode **`post_command_reset`**.  
`ManagerBasedRLEnv` applies this **after** `command_manager.reset`, so goal-conditioned matching uses the **resampled** `object_pose` command — not the pre-reset buffer.

## Snapping the cube to the dataset row

- **Single call:** `align_ee_to_teacher_trajectory_start(..., reset_object_from_dataset=True)`  
- **Two terms:** declare `reset_object_pose_from_trajectory_dataset` **above** `align_ee_to_teacher_trajectory_start` in the same `post_command_reset` event config (both use the same matching logic; the combined flag avoids duplicate matching when you only need one pass).
