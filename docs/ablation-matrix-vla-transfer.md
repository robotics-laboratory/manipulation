# VLA Transfer Ablation Matrix (SO-101 Lift-Cube)

This document tracks the experiment matrix for trajectory-guided transfer in Isaac Lab.

## Goal

Evaluate whether teacher trajectory guidance improves sparse-reward learning and how scheduling affects final performance.

## Common Settings (for fair comparison)

- Robot/task family: SO-101 Lift-Cube
- Same PPO hyperparameters across runs
- Same `num_envs`, `max_iterations`, and simulator settings
- Same seed set for all variants (recommended: `42`, `123`, `999`)
- Run in Docker (`manipulation/docker`)

## Core Matrix

| ID | Student Env | Teacher | Guidance | Lambda Schedule | Status | Notes |
|---|---|---|---|---|---|---|
| A | `Isaac-SO-ARM101-Lift-Cube-Sparse-v0` | None | Off | N/A | planned | Sparse RL baseline |
| B | `Isaac-SO-ARM101-Guided-Lift-Cube-Sparse-v0` | Dense teacher trajectories | On | Fixed | planned | Sparse student + dense trajectory guidance |
| C | `Isaac-SO-ARM101-Guided-Lift-Cube-Sparse-v0` | Dense teacher trajectories | On | Decay to ~0 | planned | Best-practice guided sparse run |
| D | `Isaac-SO-ARM101-Guided-Lift-Cube-Sparse-Discriminator-v0` | Dense teacher trajectories | Discriminator `log D` | N/A | planned | Sparse student + learned discriminator guidance |

## Extended Matrix (optional but recommended)

| ID | Student Env | Teacher | Guidance | Lambda Schedule | Purpose |
|---|---|---|---|---|---|
| D | Sparse | Sparse teacher | On | Decay | Compare teacher quality (dense vs sparse teacher) |
| E | Dense | Dense teacher | On | Decay | Check guidance utility when task reward is already dense |
| F | Dense | None | Off | N/A | Dense baseline reference |

## Commands

All commands below are executed **inside** the docker container shell (`isaac-bridge`).

### A) Sparse baseline

```bash
isaaclab -p -m isaac_so_arm101.scripts.rsl_rl.train \
  --task Isaac-SO-ARM101-Lift-Cube-Sparse-v0 \
  --disable_task_cameras \
  --headless
```

### Teacher trajectory collection (dense teacher)

Use a **mounted output path** so trajectories persist across container restarts:

```bash
isaaclab -p -m isaac_so_arm101.scripts.rsl_rl.collect_trajectories \
  --task Isaac-SO-ARM101-Lift-Cube-v0 \
  --checkpoint /workspace/isaac-bridge/isaac_so_arm101/logs/rsl_rl/lift/<run>/model_1499.pt \
  --num_episodes 500 \
  --output /workspace/isaac-bridge/output/teacher_trajectories/so101_lift_cube_teacher.pt \
  --disable_task_cameras \
  --headless
```

### B) Guided run (fixed lambda)

```bash
export ISAAC_SO_ARM101_TRAJECTORY_FILE=/workspace/isaac-bridge/output/teacher_trajectories/so101_lift_cube_teacher.pt

isaaclab -p -m isaac_so_arm101.scripts.rsl_rl.train \
  --task Isaac-SO-ARM101-Guided-Lift-Cube-Sparse-v0 \
  --disable_task_cameras \
  --headless
```

### C) Guided run (decay lambda)

Requires adding a reward-weight schedule for trajectory guidance in env curriculum.
Use the same sparse-guided task id:

```bash
export ISAAC_SO_ARM101_TRAJECTORY_FILE=/workspace/isaac-bridge/output/teacher_trajectories/so101_lift_cube_teacher.pt

isaaclab -p -m isaac_so_arm101.scripts.rsl_rl.train \
  --task Isaac-SO-ARM101-Guided-Lift-Cube-Sparse-v0 \
  --disable_task_cameras \
  --headless
```

### D) Discriminator-guided run

Train a discriminator checkpoint (offline) using the same teacher trajectories:

```bash
export TEACHER_TRAJ=/workspace/isaac-bridge/output/teacher_trajectories/so101_lift_cube_teacher.pt
export DISC_OUT=/workspace/isaac-bridge/output/teacher_trajectories/trajectory_discriminator_lift_cube.pt

isaaclab -p -m isaac_so_arm101.scripts.train_trajectory_discriminator \
  --teacher_file $TEACHER_TRAJ \
  --output $DISC_OUT \
  --g_mode object_only
```

Then run the discriminator-guided sparse student:

```bash
export ISAAC_SO_ARM101_TRAJECTORY_DISCRIMINATOR_FILE=/workspace/isaac-bridge/output/teacher_trajectories/trajectory_discriminator_lift_cube.pt

isaaclab -p -m isaac_so_arm101.scripts.rsl_rl.train \
  --task Isaac-SO-ARM101-Guided-Lift-Cube-Sparse-Discriminator-v0 \
  --disable_task_cameras \
  --headless
```

## Metrics to Report

- Success rate vs environment steps
- Mean episodic return vs steps
- Final-window success rate (last 10% training)
- Seed variance (std across 3 seeds)
- Qualitative rollout videos for representative checkpoints

## Logging Conventions

Use explicit run names to make TensorBoard comparison simple:

- `A_sparse_baseline_s{seed}`
- `B_guided_fixed_s{seed}`
- `C_guided_decay_s{seed}`

## Notes

- Sparse baseline is expected to learn slowly and can remain near-zero reward for long periods.
- Guidance is intended to improve exploration/sample efficiency.
- For final performance, decaying guidance weight is usually safer than fixed guidance.
