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

## SmolVLA RL Finetuning Matrix

| ID | Student Policy | Teacher | Guidance | RL Method | Status | Notes |
|---|---|---|---|---|---|---|
| E | SmolVLA (frozen backbone) + MLP PPO heads | MLP teacher trajectories | Traj guidance | PPO | existing | `Isaac-SO-ARM101-SmolVLA-Guided-Lift-Cube-v0` |
| F | SmolVLA (action head only) | MLP teacher trajectories | Traj guidance | RWR | planned | `Isaac-SO-ARM101-SmolVLA-RL-v0`, `train_smolvla_rl.py` |
| G | SmolVLA (action head + LoRA backbone) | MLP teacher trajectories | Traj guidance | RWR | planned | Same as F with `--use_lora` |
| H | SmolVLA SFT (offline BC on MLP demos) | MLP teacher → LeRobot dataset | None | Supervised | planned | `lerobot_train` on `collect_lerobot_dataset` output |
| I | SmolVLA (action head only) | MLP teacher trajectories | Traj guidance | PPO (Flow-SDE) | planned | `Isaac-SO-ARM101-SmolVLA-RL-v0`, `train_smolvla_ppo.py` |
| J | SmolVLA (action head + LoRA backbone) | MLP teacher trajectories | Traj guidance | PPO (Flow-SDE) | planned | Same as I with `--use_lora` |
| K | SmolVLA (action head only) | MLP teacher trajectories | Traj guidance | RECAP | planned | `Isaac-SO-ARM101-SmolVLA-RL-v0`, `train_smolvla_recap.py` |
| L | SmolVLA (action head + LoRA backbone) | MLP teacher trajectories | Traj guidance | RECAP | planned | Same as K with `--use_lora` |

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

## SmolVLA RWR Training Commands

All commands inside the Docker container or with Isaac Lab's Python.

### F) SmolVLA RWR – action head only

```bash
export ISAAC_SO_ARM101_TRAJECTORY_FILE=/workspace/isaac-bridge/isaac_so_arm101/logs/rsl_rl/teacher_trajectories/so101_fixed_layout_teacher.pt

isaaclab -p src/isaac_so_arm101/scripts/rsl_rl/train_smolvla_rl.py \
  --task Isaac-SO-ARM101-SmolVLA-RL-v0 \
  --policy igor-saprygin/so101-fixed-layout-smolvla \
  --instruction "Pick the cube." \
  --num_envs 8 \
  --num_rollout_episodes 32 \
  --max_iterations 200 \
  --top_k_percentile 30 \
  --lr 1e-5 \
  --experiment_name smolvla_rwr_action_head \
  --headless
```

### G) SmolVLA RWR – action head + LoRA backbone

```bash
isaaclab -p src/isaac_so_arm101/scripts/rsl_rl/train_smolvla_rl.py \
  --task Isaac-SO-ARM101-SmolVLA-RL-v0 \
  --policy igor-saprygin/so101-fixed-layout-smolvla \
  --num_envs 8 \
  --use_lora \
  --lora_rank 8 \
  --lr 1e-5 \
  --lora_lr 5e-6 \
  --experiment_name smolvla_rwr_lora \
  --headless
```

### Resuming a run

```bash
isaaclab -p src/isaac_so_arm101/scripts/rsl_rl/train_smolvla_rl.py \
  --task Isaac-SO-ARM101-SmolVLA-RL-v0 \
  --policy igor-saprygin/so101-fixed-layout-smolvla \
  --resume /path/to/logs/smolvla_rl/RUN_DATE/checkpoints/smolvla_rl_00019.pt \
  --headless
```

## SmolVLA PPO (Flow-SDE) Training Commands

These use the proper PPO algorithm (arXiv:2510.25889) rather than RWR, enabled by
converting the deterministic flow-matching ODE to a stochastic SDE at ONE random
denoising step per chunk, making log π(a|s) tractable.

### I) SmolVLA PPO – action head only

```bash
export ISAAC_SO_ARM101_TRAJECTORY_FILE=/workspace/isaac-bridge/isaac_so_arm101/logs/rsl_rl/teacher_trajectories/so101_fixed_layout_teacher.pt

isaaclab -p src/isaac_so_arm101/scripts/rsl_rl/train_smolvla_ppo.py \
  --task Isaac-SO-ARM101-SmolVLA-RL-v0 \
  --policy igor-saprygin/so101-fixed-layout-smolvla \
  --instruction "Pick the cube." \
  --num_envs 8 \
  --noise_level 0.5 \
  --num_denoise_steps 10 \
  --clip_param 0.2 \
  --lr 5e-6 \
  --critic_lr 1e-4 \
  --num_ppo_epochs 4 \
  --max_iterations 500 \
  --experiment_name smolvla_ppo_I \
  --headless
```

### J) SmolVLA PPO – action head + LoRA backbone

```bash
isaaclab -p src/isaac_so_arm101/scripts/rsl_rl/train_smolvla_ppo.py \
  --task Isaac-SO-ARM101-SmolVLA-RL-v0 \
  --policy igor-saprygin/so101-fixed-layout-smolvla \
  --instruction "Pick the cube." \
  --num_envs 8 \
  --noise_level 0.5 \
  --num_denoise_steps 10 \
  --use_lora \
  --lora_rank 8 \
  --lora_lr 5e-6 \
  --lr 5e-6 \
  --critic_lr 1e-4 \
  --max_iterations 500 \
  --experiment_name smolvla_ppo_J \
  --headless
```

### Resuming a PPO run

```bash
isaaclab -p src/isaac_so_arm101/scripts/rsl_rl/train_smolvla_ppo.py \
  --task Isaac-SO-ARM101-SmolVLA-RL-v0 \
  --policy igor-saprygin/so101-fixed-layout-smolvla \
  --resume /path/to/logs/smolvla_ppo/RUN_DATE/checkpoints/smolvla_ppo_00099.pt \
  --headless
```

### Hyperparameter tuning tips (from piRL paper)

- If train performance drops while eval oscillates: increase `--num_denoise_steps` to reduce ODE→SDE discretisation error.
- If KL divergence increases unstably: add `--gae_lambda 0.95` + lower `--lr 1e-6`.
- Lower `--noise_level` (e.g. 0.2) reduces exploration but is more stable at low lr.
- Avoid `--noise_level` < 0.2 — causes very large gradients and instability.

## SmolVLA RECAP Training Commands

RECAP (from the pi*0.6 paper) converts RL into advantage-conditioned supervised learning.
No SDE math, no `log_prob` computation — just a value function MLP and relabeled language instructions.
Uses ALL collected data (positive and negative), unlike RWR which discards the bottom episodes.

### K) SmolVLA RECAP – action head only

```bash
export ISAAC_SO_ARM101_TRAJECTORY_FILE=/workspace/isaac-bridge/isaac_so_arm101/logs/rsl_rl/teacher_trajectories/so101_fixed_layout_teacher.pt

isaaclab -p src/isaac_so_arm101/scripts/rsl_rl/train_smolvla_recap.py \
  --task Isaac-SO-ARM101-SmolVLA-RL-v0 \
  --policy igor-saprygin/so101-fixed-layout-smolvla \
  --instruction "Pick the cube." \
  --num_envs 8 \
  --positive_percentile 30 \
  --advantage_dropout 0.3 \
  --lr 5e-6 \
  --value_lr 1e-4 \
  --num_rollout_episodes 32 \
  --max_iterations 500 \
  --experiment_name smolvla_recap_K \
  --headless
```

### L) SmolVLA RECAP – action head + LoRA backbone

```bash
export ISAAC_SO_ARM101_TRAJECTORY_FILE=/workspace/isaac-bridge/isaac_so_arm101/logs/rsl_rl/teacher_trajectories/so101_fixed_layout_teacher.pt

isaaclab -p src/isaac_so_arm101/scripts/rsl_rl/train_smolvla_recap.py \
  --task Isaac-SO-ARM101-SmolVLA-RL-v0 \
  --policy igor-saprygin/so101-fixed-layout-smolvla \
  --instruction "Pick the cube." \
  --num_envs 8 \
  --positive_percentile 30 \
  --advantage_dropout 0.3 \
  --lr 5e-6 \
  --lora_lr 5e-6 \
  --value_lr 1e-4 \
  --use_lora \
  --lora_rank 8 \
  --num_rollout_episodes 32 \
  --max_iterations 500 \
  --experiment_name smolvla_recap_L \
  --headless
```

### Resuming a RECAP run

```bash
isaaclab -p src/isaac_so_arm101/scripts/rsl_rl/train_smolvla_recap.py \
  --task Isaac-SO-ARM101-SmolVLA-RL-v0 \
  --policy igor-saprygin/so101-fixed-layout-smolvla \
  --resume /path/to/logs/smolvla_recap/RUN_DATE/checkpoints/smolvla_recap_00099.pt \
  --headless
```

### Hyperparameter tuning tips (RECAP)

- `--positive_percentile`: Higher (e.g. 50%) = more positive labels = less contrastive signal.
  Lower (e.g. 20%) = stricter selection = stronger signal but slower learning.
  Default 30% matches the pi*0.6 paper.
- `--advantage_dropout`: Controls classifier-free guidance dropout.
  0.3 allows inference-time guidance scaling; 0.0 disables CFG.
- `--value_epochs`: More epochs (e.g. 10) gives a more accurate baseline but adds cost per iteration.
  If value loss is very high, increase this first.
- `--gamma`: Use 0.99 if episodes are long (>100 steps); 1.0 (undiscounted) is fine for short tasks.
- If loss collapses to 0: the model has overfit to the advantage labels — lower `--lr` or reduce `--num_update_epochs`.

## Notes

- Sparse baseline is expected to learn slowly and can remain near-zero reward for long periods.
- Guidance is intended to improve exploration/sample efficiency.
- For final performance, decaying guidance weight is usually safer than fixed guidance.
- SmolVLA RWR/PPO: the teacher is only used to provide gripper EE trajectories (reward shaping).
  The VLA (SmolVLA) is the student trained end-to-end.
- PPO (Flow-SDE) is expected to outperform RWR significantly (piRL: +29–40% on manipulation benchmarks).
- Checkpoints are in `logs/rsl_rl/smolvla_ppo*/RUN_DATE/checkpoints/`.
- TensorBoard logs: `tensorboard --logdir logs/rsl_rl/smolvla_ppo*/RUN_DATE/tb`.
