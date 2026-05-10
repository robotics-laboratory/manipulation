# Sparse vs Sparse+Guidance PPO

This note documents reproducible commands for:
- collecting trajectory references from a dense RSL-RL SO-101 teacher,
- running sparse baseline rollout collection + PPO finetuning,
- running sparse+guidance rollout collection + PPO finetuning.

## New task IDs

- `LeIsaac-SO101-LiftCube-Sparse-Train-v0`
- `LeIsaac-SO101-LiftCube-SparseGuidance-Train-v0`

Both are state-only SO-101 training tasks. The sparse+guidance task includes a reset hook that can override cube initial pose from trajectory metadata when enabled in rollout collection.

## 1) Collect teacher trajectories

Use your dense MLP checkpoint from `manipulation/good_checkpoints`:

```bash
isaaclab -p /workspace/manipulation/scripts/leisaac/scripts/training/collect_teacher_trajectories_so101.py \
  --task LeIsaac-SO101-LiftCube-RewardDense-v0 \
  --checkpoint /workspace/manipulation/good_checkpoints/model_latest.pt \
  --num_episodes 50 \
  --max_steps 128 \
  --success_only \
  --seed 7 \
  --output_dir /workspace/manipulation/rollouts/teacher_trajectories_so101 \
  --headless
```

Saved files:
- `trajectory_*.npz` with `ee_pos_w`, `ee_quat_w`, `gripper_state`, `initial_cube_pose_w`, `meta_json`
- `summary.jsonl`

## 2) Sparse baseline rollouts (env sparse reward)

```bash
isaaclab -p /workspace/manipulation/scripts/leisaac/scripts/training/online_pi_fast_so101.py \
  --task LeIsaac-SO101-LiftCube-Sparse-Train-v0 \
  --policy_host localhost \
  --policy_port 8000 \
  --prompt "Lift the red cube up." \
  --episodes 20 \
  --max_steps 64 \
  --action_horizon 10 \
  --reward_mode env_sparse \
  --output_dir /workspace/manipulation/rollouts/sparse_baseline \
  --seed 7 \
  --headless \
  --enable_cameras
```

Then train PPO from these rollouts:

```bash
python3 /workspace/openpi/scripts/train_fast_ppo_rollouts.py \
  --config pi0_fast_so101_lift_cube \
  --rollout_dir /workspace/manipulation/rollouts/sparse_baseline \
  --init_checkpoint /workspace/openpi/checkpoints/pi0_fast_so101_lift_cube/so101_h60_sft/7999 \
  --save_dir /workspace/manipulation/rollouts/ppo_sparse \
  --num_train_steps 1 \
  --batch_size 1 \
  --seed 7
```

## 3) Sparse+guidance rollouts (collector guidance reward)

```bash
isaaclab -p /workspace/manipulation/scripts/leisaac/scripts/training/online_pi_fast_so101.py \
  --task LeIsaac-SO101-LiftCube-SparseGuidance-Train-v0 \
  --policy_host localhost \
  --policy_port 8000 \
  --prompt "Lift the red cube up." \
  --episodes 20 \
  --max_steps 64 \
  --action_horizon 10 \
  --reward_mode guidance \
  --traj_db /workspace/manipulation/rollouts/teacher_trajectories_so101 \
  --traj_sampling_strategy round_robin \
  --reset_from_traj \
  --guidance_progress_weight 1.0 \
  --guidance_xy_weight 0.25 \
  --guidance_gripper_weight 0.25 \
  --guidance_xy_scale 0.08 \
  --guidance_gripper_scale 15.0 \
  --guidance_progress_power 1.0 \
  --output_dir /workspace/manipulation/rollouts/sparse_guidance \
  --seed 7 \
  --guidance_seed 7 \
  --headless \
  --enable_cameras
```

Then train PPO:

```bash
python3 /workspace/openpi/scripts/train_fast_ppo_rollouts.py \
  --config pi0_fast_so101_lift_cube \
  --rollout_dir /workspace/manipulation/rollouts/sparse_guidance \
  --init_checkpoint /workspace/openpi/checkpoints/pi0_fast_so101_lift_cube/so101_h60_sft/7999 \
  --save_dir /workspace/manipulation/rollouts/ppo_sparse_guidance \
  --num_train_steps 1 \
  --batch_size 1 \
  --seed 7
```

## Smoke validation checklist

- teacher collector runs for `1-2` episodes and writes `trajectory_*.npz`
- sparse run writes non-zero `rewards` only on success steps
- guidance run writes `guidance_progress`, `guidance_xy_error`, `guidance_gripper_error`, and `guidance_env_reward`
- guidance run metadata includes `selected_traj_id` and `selected_initial_cube_pose_w`
- one PPO train step succeeds for each rollout directory

## One-command orchestration

Use:

```bash
/workspace/manipulation/scripts/leisaac/scripts/training/run_sparse_guidance_ablation.sh
```

Configure via environment variables:
- `TEACHER_CHECKPOINT`
- `OPENPI_INIT_CHECKPOINT`
- `OUTPUT_ROOT`
- `PROMPT`
- `SEED`
- `ISAACLAB_BIN`
- `PYTHON_BIN`
