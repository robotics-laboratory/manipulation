#!/usr/bin/env bash
set -euo pipefail

# End-to-end sparse vs sparse+guidance ablation launcher for SO-101 pi0-FAST PPO rollouts.
#
# Stages:
#   1) Collect teacher trajectories from dense RSL-RL checkpoint
#   2) Collect sparse baseline rollouts (env sparse reward) + run PPO learner
#   3) Collect sparse+guidance rollouts (trajectory guidance reward) + run PPO learner

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
MANIP_DIR="${ROOT_DIR}/manipulation"
OPENPI_DIR="${ROOT_DIR}/openpi"

ISAACLAB_BIN="${ISAACLAB_BIN:-isaaclab}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${MANIP_DIR}/good_checkpoints/model_latest.pt}"
OPENPI_INIT_CHECKPOINT="${OPENPI_INIT_CHECKPOINT:-${OPENPI_DIR}/checkpoints/pi0_fast_so101_lift_cube/so101_h60_sft/7999}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MANIP_DIR}/rollouts/sparse_guidance_ablation}"
PROMPT="${PROMPT:-Lift the red cube up.}"
SEED="${SEED:-7}"

TEACHER_TRAJ_DIR="${OUTPUT_ROOT}/teacher_trajectories"
SPARSE_ROLLOUT_DIR="${OUTPUT_ROOT}/sparse_rollouts"
GUIDANCE_ROLLOUT_DIR="${OUTPUT_ROOT}/sparse_guidance_rollouts"
SPARSE_PPO_SAVE_DIR="${OUTPUT_ROOT}/ppo_sparse"
GUIDANCE_PPO_SAVE_DIR="${OUTPUT_ROOT}/ppo_sparse_guidance"

mkdir -p "${OUTPUT_ROOT}"

echo "[1/3] Collecting teacher trajectories -> ${TEACHER_TRAJ_DIR}"
"${ISAACLAB_BIN}" -p "${MANIP_DIR}/scripts/leisaac/scripts/training/collect_teacher_trajectories_so101.py" \
  --task LeIsaac-SO101-LiftCube-RewardDense-v0 \
  --checkpoint "${TEACHER_CHECKPOINT}" \
  --num_episodes 50 \
  --max_steps 128 \
  --success_only \
  --seed "${SEED}" \
  --output_dir "${TEACHER_TRAJ_DIR}" \
  --headless

echo "[2/3] Sparse baseline rollouts + PPO"
"${ISAACLAB_BIN}" -p "${MANIP_DIR}/scripts/leisaac/scripts/training/online_pi_fast_so101.py" \
  --task LeIsaac-SO101-LiftCube-Sparse-Train-v0 \
  --episodes 20 \
  --max_steps 64 \
  --action_horizon 10 \
  --prompt "${PROMPT}" \
  --reward_mode env_sparse \
  --seed "${SEED}" \
  --output_dir "${SPARSE_ROLLOUT_DIR}" \
  --policy_host localhost \
  --policy_port 8000 \
  --headless \
  --enable_cameras

"${PYTHON_BIN}" "${OPENPI_DIR}/scripts/train_fast_ppo_rollouts.py" \
  --config pi0_fast_so101_lift_cube \
  --rollout_dir "${SPARSE_ROLLOUT_DIR}" \
  --init_checkpoint "${OPENPI_INIT_CHECKPOINT}" \
  --save_dir "${SPARSE_PPO_SAVE_DIR}" \
  --num_train_steps 1 \
  --batch_size 1 \
  --seed "${SEED}"

echo "[3/3] Sparse+guidance rollouts + PPO"
"${ISAACLAB_BIN}" -p "${MANIP_DIR}/scripts/leisaac/scripts/training/online_pi_fast_so101.py" \
  --task LeIsaac-SO101-LiftCube-SparseGuidance-Train-v0 \
  --episodes 20 \
  --max_steps 64 \
  --action_horizon 10 \
  --prompt "${PROMPT}" \
  --reward_mode guidance \
  --traj_db "${TEACHER_TRAJ_DIR}" \
  --traj_sampling_strategy round_robin \
  --reset_from_traj \
  --seed "${SEED}" \
  --guidance_seed "${SEED}" \
  --output_dir "${GUIDANCE_ROLLOUT_DIR}" \
  --policy_host localhost \
  --policy_port 8000 \
  --headless \
  --enable_cameras

"${PYTHON_BIN}" "${OPENPI_DIR}/scripts/train_fast_ppo_rollouts.py" \
  --config pi0_fast_so101_lift_cube \
  --rollout_dir "${GUIDANCE_ROLLOUT_DIR}" \
  --init_checkpoint "${OPENPI_INIT_CHECKPOINT}" \
  --save_dir "${GUIDANCE_PPO_SAVE_DIR}" \
  --num_train_steps 1 \
  --batch_size 1 \
  --seed "${SEED}"

echo "Ablation run complete."
echo "Teacher trajectories: ${TEACHER_TRAJ_DIR}"
echo "Sparse rollouts:      ${SPARSE_ROLLOUT_DIR}"
echo "Guidance rollouts:    ${GUIDANCE_ROLLOUT_DIR}"
echo "Sparse PPO ckpts:     ${SPARSE_PPO_SAVE_DIR}"
echo "Guidance PPO ckpts:   ${GUIDANCE_PPO_SAVE_DIR}"
