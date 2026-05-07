# Restore OpenPI pi0-FAST PPO/Logprob Work

This bundle captures the OpenPI + LeIsaac work needed to resume the SO-101
LiftCube online PPO experiments on a fresh GPU instance.

## Files

- `openpi_ppo_logprob.patch`: OpenPI changes for pi0-FAST token logprobs, PPO
  serving, SO-101 LoRA config, and learner scripts.
- `manipulation_leisaac_openpi_rollout.patch`: LeIsaac rollout script/config
  additions for OpenPI policy rollouts and PPO metadata.
- `restore_openpi_ppo.sh`: Applies both patches and optionally runs install
  steps.

The patch files intentionally do not include generated rollout `.npz` files,
temporary checkpoints, virtual environments, or Isaac Sim binaries.

## Fresh Instance Flow

Start from fresh checkouts at the usual paths:

```bash
cd /workspace
git clone <openpi-repo-or-fork> openpi
git clone <manipulation-repo-or-fork> manipulation
```

Copy or commit this `openpi_ppo_restore` directory into the new manipulation
checkout, then run:

```bash
cd /workspace/manipulation
OPENPI_DIR=/workspace/openpi \
MANIPULATION_DIR=/workspace/manipulation \
bash scripts/leisaac/scripts/training/openpi_ppo_restore/restore_openpi_ppo.sh
```

If the machine already has Python/IsaacLab installed and you only want patches:

```bash
SKIP_INSTALL=1 \
OPENPI_DIR=/workspace/openpi \
MANIPULATION_DIR=/workspace/manipulation \
bash scripts/leisaac/scripts/training/openpi_ppo_restore/restore_openpi_ppo.sh
```

## Manual Patch Commands

If you prefer to apply patches yourself:

```bash
git -C /workspace/openpi apply \
  /workspace/manipulation/scripts/leisaac/scripts/training/openpi_ppo_restore/openpi_ppo_logprob.patch

git -C /workspace/manipulation apply \
  /workspace/manipulation/scripts/leisaac/scripts/training/openpi_ppo_restore/manipulation_leisaac_openpi_rollout.patch
```

Then install:

```bash
cd /workspace/openpi
/root/.local/bin/uv sync
/root/.local/bin/uv pip install -e .

ln -sfn /workspace/manipulation/scripts/leisaac/dependencies/IsaacLab /workspace/isaaclab
ln -sfn /isaac-sim /workspace/isaaclab/_isaac_sim
TERM=xterm /workspace/isaaclab/isaaclab.sh -i all
```

## Smoke Commands

Start a logprob-enabled policy server from a checkpoint:

```bash
cd /workspace/openpi
/root/.local/bin/uv run scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi0_fast_so101_lift_cube \
  --policy.dir=/workspace/openpi/checkpoints/pi0_fast_so101_lift_cube/ppo_real_logprob_smoke \
  --policy.return-token-logprobs \
  --policy.temperature=1.0 \
  --policy.max-decoding-steps=32
```

Collect a tiny LeIsaac rollout:

```bash
TERM=xterm /workspace/isaaclab/isaaclab.sh -p \
  /workspace/manipulation/scripts/leisaac/scripts/training/online_pi_fast_so101.py \
  --headless \
  --enable_cameras \
  --episodes 1 \
  --max_steps 1 \
  --policy_host localhost \
  --policy_port 8000 \
  --output_dir /workspace/manipulation/scripts/leisaac/rollouts/pi_fast_so101_ppo_schema_smoke
```

Stop the policy server before GPU training, then run PPO:

```bash
cd /workspace/openpi
/root/.local/bin/uv run scripts/train_fast_ppo_rollouts.py \
  --config pi0_fast_so101_lift_cube \
  --rollout_dir /workspace/manipulation/scripts/leisaac/rollouts/pi_fast_so101_ppo_schema_smoke \
  --num_train_steps 1 \
  --batch_size 1 \
  --max_examples 8 \
  --init_checkpoint /workspace/openpi/checkpoints/pi0_fast_so101_lift_cube/ppo_real_logprob_smoke \
  --save_dir /workspace/openpi/checkpoints/pi0_fast_so101_lift_cube/ppo_next
```

For smoke tests on smaller GPUs or while the server is still using the GPU:

```bash
JAX_PLATFORMS=cpu /root/.local/bin/uv run scripts/train_fast_ppo_rollouts.py \
  --config pi0_fast_so101_lift_cube \
  --rollout_dir /workspace/manipulation/scripts/leisaac/rollouts/pi_fast_so101_ppo_schema_smoke \
  --num_train_steps 1 \
  --batch_size 1 \
  --max_examples 8 \
  --init_checkpoint /workspace/openpi/checkpoints/pi0_fast_so101_lift_cube/ppo_real_logprob_smoke \
  --save_dir /workspace/openpi/checkpoints/pi0_fast_so101_lift_cube/ppo_cpu_smoke
```

## Notes

- On a single GPU, collect with the server running, stop it, train PPO, then
  restart from the new checkpoint.
- Keep `--batch_size 1` and reduce `--policy.max-decoding-steps` to `16` if
  VRAM is tight.
- Copy checkpoints separately if you need to continue from an existing run; this
  bundle only restores code.
