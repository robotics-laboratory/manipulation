# SO-101 RLinf Overlay

This folder keeps the SO-101 RLinf/GR00T/OpenPI integration inside the
`manipulation` repository so it can be pushed independently of a local RLinf
checkout.

## Apply To RLinf

Clone or mount RLinf, then apply the overlay:

```bash
bash /workspace/manipulation/rlinf_so101/apply_overlay.sh /workspace/RLinf
```

The script copies files from `overlays/rlinf/` into the target RLinf checkout.

## Build Runtime Image

Build from the `manipulation` repo root, not from the parent IsaacLab checkout:

```bash
cd /workspace/manipulation
docker build -f rlinf_so101/Dockerfile.rlinf-so101-openpi -t rlinf-so101-openpi .
```

## Run Smoke Training

```bash
docker run -it --rm --gpus all \
  --shm-size 20g \
  --network host \
  --ipc host \
  --pid host \
  --privileged \
  --name rlinf \
  -v "/workspace/RLinf:/workspace/RLinf" \
  -v "/workspace/manipulation:/workspace/manipulation" \
  -w /workspace/RLinf \
  rlinf-so101-openpi
```

Inside the container:

```bash
source /workspace/RLinf/isaac_sim/setup_conda_env.sh

/opt/venv/gr00t/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path /workspace/RLinf/examples/embodiment/config \
  --config-name isaaclab_so101_lift_cube_ppo_openpi_pi05 \
  runner.logger.log_path=/workspace/RLinf/logs/so101_openpi_ppo_smoke \
  actor.model.model_path=/workspace/RLinf/pi_model/RLinf-pi05-SFT-Stack-cube
```

## Run SmolVLA SO101 PPO

```bash
source /workspace/RLinf/isaac_sim/setup_conda_env.sh

/opt/venv/gr00t/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path /workspace/RLinf/examples/embodiment/config \
  --config-name isaaclab_so101_lift_cube_ppo_smolvla \
  runner.max_epochs=5 \
  runner.logger.log_path=/workspace/RLinf/logs/so101_smolvla_ppo \
  actor.model.model_path=/workspace/RLinf/models/SmolVLA
```

## SmolVLA Smoke Run

```bash
source /workspace/RLinf/isaac_sim/setup_conda_env.sh

/opt/venv/gr00t/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path /workspace/RLinf/examples/embodiment/config \
  --config-name isaaclab_so101_lift_cube_ppo_smolvla \
  runner.max_epochs=1 \
  env.train.total_num_envs=1 \
  env.train.max_steps_per_rollout_epoch=16 \
  env.train.max_episode_steps=16 \
  runner.logger.log_path=/workspace/RLinf/logs/so101_smolvla_smoke \
  actor.model.model_path=/workspace/RLinf/models/SmolVLA
```
