# Pure Isaac SO-101 + SmolVLA inference

Isaac Lab SO-101 pick-cube (or reach) environment with **SmolVLA** as an inference-only policy. Simulation is pure Isaac Lab. End-effector position and deltas are exposed via a thin wrapper.

**SO-101** is provided by the **isaac_so_arm101** extension included in this repo (robot URDF, reach and lift-cube tasks). The run script automatically registers its envs when you run from the Isaac directory.

## Prerequisites

1. **Isaac Sim** and **Isaac Lab** installed (see [Isaac Lab installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)).
2. **This repo** (including the `isaac_so_arm101/` extension):
   - Clone this repo. The extension lives under `isaac_so_arm101/` (SO-100/SO-101 URDFs, tasks, scripts).
   - **Optional**: Install the extension so it’s on `PYTHONPATH` when using Isaac Lab:
     ```bash
     cd /path/to/Isaac
     pip install -e isaac_so_arm101
     ```
     If you don’t install it, the run script adds `isaac_so_arm101/src` to `sys.path` so the envs are still registered.

## Install extra dependencies

From the **Isaac Lab** environment (the one you use with `./isaaclab.sh`), install LeRobot with SmolVLA and this project’s extras:

```bash
pip install -r /path/to/Isaac/requirements.txt
```

Or:

```bash
pip install "lerobot[smolvla]" gymnasium numpy torch huggingface_hub
```

## How to run

Scripts must be run via Isaac Lab’s Python so that `isaaclab` and the SO-101 envs are available. From your **Isaac Lab** repo root:

```bash
./isaaclab.sh -p /path/to/Isaac/scripts/run_smolvla_isaac.py --task Isaac-SO-ARM101-Lift-Cube-v0
```

**Task IDs** (from the in-repo isaac_so_arm101 extension):

| Task | Description |
|------|-------------|
| `Isaac-SO-ARM101-Lift-Cube-v0` | SO-101 pick cube and lift to target (default) |
| `Isaac-SO-ARM101-Lift-Cube-Play-v0` | Same, smaller scene for play |
| `Isaac-SO-ARM101-Reach-v0` | SO-101 reach target pose |
| `Isaac-SO-ARM101-Reach-Play-v0` | Same, play variant |
| `Isaac-SO-ARM100-Lift-Cube-v0`, `Isaac-SO-ARM100-Reach-v0` | SO-100 (same tasks) |

Options:

- `--task`: Gym task id (default: `Isaac-SO-ARM101-Lift-Cube-v0`).
- `--headless`: Run without GUI.
- `--num_envs`: Number of parallel envs (default: 1).
- `--policy`: HuggingFace policy repo or local path (default: `lerobot/smolvla_base`).
- `--instruction`: Language instruction for the policy (default: `"Pick the cube."`).
- `--max_steps`, `--episodes`: Episode length and number of episodes.
- `--camera_usd`: Load camera pose + intrinsics from a USD.
- `--camera_json`: Override camera pose from JSON (`translate` + `orient`).
- `--robot_name`: Scene articulation name (default: `robot`, as in isaac_so_arm101).
- `--ee_link_name`: End-effector link for EE pose (default: `gripper_link` for SO-101).
- `--no_ee_in_obs`: Do not add `ee_pos` / `ee_quat` / `ee_pos_delta` into the observation dict (EE still available via `get_ee_state()` and in `info["ee_state"]`).

Example (headless, 2 episodes, reach task):

```bash
./isaaclab.sh -p /path/to/Isaac/scripts/run_smolvla_isaac.py --task Isaac-SO-ARM101-Reach-v0 --headless --episodes 2 --instruction "Reach the target."
```

## Running with Docker

The Docker image is based on the official Isaac Lab container and bundles all dependencies (Isaac Sim, LeRobot, SmolVLA). This is the easiest way to get started — no local Isaac Sim install needed.

**Requirements:** Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (`nvidia-ctk`).

### Build

```bash
cd docker && docker compose build
```

To use a different base image (e.g. a newer Isaac Lab release):

```bash
BASE_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.2 docker compose build
```

### Run (headless)

```bash
cd docker && docker compose run --rm isaac-bridge
```

This drops you into a bash shell inside the container. From there, use the built-in aliases:

```bash
run-smolvla --task Isaac-SO-ARM101-Lift-Cube-v0 --headless --enable_cameras
save-cameras --task Isaac-SO-ARM101-Lift-Cube-v0
```

Or call `isaaclab.sh` directly:

```bash
isaaclab -p scripts/run_smolvla_isaac.py --task Isaac-SO-ARM101-Reach-v0 --headless --episodes 2
```

### Run (with X11 GUI)

For visualization and camera debugging, use the X11 service. On the host, make sure `DISPLAY` is set and allow Docker to connect:

```bash
export DISPLAY=:1   # use :0, :1, etc. — match your active X display
xhost +local:docker
cd docker && docker compose run --rm isaac-bridge-x11
```

Then inside the container, run without `--headless` to get the Isaac Sim viewport:

```bash
run-smolvla --task Isaac-SO-ARM101-Lift-Cube-v0 --enable_cameras
```

### Live editing

The compose file bind-mounts `src/`, `scripts/`, and `isaac_so_arm101/src/` from the host into the container. Edits on the host are reflected immediately — no rebuild needed for code changes.

### Persistent caches

Model weights (HuggingFace), pip packages, and Isaac Sim caches are stored in Docker named volumes, so they survive `docker compose down` and don't re-download between runs.

### Output

The `output/` directory is mounted from the host, so any data saved there (logs, images, checkpoints) persists after the container exits.

**Curated RSL-RL weights:** After training, copy your best `model_*.pt` into [`isaac_so_arm101/checkpoints/`](isaac_so_arm101/checkpoints/README.md) with a stable name (e.g. `so101_lift_cube.pt`). That directory is bind-mounted in Docker like `logs/`, so you can point `play.py` / dataset scripts at a fixed path without retraining.

## End-effector position and deltas

- **In the wrapper**: After each `step()` and `reset()`, the wrapper reads the SO-101 articulation’s end-effector link pose from the Isaac Lab scene and computes position deltas from the previous step.
- **API**:
  - **`get_ee_state()`**: Call on the wrapped env to get `ee_pos`, `ee_quat`, `ee_pos_delta`, `ee_quat_delta` (each numpy arrays).
  - **`info["ee_state"]`**: The same dict is written into `info` at every `step()` and `reset()`.
- **Observations**: By default the wrapper adds `ee_pos`, `ee_quat`, and `ee_pos_delta` to the observation dict so the policy (and your code) can use them. Disable with `--no_ee_in_obs` if you only need them from `get_ee_state()` / `info`.

Usage in your code:

```python
import isaac_so_arm101.tasks.reach  # noqa: F401
import isaac_so_arm101.tasks.lift   # noqa: F401
import gymnasium as gym
from env_wrapper import IsaacEEWrapper

env = gym.make("Isaac-SO-ARM101-Lift-Cube-v0", num_envs=1)
env = IsaacEEWrapper(env, robot_name="robot", ee_link_name="gripper_link", add_ee_to_obs=True)
obs, info = env.reset()
# EE state in info
ee = info["ee_state"]  # ee["ee_pos"], ee["ee_quat"], ee["ee_pos_delta"], ee["ee_quat_delta"]
# or on the wrapper
ee = env.get_ee_state()
```

## Vision / SmolVLA and cameras

The **Lift** task includes **top** and **wrist** cameras and exposes them as `observation.images_top` and `observation.images_wrist` (plus legacy aliases `observation.images_side` / `observation.images_up` for compatibility). To get real images (and avoid all-zero camera slots), run with **`--enable_cameras`** (see command below).

Example with cameras enabled:

```bash
./isaaclab.sh -p /path/to/Isaac/scripts/run_smolvla_isaac.py --task Isaac-SO-ARM101-Lift-Cube-v0 --enable_cameras
```

Override camera poses from JSON at runtime:

```bash
./isaaclab.sh -p scripts/run_smolvla_isaac.py --task Isaac-SO-ARM101-Lift-Cube-v0 --enable_cameras --camera_json manipulation/camera_poses.example.json
```

By default, if `--camera_json` is not provided, scripts use `manipulation/camera_poses.json`.

JSON schema (Euler degrees are converted to quaternion at launch):

```json
{
  "camera_top": { "translate": [0.06488, 0.00395, 1.06841], "orient": [-3.732, -5.338, -89.595], "euler_mode": "ui", "convention": "opengl" },
  "camera_wrist": { "translate": [0.00165, 0.10846, -0.02989], "orient": [-49.519, -10.504, -4.027], "convention": "opengl" }
}
```

If you paste rotation values directly from Isaac Sim's Transform panel, set `"euler_mode": "ui"` for that camera.

The script uses a default rename map so that top/wrist images map to `camera1`/`camera2` (with legacy side/up fallback), and the third slot is zeros (`--empty_cameras 1`). **Reach** and other tasks do not define cameras (see [Isaac Lab Camera](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/camera.html) to add them).

## Alternative checkpoints

- Default: **`lerobot/smolvla_base`** (multi-task, SO-100/SO-101–friendly action space).
- You can pass another HuggingFace repo, e.g. **`kenmacken/smolvla_policy`**, via `--policy`. SO-101 aligns well with SmolVLA’s typical SO-100/SO-101 action space; if the checkpoint uses a different robot, you may need to adjust scaling in `adapters.policy_action_to_env()` or add a small mapping in this repo.

## LeRobot compatibility

Isaac here is the **environment interface** (Isaac Lab SO-101 + wrapper); LeRobot is the **framework for training and data**. You can use both: make the Isaac env available to LeRobot so you can reuse LeRobot’s data collection, training, and evaluation.

- **`src/env.py`** exposes **`make_env(n_envs, use_async_envs, cfg)`** so LeRobot (and EnvHub) can load this env. It returns a VectorEnv-compatible wrapper so LeRobot does not try to clone the env into multiple processes.
- **`src/env_lerobot.py`** implements that wrapper (`IsaacAsVectorEnv`).
- **Usage**: Run your script via **Isaac Lab’s Python** (`isaaclab.sh`); the simulation app must be started by the caller before calling `make_env()`. Example:

```python
# Run this script with: ./isaaclab.sh -p your_script.py
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
from env import make_env

env = make_env(n_envs=2, task="Isaac-SO-ARM101-Lift-Cube-v0")
obs, info = env.reset()
# Use env with LeRobot's data collection or training APIs...
env.close()
simulation_app.close()
```

For **EnvHub**: put this repo on the Hub and load with `make_env("username/your-repo", trust_remote_code=True)` from a process that has already started the Isaac app (e.g. a script launched with `isaaclab.sh`).

## Collecting a LeRobot dataset (simulation, RSL-RL teacher)

You can record a **LeRobot v3** dataset (Parquet + MP4) by rolling out a **trained RSL-RL MLP** policy in Isaac Sim: top + wrist + side RGB, proprioceptive joint state (degrees), and actions. This matches the workflow used for VLA / SmolVLA finetuning data.

**Script:** `isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/collect_lerobot_dataset.py`

**Run it with a file path** (not `python -m ...`): the script imports sibling modules (`cli_args`, `log_paths`) from the same directory, so `isaaclab` must be given the `.py` path. From this repo root (or `/workspace/isaac-bridge` in Docker):

```bash
isaaclab -p isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/collect_lerobot_dataset.py \
  --task Isaac-SO-ARM101-FixedLayout-Lift-Cube-Play-v0 \
  --checkpoint /path/to/lift_fixed_layout/.../model_1499.pt \
  --num_episodes 50 \
  --output_dir output/datasets/my_so101_lift \
  --repo_id HF_USER/my-so101-lift \
  --success_only \
  --push_to_hub \
  --headless
```

For Hub upload, provide your token: `docker compose run --rm -e HF_TOKEN=hf_... isaac-bridge` (or `export HF_TOKEN=hf_...` inside the container before running).

**Typical flags**

| Flag | Role |
|------|------|
| `--task` | Gym id; use the same MDP as the teacher (e.g. fixed-layout dense teacher → `Isaac-SO-ARM101-FixedLayout-Lift-Cube-Play-v0`). Prefer `*-Play-*` for clean observations. |
| `--checkpoint` | Path to the RSL-RL checkpoint (`model_*.pt`). |
| `--num_episodes` | Episodes to record. |
| `--output_dir` | Local folder for the LeRobot dataset. |
| `--repo_id` | Dataset id for metadata (e.g. `HF_USER/my-dataset`). |
| `--task_description` | Language string stored per frame (default: pick-cube phrasing). |
| `--success_only` | Drop episodes where the cube was never lifted. |
| `--push_to_hub` | Upload after `finalize()`; requires Hub auth (see below). |
| `--headless` | Use in Docker / no display; the script also forces headless when `DISPLAY` is unset. |

**Teacher checkpoint:** Train the MLP on the matching task first (e.g. fixed-layout dense lift). See `docs/FIXED_LAYOUT_LIFT_EXPERIMENT.md` and run `isaaclab -p isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/train.py ...` (script path, same as above).

**Docker:** `docker compose run --rm isaac-bridge` mounts `output/` on the host — save under `output/datasets/...` to keep data after the container exits. For Hub upload, pass a token, e.g. `docker compose run --rm -e HF_TOKEN isaac-bridge`. The image sets `PATH` so `hf` / `huggingface-cli` resolve; use `/isaac-sim/python.sh` if anything still resolves to the wrong interpreter.

**Hugging Face upload:** `--push_to_hub` does not run unless you are logged in. Options: `export HF_TOKEN=hf_...`, or `hf auth login` using the same env as Isaac (`export PATH="/isaac-sim/kit/python/bin:$PATH"` and prefer `/isaac-sim/python.sh -m huggingface_hub.cli.hf auth login` if bare `hf` fails to import deps).

**SmolVLA / offline `lerobot-train`:** Base policy configs expect image keys `observation.images.camera1`, `camera2`, `camera3`. This collector writes `observation.images.top`, `observation.images.wrist`, and `observation.images.side`. Pass a **rename map** when training:

```text
--rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2", "observation.images.side": "observation.images.camera3"}'
```

Run training with the **same Python** that has LeRobot installed (avoid a mismatched `lerobot-train` on `PATH`). Prefer the module form:

```bash
python3 -m lerobot.scripts.lerobot_train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=HF_USER/so101-fixed-layout-smolvla \
  --policy.push_to_hub=true \
  --dataset.repo_id=HF_USER/so101-fixed-layout-vla \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2", "observation.images.side": "observation.images.camera3"}' \
  --steps=20000 \
  --batch_size=32 \
  --output_dir=outputs/train/so101_fixed_layout_smolvla \
  --job_name=so101_fixed_layout_smolvla \
  --policy.device=cuda \
  --wandb.enable=false
```

- Replace **`HF_USER`** with your Hugging Face namespace (`--dataset.repo_id` = Hub dataset; `--policy.repo_id` = where the finetuned policy is pushed).
- **`--output_dir`** must not already exist unless you resume a run; checkpoints and logs are written there locally.
- If the process is **Killed** with no traceback, reduce **`batch_size`** / **`num_workers`** — usually host or container **OOM** (see LeRobot docs for `lerobot-train --help`).
- Hub push requires login: `python3 -m huggingface_hub.cli.hf auth login` or `HF_TOKEN` on the same interpreter.

If training still expects a third view, check LeRobot / SmolVLA docs for padding or duplicate-camera behavior. Offline training also needs **FFmpeg** libraries for video decoding (`torchcodec`); the project `docker/Dockerfile` installs `ffmpeg` for that.

## Project layout

| File / folder | Purpose |
|---------------|--------|
| `src/` | Library modules (`adapters.py`, `env.py`, `env_lerobot.py`, `env_wrapper.py`, `camera_usd_loader.py`, `camera_json_loader.py`). |
| `scripts/` | Entrypoint scripts (`run_smolvla_isaac.py`, `save_env_cameras.py`, `view_training_cameras.py`). |
| `docker/` | `Dockerfile` and `docker-compose.yaml`. Build with `cd docker && docker compose build`. |
| `tools/` | Standalone utilities (`verify_versions.py`). |
| `isaac_so_arm101/` | **In-repo extension**: SO-100/SO-101 URDFs, reach and lift-cube tasks, RSL-RL scripts (`scripts/rsl_rl/train.py`, `collect_lerobot_dataset.py`, …). Run script adds `isaac_so_arm101/src` to the path to register envs. |
| `requirements.txt` | Extra deps (lerobot[smolvla], gymnasium, torch, etc.). |
| `README.md` | This file. |

Isaac Sim and Isaac Lab are **not** listed in `requirements.txt`; they are installed and run separately. The SO-101 robot and tasks come from the bundled **isaac_so_arm101** extension (original repo: [MuammerBay/isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101)).
