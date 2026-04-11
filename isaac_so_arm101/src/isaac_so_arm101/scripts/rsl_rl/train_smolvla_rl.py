"""Train SmolVLA via RWR (Reward-Weighted Regression) with trajectory guidance.

Algorithm overview
------------------
Each iteration:
  1. Roll out SmolVLA in a vectorised Isaac Sim env for ``num_rollout_episodes`` episodes.
     Trajectory guidance rewards (path progress + gripper alignment + sparse lift) provide
     the RL signal.  The MLP teacher's EE polylines are already baked into the env cfg.
  2. Filter the top-``top_k_percentile``% episodes by cumulative return.
  3. Update the SmolVLA **action head** (``action_in_proj``, ``action_time_mlp_*``,
     ``action_out_proj``, ``lm_expert``) using the standard flow-matching MSE loss on
     the collected (obs, action_chunk) pairs.  The VLM backbone + vision encoder are
     frozen by default.

Policy architecture
-------------------
SmolVLA uses conditional flow matching (not PPO).  There is no tractable log-prob.
RWR avoids this by directly reusing the flow-matching training objective, weighted/filtered
by episode return — so no modifications to the model architecture are needed.

Usage (from manipulation/isaac_so_arm101/)::

    isaaclab -p src/isaac_so_arm101/scripts/rsl_rl/train_smolvla_rl.py \\
        --task Isaac-SO-ARM101-SmolVLA-RL-v0 \\
        --policy lerobot/smolvla_base \\
        --headless

Or with a finetuned checkpoint::

    isaaclab -p src/isaac_so_arm101/scripts/rsl_rl/train_smolvla_rl.py \\
        --task Isaac-SO-ARM101-SmolVLA-RL-v0 \\
        --policy igor-saprygin/so101-fixed-layout-smolvla \\
        --num_envs 8 \\
        --headless
"""

# -------------------------------------------------------------------------
# LeRobot / torch must be imported BEFORE AppLauncher to avoid
# inspect.getfile crash on namespace packages inside Isaac Sim's Kit Python.
# -------------------------------------------------------------------------
import sys

try:
    import torch
    import torchvision  # noqa: F401 — ensures torchvision fake ops register early
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
except ImportError as _err:
    print(
        "LeRobot/SmolVLA not installed.  Install with: pip install 'lerobot[smolvla]'",
        file=sys.stderr,
    )
    raise SystemExit(1) from _err

import argparse
import os

from isaaclab.app import AppLauncher

# -------------------------------------------------------------------------
# Argument parsing
# -------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="SmolVLA RWR finetuning with trajectory guidance in Isaac Lab."
)
# --- Environment ---
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-SO-ARM101-SmolVLA-RL-v0",
    help="Gym task ID (must have cameras and trajectory guidance enabled).",
)
parser.add_argument("--num_envs", type=int, default=None, help="Parallel envs (default from env cfg).")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
# --- Policy ---
parser.add_argument(
    "--policy",
    type=str,
    default="lerobot/smolvla_base",
    help="HuggingFace SmolVLA checkpoint (repo id or local path).",
)
parser.add_argument(
    "--instruction",
    type=str,
    default="Pick the cube.",
    help="Language instruction for SmolVLA.",
)
parser.add_argument(
    "--camera_mapping",
    type=str,
    default=None,
    help=(
        'JSON map {isaac_sensor_name: policy_key}, '
        'e.g. \'{"camera_top":"observation.images.camera1",'
        '"camera_wrist":"observation.images.camera2"}\'. '
        "Default: camera_top→camera1, camera_wrist→camera2."
    ),
)
# --- RWR hyperparameters ---
parser.add_argument(
    "--max_iterations",
    type=int,
    default=200,
    help="Number of collect-then-update iterations.",
)
parser.add_argument(
    "--num_rollout_episodes",
    type=int,
    default=32,
    help="Episodes to collect per iteration (across all envs).",
)
parser.add_argument(
    "--max_episode_steps",
    type=int,
    default=250,
    help="Max env steps per episode before forced reset.",
)
parser.add_argument(
    "--top_k_percentile",
    type=float,
    default=30.0,
    help="Keep this fraction (%) of episodes (by return) for the update step.",
)
parser.add_argument(
    "--num_update_epochs",
    type=int,
    default=4,
    help="Gradient update epochs per iteration on the filtered data.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
    help="Mini-batch size for the flow-matching update (decision points per step).",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-5,
    help="Learning rate for the action head (and lm_expert).",
)
parser.add_argument(
    "--grad_clip",
    type=float,
    default=1.0,
    help="Max gradient norm for clipping.",
)
# --- LoRA (optional) ---
parser.add_argument(
    "--use_lora",
    action="store_true",
    default=False,
    help="Add LoRA adapters to the VLM backbone in addition to training the action head.",
)
parser.add_argument(
    "--lora_rank",
    type=int,
    default=8,
    help="LoRA rank (used when --use_lora is set).",
)
parser.add_argument(
    "--lora_lr",
    type=float,
    default=5e-6,
    help="Learning rate for LoRA parameters (lower than action head lr).",
)
# --- Checkpointing ---
parser.add_argument(
    "--save_interval",
    type=int,
    default=20,
    help="Save checkpoint every N iterations.",
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Path to a previous SmolVLA-RL checkpoint to resume from.",
)
# --- Action space ---
parser.add_argument(
    "--n_arm_joints",
    type=int,
    default=5,
    help="Number of arm (non-gripper) joints for degree→radian conversion.",
)
parser.add_argument(
    "--n_state_joints",
    type=int,
    default=6,
    help="Number of joints to extract from robot state for SmolVLA input.",
)
# --- Misc ---
parser.add_argument(
    "--experiment_name",
    type=str,
    default="smolvla_rl",
    help="Experiment name for log directory.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, _hydra_args = parser.parse_known_args()
# Cameras must be enabled for SmolVLA visual input.
args_cli.enable_cameras = True

# Force headless when no display is available (common in Docker / SSH).
_has_display = bool(os.environ.get("DISPLAY", "").strip()) or bool(
    os.environ.get("WAYLAND_DISPLAY", "").strip()
)
if not _has_display and not getattr(args_cli, "headless", False):
    print(
        "[INFO] No DISPLAY detected — forcing --headless (camera rendering still works).",
        flush=True,
    )
    args_cli.headless = True

sys.argv = [sys.argv[0]] + _hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------------------------------------------------------------
# Remaining imports (after AppLauncher)
# -------------------------------------------------------------------------

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import isaaclab as _il_ns
if not getattr(_il_ns, "__file__", None):
    _il_ns.__file__ = next(iter(_il_ns.__path__), __file__)

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks.lift  # noqa: F401
import isaac_so_arm101.tasks.reach  # noqa: F401

from isaac_so_arm101.tasks.lift.smolvla_rollout import (
    DEFAULT_CAMERA_MAPPING,
    SmolVLARolloutBuffer,
    collect_rollouts,
)
from log_paths import rsl_rl_root

logger = logging.getLogger(__name__)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _build_camera_mapping(args_cli_camera_mapping: str | None) -> dict[str, str]:
    if args_cli_camera_mapping:
        return json.loads(args_cli_camera_mapping)
    return dict(DEFAULT_CAMERA_MAPPING)


_ACTION_HEAD_PREFIXES = (
    "action_in_proj",
    "action_out_proj",
    "action_time_mlp_in",
    "action_time_mlp_out",
    "vlm_with_expert.lm_expert",
    "state_proj",
)


def _get_trainable_params(policy: SmolVLAPolicy, use_lora: bool) -> list[dict]:
    """Return optimizer parameter groups and freeze everything else.

    Action head (always trained):
        model.action_in_proj, model.action_time_mlp_*, model.action_out_proj,
        model.state_proj, model.vlm_with_expert.lm_expert (the cross-attending expert LM).

    LoRA (optional, Phase 3):
        Adapter weights added to the VLM backbone's attention layers via peft.
    """
    # Freeze ALL policy parameters first (including any normalizer modules).
    for param in policy.parameters():
        param.requires_grad_(False)

    action_head_params: list[torch.nn.Parameter] = []
    lora_params: list[torch.nn.Parameter] = []

    for name, param in policy.model.named_parameters():
        is_action_head = any(name.startswith(prefix) for prefix in _ACTION_HEAD_PREFIXES)
        is_lora = "lora_" in name  # peft LoRA naming convention

        if is_action_head:
            param.requires_grad_(True)
            action_head_params.append(param)
        elif use_lora and is_lora:
            param.requires_grad_(True)
            lora_params.append(param)
        # else: already frozen above

    if not action_head_params:
        # Fallback: if the above prefixes matched nothing (model architecture changed),
        # warn and unfreeze the entire model so training still proceeds.
        print(
            "[SmolVLA-RL] WARNING: no action-head parameters matched – unfreezing full model.",
            flush=True,
        )
        for param in policy.model.parameters():
            param.requires_grad_(True)
        action_head_params = [p for p in policy.model.parameters()]

    param_groups = [{"params": action_head_params, "lr": args_cli.lr}]
    if lora_params:
        param_groups.append({"params": lora_params, "lr": args_cli.lora_lr})

    n_action = sum(p.numel() for p in action_head_params)
    n_lora = sum(p.numel() for p in lora_params)
    n_total = sum(p.numel() for p in policy.parameters())
    print(
        f"[SmolVLA-RL] Trainable: action_head={n_action:,}, lora={n_lora:,} "
        f"/ total={n_total:,} params",
        flush=True,
    )
    return param_groups


def _apply_lora(policy: SmolVLAPolicy, lora_rank: int) -> None:
    """Wrap the VLM backbone with LoRA adapters using peft."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "peft is required for LoRA finetuning.  Install with: pip install peft"
        ) from exc

    default_targets = policy._get_default_peft_targets()
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=default_targets["target_modules"],
        modules_to_save=default_targets.get("modules_to_save", []),
        bias="none",
    )
    policy.model = get_peft_model(policy.model, lora_cfg)
    policy.model.print_trainable_parameters()


def _rwr_update(
    policy: SmolVLAPolicy,
    samples: list[dict],
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    batch_size: int,
    grad_clip: float,
    device: torch.device,
) -> dict[str, float]:
    """Run ``num_epochs`` of flow-matching RWR update on the filtered samples.

    Args:
        policy:     SmolVLAPolicy (action head in train mode).
        samples:    List of per-decision-point batch dicts from the buffer.
        optimizer:  Adam/AdamW targeting action-head parameters.
        num_epochs: Number of passes through the sample dataset.
        batch_size: Number of samples per mini-batch.
        grad_clip:  Max gradient norm.
        device:     Compute device.

    Returns:
        Dict of aggregate metrics (mean loss, num updates, etc.).
    """
    from lerobot.utils.constants import ACTION

    if not samples:
        return {"loss": float("nan"), "num_updates": 0}

    policy.train()
    total_loss = 0.0
    num_updates = 0

    indices = list(range(len(samples)))
    for epoch in range(num_epochs):
        random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            mini_idx = indices[start: start + batch_size]
            if not mini_idx:
                continue

            # Collate mini-batch: cat along batch dim.
            batch: dict[str, torch.Tensor] = {}
            for key in samples[mini_idx[0]].keys():
                vals = []
                for i in mini_idx:
                    v = samples[i].get(key)
                    if isinstance(v, torch.Tensor):
                        vals.append(v.to(device))
                if vals:
                    batch[key] = torch.cat(vals, dim=0)

            if str(ACTION) not in batch:
                continue

            # Forward pass with per-sample loss reduction for optional weighting.
            optimizer.zero_grad(set_to_none=True)
            loss, _loss_dict = policy.forward(batch, reduction="mean")

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"]
                     if p.grad is not None],
                    max_norm=grad_clip,
                )
            optimizer.step()

            total_loss += loss.item()
            num_updates += 1

    mean_loss = total_loss / max(num_updates, 1)
    return {"loss": mean_loss, "num_updates": num_updates}


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)

    # ------------------------------------------------------------------ #
    # Logging / checkpointing dirs
    # ------------------------------------------------------------------ #
    log_root = os.path.abspath(os.path.join(rsl_rl_root(), args_cli.experiment_name))
    run_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))
    print(f"[SmolVLA-RL] Logging to: {run_dir}", flush=True)

    # ------------------------------------------------------------------ #
    # Load SmolVLA policy
    # ------------------------------------------------------------------ #
    print(f"[SmolVLA-RL] Loading policy from: {args_cli.policy}", flush=True)
    policy = SmolVLAPolicy.from_pretrained(args_cli.policy).to(device)

    # Build pre/post processors with fallback to smolvla_base for missing artifacts.
    _processor_sources = [args_cli.policy, "lerobot/smolvla_base"]
    preprocess = postprocess = None
    for src in _processor_sources:
        try:
            preprocess, postprocess = make_pre_post_processors(
                policy.config,
                src,
                preprocessor_overrides={"device_processor": {"device": str(device)}},
            )
            if src != args_cli.policy:
                print(f"[SmolVLA-RL] Processor fallback source: {src}", flush=True)
            break
        except FileNotFoundError:
            continue
    if preprocess is None:
        raise RuntimeError(
            "Could not load SmolVLA preprocessors from any source.  "
            "Try: --policy lerobot/smolvla_base"
        )

    # Optionally add LoRA adapters to the VLM backbone before freezing.
    if args_cli.use_lora:
        print(f"[SmolVLA-RL] Applying LoRA (rank={args_cli.lora_rank}) to backbone.", flush=True)
        _apply_lora(policy, args_cli.lora_rank)

    # Freeze backbone / set up trainable parameter groups.
    param_groups = _get_trainable_params(policy, use_lora=args_cli.use_lora)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    # Resume from checkpoint if requested.
    start_iteration = 0
    if args_cli.resume:
        ckpt = torch.load(args_cli.resume, map_location=device)
        policy.load_state_dict(ckpt["policy_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iteration = ckpt.get("iteration", 0) + 1
        print(f"[SmolVLA-RL] Resumed from iteration {start_iteration}.", flush=True)

    # ------------------------------------------------------------------ #
    # Create Isaac Lab env
    # ------------------------------------------------------------------ #
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=str(device),
        num_envs=args_cli.num_envs if args_cli.num_envs is not None else None,
    )
    # Silence debug visualisers — markers pollute camera images.
    if hasattr(getattr(env_cfg, "scene", None), "ee_frame"):
        env_cfg.scene.ee_frame.debug_vis = False
    if hasattr(getattr(env_cfg, "commands", None), "object_pose"):
        env_cfg.commands.object_pose.debug_vis = False

    # Set episode length.
    step_dt = env_cfg.sim.dt * env_cfg.decimation
    env_cfg.episode_length_s = args_cli.max_episode_steps * step_dt

    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    num_envs = env.unwrapped.num_envs

    camera_mapping = _build_camera_mapping(args_cli.camera_mapping)
    print(f"[SmolVLA-RL] Camera mapping: {camera_mapping}", flush=True)
    print(f"[SmolVLA-RL] num_envs={num_envs}, device={device}", flush=True)

    # ------------------------------------------------------------------ #
    # Save run config
    # ------------------------------------------------------------------ #
    run_cfg_path = os.path.join(run_dir, "train_cfg.json")
    with open(run_cfg_path, "w") as f:
        json.dump(vars(args_cli), f, indent=2, default=str)

    # ------------------------------------------------------------------ #
    # Main RWR training loop
    # ------------------------------------------------------------------ #
    print(f"[SmolVLA-RL] Starting training for {args_cli.max_iterations} iterations.", flush=True)

    for iteration in range(start_iteration, args_cli.max_iterations):
        iter_start = time.time()

        # ---------------------------------------------------------------- #
        # 1. Rollout collection
        # ---------------------------------------------------------------- #
        buffer = collect_rollouts(
            policy=policy,
            env=env,
            preprocess=preprocess,
            postprocess=postprocess,
            language_instruction=args_cli.instruction,
            camera_mapping=camera_mapping,
            num_episodes=args_cli.num_rollout_episodes,
            max_episode_steps=args_cli.max_episode_steps,
            device=device,
            n_arm_joints=args_cli.n_arm_joints,
            n_state_joints=args_cli.n_state_joints,
        )
        rollout_summary = buffer.summary()

        # ---------------------------------------------------------------- #
        # 2. Filter top-K% and build training samples
        # ---------------------------------------------------------------- #
        samples = buffer.get_filtered_batches(
            top_k_percentile=args_cli.top_k_percentile,
            preprocess=preprocess,
            language_instruction=args_cli.instruction,
            camera_mapping=camera_mapping,
            device=device,
        )
        buffer.clear()

        n_total = rollout_summary.get("num_episodes", 0)
        n_selected = len(samples)
        print(
            f"[SmolVLA-RL] iter={iteration:04d} | "
            f"episodes={n_total} | selected_dps={n_selected} | "
            f"mean_return={rollout_summary.get('mean_return', 0):.3f} | "
            f"success={rollout_summary.get('success_rate', 0)*100:.1f}%",
            flush=True,
        )

        # ---------------------------------------------------------------- #
        # 3. RWR update
        # ---------------------------------------------------------------- #
        update_metrics = _rwr_update(
            policy=policy,
            samples=samples,
            optimizer=optimizer,
            num_epochs=args_cli.num_update_epochs,
            batch_size=args_cli.batch_size,
            grad_clip=args_cli.grad_clip,
            device=device,
        )

        iter_elapsed = time.time() - iter_start

        # ---------------------------------------------------------------- #
        # 4. Logging
        # ---------------------------------------------------------------- #
        global_step = iteration + 1
        writer.add_scalar("rollout/mean_return", rollout_summary.get("mean_return", 0), global_step)
        writer.add_scalar("rollout/max_return", rollout_summary.get("max_return", 0), global_step)
        writer.add_scalar("rollout/success_rate", rollout_summary.get("success_rate", 0), global_step)
        writer.add_scalar("rollout/num_episodes", rollout_summary.get("num_episodes", 0), global_step)
        writer.add_scalar("rollout/num_selected_dps", n_selected, global_step)
        writer.add_scalar("train/loss", update_metrics["loss"], global_step)
        writer.add_scalar("train/num_updates", update_metrics["num_updates"], global_step)
        writer.add_scalar("perf/iter_time_s", iter_elapsed, global_step)

        print(
            f"          loss={update_metrics['loss']:.5f} | "
            f"updates={update_metrics['num_updates']} | "
            f"iter_time={iter_elapsed:.1f}s",
            flush=True,
        )

        # ---------------------------------------------------------------- #
        # 5. Checkpoint
        # ---------------------------------------------------------------- #
        if (iteration + 1) % args_cli.save_interval == 0 or iteration == args_cli.max_iterations - 1:
            ckpt_path = os.path.join(ckpt_dir, f"smolvla_rl_{iteration:05d}.pt")
            torch.save(
                {
                    "iteration": iteration,
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "rollout_summary": rollout_summary,
                    "train_metrics": update_metrics,
                    "args": vars(args_cli),
                },
                ckpt_path,
            )
            print(f"[SmolVLA-RL] Checkpoint saved: {ckpt_path}", flush=True)

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    writer.close()
    env.close()
    print("[SmolVLA-RL] Training complete.", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
