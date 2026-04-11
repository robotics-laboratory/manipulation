"""Train SmolVLA via RECAP (RL with Experience and Corrections via Advantage-conditioned Policies).

Algorithm overview
------------------
RECAP converts RL finetuning into advantage-conditioned supervised learning.
Reference: pi*0.6 paper (arXiv:2511.14759).

Each iteration:
  1. Collect ALL rollouts with "Advantage: positive. " prepended to the task instruction
     (always-positive conditioning at inference time).
  2. Train/update the value function V(s) on Monte Carlo returns from the collected episodes
     (a few SGD epochs, MLP on frozen VLM prefix features).
  3. Compute per-decision-point advantages: A = MC_return - V(s).
  4. Binarize advantages: top ``positive_percentile``% → "positive", rest → "negative".
  5. Build training batches with advantage-conditioned language:
       - "Advantage: positive. Pick the cube."
       - "Advantage: negative. Pick the cube."
       - With ``advantage_dropout`` probability: drop the advantage prefix entirely
         (classifier-free guidance dropout).
  6. Run standard ``policy.forward(batch, reduction="mean")`` flow-matching MSE update
     on ALL collected decision points (positive and negative).
  7. Log and checkpoint.

Key advantages over RWR:
  - Uses ALL data (not just top-K) — more efficient.
  - Stable supervised objective — same as pretraining.
  - No SDE math, no log_prob computation, no critic needed during rollout.
  - Enables classifier-free guidance at inference: scale the advantage conditioning.

Usage (from manipulation/isaac_so_arm101/)::

    isaaclab -p src/isaac_so_arm101/scripts/rsl_rl/train_smolvla_recap.py \\
        --task Isaac-SO-ARM101-SmolVLA-RL-v0 \\
        --policy igor-saprygin/so101-fixed-layout-smolvla \\
        --instruction "Pick the cube." \\
        --num_envs 8 \\
        --positive_percentile 30 \\
        --advantage_dropout 0.3 \\
        --lr 5e-6 \\
        --value_lr 1e-4 \\
        --max_iterations 500 \\
        --experiment_name smolvla_recap_K \\
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
    description="SmolVLA RECAP finetuning with trajectory guidance in Isaac Lab."
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
    help="Base language instruction for SmolVLA (advantage prefix is prepended automatically).",
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
# --- RECAP hyperparameters ---
parser.add_argument(
    "--max_iterations",
    type=int,
    default=500,
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
    "--positive_percentile",
    type=float,
    default=30.0,
    help=(
        "Fraction (%%) of decision points (by advantage) labeled as 'positive'. "
        "The rest are labeled 'negative'. Default: 30."
    ),
)
parser.add_argument(
    "--advantage_dropout",
    type=float,
    default=0.3,
    help=(
        "Probability of dropping the advantage prefix entirely during training "
        "(classifier-free guidance dropout). Default: 0.3."
    ),
)
parser.add_argument(
    "--num_update_epochs",
    type=int,
    default=4,
    help="Gradient update epochs per iteration on the advantage-labeled data.",
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
    default=5e-6,
    help="Learning rate for the action head (and lm_expert).",
)
parser.add_argument(
    "--grad_clip",
    type=float,
    default=1.0,
    help="Max gradient norm for clipping.",
)
# --- Value function hyperparameters ---
parser.add_argument(
    "--value_lr",
    type=float,
    default=1e-4,
    help="Learning rate for the value function MLP.",
)
parser.add_argument(
    "--value_epochs",
    type=int,
    default=5,
    help="SGD epochs for value function per iteration.",
)
parser.add_argument(
    "--value_batch_size",
    type=int,
    default=16,
    help="Mini-batch size for value function training.",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=1.0,
    help="Discount factor for Monte Carlo returns. Default: 1.0 (undiscounted).",
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
    help="Path to a previous RECAP checkpoint to resume from.",
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
    default="smolvla_recap",
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
from isaac_so_arm101.tasks.lift.smolvla_value_fn import SmolVLAValueFunction
from log_paths import rsl_rl_root

logger = logging.getLogger(__name__)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

ADVANTAGE_POSITIVE_PREFIX = "Advantage: positive. "
ADVANTAGE_NEGATIVE_PREFIX = "Advantage: negative. "

_ACTION_HEAD_PREFIXES = (
    "action_in_proj",
    "action_out_proj",
    "action_time_mlp_in",
    "action_time_mlp_out",
    "vlm_with_expert.lm_expert",
    "state_proj",
)


def _build_camera_mapping(args_cli_camera_mapping: str | None) -> dict[str, str]:
    if args_cli_camera_mapping:
        return json.loads(args_cli_camera_mapping)
    return dict(DEFAULT_CAMERA_MAPPING)


def _get_trainable_params(
    policy: SmolVLAPolicy,
    use_lora: bool,
    action_lr: float,
    lora_lr: float,
) -> list[dict]:
    """Return AdamW parameter groups.  Freezes everything except action head (and LoRA)."""
    for param in policy.parameters():
        param.requires_grad_(False)

    action_head_params: list[torch.nn.Parameter] = []
    lora_params: list[torch.nn.Parameter] = []

    for name, param in policy.model.named_parameters():
        is_action_head = any(name.startswith(pfx) for pfx in _ACTION_HEAD_PREFIXES)
        is_lora = "lora_" in name
        if is_action_head:
            param.requires_grad_(True)
            action_head_params.append(param)
        elif is_lora and use_lora:
            param.requires_grad_(True)
            lora_params.append(param)

    if not action_head_params:
        print(
            "[SmolVLA-RECAP] WARNING: no action-head params found — "
            "unfreezing full flow_model as fallback.",
            flush=True,
        )
        for p in policy.model.parameters():
            p.requires_grad_(True)
        action_head_params = list(policy.model.parameters())

    groups: list[dict] = [
        {"params": action_head_params, "lr": action_lr, "name": "action_head"},
    ]
    if lora_params:
        groups.append({"params": lora_params, "lr": lora_lr, "name": "lora"})

    n_action = sum(p.numel() for p in action_head_params)
    n_lora = sum(p.numel() for p in lora_params)
    print(
        f"[SmolVLA-RECAP] Trainable: action_head={n_action:,}, lora={n_lora:,}",
        flush=True,
    )
    return groups


def _apply_lora(policy: SmolVLAPolicy, lora_rank: int) -> None:
    """Add LoRA adapters to the VLM backbone."""
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


def _build_recap_samples(
    labeled_dps,  # list of (DecisionPointData, label, mc_return) from SmolVLAValueFunction
    base_instruction: str,
    preprocess,
    device: torch.device,
    advantage_dropout: float,
) -> list[dict]:
    """Convert labeled decision points into training batch dicts.

    For each decision point:
        - With probability (1 - advantage_dropout): prepend the advantage prefix.
          "Advantage: positive. <instruction>" or "Advantage: negative. <instruction>"
        - With probability advantage_dropout: use the plain instruction
          (classifier-free guidance dropout).

    Returns:
        List of per-decision-point training dicts ready for ``policy.forward``.
    """
    from lerobot.utils.constants import ACTION, OBS_STATE

    samples: list[dict] = []
    for dp, label, _ret in labeled_dps:
        # Choose instruction string.
        if random.random() < advantage_dropout:
            instruction = base_instruction
        elif label == "positive":
            instruction = ADVANTAGE_POSITIVE_PREFIX + base_instruction
        else:
            instruction = ADVANTAGE_NEGATIVE_PREFIX + base_instruction

        frame: dict = {
            "language_instruction": instruction,
            "task": instruction,
        }
        for policy_key, img_np in dp.images_np.items():
            frame[policy_key] = np.expand_dims(img_np.astype(np.float32), axis=0)
        state_arr = dp.state_np[:6] if dp.state_np.shape[0] >= 6 else dp.state_np
        frame[str(OBS_STATE)] = np.expand_dims(state_arr.astype(np.float32), axis=0)

        batch = preprocess(frame)

        tensor_batch: dict[str, torch.Tensor] = {}
        for key, val in batch.items():
            if isinstance(val, np.ndarray):
                tensor_batch[key] = torch.as_tensor(val, device=device)
            elif isinstance(val, torch.Tensor):
                tensor_batch[key] = val.to(device)
            else:
                tensor_batch[key] = val

        # Action chunk (model-output space): (1, chunk_size, action_dim).
        tensor_batch[str(ACTION)] = dp.action_chunk.unsqueeze(0).to(device)

        samples.append(tensor_batch)
    return samples


def _recap_update(
    policy: SmolVLAPolicy,
    samples: list[dict],
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    batch_size: int,
    grad_clip: float,
    device: torch.device,
) -> dict[str, float]:
    """Flow-matching MSE update on advantage-labeled samples (all positive + negative).

    The advantage conditioning is already baked into the language instruction strings
    inside each ``sample`` dict.  The update is identical to RWR but uses all data
    instead of filtering top-K.

    Args:
        policy:     SmolVLAPolicy in train mode.
        samples:    Per-decision-point batch dicts with advantage-conditioned language.
        optimizer:  AdamW over action head (and optional LoRA).
        num_epochs: Passes through the sample dataset.
        batch_size: Samples per mini-batch.
        grad_clip:  Max gradient norm.
        device:     Compute device.

    Returns:
        Dict with ``loss`` and ``num_updates``.
    """
    from lerobot.utils.constants import ACTION

    if not samples:
        return {"loss": float("nan"), "num_updates": 0}

    policy.train()
    total_loss = 0.0
    num_updates = 0

    indices = list(range(len(samples)))
    for _epoch in range(num_epochs):
        random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            mini_idx = indices[start: start + batch_size]
            if not mini_idx:
                continue

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

    return {"loss": total_loss / max(num_updates, 1), "num_updates": num_updates}


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
    print(f"[SmolVLA-RECAP] Logging to: {run_dir}", flush=True)

    # ------------------------------------------------------------------ #
    # Load SmolVLA policy
    # ------------------------------------------------------------------ #
    print(f"[SmolVLA-RECAP] Loading policy from: {args_cli.policy}", flush=True)
    policy = SmolVLAPolicy.from_pretrained(args_cli.policy).to(device)

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
                print(f"[SmolVLA-RECAP] Processor fallback source: {src}", flush=True)
            break
        except FileNotFoundError:
            continue
    if preprocess is None:
        raise RuntimeError(
            "Could not load SmolVLA preprocessors from any source.  "
            "Try: --policy lerobot/smolvla_base"
        )

    # ------------------------------------------------------------------ #
    # Build value function (frozen VLM + trainable MLP head)
    # ------------------------------------------------------------------ #
    print("[SmolVLA-RECAP] Building value function.", flush=True)
    value_fn = SmolVLAValueFunction(
        policy=policy,
        device=device,
        gamma=args_cli.gamma,
    )
    value_optimizer = torch.optim.AdamW(value_fn.mlp.parameters(), lr=args_cli.value_lr)

    # ------------------------------------------------------------------ #
    # Optionally add LoRA and freeze backbone
    # ------------------------------------------------------------------ #
    if args_cli.use_lora:
        print(f"[SmolVLA-RECAP] Applying LoRA (rank={args_cli.lora_rank}) to backbone.", flush=True)
        _apply_lora(policy, args_cli.lora_rank)

    param_groups = _get_trainable_params(
        policy,
        use_lora=args_cli.use_lora,
        action_lr=args_cli.lr,
        lora_lr=args_cli.lora_lr,
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    # Resume from checkpoint if requested.
    start_iteration = 0
    if args_cli.resume:
        ckpt = torch.load(args_cli.resume, map_location=device)
        policy.load_state_dict(ckpt["policy_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "value_fn_state_dict" in ckpt:
            value_fn.mlp.load_state_dict(ckpt["value_fn_state_dict"])
            value_optimizer.load_state_dict(ckpt["value_optimizer_state_dict"])
        start_iteration = ckpt.get("iteration", 0) + 1
        print(f"[SmolVLA-RECAP] Resumed from iteration {start_iteration}.", flush=True)

    # ------------------------------------------------------------------ #
    # Create Isaac Lab env
    # ------------------------------------------------------------------ #
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=str(device),
        num_envs=args_cli.num_envs if args_cli.num_envs is not None else None,
    )
    if hasattr(getattr(env_cfg, "scene", None), "ee_frame"):
        env_cfg.scene.ee_frame.debug_vis = False
    if hasattr(getattr(env_cfg, "commands", None), "object_pose"):
        env_cfg.commands.object_pose.debug_vis = False

    step_dt = env_cfg.sim.dt * env_cfg.decimation
    env_cfg.episode_length_s = args_cli.max_episode_steps * step_dt
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    num_envs = env.unwrapped.num_envs

    camera_mapping = _build_camera_mapping(args_cli.camera_mapping)
    print(f"[SmolVLA-RECAP] Camera mapping: {camera_mapping}", flush=True)
    print(f"[SmolVLA-RECAP] num_envs={num_envs}, device={device}", flush=True)

    # Save run config.
    run_cfg_path = os.path.join(run_dir, "train_cfg.json")
    with open(run_cfg_path, "w") as f:
        json.dump(vars(args_cli), f, indent=2, default=str)

    # ------------------------------------------------------------------ #
    # Language instruction strings
    # ------------------------------------------------------------------ #
    # During rollout we always use the "positive" conditioning (deploy mode).
    rollout_instruction = ADVANTAGE_POSITIVE_PREFIX + args_cli.instruction
    base_instruction = args_cli.instruction

    print(
        f"[SmolVLA-RECAP] Starting training for {args_cli.max_iterations} iterations.",
        flush=True,
    )

    # ------------------------------------------------------------------ #
    # Main RECAP training loop
    # ------------------------------------------------------------------ #
    for iteration in range(start_iteration, args_cli.max_iterations):
        iter_start = time.time()

        # ---------------------------------------------------------------- #
        # 1. Rollout collection (ALL episodes, positive conditioning)
        # ---------------------------------------------------------------- #
        buffer = collect_rollouts(
            policy=policy,
            env=env,
            preprocess=preprocess,
            postprocess=postprocess,
            language_instruction=rollout_instruction,
            camera_mapping=camera_mapping,
            num_episodes=args_cli.num_rollout_episodes,
            max_episode_steps=args_cli.max_episode_steps,
            device=device,
            n_arm_joints=args_cli.n_arm_joints,
            n_state_joints=args_cli.n_state_joints,
        )
        rollout_summary = buffer.summary()
        episodes = buffer._completed_episodes  # all episodes, unfiltered

        # ---------------------------------------------------------------- #
        # 2. Train value function V(s) on Monte Carlo returns
        # ---------------------------------------------------------------- #
        vf_metrics = value_fn.train_on_episodes(
            episodes=episodes,
            preprocess=preprocess,
            language_instruction=base_instruction,
            camera_mapping=camera_mapping,
            num_epochs=args_cli.value_epochs,
            batch_size=args_cli.value_batch_size,
            lr=args_cli.value_lr,
            optimizer=value_optimizer,
        )

        # ---------------------------------------------------------------- #
        # 3. Compute advantages and binarize (positive / negative labels)
        # ---------------------------------------------------------------- #
        labeled_dps = value_fn.compute_advantages(
            episodes=episodes,
            preprocess=preprocess,
            language_instruction=base_instruction,
            camera_mapping=camera_mapping,
            positive_percentile=args_cli.positive_percentile,
        )
        buffer.clear()

        # ---------------------------------------------------------------- #
        # 4. Build advantage-conditioned training samples
        # ---------------------------------------------------------------- #
        samples = _build_recap_samples(
            labeled_dps=labeled_dps,
            base_instruction=base_instruction,
            preprocess=preprocess,
            device=device,
            advantage_dropout=args_cli.advantage_dropout,
        )

        n_total_dps = len(labeled_dps)
        n_pos = sum(1 for _, lbl, _ in labeled_dps if lbl == "positive")
        n_neg = n_total_dps - n_pos

        print(
            f"[SmolVLA-RECAP] iter={iteration:04d} | "
            f"episodes={rollout_summary.get('num_episodes', 0)} | "
            f"dps={n_total_dps} (pos={n_pos}, neg={n_neg}) | "
            f"mean_return={rollout_summary.get('mean_return', 0):.3f} | "
            f"success={rollout_summary.get('success_rate', 0)*100:.1f}%",
            flush=True,
        )

        # ---------------------------------------------------------------- #
        # 5. RECAP update (flow-matching MSE on advantage-conditioned data)
        # ---------------------------------------------------------------- #
        update_metrics = _recap_update(
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
        # 6. Logging
        # ---------------------------------------------------------------- #
        global_step = iteration + 1
        writer.add_scalar("rollout/mean_return", rollout_summary.get("mean_return", 0), global_step)
        writer.add_scalar("rollout/max_return", rollout_summary.get("max_return", 0), global_step)
        writer.add_scalar("rollout/success_rate", rollout_summary.get("success_rate", 0), global_step)
        writer.add_scalar("rollout/num_episodes", rollout_summary.get("num_episodes", 0), global_step)
        writer.add_scalar("rollout/num_total_dps", n_total_dps, global_step)
        writer.add_scalar("rollout/num_positive_dps", n_pos, global_step)
        writer.add_scalar("rollout/frac_positive", n_pos / max(n_total_dps, 1), global_step)
        writer.add_scalar("train/loss", update_metrics["loss"], global_step)
        writer.add_scalar("train/num_updates", update_metrics["num_updates"], global_step)
        writer.add_scalar("value_fn/mean_loss", vf_metrics["mean_loss"], global_step)
        writer.add_scalar("value_fn/num_updates", vf_metrics["num_updates"], global_step)
        writer.add_scalar("perf/iter_time_s", iter_elapsed, global_step)

        print(
            f"          loss={update_metrics['loss']:.5f} | "
            f"vf_loss={vf_metrics['mean_loss']:.5f} | "
            f"updates={update_metrics['num_updates']} | "
            f"iter_time={iter_elapsed:.1f}s",
            flush=True,
        )

        # ---------------------------------------------------------------- #
        # 7. Checkpoint
        # ---------------------------------------------------------------- #
        if (iteration + 1) % args_cli.save_interval == 0 or iteration == args_cli.max_iterations - 1:
            ckpt_path = os.path.join(ckpt_dir, f"smolvla_recap_{iteration:05d}.pt")
            torch.save(
                {
                    "iteration": iteration,
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "value_fn_state_dict": value_fn.mlp.state_dict(),
                    "value_optimizer_state_dict": value_optimizer.state_dict(),
                    "rollout_summary": rollout_summary,
                    "train_metrics": update_metrics,
                    "vf_metrics": vf_metrics,
                    "args": vars(args_cli),
                },
                ckpt_path,
            )
            print(f"[SmolVLA-RECAP] Checkpoint saved: {ckpt_path}", flush=True)

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    writer.close()
    env.close()
    print("[SmolVLA-RECAP] Training complete.", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
