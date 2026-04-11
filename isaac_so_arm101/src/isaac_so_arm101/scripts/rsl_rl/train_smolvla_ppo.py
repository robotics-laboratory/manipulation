"""Train SmolVLA via PPO with Flow-SDE and trajectory guidance rewards.

Algorithm overview (piRL-style, arXiv:2510.25889)
--------------------------------------------------
Each iteration:
  1. ROLLOUT: Roll out SmolVLA in the vectorised Isaac Sim env for
     ``num_rollout_episodes`` episodes.  At each decision point, the
     hybrid ODE-SDE sampler produces an action chunk and a tractable
     log π_old(a|s) from ONE stochastic denoising step.  Per-step rewards
     come from trajectory guidance + sparse lift.

  2. GAE: Compute Generalized Advantage Estimates at the chunk (macro-step)
     level.  Each decision point is treated as one MDP transition with
     reward = sum of sub-step rewards during that chunk.

  3. PPO UPDATE: For ``num_ppo_epochs`` epochs, sample mini-batches of
     transitions and apply the clipped surrogate loss + value function loss.
     Only the action head and critic MLP (+ optional LoRA) are updated.

Why not standard PPO?
---------------------
SmolVLA uses Conditional Flow Matching — no tractable log π.  The Flow-SDE
converts the deterministic ODE to a stochastic one, making each denoising
step a Gaussian transition with computable log-probability.  See
``flow_sde.py`` for the SDE math.

Usage (from manipulation/isaac_so_arm101/)::

    isaaclab -p src/isaac_so_arm101/scripts/rsl_rl/train_smolvla_ppo.py \\
        --task Isaac-SO-ARM101-SmolVLA-RL-v0 \\
        --policy igor-saprygin/so101-fixed-layout-smolvla \\
        --instruction "Pick the cube." \\
        --num_envs 8 \\
        --noise_level 0.5 \\
        --num_denoise_steps 10 \\
        --clip_param 0.2 \\
        --lr 5e-6 \\
        --critic_lr 1e-4 \\
        --num_ppo_epochs 4 \\
        --max_iterations 500 \\
        --experiment_name smolvla_ppo_flow_sde \\
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
    description="SmolVLA PPO (Flow-SDE) finetuning with trajectory guidance in Isaac Lab."
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

# --- Rollout ---
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

# --- Flow-SDE hyperparameters ---
parser.add_argument(
    "--noise_level",
    type=float,
    default=0.5,
    help="SDE noise magnitude `a` (controls exploration; 0 = pure ODE). Default 0.5 per piRL.",
)
parser.add_argument(
    "--num_denoise_steps",
    type=int,
    default=10,
    help="Number of denoising steps K. Can reduce to 4 for speed.",
)

# --- PPO hyperparameters ---
parser.add_argument(
    "--clip_param",
    type=float,
    default=0.2,
    help="PPO clipping parameter ε.",
)
parser.add_argument(
    "--value_loss_coef",
    type=float,
    default=0.5,
    help="Coefficient for the value function loss term.",
)
parser.add_argument(
    "--entropy_coef",
    type=float,
    default=0.0,
    help="Entropy bonus coefficient (0 = disabled for SDE policies).",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    help="Discount factor γ.",
)
parser.add_argument(
    "--gae_lambda",
    type=float,
    default=0.95,
    help="GAE λ parameter for advantage estimation.",
)
parser.add_argument(
    "--num_ppo_epochs",
    type=int,
    default=4,
    help="Number of PPO update epochs per iteration.",
)
parser.add_argument(
    "--num_mini_batches",
    type=int,
    default=4,
    help="Number of mini-batches per PPO epoch.",
)
parser.add_argument(
    "--lr",
    type=float,
    default=5e-6,
    help="Learning rate for the action head.",
)
parser.add_argument(
    "--critic_lr",
    type=float,
    default=1e-4,
    help="Learning rate for the critic MLP.",
)
parser.add_argument(
    "--grad_clip",
    type=float,
    default=1.0,
    help="Max gradient norm for clipping.",
)
parser.add_argument(
    "--normalize_advantages",
    action="store_true",
    default=True,
    help="Normalize advantages per mini-batch.",
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
    help="Learning rate for LoRA parameters.",
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
    help="Path to a previous checkpoint to resume from.",
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
    default="smolvla_ppo",
    help="Experiment name for log directory.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, _hydra_args = parser.parse_known_args()
# Cameras must be enabled for SmolVLA visual input.
args_cli.enable_cameras = True

# Force headless when no display is available.
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

import dataclasses
import json
import logging
import math
import random
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import isaaclab as _il_ns
if not getattr(_il_ns, "__file__", None):
    _il_ns.__file__ = next(iter(_il_ns.__path__), __file__)

from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks.lift  # noqa: F401
import isaac_so_arm101.tasks.reach  # noqa: F401

from isaac_so_arm101.tasks.lift.smolvla_rollout import (
    DEFAULT_CAMERA_MAPPING,
    _extract_images_numpy,
    _extract_state_degrees_numpy,
    smolvla_actions_to_env,
)
from isaac_so_arm101.tasks.lift.smolvla_ppo_actor import SmolVLAPPOActor
from log_paths import rsl_rl_root

logger = logging.getLogger(__name__)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# PPO rollout buffer
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PPOTransition:
    """One decision-point transition for PPO training."""

    # Observation (for recomputing log-prob in update step)
    images_np: dict[str, np.ndarray]   # {policy_key: (3, H, W) float32}
    state_np: np.ndarray               # (n_joints,) float32 degrees

    # SDE cache (for efficient log-prob recomputation — only 1 denoise_step needed)
    x_k_sde: torch.Tensor             # CPU (1, chunk_size, max_action_dim)
    x_next_sde: torch.Tensor          # CPU (1, chunk_size, max_action_dim)
    k_sde: int                         # which denoising step was stochastic

    # PPO signals
    old_log_prob: float               # log π_old(a|s)
    value: float                      # V_old(s)
    reward: float                     # sum of chunk sub-step rewards
    done: bool                        # episode terminated/truncated after this chunk


class PPORolloutBuffer:
    """Stores transitions for one PPO iteration and computes GAE advantages."""

    def __init__(self) -> None:
        self._transitions: list[PPOTransition] = []
        self._episode_returns: list[float] = []
        self._episode_successes: list[bool] = []

        # Per-env accumulators
        self._in_progress: dict[int, list[PPOTransition]] = defaultdict(list)
        self._in_progress_return: dict[int, float] = defaultdict(float)
        self._in_progress_success: dict[int, bool] = defaultdict(bool)

    def add_transition(self, env_id: int, t: PPOTransition) -> None:
        self._in_progress[env_id].append(t)
        self._in_progress_return[env_id] += t.reward

    def mark_success(self, env_id: int) -> None:
        self._in_progress_success[env_id] = True

    def finish_episode(self, env_ids: list[int]) -> None:
        for eid in env_ids:
            ts = self._in_progress.get(eid)
            if not ts:
                continue
            self._transitions.extend(ts)
            self._episode_returns.append(self._in_progress_return[eid])
            self._episode_successes.append(self._in_progress_success[eid])
            self._in_progress[eid] = []
            self._in_progress_return[eid] = 0.0
            self._in_progress_success[eid] = False

    def num_transitions(self) -> int:
        return len(self._transitions)

    def summary(self) -> dict[str, float]:
        if not self._episode_returns:
            return {"num_episodes": 0, "mean_return": 0.0, "success_rate": 0.0}
        return {
            "num_episodes": len(self._episode_returns),
            "mean_return": float(np.mean(self._episode_returns)),
            "max_return": float(np.max(self._episode_returns)),
            "min_return": float(np.min(self._episode_returns)),
            "success_rate": float(np.mean(self._episode_successes)),
        }

    def compute_gae(
        self,
        last_values: dict[int, float],
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute GAE advantages and returns in-place for all completed transitions.

        Args:
            last_values: V(s) at the step AFTER the last recorded transition per env.
                         Use 0.0 for terminal states.
            gamma:       Discount factor.
            gae_lambda:  GAE lambda.
        """
        # Process all transitions in order.
        # Since episodes may be interleaved across envs, we process per-episode.
        # We group self._transitions by episode boundary (done=True).
        episodes: list[list[PPOTransition]] = []
        current: list[PPOTransition] = []
        for t in self._transitions:
            current.append(t)
            if t.done:
                episodes.append(current)
                current = []
        if current:
            episodes.append(current)

        self._advantages: list[float] = []
        self._returns: list[float] = []

        for ep in episodes:
            gae = 0.0
            ep_adv: list[float] = [0.0] * len(ep)
            ep_ret: list[float] = [0.0] * len(ep)

            next_val = 0.0  # terminal
            for i in reversed(range(len(ep))):
                t = ep[i]
                td_error = t.reward + gamma * next_val * (1.0 - float(t.done)) - t.value
                gae = td_error + gamma * gae_lambda * (1.0 - float(t.done)) * gae
                ep_adv[i] = gae
                ep_ret[i] = gae + t.value
                next_val = t.value

            self._advantages.extend(ep_adv)
            self._returns.extend(ep_ret)

    def get_mini_batches(
        self,
        num_mini_batches: int,
        preprocess,
        language_instruction: str,
        camera_mapping: dict[str, str],
        device: torch.device,
        normalize_advantages: bool = True,
    ) -> list[list[dict]]:
        """Shuffle transitions and split into mini-batches.

        Each element of the inner list is a dict ready for
        ``SmolVLAPPOActor.recompute_logprob_and_value``.

        Returns a list of mini-batches, where each mini-batch is a list of
        per-transition dicts.
        """
        from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

        n = len(self._transitions)
        indices = list(range(n))
        random.shuffle(indices)

        adv_arr = np.array(self._advantages, dtype=np.float32)
        ret_arr = np.array(self._returns, dtype=np.float32)

        if normalize_advantages and len(adv_arr) > 1:
            adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        mb_size = max(1, n // num_mini_batches)
        mini_batches: list[list[dict]] = []

        for start in range(0, n, mb_size):
            mb_indices = indices[start: start + mb_size]
            if not mb_indices:
                continue

            items: list[dict] = []
            for idx in mb_indices:
                t = self._transitions[idx]

                # Build preprocessed observation dict.
                frame: dict = {
                    "language_instruction": language_instruction,
                    "task": language_instruction,
                }
                for policy_key, img_np in t.images_np.items():
                    frame[policy_key] = np.expand_dims(img_np.astype(np.float32), axis=0)
                state_arr = t.state_np[:6] if t.state_np.shape[0] >= 6 else t.state_np
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

                items.append({
                    "obs_batch": tensor_batch,
                    "x_k_sde": t.x_k_sde,                       # CPU
                    "x_next_sde": t.x_next_sde,                  # CPU
                    "k_sde": t.k_sde,
                    "old_log_prob": torch.tensor(t.old_log_prob, dtype=torch.float32, device=device),
                    "advantage": torch.tensor(adv_arr[idx], dtype=torch.float32, device=device),
                    "return_": torch.tensor(ret_arr[idx], dtype=torch.float32, device=device),
                })

            mini_batches.append(items)

        return mini_batches

    def clear(self) -> None:
        self._transitions.clear()
        self._episode_returns.clear()
        self._episode_successes.clear()
        self._in_progress.clear()
        self._in_progress_return.clear()
        self._in_progress_success.clear()


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------


def collect_ppo_rollouts(
    actor: SmolVLAPPOActor,
    env,
    preprocess,
    postprocess,
    language_instruction: str,
    camera_mapping: dict[str, str],
    num_episodes: int,
    max_episode_steps: int,
    device: torch.device,
    n_arm_joints: int = 5,
    n_state_joints: int = 6,
    min_lift_height: float = 0.025,
) -> PPORolloutBuffer:
    """Roll out SmolVLA (with Flow-SDE) and collect PPO transitions.

    At each decision point, ``SmolVLAPPOActor.sample_actions_with_logprob``
    is called to produce an action chunk and a log-probability.

    Returns:
        PPORolloutBuffer with all completed episode transitions.
        Note: GAE is NOT computed here; call buffer.compute_gae() afterwards.
    """
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

    env_unwrapped = env.unwrapped
    num_envs = env_unwrapped.num_envs
    chunk_size = actor.policy.config.chunk_size
    action_dim = actor.policy.config.action_feature.shape[0]

    buffer = PPORolloutBuffer()
    episodes_completed = 0
    episode_step = torch.zeros(num_envs, dtype=torch.long)

    env.reset()
    print(
        f"[SmolVLA-PPO] Rollout start: envs={num_envs}, "
        f"chunk_size={chunk_size}, target_eps={num_episodes}",
        flush=True,
    )

    while episodes_completed < num_episodes:
        # -------------------------------------------------------------- #
        # 1. Build per-env batches and generate action chunks with log-prob
        # -------------------------------------------------------------- #
        per_env_data: list[dict] = []
        processed_frames = []

        for env_id in range(num_envs):
            images_np = _extract_images_numpy(env_unwrapped, env_id, camera_mapping)
            state_np = _extract_state_degrees_numpy(env_unwrapped, env_id, n_state_joints)

            frame: dict = {
                "language_instruction": language_instruction,
                "task": language_instruction,
            }
            for policy_key, img_np in images_np.items():
                frame[policy_key] = np.expand_dims(img_np, axis=0)
            frame[str(OBS_STATE)] = np.expand_dims(state_np, axis=0)

            per_env_data.append({"images_np": images_np, "state_np": state_np})
            processed_frames.append(preprocess(frame))

        # Stack into a single batched dict.
        batched: dict[str, Any] = {}
        for key in processed_frames[0]:
            vals = []
            for pf in processed_frames:
                v = pf[key]
                if isinstance(v, np.ndarray):
                    vals.append(torch.as_tensor(v, device=device))
                elif isinstance(v, torch.Tensor):
                    vals.append(v.to(device))
                else:
                    vals.append(v)
            if isinstance(vals[0], torch.Tensor):
                batched[key] = torch.cat(vals, dim=0)
            else:
                batched[key] = vals[0]

        # Hybrid ODE-SDE sampling → action_chunk, log_prob, value, sde_cache.
        actor.policy.eval()
        out = actor.sample_actions_with_logprob(batched, device)

        action_chunks = out["action_chunk"]   # (B, chunk_size, action_dim)
        log_probs = out["log_prob"]           # (B,) CPU
        values = out["value"]                  # (B,) CPU
        k_sde = out["k_sde"]
        x_k_sde_batch = out["x_k_sde"]       # (B, chunk_size, max_action_dim) CPU
        x_next_sde_batch = out["x_next_sde"] # (B, chunk_size, max_action_dim) CPU

        # -------------------------------------------------------------- #
        # 2. Execute the chunk step-by-step, accumulating rewards
        # -------------------------------------------------------------- #
        chunk_rewards = torch.zeros(num_envs, device=device)
        env_reset_at: dict[int, int] = {}

        for sub_step in range(chunk_size):
            step_action_norm = action_chunks[:, sub_step, :]
            try:
                step_action_deg = postprocess(step_action_norm)
                if not isinstance(step_action_deg, torch.Tensor):
                    step_action_deg = torch.as_tensor(step_action_deg, device=device, dtype=torch.float32)
                elif step_action_deg.device != device:
                    step_action_deg = step_action_deg.to(device)
                if step_action_deg.shape != step_action_norm.shape:
                    raise ValueError("shape mismatch")
            except Exception:
                per_env = []
                for env_id_p in range(num_envs):
                    a = postprocess(step_action_norm[env_id_p])
                    if not isinstance(a, torch.Tensor):
                        a = torch.as_tensor(a, dtype=torch.float32)
                    per_env.append(a.to(device))
                step_action_deg = torch.stack(per_env, dim=0)

            env_action = smolvla_actions_to_env(step_action_deg, n_arm_joints=n_arm_joints)
            env_action = env_action.clamp(-1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(env_action)
            reward = reward.to(device=device, dtype=torch.float32)
            done = (terminated | truncated).to(device=device)

            for env_id in range(num_envs):
                if env_id not in env_reset_at:
                    chunk_rewards[env_id] += reward[env_id]

            episode_step += 1

            # Track success.
            try:
                obj = env_unwrapped.scene.rigid_objects["object"]
                lifted = obj.data.root_pos_w[:, 2] > min_lift_height
                for env_id in range(num_envs):
                    if env_id not in env_reset_at and lifted[env_id].item():
                        buffer.mark_success(env_id)
            except (KeyError, AttributeError):
                pass

            for env_id in range(num_envs):
                if done[env_id].item() and env_id not in env_reset_at:
                    env_reset_at[env_id] = sub_step

        # -------------------------------------------------------------- #
        # 3. Store PPO transitions
        # -------------------------------------------------------------- #
        for env_id in range(num_envs):
            is_done = env_id in env_reset_at
            t = PPOTransition(
                images_np=per_env_data[env_id]["images_np"],
                state_np=per_env_data[env_id]["state_np"],
                x_k_sde=x_k_sde_batch[env_id: env_id + 1].cpu(),    # (1, chunk_size, max_action_dim)
                x_next_sde=x_next_sde_batch[env_id: env_id + 1].cpu(),
                k_sde=k_sde,
                old_log_prob=log_probs[env_id].item(),
                value=values[env_id].item(),
                reward=chunk_rewards[env_id].item(),
                done=is_done,
            )
            buffer.add_transition(env_id, t)

        # -------------------------------------------------------------- #
        # 4. Finish episodes for resets and force-done envs
        # -------------------------------------------------------------- #
        done_now = list(env_reset_at.keys())
        force_done = [
            env_id
            for env_id in range(num_envs)
            if episode_step[env_id].item() >= max_episode_steps and env_id not in done_now
        ]
        all_done = done_now + force_done

        if all_done:
            # Mark the last transition for each done env as done=True.
            for eid in all_done:
                if buffer._in_progress.get(eid):
                    buffer._in_progress[eid][-1] = dataclasses.replace(
                        buffer._in_progress[eid][-1], done=True
                    )
            buffer.finish_episode(all_done)
            episodes_completed += len(all_done)
            for env_id in all_done:
                episode_step[env_id] = 0

    # Flush remaining in-progress episodes.
    for eid in range(num_envs):
        if buffer._in_progress.get(eid):
            buffer._in_progress[eid][-1] = dataclasses.replace(
                buffer._in_progress[eid][-1], done=True
            )
    buffer.finish_episode(list(range(num_envs)))

    summ = buffer.summary()
    print(
        f"[SmolVLA-PPO] Rollout done: episodes={summ['num_episodes']}, "
        f"transitions={buffer.num_transitions()}, "
        f"mean_return={summ['mean_return']:.3f}, "
        f"success_rate={summ.get('success_rate', 0)*100:.1f}%",
        flush=True,
    )
    return buffer


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------


def ppo_update(
    actor: SmolVLAPPOActor,
    mini_batches_per_epoch: list[list[dict]],
    optimizer: torch.optim.Optimizer,
    clip_param: float,
    value_loss_coef: float,
    grad_clip: float,
    device: torch.device,
) -> dict[str, float]:
    """Run one PPO update epoch on the provided mini-batches.

    Args:
        actor:                 SmolVLAPPOActor with current weights.
        mini_batches_per_epoch: List of mini-batches (from PPORolloutBuffer.get_mini_batches).
        optimizer:             Optimizer targeting action head + critic params.
        clip_param:            PPO clipping ε.
        value_loss_coef:       Weight for the value function loss.
        grad_clip:             Max gradient norm.
        device:                Compute device.

    Returns:
        Dict with mean policy_loss, value_loss, total_loss, clip_fraction, approx_kl.
    """
    actor.policy.train()
    actor.critic.train()

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_loss_sum = 0.0
    total_clip_frac = 0.0
    total_kl = 0.0
    n_updates = 0

    for mb_items in mini_batches_per_epoch:
        if not mb_items:
            continue

        # Collate mini-batch tensors.
        old_log_probs = torch.stack([item["old_log_prob"] for item in mb_items]).to(device)  # (M,)
        advantages = torch.stack([item["advantage"] for item in mb_items]).to(device)        # (M,)
        returns = torch.stack([item["return_"] for item in mb_items]).to(device)             # (M,)

        # Recompute log π_new and V_new for each sample individually.
        # This is necessary because each sample may have a different k_sde and
        # the prefix KV cache depends on the observation.
        new_log_probs_list = []
        new_values_list = []

        for item in mb_items:
            obs = item["obs_batch"]
            x_k = item["x_k_sde"]    # CPU (1, chunk_size, max_action_dim)
            x_next = item["x_next_sde"]  # CPU (1, chunk_size, max_action_dim)
            k_sde = item["k_sde"]

            new_lp, new_v = actor.recompute_logprob_and_value(
                batch=obs,
                x_k_sde=x_k,
                x_next_sde=x_next,
                k_sde=k_sde,
                device=device,
            )
            new_log_probs_list.append(new_lp)   # (1,)
            new_values_list.append(new_v)        # (1,)

        new_log_probs = torch.cat(new_log_probs_list, dim=0)  # (M,)
        new_values = torch.cat(new_values_list, dim=0)         # (M,)

        # Policy (actor) loss — clipped surrogate.
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))  # numerical stability

        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        # Value function loss.
        value_loss = torch.nn.functional.mse_loss(new_values, returns)

        loss = policy_loss + value_loss_coef * value_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            all_params = [
                p for group in optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=grad_clip)
        optimizer.step()

        # Diagnostics.
        with torch.no_grad():
            clip_frac = ((ratio - 1.0).abs() > clip_param).float().mean().item()
            approx_kl = (old_log_probs - new_log_probs).mean().item()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_loss_sum += loss.item()
        total_clip_frac += clip_frac
        total_kl += approx_kl
        n_updates += 1

    n = max(n_updates, 1)
    return {
        "policy_loss": total_policy_loss / n,
        "value_loss": total_value_loss / n,
        "total_loss": total_loss_sum / n,
        "clip_fraction": total_clip_frac / n,
        "approx_kl": total_kl / n,
        "num_updates": n_updates,
    }


# ---------------------------------------------------------------------------
# Camera mapping helper
# ---------------------------------------------------------------------------


def _build_camera_mapping(raw: str | None) -> dict[str, str]:
    if raw:
        return json.loads(raw)
    return dict(DEFAULT_CAMERA_MAPPING)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
    print(f"[SmolVLA-PPO] Logging to: {run_dir}", flush=True)

    # ------------------------------------------------------------------ #
    # Load SmolVLA policy
    # ------------------------------------------------------------------ #
    print(f"[SmolVLA-PPO] Loading policy: {args_cli.policy}", flush=True)
    policy = SmolVLAPolicy.from_pretrained(args_cli.policy).to(device)

    # Build preprocessors.
    preprocess = postprocess = None
    for src in [args_cli.policy, "lerobot/smolvla_base"]:
        try:
            preprocess, postprocess = make_pre_post_processors(
                policy.config,
                src,
                preprocessor_overrides={"device_processor": {"device": str(device)}},
            )
            if src != args_cli.policy:
                print(f"[SmolVLA-PPO] Preprocessor fallback: {src}", flush=True)
            break
        except FileNotFoundError:
            continue
    if preprocess is None:
        raise RuntimeError("Could not load SmolVLA preprocessors. Try: --policy lerobot/smolvla_base")

    # ------------------------------------------------------------------ #
    # Build actor-critic
    # ------------------------------------------------------------------ #
    actor = SmolVLAPPOActor(
        policy=policy,
        noise_level=args_cli.noise_level,
        num_steps=args_cli.num_denoise_steps,
        lora_rank=args_cli.lora_rank if args_cli.use_lora else 0,
    )
    actor.critic.to(device)

    # Freeze backbone, get trainable param groups.
    param_groups = actor.get_trainable_param_groups(
        action_lr=args_cli.lr,
        critic_lr=args_cli.critic_lr,
        lora_lr=args_cli.lora_lr,
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    # ------------------------------------------------------------------ #
    # Resume from checkpoint
    # ------------------------------------------------------------------ #
    start_iteration = 0
    if args_cli.resume:
        ckpt = torch.load(args_cli.resume, map_location=device)
        policy.load_state_dict(ckpt.get("policy_state_dict", {}), strict=False)
        actor.critic.load_state_dict(ckpt.get("critic_state_dict", {}), strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iteration = ckpt.get("iteration", 0) + 1
        print(f"[SmolVLA-PPO] Resumed from iteration {start_iteration}.", flush=True)

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
    print(f"[SmolVLA-PPO] num_envs={num_envs}, K={actor.K}, noise_level={actor.noise_level}", flush=True)

    # Save run config.
    with open(os.path.join(run_dir, "train_cfg.json"), "w") as f:
        json.dump(vars(args_cli), f, indent=2, default=str)

    # ------------------------------------------------------------------ #
    # Main PPO training loop
    # ------------------------------------------------------------------ #
    print(f"[SmolVLA-PPO] Starting training for {args_cli.max_iterations} iterations.", flush=True)

    for iteration in range(start_iteration, args_cli.max_iterations):
        iter_start = time.time()

        # ------------------------------------------------------------ #
        # 1. Rollout collection
        # ------------------------------------------------------------ #
        buffer = collect_ppo_rollouts(
            actor=actor,
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

        # ------------------------------------------------------------ #
        # 2. GAE computation
        # ------------------------------------------------------------ #
        buffer.compute_gae(
            last_values={},  # 0.0 for all terminal states
            gamma=args_cli.gamma,
            gae_lambda=args_cli.gae_lambda,
        )

        # ------------------------------------------------------------ #
        # 3. PPO update (num_ppo_epochs passes)
        # ------------------------------------------------------------ #
        all_update_metrics = []
        for epoch in range(args_cli.num_ppo_epochs):
            mini_batches = buffer.get_mini_batches(
                num_mini_batches=args_cli.num_mini_batches,
                preprocess=preprocess,
                language_instruction=args_cli.instruction,
                camera_mapping=camera_mapping,
                device=device,
                normalize_advantages=args_cli.normalize_advantages,
            )
            metrics = ppo_update(
                actor=actor,
                mini_batches_per_epoch=mini_batches,
                optimizer=optimizer,
                clip_param=args_cli.clip_param,
                value_loss_coef=args_cli.value_loss_coef,
                grad_clip=args_cli.grad_clip,
                device=device,
            )
            all_update_metrics.append(metrics)

        buffer.clear()

        # Average metrics over epochs.
        n_e = len(all_update_metrics)
        avg_metrics = {
            k: sum(m[k] for m in all_update_metrics) / max(n_e, 1)
            for k in all_update_metrics[0]
        } if all_update_metrics else {}

        iter_elapsed = time.time() - iter_start

        # ------------------------------------------------------------ #
        # 4. Logging
        # ------------------------------------------------------------ #
        gs = iteration + 1
        writer.add_scalar("rollout/mean_return", rollout_summary.get("mean_return", 0), gs)
        writer.add_scalar("rollout/max_return", rollout_summary.get("max_return", 0), gs)
        writer.add_scalar("rollout/success_rate", rollout_summary.get("success_rate", 0), gs)
        writer.add_scalar("rollout/num_episodes", rollout_summary.get("num_episodes", 0), gs)
        writer.add_scalar("train/policy_loss", avg_metrics.get("policy_loss", 0), gs)
        writer.add_scalar("train/value_loss", avg_metrics.get("value_loss", 0), gs)
        writer.add_scalar("train/total_loss", avg_metrics.get("total_loss", 0), gs)
        writer.add_scalar("train/clip_fraction", avg_metrics.get("clip_fraction", 0), gs)
        writer.add_scalar("train/approx_kl", avg_metrics.get("approx_kl", 0), gs)
        writer.add_scalar("perf/iter_time_s", iter_elapsed, gs)

        print(
            f"[SmolVLA-PPO] iter={iteration:04d} | "
            f"eps={rollout_summary.get('num_episodes', 0)} | "
            f"mean_ret={rollout_summary.get('mean_return', 0):.3f} | "
            f"success={rollout_summary.get('success_rate', 0)*100:.1f}% | "
            f"pi_loss={avg_metrics.get('policy_loss', 0):.5f} | "
            f"v_loss={avg_metrics.get('value_loss', 0):.5f} | "
            f"kl={avg_metrics.get('approx_kl', 0):.4f} | "
            f"clip={avg_metrics.get('clip_fraction', 0):.3f} | "
            f"t={iter_elapsed:.1f}s",
            flush=True,
        )

        # ------------------------------------------------------------ #
        # 5. Checkpoint
        # ------------------------------------------------------------ #
        if (iteration + 1) % args_cli.save_interval == 0 or iteration == args_cli.max_iterations - 1:
            ckpt_path = os.path.join(ckpt_dir, f"smolvla_ppo_{iteration:05d}.pt")
            torch.save(
                {
                    "iteration": iteration,
                    "policy_state_dict": policy.state_dict(),
                    "critic_state_dict": actor.critic.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "rollout_summary": rollout_summary,
                    "train_metrics": avg_metrics,
                    "args": vars(args_cli),
                },
                ckpt_path,
            )
            print(f"[SmolVLA-PPO] Checkpoint: {ckpt_path}", flush=True)

    writer.close()
    env.close()
    print("[SmolVLA-PPO] Training complete.", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
