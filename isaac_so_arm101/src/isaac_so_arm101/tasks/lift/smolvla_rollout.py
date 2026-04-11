"""SmolVLA rollout collection utilities for RWR (Reward-Weighted Regression) finetuning.

Overview
--------
At each *decision point* (every ``chunk_size`` env steps), SmolVLA generates a full
action chunk from current cameras + joint state.  The chunk is executed step-by-step in
the Isaac Lab env, and per-step rewards (trajectory guidance + sparse lift) accumulate.
After enough episodes, the ``SmolVLARolloutBuffer`` filters the top-K% by episodic return
and returns training-ready batches for the flow-matching update.

Normalization contract
----------------------
* Images: stored as ``(3, H, W)`` float32 in [0, 1].  ``SmolVLAPolicy.prepare_images``
  (called inside ``policy.forward``) resizes and normalises to [-1, 1] internally.
* State: stored as raw joint positions in **degrees** (SO-101 convention).
  ``preprocess`` (from ``make_pre_post_processors``) is used to handle dataset-specific
  mean/std normalisation before passing to the policy.
* Action chunks: stored as the raw output of ``predict_action_chunk`` (model-output
  space, post-denoising, pre-postprocess).  These are used directly as training targets
  in ``policy.forward`` (same normalisation as the offline training pipeline).
"""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Arm joints for SO-101 (shoulder_pan…wrist_roll); gripper is the last DOF.
SO101_N_ARM_JOINTS = 5
_DEG2RAD = math.pi / 180.0

# Isaac sensor names → policy image-key names (can be overridden by callers)
DEFAULT_CAMERA_MAPPING: dict[str, str] = {
    "camera_top": "observation.images.camera1",
    "camera_wrist": "observation.images.camera2",
}


# ---------------------------------------------------------------------------
# Buffer data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DecisionPointData:
    """All data recorded at one SmolVLA decision point (one chunk generation) for ONE env."""

    # Raw image per policy camera key: {policy_key: (3, H, W)} float32 [0, 1] numpy
    images_np: dict[str, np.ndarray]
    # Joint positions in degrees: (n_joints,) float32 numpy
    state_np: np.ndarray
    # Action chunk (model-output space, normalised): (chunk_size, action_dim) float32
    action_chunk: torch.Tensor
    # Sum of step rewards collected while executing this chunk
    chunk_reward: float
    # Step index within the episode at which this decision point occurred
    step_in_episode: int


@dataclasses.dataclass
class EpisodeData:
    """Complete episode data for a single environment."""

    decision_points: list[DecisionPointData]
    total_return: float
    success: bool  # cube was lifted at least once


class SmolVLARolloutBuffer:
    """Accumulates decision-point data across envs and episodes.

    After collection, call ``get_filtered_batches`` to obtain training samples
    from the top-K% episodes sorted by episode return.
    """

    def __init__(self) -> None:
        self._in_progress: dict[int, list[DecisionPointData]] = defaultdict(list)
        self._in_progress_reward: dict[int, float] = defaultdict(float)
        self._in_progress_success: dict[int, bool] = defaultdict(bool)
        self._completed_episodes: list[EpisodeData] = []

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def add_decision_point(self, env_id: int, dp: DecisionPointData) -> None:
        self._in_progress[env_id].append(dp)
        self._in_progress_reward[env_id] += dp.chunk_reward

    def mark_success(self, env_id: int) -> None:
        self._in_progress_success[env_id] = True

    def finish_episode(self, env_ids: list[int]) -> None:
        """Finalise episodes for the given env IDs and move them to completed list."""
        for env_id in env_ids:
            if not self._in_progress[env_id]:
                continue
            ep = EpisodeData(
                decision_points=list(self._in_progress[env_id]),
                total_return=self._in_progress_reward[env_id],
                success=self._in_progress_success[env_id],
            )
            self._completed_episodes.append(ep)
            self._in_progress[env_id] = []
            self._in_progress_reward[env_id] = 0.0
            self._in_progress_success[env_id] = False

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def num_episodes(self) -> int:
        return len(self._completed_episodes)

    def summary(self) -> dict[str, Any]:
        if not self._completed_episodes:
            return {"num_episodes": 0, "mean_return": 0.0, "success_rate": 0.0}
        returns = [ep.total_return for ep in self._completed_episodes]
        successes = [ep.success for ep in self._completed_episodes]
        return {
            "num_episodes": len(self._completed_episodes),
            "mean_return": sum(returns) / len(returns),
            "max_return": max(returns),
            "min_return": min(returns),
            "success_rate": sum(successes) / len(successes),
        }

    def get_filtered_batches(
        self,
        top_k_percentile: float,
        preprocess,
        language_instruction: str,
        camera_mapping: dict[str, str],
        device: torch.device,
        min_episodes: int = 1,
    ) -> list[dict[str, torch.Tensor]]:
        """Return per-decision-point training dicts from top-K% episodes.

        Each dict has the following keys ready for ``SmolVLAPolicy.forward``:
            ``observation.images.*``  (3, H, W) float32 [0,1] wrapped in (1, …)
            ``observation.state``     (1, state_dim) – preprocessed (normalised)
            ``observation.language_tokens``           (1, seq_len) int64
            ``observation.language_attention_mask``   (1, seq_len) bool
            ``action``                (1, chunk_size, action_dim)

        ``preprocess`` is the function from ``make_pre_post_processors``.  It is called
        once per (env, decision-point) sample so normalisation stats are applied correctly.
        """
        from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

        if not self._completed_episodes:
            return []

        returns = [ep.total_return for ep in self._completed_episodes]
        sorted_returns = sorted(returns, reverse=True)
        k = max(min_episodes, math.ceil(len(sorted_returns) * top_k_percentile / 100.0))
        threshold = sorted_returns[min(k - 1, len(sorted_returns) - 1)]

        samples: list[dict[str, torch.Tensor]] = []
        for ep in self._completed_episodes:
            if ep.total_return < threshold:
                continue
            for dp in ep.decision_points:
                # Build a single-env frame dict that preprocess understands.
                frame: dict[str, Any] = {
                    "language_instruction": language_instruction,
                    "task": language_instruction,
                }
                for policy_key, img_np in dp.images_np.items():
                    # preprocess expects (1, 3, H, W) float32 in [0, 1].
                    frame[policy_key] = np.expand_dims(img_np.astype(np.float32), axis=0)

                # State: (1, state_dim) float32 degrees (preprocess normalises).
                state_truncated = dp.state_np[:6] if dp.state_np.shape[0] >= 6 else dp.state_np
                frame[str(OBS_STATE)] = np.expand_dims(state_truncated.astype(np.float32), axis=0)

                # Apply policy preprocessor (tokenise lang, normalise state/images).
                batch = preprocess(frame)

                # Convert numpy values to tensors on the target device.
                tensor_batch: dict[str, torch.Tensor] = {}
                for key, val in batch.items():
                    if isinstance(val, np.ndarray):
                        tensor_batch[key] = torch.as_tensor(val, device=device)
                    elif isinstance(val, torch.Tensor):
                        tensor_batch[key] = val.to(device)
                    else:
                        tensor_batch[key] = val  # strings, etc.

                # Add action chunk (model-output space): (1, chunk_size, action_dim).
                tensor_batch[str(ACTION)] = dp.action_chunk.unsqueeze(0).to(device)

                samples.append(tensor_batch)
        return samples

    def clear(self) -> None:
        self._completed_episodes.clear()
        self._in_progress.clear()
        self._in_progress_reward.clear()
        self._in_progress_success.clear()


# ---------------------------------------------------------------------------
# Isaac env helpers
# ---------------------------------------------------------------------------


def _extract_images_numpy(
    env_unwrapped: "ManagerBasedRLEnv",
    env_id: int,
    camera_mapping: dict[str, str],
) -> dict[str, np.ndarray]:
    """Extract (3, H, W) float32 [0, 1] numpy image arrays from Isaac camera sensors."""
    images: dict[str, np.ndarray] = {}
    for sensor_name, policy_key in camera_mapping.items():
        sensor = env_unwrapped.scene.sensors.get(sensor_name)
        if sensor is None:
            continue
        rgb = sensor.data.output["rgb"]  # (B, H, W, 3|4) uint8 or float tensor
        # Grab the single env frame.
        frame = rgb[env_id]  # (H, W, C)
        frame_np = frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame)
        frame_np = frame_np[:, :, :3]  # drop alpha if present
        if frame_np.dtype == np.uint8:
            img_chw = np.transpose(frame_np, (2, 0, 1)).astype(np.float32) / 255.0
        else:
            img_chw = np.transpose(frame_np, (2, 0, 1)).astype(np.float32)
            if img_chw.max() > 1.5:
                img_chw = img_chw / 255.0
        images[policy_key] = img_chw  # (3, H, W) float32 [0, 1]
    return images


def _extract_state_degrees_numpy(
    env_unwrapped: "ManagerBasedRLEnv",
    env_id: int,
    n_joints: int = 6,
) -> np.ndarray:
    """Return joint positions in degrees for one env as a (n_joints,) float32 numpy array."""
    robot = env_unwrapped.scene.articulations["robot"]
    joint_pos_rad = robot.data.joint_pos[env_id]  # (n_joints,) radians tensor
    joint_pos_deg = torch.rad2deg(joint_pos_rad).detach().cpu().numpy().astype(np.float32)
    if joint_pos_deg.shape[0] >= n_joints:
        return joint_pos_deg[:n_joints]
    return np.pad(joint_pos_deg, (0, n_joints - joint_pos_deg.shape[0]))


def smolvla_actions_to_env(
    actions: torch.Tensor,
    n_arm_joints: int = SO101_N_ARM_JOINTS,
) -> torch.Tensor:
    """Convert SmolVLA actions (degrees for arm) to Isaac Lab env actions (radians for arm).

    Args:
        actions: (..., action_dim) in SmolVLA space (degrees for first n_arm_joints).
        n_arm_joints: Number of non-gripper joints.

    Returns:
        Tensor with arm joints in radians, gripper unchanged.
    """
    env_actions = actions.clone().float()
    env_actions[..., :n_arm_joints] = actions[..., :n_arm_joints] * _DEG2RAD
    return env_actions


# ---------------------------------------------------------------------------
# Main rollout function
# ---------------------------------------------------------------------------


def collect_rollouts(
    policy,  # SmolVLAPolicy
    env,
    preprocess,
    postprocess,
    language_instruction: str,
    camera_mapping: dict[str, str],
    num_episodes: int,
    max_episode_steps: int,
    device: torch.device,
    n_arm_joints: int = SO101_N_ARM_JOINTS,
    n_state_joints: int = 6,
    min_lift_height: float = 0.025,
) -> SmolVLARolloutBuffer:
    """Roll out SmolVLA in the vectorised Isaac Sim env and collect RWR training data.

    Decision points occur every ``policy.config.chunk_size`` env steps (execute the full
    chunk, one action at a time, then re-plan).  Per-step rewards come from the env's
    trajectory-guidance reward terms.  Episode return = sum of all step rewards.

    Args:
        policy:             Loaded SmolVLAPolicy (set to eval mode by predict_action_chunk).
        env:                Isaac Lab gymnasium env (NOT wrapped by RslRlVecEnvWrapper).
        preprocess:         Function from make_pre_post_processors – normalises inputs.
        postprocess:        Function from make_pre_post_processors – denormalises outputs.
        language_instruction: Natural language task description.
        camera_mapping:     {isaac_sensor_name: policy_image_key}.
        num_episodes:       Total completed episodes to collect across all envs.
        max_episode_steps:  Force-reset after this many steps per env.
        device:             Compute device (should match env device).
        n_arm_joints:       Number of arm (non-gripper) joints for degree→radian conversion.
        n_state_joints:     Number of joints to extract from robot state.
        min_lift_height:    Cube Z height (m) above which success is flagged.

    Returns:
        ``SmolVLARolloutBuffer`` with all completed episodes.
    """
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

    env_unwrapped = env.unwrapped
    num_envs = env_unwrapped.num_envs
    chunk_size = policy.config.chunk_size
    action_dim = policy.config.action_feature.shape[0]

    buffer = SmolVLARolloutBuffer()
    episodes_completed = 0
    episode_step = torch.zeros(num_envs, dtype=torch.long)

    # Reset the environment to start.
    env.reset()

    print(
        f"[SmolVLA-RL] Starting rollout collection: "
        f"num_envs={num_envs}, chunk_size={chunk_size}, "
        f"action_dim={action_dim}, target_episodes={num_episodes}",
        flush=True,
    )

    while episodes_completed < num_episodes:
        # ------------------------------------------------------------------ #
        # 1. Build per-env batches and generate action chunks.
        # ------------------------------------------------------------------ #
        per_env_frames: list[dict] = []
        for env_id in range(num_envs):
            images_np = _extract_images_numpy(env_unwrapped, env_id, camera_mapping)
            state_np = _extract_state_degrees_numpy(env_unwrapped, env_id, n_state_joints)

            frame: dict = {
                "language_instruction": language_instruction,
                "task": language_instruction,
            }
            for policy_key, img_np in images_np.items():
                frame[policy_key] = np.expand_dims(img_np, axis=0)  # (1, 3, H, W)
            frame[str(OBS_STATE)] = np.expand_dims(state_np, axis=0)  # (1, state_dim)
            per_env_frames.append({"frame": frame, "images_np": images_np, "state_np": state_np})

        # Pre-process each env's frame and stack into a single batched dict.
        batched: dict[str, Any] = {}
        processed_frames = [preprocess(ef["frame"]) for ef in per_env_frames]
        for key in processed_frames[0]:
            vals = []
            for pf in processed_frames:
                v = pf[key]
                if isinstance(v, np.ndarray):
                    vals.append(torch.as_tensor(v, device=device))
                elif isinstance(v, torch.Tensor):
                    vals.append(v.to(device))
                else:
                    vals.append(v)  # e.g. string keys
            if isinstance(vals[0], torch.Tensor):
                batched[key] = torch.cat(vals, dim=0)  # (B, ...)
            else:
                batched[key] = vals[0]  # strings: same for all envs

        # Generate full action chunk for all envs: (B, chunk_size, action_dim).
        action_chunks = policy.predict_action_chunk(batched)  # normalised model space
        # predict_action_chunk calls self.eval() internally.

        # ------------------------------------------------------------------ #
        # 2. Execute the chunk step-by-step, accumulating rewards.
        # ------------------------------------------------------------------ #
        chunk_rewards = torch.zeros(num_envs, device=device)
        # Track which envs reset mid-chunk so we split their episodes correctly.
        env_reset_at: dict[int, int] = {}  # env_id -> sub_step at which it reset

        for sub_step in range(chunk_size):
            # Denormalise the full batch of sub_step actions in one call if possible,
            # then fall back to per-env processing if postprocess rejects batched input.
            step_action_norm = action_chunks[:, sub_step, :]  # (B, action_dim) normalised
            try:
                step_action_deg = postprocess(step_action_norm)
                if not isinstance(step_action_deg, torch.Tensor):
                    step_action_deg = torch.as_tensor(
                        step_action_deg, device=device, dtype=torch.float32
                    )
                elif step_action_deg.device != device:
                    step_action_deg = step_action_deg.to(device)
                if step_action_deg.shape != step_action_norm.shape:
                    raise ValueError("shape mismatch after batched postprocess")
            except Exception:
                # Fall back: call postprocess per env.
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

            # Accumulate rewards only for envs that haven't reset yet this chunk.
            for env_id in range(num_envs):
                if env_id not in env_reset_at:
                    chunk_rewards[env_id] += reward[env_id]

            episode_step += 1

            # Flag success: cube lifted for still-running envs.
            try:
                obj = env_unwrapped.scene.rigid_objects["object"]
                lifted = obj.data.root_pos_w[:, 2] > min_lift_height
                for env_id in range(num_envs):
                    if env_id not in env_reset_at and lifted[env_id].item():
                        buffer.mark_success(env_id)
            except (KeyError, AttributeError):
                pass

            # Track resets within the chunk.
            for env_id in range(num_envs):
                if done[env_id].item() and env_id not in env_reset_at:
                    env_reset_at[env_id] = sub_step

        # ------------------------------------------------------------------ #
        # 3. Store decision-point data for each env.
        # ------------------------------------------------------------------ #
        for env_id in range(num_envs):
            dp = DecisionPointData(
                images_np=per_env_frames[env_id]["images_np"],
                state_np=per_env_frames[env_id]["state_np"],
                action_chunk=action_chunks[env_id].detach().cpu(),
                chunk_reward=chunk_rewards[env_id].item(),
                step_in_episode=episode_step[env_id].item() - chunk_size,
            )
            buffer.add_decision_point(env_id, dp)

        # ------------------------------------------------------------------ #
        # 4. Finish episodes for envs that reset mid-chunk or exceeded max steps.
        # ------------------------------------------------------------------ #
        done_now = list(env_reset_at.keys())
        force_done = [
            env_id
            for env_id in range(num_envs)
            if episode_step[env_id].item() >= max_episode_steps and env_id not in done_now
        ]
        all_done = done_now + force_done

        if all_done:
            buffer.finish_episode(all_done)
            episodes_completed += len(all_done)
            for env_id in all_done:
                episode_step[env_id] = 0

    # Flush any remaining in-progress episodes.
    buffer.finish_episode(list(range(num_envs)))

    summ = buffer.summary()
    print(
        f"[SmolVLA-RL] Rollout done: "
        f"episodes={summ['num_episodes']}, "
        f"mean_return={summ['mean_return']:.3f}, "
        f"success_rate={summ.get('success_rate', 0.0)*100:.1f}%",
        flush=True,
    )
    return buffer
