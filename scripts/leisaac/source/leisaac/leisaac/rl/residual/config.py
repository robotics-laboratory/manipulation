from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ResidualRLConfig:
    """Configuration for residual off-policy RL.

    This is intentionally generic. Keep policy/env integration outside this
    module so it can be reused with different base policies (SmolVLA, OpenPI,
    etc.).
    """

    # -------------------- environment / rollout --------------------
    task: str = "LeIsaac-SO101-LiftCube-v0"
    seed: int = 7
    device: str = "cuda"
    max_env_steps: int = 300_000
    warmup_steps: int = 5_000
    action_horizon: int = 16

    # -------------------- observation / action dims --------------------
    # NOTE: Fill these from your actual env observation encoding.
    obs_dim: int = 6
    act_dim: int = 6

    # -------------------- reward shaping --------------------
    use_guidance_reward: bool = True
    lambda_guidance: float = 0.5
    lambda_residual_l2: float = 1.0e-3

    # -------------------- residual scaling / safety --------------------
    residual_alpha_start: float = 0.0
    residual_alpha_end: float = 0.3
    residual_alpha_ramp_steps: int = 20_000
    residual_arm_limit_rad: float = 0.15
    residual_gripper_limit: float = 0.20

    # -------------------- replay / update --------------------
    replay_capacity: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    n_step: int = 3
    utd: int = 4
    actor_update_interval: int = 2

    # -------------------- TD3-style stability --------------------
    target_noise_std: float = 0.10
    target_noise_clip: float = 0.20
    tau: float = 0.005

    # -------------------- optimization --------------------
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0

    # -------------------- networks --------------------
    actor_hidden: tuple[int, int] = (256, 256)
    critic_hidden: tuple[int, int] = (256, 256)
    layer_norm_critic: bool = True
    layer_norm_actor: bool = False

    # -------------------- logging / checkpoints --------------------
    log_interval: int = 200
    checkpoint_interval: int = 10_000
    output_dir: str = "logs/residual_smolvla"

    def residual_alpha(self, env_step: int) -> float:
        """Linearly ramp residual influence for safer early training."""
        if self.residual_alpha_ramp_steps <= 0:
            return self.residual_alpha_end
        t = min(max(env_step, 0), self.residual_alpha_ramp_steps)
        ratio = float(t) / float(self.residual_alpha_ramp_steps)
        return self.residual_alpha_start + ratio * (self.residual_alpha_end - self.residual_alpha_start)
