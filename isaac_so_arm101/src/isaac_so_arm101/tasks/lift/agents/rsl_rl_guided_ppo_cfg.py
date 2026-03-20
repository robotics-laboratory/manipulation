from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from .rsl_rl_ppo_cfg import LiftCubePPORunnerCfg


@configclass
class GuidedLiftCubePPORunnerCfg(LiftCubePPORunnerCfg):
    experiment_name = "guided_lift_cube"


@configclass
class GuidedDiscriminatorLiftCubePPORunnerCfg(LiftCubePPORunnerCfg):
    """PPO settings tuned for discriminator-shaped rewards (avoid critic/actor blow-up)."""

    experiment_name = "guided_discriminator_lift_cube"
    # @configclass does not expose default nested fields on the class object (no
    # LiftCubePPORunnerCfg.policy); spell out overrides explicitly.
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.45,
        # ``log`` parametrization keeps exploration std > 0 (avoids PPO Normal.sample crash).
        noise_std_type="log",
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        # Very conservative updates — discriminator + centered shaping can still blow value / std.
        value_loss_coef=0.35,
        use_clipped_value_loss=True,
        clip_param=0.1,
        normalize_advantage_per_mini_batch=True,
        entropy_coef=0.012,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.006,
        max_grad_norm=0.15,
    )
