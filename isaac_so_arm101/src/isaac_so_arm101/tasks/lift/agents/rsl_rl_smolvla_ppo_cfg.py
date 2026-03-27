"""RSL-RL runner config for SmolVLA-backed PPO training."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class SmolVLAPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    """Policy config that targets ``SmolVLAActorCritic`` and carries SmolVLA-specific fields.

    Extra fields are serialised by ``to_dict()`` and forwarded as ``**kwargs``
    to the ``SmolVLAActorCritic`` constructor by RSL-RL's ``eval(class_name)``
    instantiation path.
    """

    class_name: str = "SmolVLAActorCritic"

    # MLP heads (relatively small; visual features provide most capacity)
    actor_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"
    init_noise_std: float = 0.5
    noise_std_type: str = "log"
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False

    # SmolVLA-specific — passed through to the wrapper
    smolvla_model_path: str = "lerobot/smolvla_base"
    language_instruction: str = "Pick the cube."
    freeze_backbone: bool = True
    use_lora: bool = False


@configclass
class SmolVLAPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Conservative PPO hyper-params for vision-conditioned training."""

    value_loss_coef: float = 0.5
    use_clipped_value_loss: bool = True
    clip_param: float = 0.15
    normalize_advantage_per_mini_batch: bool = True
    entropy_coef: float = 0.01
    num_learning_epochs: int = 4
    num_mini_batches: int = 4
    learning_rate: float = 3.0e-5
    schedule: str = "adaptive"
    gamma: float = 0.98
    lam: float = 0.95
    desired_kl: float = 0.008
    max_grad_norm: float = 0.5


@configclass
class SmolVLAGuidedLiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Full runner config for SmolVLA trajectory-guided lift-cube training."""

    num_steps_per_env: int = 16
    max_iterations: int = 3000
    save_interval: int = 100
    experiment_name: str = "smolvla_guided_lift_cube"
    empirical_normalization: bool = False

    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "smolvla_features"],
        "critic": ["policy", "smolvla_features"],
    }

    policy: SmolVLAPpoActorCriticCfg = SmolVLAPpoActorCriticCfg()
    algorithm: SmolVLAPpoAlgorithmCfg = SmolVLAPpoAlgorithmCfg()
