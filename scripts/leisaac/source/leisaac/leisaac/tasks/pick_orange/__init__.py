import gymnasium as gym

from . import agents

gym.register(
    id="LeIsaac-SO101-PickOrange-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_orange_env_cfg:PickOrangeEnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-PickOrange-RewardDense-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_orange_env_cfg:PickOrangeRewardDenseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickOrangeRewardDensePPORunnerCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-PickOrange-RewardDense-Train-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_orange_env_cfg:PickOrangeRewardDenseTrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickOrangeRewardDensePPORunnerCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-PickOrange-Mimic-v0",
    entry_point=f"leisaac.enhance.envs:ManagerBasedRLLeIsaacMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_orange_mimic_env_cfg:PickOrangeMimicEnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-PickOrange-Direct-v0",
    entry_point=f"{__name__}.direct.pick_orange_env:PickOrangeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct.pick_orange_env:PickOrangeEnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-PickOrange-Eureka-Direct-v0",
    entry_point=f"{__name__}.direct.pick_orange_env:PickOrangeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct.pick_orange_env:PickOrangeEurekaEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickOrangeEurekaDirectPPORunnerCfg",
    },
)
