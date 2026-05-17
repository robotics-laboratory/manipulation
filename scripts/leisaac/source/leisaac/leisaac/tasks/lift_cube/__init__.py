import gymnasium as gym

from . import agents

gym.register(
    id="LeIsaac-SO101-LiftCube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube_env_cfg:LiftCubeEnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-LiftCube-RewardDense-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube_env_cfg:LiftCubeRewardDenseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubeRewardDensePPORunnerCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-LiftCube-RewardDense-Train-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube_env_cfg:LiftCubeRewardDenseTrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubeRewardDensePPORunnerCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-LiftCube-RewardDense-Collect-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube_env_cfg:LiftCubeRewardDenseCollectEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubeRewardDensePPORunnerCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-LiftCube-GuidanceSparse-Collect-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube_env_cfg:LiftCubeGuidanceSparseCollectEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubeRewardDensePPORunnerCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-LiftCube-RewardDense-AwkwardReset-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube_env_cfg:LiftCubeRewardDenseAwkwardResetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubeRewardDensePPORunnerCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-LiftCube-RewardDense-AwkwardReset-Train-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube_env_cfg:LiftCubeRewardDenseAwkwardResetTrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubeRewardDensePPORunnerCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-LiftCube-RewardDense-AwkwardReset-Collect-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube_env_cfg:LiftCubeRewardDenseAwkwardResetCollectEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubeRewardDensePPORunnerCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-LiftCube-RewardDense-Vision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube_env_cfg:LiftCubeRewardDenseVisionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubeRewardDenseVisionPPORunnerCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-LiftCube-RewardDense-Vision-Train-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube_env_cfg:LiftCubeRewardDenseVisionTrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubeRewardDenseVisionPPORunnerCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-LiftCube-RewardDense-Vision-Collect-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube_env_cfg:LiftCubeRewardDenseVisionCollectEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubeRewardDenseVisionPPORunnerCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-LiftCube-Mimic-v0",
    entry_point=f"leisaac.enhance.envs:ManagerBasedRLLeIsaacMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube_mimic_env_cfg:LiftCubeMimicEnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-LiftCube-Direct-v0",
    entry_point=f"{__name__}.direct.lift_cube_env:LiftCubeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct.lift_cube_env:LiftCubeEnvCfg",
    },
)
