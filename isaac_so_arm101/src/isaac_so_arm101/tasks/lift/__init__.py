# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-SO-ARM100-Lift-Cube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100LiftCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM100-Target-Cube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100TargetCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TargetCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM100-Target-Cube-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100TargetCubeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TargetCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM100-Lift-Cube-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100LiftCubeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Target-Cube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101TargetCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TargetCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Target-Cube-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101TargetCubeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TargetCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Lift-Cube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101LiftCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Lift-Cube-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101LiftCubeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Lift-Cube-Sparse-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101LiftCubeSparseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Lift-Cube-Sparse-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101LiftCubeSparseEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Guided-Lift-Cube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.guided_env_cfg:SoArm101GuidedLiftCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_guided_ppo_cfg:GuidedLiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Guided-Lift-Cube-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.guided_env_cfg:SoArm101GuidedLiftCubeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_guided_ppo_cfg:GuidedLiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Guided-Lift-Cube-Sparse-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.guided_env_cfg:SoArm101GuidedLiftCubeSparseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_guided_ppo_cfg:GuidedLiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Guided-Lift-Cube-Sparse-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.guided_env_cfg:SoArm101GuidedLiftCubeSparseEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_guided_ppo_cfg:GuidedLiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Guided-Lift-Cube-Sparse-Discriminator-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.guided_env_cfg:SoArm101GuidedLiftCubeSparseDiscriminatorEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_guided_ppo_cfg:GuidedDiscriminatorLiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Guided-Lift-Cube-Sparse-Discriminator-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.guided_env_cfg:SoArm101GuidedLiftCubeSparseDiscriminatorEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_guided_ppo_cfg:GuidedDiscriminatorLiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

# Fixed cube + fixed goal (aligned dense teacher → trajectory dataset → guided student)
gym.register(
    id="Isaac-SO-ARM101-FixedLayout-Lift-Cube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fixed_layout_env_cfg:SoArm101FixedLayoutLiftCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FixedLayoutLiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-FixedLayout-Lift-Cube-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fixed_layout_env_cfg:SoArm101FixedLayoutLiftCubeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FixedLayoutLiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-FixedLayout-Lift-Cube-Sparse-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fixed_layout_env_cfg:SoArm101FixedLayoutLiftCubeSparseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FixedLayoutLiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-FixedLayout-Lift-Cube-Sparse-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fixed_layout_env_cfg:SoArm101FixedLayoutLiftCubeSparseEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FixedLayoutLiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-FixedLayout-Guided-Lift-Cube-Sparse-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fixed_layout_env_cfg:SoArm101FixedLayoutGuidedLiftCubeSparseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_guided_ppo_cfg:FixedLayoutGuidedLiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-FixedLayout-Guided-Lift-Cube-Sparse-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fixed_layout_env_cfg:SoArm101FixedLayoutGuidedLiftCubeSparseEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_guided_ppo_cfg:FixedLayoutGuidedLiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)
