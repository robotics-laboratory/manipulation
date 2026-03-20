# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL log and artifact paths for isaac_so_arm101.

Default layout: ``isaac_so_arm101/logs/rsl_rl/...`` relative to the process cwd.

In Docker (``WORKDIR=/workspace/isaac-bridge``), this resolves to
``/workspace/isaac-bridge/isaac_so_arm101/logs/rsl_rl``, which is bind-mounted from the host
(see ``manipulation/docker/docker-compose.yaml``) so checkpoints and TensorBoard data persist.

Override the parent of per-experiment folders with env
``ISAAC_SO_ARM101_RSL_RL_LOG_ROOT`` (e.g. ``logs/rsl_rl`` for legacy layouts).
"""

from __future__ import annotations

import os


def rsl_rl_root() -> str:
    """Parent directory of per-experiment folders (``lift/``, ``guided_lift_cube/``, ...)."""
    return os.environ.get(
        "ISAAC_SO_ARM101_RSL_RL_LOG_ROOT",
        os.path.join("isaac_so_arm101", "logs", "rsl_rl"),
    )


def rsl_rl_experiment_dir(experiment_name: str) -> str:
    """Absolute path: ``<rsl_rl_root>/<experiment_name>``."""
    return os.path.abspath(os.path.join(rsl_rl_root(), experiment_name))
