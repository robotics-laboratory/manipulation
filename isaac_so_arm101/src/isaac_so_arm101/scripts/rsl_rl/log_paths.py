# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL log and artifact paths for isaac_so_arm101.

Default layout: ``isaac_so_arm101/logs/rsl_rl/...`` relative to the process cwd.

In Docker (``WORKDIR=/workspace/isaac-bridge``), this resolves to
``/workspace/isaac-bridge/isaac_so_arm101/logs/rsl_rl``, which is bind-mounted from the host
(see ``manipulation/docker/docker-compose.yaml``) so checkpoints and TensorBoard data persist.

Curated final weights (copies of ``model_*.pt``) live under ``isaac_so_arm101/checkpoints/``
(see that folder's README). Override with ``ISAAC_SO_ARM101_FINAL_CHECKPOINTS_DIR``.

Override the parent of per-experiment folders with env
``ISAAC_SO_ARM101_RSL_RL_LOG_ROOT`` (e.g. ``logs/rsl_rl`` for legacy layouts).
"""

from __future__ import annotations

import os


def rsl_rl_root() -> str:
    """Parent directory of per-experiment folders (``lift/``, ``lift_fixed_layout/``, ``guided_lift_cube/``, ...)."""
    return os.environ.get(
        "ISAAC_SO_ARM101_RSL_RL_LOG_ROOT",
        os.path.join("isaac_so_arm101", "logs", "rsl_rl"),
    )


def rsl_rl_experiment_dir(experiment_name: str) -> str:
    """Absolute path: ``<rsl_rl_root>/<experiment_name>``."""
    return os.path.abspath(os.path.join(rsl_rl_root(), experiment_name))


def final_checkpoints_dir() -> str:
    """Directory for stable RSL-RL checkpoint copies (per-task ``*.pt``).

    Default: ``isaac_so_arm101/checkpoints`` relative to cwd. Bind-mounted in Docker
    (see ``manipulation/docker/docker-compose.yaml``).
    """
    return os.environ.get(
        "ISAAC_SO_ARM101_FINAL_CHECKPOINTS_DIR",
        os.path.join("isaac_so_arm101", "checkpoints"),
    )


def resolve_checkpoint_cli_path(path: str) -> str:
    """Resolve ``--checkpoint`` when the process cwd is not the bridge project root.

    ``isaaclab.sh`` often sets cwd to the Isaac Lab repo; relative paths like
    ``isaac_so_arm101/checkpoints/foo.pt`` then fail in :func:`retrieve_file_path`.
    If ``PROJECT_DIR`` is set (Docker: ``/workspace/isaac-bridge``), try
    ``os.path.join(PROJECT_DIR, path)`` first.

    Remote HTTP(S) and Omniverse-style URLs are returned unchanged.
    """
    if not path:
        return path
    p = path.strip()
    if os.path.isfile(p):
        return os.path.abspath(p)
    if p.startswith(("http://", "https://", "omniverse://")) or p.startswith("nvidia::"):
        return p
    project_dir = os.environ.get("PROJECT_DIR")
    if project_dir and not os.path.isabs(p):
        candidate = os.path.join(project_dir, p)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    if not os.path.isabs(p):
        candidate = os.path.join(os.getcwd(), p)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return p
