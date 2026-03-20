# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for the extension."""

##
# Register Gym environments.
##

import os

# Offline tools (e.g. train_trajectory_discriminator) only need submodules under
# tasks/lift and must not execute import_packages here: that pulls in all of
# isaaclab_tasks -> isaaclab.envs -> controllers -> isaaclab.utils (and USD/pxr).
# Set ISAAC_SO_ARM101_SKIP_TASK_AUTOIMPORT=1 before importing isaac_so_arm101.tasks.
if os.environ.get("ISAAC_SO_ARM101_SKIP_TASK_AUTOIMPORT", "").lower() not in ("1", "true", "yes"):
    from isaaclab_tasks.utils import import_packages

    # The blacklist is used to prevent importing configs from sub-packages
    _BLACKLIST_PKGS = ["utils", ".mdp"]
    # Import all configs in this package
    import_packages(__name__, _BLACKLIST_PKGS)