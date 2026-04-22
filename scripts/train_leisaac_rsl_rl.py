#!/usr/bin/env python3
"""Train LeIsaac tasks using IsaacLab's native RSL-RL trainer.

Usage:
  isaaclab -p scripts/train_leisaac_rsl_rl.py --task LeIsaac-... --headless ...
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path


def _resolve_train_module_dir(manipulation_root: Path) -> Path:
    candidates = [
        # Container layout (official docker image)
        Path("/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl"),
        # Host layout (repository root one level above manipulation/)
        manipulation_root.parent / "scripts" / "reinforcement_learning" / "rsl_rl",
    ]
    for path in candidates:
        if (path / "train.py").exists():
            return path
    raise FileNotFoundError("Could not locate IsaacLab RSL-RL train.py directory.")


def main() -> None:
    manipulation_root = Path(__file__).resolve().parents[1]
    leisaac_src = manipulation_root / "scripts" / "leisaac" / "source" / "leisaac"
    train_module_dir = _resolve_train_module_dir(manipulation_root)

    # Make IsaacLab RSL-RL trainer importable as `import train` and allow `import cli_args`.
    if str(train_module_dir) not in sys.path:
        sys.path.insert(0, str(train_module_dir))

    # Import trainer module first so AppLauncher / simulation app setup happens normally.
    trainer = import_module("train")

    # Register LeIsaac tasks before invoking hydra-wrapped trainer main.
    if str(leisaac_src) not in sys.path:
        sys.path.insert(0, str(leisaac_src))
    import leisaac  # noqa: F401

    try:
        trainer.main()
        print("Training completed")
    finally:
        if hasattr(trainer, "simulation_app"):
            trainer.simulation_app.close()


if __name__ == "__main__":
    main()
