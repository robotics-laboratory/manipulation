#!/usr/bin/env python3
"""Train LeIsaac tasks using IsaacLab's native RSL-RL trainer.

Usage:
  isaaclab -p scripts/train_leisaac_rsl_rl.py --task LeIsaac-... --headless ...
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Iterable


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


def _resolve_leisaac_src(manipulation_root: Path) -> Path:
    candidates = [
        manipulation_root / "scripts" / "leisaac" / "source" / "leisaac",
        manipulation_root / "leisaac" / "source" / "leisaac",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not locate LeIsaac source directory.")


def _extract_flag_value(argv: Iterable[str], flag: str) -> str | None:
    argv = list(argv)
    for index, value in enumerate(argv):
        if value == flag and index + 1 < len(argv):
            return argv[index + 1]
    return None


def main() -> None:
    manipulation_root = Path(__file__).resolve().parents[1]
    leisaac_src = _resolve_leisaac_src(manipulation_root)
    train_module_dir = _resolve_train_module_dir(manipulation_root)
    task_name = _extract_flag_value(sys.argv, "--task")

    # Make IsaacLab RSL-RL trainer importable as `import train` and allow `import cli_args`.
    if str(train_module_dir) not in sys.path:
        sys.path.insert(0, str(train_module_dir))

    # Import trainer module so AppLauncher / simulation app setup happens normally.
    trainer = import_module("train")

    # Register LeIsaac tasks after trainer import to avoid early pxr import failures.
    if str(leisaac_src) not in sys.path:
        sys.path.insert(0, str(leisaac_src))
    import leisaac

    if task_name:
        import gymnasium as gym

        try:
            gym.spec(task_name)
        except Exception:
            # leisaac.__init__ may swallow import errors; retry a direct tasks import for clearer diagnostics.
            try:
                import_module("leisaac.tasks")
                gym.spec(task_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Task '{task_name}' is not registered after importing leisaac from '{leisaac.__file__}'. "
                    "Check for preceding '[leisaac] ERROR: Failed to import ...' logs."
                ) from exc
    else:
        print("[WARN] No '--task' was provided. Trainer may exit immediately.")

    try:
        trainer.main()
        print("Training completed")
    finally:
        if hasattr(trainer, "simulation_app"):
            trainer.simulation_app.close()


if __name__ == "__main__":
    main()
