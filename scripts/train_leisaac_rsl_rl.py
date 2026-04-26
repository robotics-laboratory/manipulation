#!/usr/bin/env python3
"""Train LeIsaac tasks using IsaacLab's native RSL-RL trainer.

Usage:
  isaaclab -p scripts/train_leisaac_rsl_rl.py --task LeIsaac-... --headless ...
"""

from __future__ import annotations

import sys
import traceback
from importlib import import_module
from pathlib import Path
from typing import Iterable


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


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
    _prepend_sys_path(train_module_dir)

    # Native train.py imports optional external tasks after AppLauncher starts but before Hydra config loading.
    # The local LeIsaac source path must already be visible for that import to register LeIsaac task configs.
    _prepend_sys_path(leisaac_src)

    # Import trainer module so AppLauncher / simulation app setup happens normally. Avoid importing leisaac here:
    # native train.py will do it after the simulation app is live, which keeps pxr-dependent modules safe.
    trainer = import_module("train")

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
        print("[leisaac-train] entering IsaacLab RSL-RL trainer.main()", flush=True)
        trainer.main()
        print("[leisaac-train] trainer.main() returned normally", flush=True)
        print("Training completed", flush=True)
    except SystemExit as exc:
        print(f"[leisaac-train] trainer.main() raised SystemExit(code={exc.code!r})", flush=True)
        raise
    except BaseException:
        print("[leisaac-train] trainer.main() raised an unexpected exception:", flush=True)
        traceback.print_exc()
        raise
    finally:
        if hasattr(trainer, "simulation_app"):
            trainer.simulation_app.close()


if __name__ == "__main__":
    main()
