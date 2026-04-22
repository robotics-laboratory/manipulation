#!/usr/bin/env python3
"""Wrapper to play LeIsaac tasks with IsaacLab RSL-RL script.

This keeps CLI usage clean:
  isaaclab -p scripts/play_leisaac_rsl_rl.py --task LeIsaac-... --video ...
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _resolve_play_script(manipulation_root: Path) -> Path:
    candidates = [
        # Container layout (official docker image)
        Path("/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/play.py"),
        # Host layout (this repository root one level above manipulation/)
        manipulation_root.parent / "scripts" / "reinforcement_learning" / "rsl_rl" / "play.py",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not locate IsaacLab RSL-RL play.py. Tried: " + ", ".join(str(p) for p in candidates))


def main() -> None:
    manipulation_root = Path(__file__).resolve().parents[1]
    leisaac_src = manipulation_root / "scripts" / "leisaac" / "source" / "leisaac"
    play_script = _resolve_play_script(manipulation_root)

    # Keep argv intact for downstream argparse/hydra parsing.
    sys.argv[0] = str(play_script)
    # IsaacLab play.py imports sibling module `cli_args.py`.
    if str(play_script.parent) not in sys.path:
        sys.path.insert(0, str(play_script.parent))

    # Load the play module without auto-running its __main__ block.
    # This lets AppLauncher initialize first before importing leisaac tasks.
    module_globals = runpy.run_path(str(play_script), run_name="__leisaac_rsl_play__")

    # Register LeIsaac tasks after IsaacLab runtime setup is in place.
    if str(leisaac_src) not in sys.path:
        sys.path.insert(0, str(leisaac_src))
    import leisaac  # noqa: F401

    try:
        module_globals["main"]()
    finally:
        simulation_app = module_globals.get("simulation_app")
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
