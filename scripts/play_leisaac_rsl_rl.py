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


def _install_lift_height_debug_print(module_globals: dict) -> None:
    """Print the same lift height used by the success termination during play."""
    # IsaacLab play.py strips sys.argv for Hydra during runpy, so read the parsed args it leaves behind.
    task_name = getattr(module_globals["args_cli"], "task", "") or ""
    if "LiftCube" not in task_name:
        return

    gym = module_globals["gym"]
    original_make = gym.make

    def make_with_lift_debug(*args, **kwargs):
        env = original_make(*args, **kwargs)
        unwrapped = env.unwrapped
        if not hasattr(unwrapped, "scene"):
            return env

        original_step = env.step
        step_count = 0

        def step_with_lift_debug(actions):
            nonlocal step_count
            result = original_step(actions)
            step_count += 1

            try:
                dones = result[2]
                if bool(dones[0].item()):
                    term = bool(unwrapped.reset_terminated[0].item())
                    trunc = bool(unwrapped.reset_time_outs[0].item())
                    ep_len = int(unwrapped.episode_length_buf[0].item())
                    parts = [
                        "[reset-debug]",
                        f"step={step_count}",
                        f"episode_len={ep_len}",
                        f"terminated={term}",
                        f"truncated={trunc}",
                    ]
                    if hasattr(unwrapped, "termination_manager"):
                        for name in unwrapped.termination_manager.active_terms:
                            term_val = bool(unwrapped.termination_manager.get_term(name)[0].item())
                            parts.append(f"{name}={term_val}")
                    print(" ".join(parts))
            except Exception as exc:
                print(f"[reset-debug] failed to read reset reason: {exc}")

            if step_count % 30 != 0:
                return result

            try:
                cube = unwrapped.scene["cube"]
                robot = unwrapped.scene["robot"]
                base_index = robot.data.body_names.index("base")
                lift_height = cube.data.root_pos_w[:, 2] - robot.data.body_pos_w[:, base_index, 2]
                threshold = unwrapped.cfg.terminations.success.params.get("height_threshold", 0.20)
                print(
                    "[lift-debug] "
                    f"step={step_count} "
                    f"cube_z-base_z={lift_height[0].item():.4f}m "
                    f"success_threshold={threshold:.4f}m "
                    f"success={bool(lift_height[0].item() > threshold)}"
                )
            except Exception as exc:
                print(f"[lift-debug] failed to read lift height: {exc}")

            return result

        env.step = step_with_lift_debug
        return env

    gym.make = make_with_lift_debug


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

    _install_lift_height_debug_print(module_globals)

    try:
        module_globals["main"]()
    finally:
        simulation_app = module_globals.get("simulation_app")
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
