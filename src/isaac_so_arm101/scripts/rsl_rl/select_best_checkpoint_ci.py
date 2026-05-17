#!/usr/bin/env python3
"""Batch checkpoint selection with Wilson confidence intervals.

Evaluates residual checkpoints for two groups (resfit-vit, privileged), computes
Wilson 95% confidence intervals for success rate, and selects the best checkpoint
in each group by maximizing the lower CI bound.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import subprocess
import sys
import tempfile
from statistics import NormalDist
from dataclasses import asdict, dataclass
from pathlib import Path
from tqdm.auto import tqdm


DEFAULT_TASK = "LeIsaac-SO101-LiftCube-RewardDense-Collect-v0"
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_TARGET_CI_HALF_WIDTH = 0.05
DEFAULT_COARSE_EPISODES = 100
DEFAULT_TOP_K = 3
DEFAULT_LATEST_K = 1


@dataclass
class CheckpointResult:
    stage: str
    group: str
    checkpoint: str
    step: int
    episodes: int
    successes: int
    sr: float
    ci_low: float
    ci_high: float
    mean_return: float
    seed: int
    base_successes: int | None = None
    base_episodes: int | None = None
    base_sr: float | None = None
    base_mean_return: float | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resfit_glob", type=str, nargs="+", required=True, help="Glob(s) for resfit-vit checkpoints.")
    parser.add_argument(
        "--privileged_glob", type=str, nargs="+", required=True, help="Glob(s) for privileged checkpoints."
    )
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help=(
            "Final-stage episodes per checkpoint. If omitted, script computes strict minimum n so that Wilson CI "
            "half-width is <= --target_ci_half_width at --confidence_level for all possible success counts."
        ),
    )
    parser.add_argument(
        "--coarse_episodes",
        type=int,
        default=DEFAULT_COARSE_EPISODES,
        help="Coarse-stage episodes used for initial filtering over all checkpoints (default: 100).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help="How many checkpoints per group to re-evaluate at final-stage episodes (default: 3).",
    )
    parser.add_argument(
        "--latest_k",
        type=int,
        default=DEFAULT_LATEST_K,
        help=(
            "Evaluate only latest k discovered checkpoints per group before selection "
            "(default: 1). Set <= 0 to evaluate all discovered checkpoints."
        ),
    )
    parser.add_argument(
        "--two_stage_selection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable two-stage selection: coarse over all checkpoints, final over top-k. "
            "Auto-disabled when --latest_k > 0."
        ),
    )
    parser.add_argument(
        "--target_ci_half_width",
        type=float,
        default=DEFAULT_TARGET_CI_HALF_WIDTH,
        help="Target half-width for Wilson confidence interval (default: 0.05).",
    )
    parser.add_argument(
        "--confidence_level",
        type=float,
        default=DEFAULT_CONFIDENCE_LEVEL,
        help="Confidence level for Wilson interval (default: 0.95).",
    )
    parser.add_argument(
        "--eval_script",
        type=str,
        default=str(Path(__file__).with_name("eval_residual_smolvla_td3_privileged.py")),
        help="Path to residual evaluation script.",
    )
    parser.add_argument(
        "--python_executable",
        type=str,
        default=sys.executable,
        help="Python executable used to run eval script (typically /isaac-sim/python.sh).",
    )
    parser.add_argument(
        "--eval_timeout_s",
        type=float,
        default=None,
        help=(
            "Per-checkpoint eval timeout in seconds. "
            "If omitted or <= 0, timeout is disabled (wait until eval finishes)."
        ),
    )
    parser.add_argument(
        "--eval_base",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also evaluate base policy and persist base metrics in CSV (default: False).",
    )
    parser.add_argument(
        "--base_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate/select using base-only metrics (skip residual eval in eval script).",
    )
    parser.add_argument("--output_dir", type=str, default="logs/residual_td3/checkpoint_ci")
    parser.add_argument(
        "--policy_checkpoint_path",
        type=str,
        default=None,
        help="Optional override passed to eval script.",
    )
    parser.add_argument(
        "--policy_backend",
        type=str,
        choices=["local", "service"],
        default=None,
        help="Optional override passed to eval script.",
    )
    parser.add_argument(
        "--policy_action_horizon",
        type=int,
        default=None,
        help="Optional override passed to eval script.",
    )
    parser.add_argument(
        "--skip_teleop_device_setup",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional override passed to eval script.",
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_cameras", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--extra_eval_arg",
        action="append",
        default=[],
        help="Extra argument forwarded to eval script. Repeatable.",
    )
    return parser.parse_args()


def _discover_checkpoints(patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        for raw in glob.glob(pattern):
            p = Path(raw).expanduser().resolve()
            if p.is_file() and p.suffix == ".pt":
                paths.add(p)
    return sorted(paths)


def _extract_step(ckpt_path: Path) -> int:
    match = re.search(r"step_(\d+)\.pt$", ckpt_path.name)
    if match:
        return int(match.group(1))
    if ckpt_path.name.endswith("final.pt"):
        return 10**12
    return -1


def _checkpoint_recency_key(ckpt_path: Path) -> tuple[int, float, str]:
    try:
        mtime = float(ckpt_path.stat().st_mtime)
    except OSError:
        mtime = 0.0
    return _extract_step(ckpt_path), mtime, ckpt_path.name


def _dedupe_final_and_last_step(checkpoints: list[Path]) -> tuple[list[Path], list[Path]]:
    """Drop max step checkpoint when final.pt is present (they are often identical snapshots)."""
    if len(checkpoints) <= 1:
        return checkpoints, []
    final_ckpts = [p for p in checkpoints if p.name.endswith("final.pt")]
    step_ckpts = [p for p in checkpoints if re.search(r"step_(\d+)\.pt$", p.name)]
    if not final_ckpts or not step_ckpts:
        return checkpoints, []
    max_step = max(_extract_step(p) for p in step_ckpts)
    to_drop = [p for p in step_ckpts if _extract_step(p) == max_step]
    if not to_drop:
        return checkpoints, []
    drop_set = set(to_drop)
    filtered = [p for p in checkpoints if p not in drop_set]
    return filtered, to_drop


def _take_latest_k(checkpoints: list[Path], k: int) -> list[Path]:
    if k <= 0 or len(checkpoints) <= k:
        return checkpoints
    return sorted(checkpoints, key=_checkpoint_recency_key, reverse=True)[:k]


def _z_from_confidence_level(confidence_level: float) -> float:
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"--confidence_level must be in (0, 1), got: {confidence_level}")
    alpha = 1.0 - confidence_level
    return float(NormalDist().inv_cdf(1.0 - alpha / 2.0))


def _wilson_interval(successes: int, n: int, z: float) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    phat = float(successes) / float(n)
    z2 = z * z
    denom = 1.0 + z2 / float(n)
    center = (phat + z2 / (2.0 * float(n))) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * float(n))) / float(n)) / denom
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return float(low), float(high)


def _max_wilson_half_width_for_n(n: int, z: float) -> float:
    if n <= 0:
        return 1.0
    max_half = 0.0
    for successes in range(n + 1):
        low, high = _wilson_interval(successes, n, z)
        half = 0.5 * (high - low)
        if half > max_half:
            max_half = half
    return float(max_half)


def _required_episodes_strict_wilson(target_half_width: float, confidence_level: float) -> int:
    if target_half_width <= 0.0 or target_half_width >= 1.0:
        raise ValueError(f"--target_ci_half_width must be in (0, 1), got: {target_half_width}")
    z = _z_from_confidence_level(confidence_level)
    # Start from normal-approx worst-case and then verify strictly.
    approx = int(math.ceil((z * z * 0.25) / (target_half_width * target_half_width)))
    n = max(1, approx - 50)
    max_n = 2_000_000
    while n <= max_n:
        if _max_wilson_half_width_for_n(n, z) <= target_half_width:
            return int(n)
        n += 1
    raise RuntimeError(
        f"Failed to find required episodes up to {max_n} for target_half_width={target_half_width} "
        f"confidence_level={confidence_level}"
    )


def _write_results_csv(results: list[CheckpointResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "stage",
                "checkpoint",
                "step",
                "episodes",
                "successes",
                "sr",
                "ci_low",
                "ci_high",
                "mean_return",
                "seed",
                "base_successes",
                "base_episodes",
                "base_sr",
                "base_mean_return",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))


def _rank_group(rows: list[CheckpointResult]) -> list[CheckpointResult]:
    return sorted(rows, key=lambda x: (x.ci_low, x.sr, x.step), reverse=True)


def _write_summary_json(
    *,
    path: Path,
    args: argparse.Namespace,
    resfit_ranked: list[CheckpointResult],
    privileged_ranked: list[CheckpointResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selection_rule": "max lower bound of Wilson CI",
        "ci_method": "wilson",
        "confidence_level": float(args.confidence_level),
        "target_ci_half_width": float(args.target_ci_half_width),
        "coarse_episodes": int(args.coarse_episodes),
        "top_k": int(args.top_k),
        "latest_k": int(args.latest_k),
        "eval_base": bool(args.eval_base),
        "base_only": bool(args.base_only),
        "two_stage_selection": bool(args.two_stage_selection),
        "task": str(args.task),
        "num_episodes": int(args.num_episodes),
        "seed": int(args.seed),
        "resfit_vit": {
            "best": asdict(resfit_ranked[0]) if resfit_ranked else None,
            "top3": [asdict(x) for x in resfit_ranked[:3]],
        },
        "privileged": {
            "best": asdict(privileged_ranked[0]) if privileged_ranked else None,
            "top3": [asdict(x) for x in privileged_ranked[:3]],
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_eval_for_checkpoint(
    args: argparse.Namespace, checkpoint: Path, episodes: int
) -> tuple[int, int, float, int | None, int | None, float | None]:
    with tempfile.TemporaryDirectory(prefix="eval_ci_") as tmp_dir:
        output_json = Path(tmp_dir) / "eval_result.json"
        cmd: list[str] = [
            str(args.python_executable),
            str(Path(args.eval_script).resolve()),
            "--residual_checkpoint",
            str(checkpoint),
            "--task",
            str(args.task),
            "--num_episodes",
            str(int(episodes)),
            "--seed",
            str(int(args.seed)),
            "--no-print_episode_debug",
            "--output_json",
            str(output_json),
        ]
        effective_eval_base = bool(args.eval_base or args.base_only)
        cmd.append("--eval_base" if effective_eval_base else "--no-eval_base")
        cmd.append("--base_only" if args.base_only else "--no-base_only")
        if args.policy_checkpoint_path:
            cmd += ["--policy_checkpoint_path", str(args.policy_checkpoint_path)]
        if args.policy_backend:
            cmd += ["--policy_backend", str(args.policy_backend)]
        if args.policy_action_horizon is not None:
            cmd += ["--policy_action_horizon", str(int(args.policy_action_horizon))]
        if args.skip_teleop_device_setup is not None:
            cmd.append("--skip_teleop_device_setup" if args.skip_teleop_device_setup else "--no-skip_teleop_device_setup")
        cmd.append("--headless" if args.headless else "--no-headless")
        cmd.append("--enable_cameras" if args.enable_cameras else "--no-enable_cameras")
        for extra in args.extra_eval_arg:
            if extra:
                cmd.append(str(extra))

        timeout_s = float(args.eval_timeout_s) if args.eval_timeout_s is not None and float(args.eval_timeout_s) > 0.0 else None
        proc = subprocess.run(cmd, text=True, timeout=timeout_s)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Eval failed for checkpoint: {checkpoint}\n"
                f"Command: {' '.join(cmd)}\n"
                f"Return code: {proc.returncode}"
            )
        if not output_json.exists():
            raise RuntimeError(f"Eval completed but no output JSON found: {output_json}")
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        primary_key = "base" if args.base_only else "residual"
        primary_payload = payload.get(primary_key, {})
        if not isinstance(primary_payload, dict):
            primary_payload = {}
        successes = int(primary_payload.get("successes", 0))
        episodes = int(primary_payload.get("episodes", int(episodes)))
        mean_return = float(primary_payload.get("mean_return", 0.0))
        base_successes: int | None = None
        base_episodes: int | None = None
        base_mean_return: float | None = None
        base_payload = payload.get("base", {})
        if isinstance(base_payload, dict):
            if "successes" in base_payload:
                base_successes = int(base_payload.get("successes", 0))
            if "episodes" in base_payload:
                base_episodes = int(base_payload.get("episodes", int(episodes)))
            if "mean_return" in base_payload:
                base_mean_return = float(base_payload.get("mean_return", 0.0))
        return successes, episodes, mean_return, base_successes, base_episodes, base_mean_return


def _rank_for_group_from_all(all_results: list[CheckpointResult], group: str) -> list[CheckpointResult]:
    final_rows = [x for x in all_results if x.group == group and x.stage == "final"]
    if final_rows:
        return _rank_group(final_rows)
    coarse_rows = [x for x in all_results if x.group == group and x.stage == "coarse"]
    return _rank_group(coarse_rows)


def _evaluate_group_stage(
    *,
    stage: str,
    group_name: str,
    checkpoints: list[Path],
    episodes: int,
    args: argparse.Namespace,
    all_results: list[CheckpointResult],
    results_csv: Path,
    summary_json: Path,
) -> list[CheckpointResult]:
    z = _z_from_confidence_level(args.confidence_level)
    group_rows: list[CheckpointResult] = []
    skipped_count = 0
    interrupt_skip_count = 0
    print(f"[INFO] Evaluating stage={stage} group={group_name} checkpoints={len(checkpoints)} episodes={episodes}")
    pbar = tqdm(checkpoints, desc=f"{group_name}:{stage}", unit="ckpt")
    for idx, ckpt in enumerate(pbar, start=1):
        print(
            f"[EVAL] stage={stage} group={group_name} {idx}/{len(checkpoints)} "
            f"ckpt={ckpt}"
        )
        pbar.set_postfix(current=ckpt.name, skipped=skipped_count, interrupted=interrupt_skip_count)
        try:
            successes, eval_episodes, mean_return, base_successes, base_episodes, base_mean_return = _run_eval_for_checkpoint(
                args, ckpt, episodes=episodes
            )
        except KeyboardInterrupt:
            interrupt_skip_count += 1
            skipped_count += 1
            print(
                f"[SKIP] stage={stage} group={group_name} {idx}/{len(checkpoints)} "
                f"ckpt={ckpt} reason=KeyboardInterrupt: interrupted current checkpoint, continuing"
            )
            pbar.set_postfix(skipped=skipped_count, interrupted=interrupt_skip_count)
            continue
        except subprocess.TimeoutExpired:
            skipped_count += 1
            timeout_desc = (
                f"exceeded {args.eval_timeout_s:.1f}s"
                if args.eval_timeout_s is not None and float(args.eval_timeout_s) > 0.0
                else "timeout expired"
            )
            print(
                f"[SKIP] stage={stage} group={group_name} {idx}/{len(checkpoints)} "
                f"ckpt={ckpt} reason=TimeoutExpired: {timeout_desc}"
            )
            continue
        except Exception as exc:
            skipped_count += 1
            print(
                f"[SKIP] stage={stage} group={group_name} {idx}/{len(checkpoints)} "
                f"ckpt={ckpt} reason={exc.__class__.__name__}: {exc}"
            )
            continue
        sr = float(successes / max(eval_episodes, 1))
        ci_low, ci_high = _wilson_interval(successes, eval_episodes, z)
        base_sr: float | None = None
        if base_successes is not None and base_episodes is not None and base_episodes > 0:
            base_sr = float(base_successes / base_episodes)
        row = CheckpointResult(
            stage=stage,
            group=group_name,
            checkpoint=str(ckpt),
            step=_extract_step(ckpt),
            episodes=int(eval_episodes),
            successes=int(successes),
            sr=sr,
            ci_low=ci_low,
            ci_high=ci_high,
            mean_return=float(mean_return),
            seed=int(args.seed),
            base_successes=base_successes,
            base_episodes=base_episodes,
            base_sr=base_sr,
            base_mean_return=base_mean_return,
        )
        group_rows.append(row)
        all_results.append(row)
        base_str = ""
        if row.base_sr is not None:
            base_ret = row.base_mean_return if row.base_mean_return is not None else 0.0
            base_succ = row.base_successes if row.base_successes is not None else 0
            base_eps = row.base_episodes if row.base_episodes is not None else 0
            base_str = f" base_sr={row.base_sr:.3f} base_ret={base_ret:.3f} base={base_succ}/{base_eps}"
        metric_label = "BASE" if args.base_only else "CI"
        print(
            f"[{metric_label}] stage={stage} group={group_name} {idx}/{len(checkpoints)} "
            f"sr={row.sr:.3f} ci=[{row.ci_low:.3f},{row.ci_high:.3f}] "
            f"successes={row.successes}/{row.episodes} step={row.step} ckpt={row.checkpoint}{base_str}"
        )
        pbar.set_postfix(
            sr=f"{row.sr:.3f}",
            ci_low=f"{row.ci_low:.3f}",
            skipped=skipped_count,
            interrupted=interrupt_skip_count,
        )
        # Persist intermediate progress after every checkpoint.
        _write_results_csv(all_results, results_csv)
        resfit_ranked = _rank_for_group_from_all(all_results, group="resfit-vit")
        privileged_ranked = _rank_for_group_from_all(all_results, group="privileged")
        _write_summary_json(
            path=summary_json,
            args=args,
            resfit_ranked=resfit_ranked,
            privileged_ranked=privileged_ranked,
        )
    pbar.close()
    print(
        f"[INFO] Finished stage={stage} group={group_name}: evaluated={len(group_rows)} "
        f"skipped={skipped_count} interrupted={interrupt_skip_count} total={len(checkpoints)}"
    )
    return group_rows


def main() -> None:
    args = _parse_args()
    if args.num_episodes is None:
        args.num_episodes = _required_episodes_strict_wilson(
            target_half_width=float(args.target_ci_half_width),
            confidence_level=float(args.confidence_level),
        )
        print(
            "[INFO] Auto-computed strict num_episodes for Wilson CI: "
            f"n={args.num_episodes} (target_half_width={args.target_ci_half_width:.4f}, "
            f"confidence={args.confidence_level:.3f})"
        )
    elif args.num_episodes <= 0:
        raise ValueError(f"--num_episodes must be > 0 when provided, got: {args.num_episodes}")
    if args.coarse_episodes <= 0:
        raise ValueError(f"--coarse_episodes must be > 0, got: {args.coarse_episodes}")
    if args.top_k <= 0:
        raise ValueError(f"--top_k must be > 0, got: {args.top_k}")

    resfit_ckpts = _discover_checkpoints(args.resfit_glob)
    privileged_ckpts = _discover_checkpoints(args.privileged_glob)
    if not resfit_ckpts:
        raise FileNotFoundError("No checkpoints matched --resfit_glob patterns.")
    if not privileged_ckpts:
        raise FileNotFoundError("No checkpoints matched --privileged_glob patterns.")
    resfit_ckpts, resfit_dropped = _dedupe_final_and_last_step(resfit_ckpts)
    privileged_ckpts, privileged_dropped = _dedupe_final_and_last_step(privileged_ckpts)
    if resfit_dropped:
        print(f"[INFO] Dropping duplicate last-step checkpoints for resfit-vit: {[str(x) for x in resfit_dropped]}")
    if privileged_dropped:
        print(f"[INFO] Dropping duplicate last-step checkpoints for privileged: {[str(x) for x in privileged_dropped]}")
    if args.latest_k > 0:
        original_resfit_count = len(resfit_ckpts)
        original_privileged_count = len(privileged_ckpts)
        resfit_ckpts = _take_latest_k(resfit_ckpts, int(args.latest_k))
        privileged_ckpts = _take_latest_k(privileged_ckpts, int(args.latest_k))
        print(
            f"[INFO] Limiting checkpoints to latest_k={args.latest_k}: "
            f"resfit-vit {original_resfit_count}->{len(resfit_ckpts)}, "
            f"privileged {original_privileged_count}->{len(privileged_ckpts)}"
        )
        if resfit_ckpts:
            print(f"[INFO] Latest resfit-vit checkpoints: {[str(x) for x in resfit_ckpts]}")
        if privileged_ckpts:
            print(f"[INFO] Latest privileged checkpoints: {[str(x) for x in privileged_ckpts]}")
        if args.two_stage_selection:
            print(
                "[INFO] Disabling two-stage selection because --latest_k > 0: "
                "running direct full-episode evaluation on selected latest checkpoints."
            )
            args.two_stage_selection = False

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / "results.csv"
    summary_json = output_dir / "summary.json"

    all_results: list[CheckpointResult] = []
    if args.two_stage_selection:
        # Stage 1: coarse pass on all checkpoints.
        resfit_coarse_rows = _evaluate_group_stage(
            stage="coarse",
            group_name="resfit-vit",
            checkpoints=resfit_ckpts,
            episodes=int(args.coarse_episodes),
            args=args,
            all_results=all_results,
            results_csv=results_csv,
            summary_json=summary_json,
        )
        privileged_coarse_rows = _evaluate_group_stage(
            stage="coarse",
            group_name="privileged",
            checkpoints=privileged_ckpts,
            episodes=int(args.coarse_episodes),
            args=args,
            all_results=all_results,
            results_csv=results_csv,
            summary_json=summary_json,
        )
        resfit_top = _rank_group(resfit_coarse_rows)[: min(args.top_k, len(resfit_coarse_rows))]
        privileged_top = _rank_group(privileged_coarse_rows)[: min(args.top_k, len(privileged_coarse_rows))]
        print(
            f"[INFO] Two-stage selection: re-evaluating top-{args.top_k} with final episodes={args.num_episodes} "
            f"(resfit_candidates={len(resfit_top)}, privileged_candidates={len(privileged_top)})"
        )
        resfit_rows = _evaluate_group_stage(
            stage="final",
            group_name="resfit-vit",
            checkpoints=[Path(x.checkpoint) for x in resfit_top],
            episodes=int(args.num_episodes),
            args=args,
            all_results=all_results,
            results_csv=results_csv,
            summary_json=summary_json,
        )
        privileged_rows = _evaluate_group_stage(
            stage="final",
            group_name="privileged",
            checkpoints=[Path(x.checkpoint) for x in privileged_top],
            episodes=int(args.num_episodes),
            args=args,
            all_results=all_results,
            results_csv=results_csv,
            summary_json=summary_json,
        )
        if not resfit_rows:
            resfit_rows = resfit_coarse_rows
        if not privileged_rows:
            privileged_rows = privileged_coarse_rows
    else:
        # Single-stage full pass.
        resfit_rows = _evaluate_group_stage(
            stage="final",
            group_name="resfit-vit",
            checkpoints=resfit_ckpts,
            episodes=int(args.num_episodes),
            args=args,
            all_results=all_results,
            results_csv=results_csv,
            summary_json=summary_json,
        )
        privileged_rows = _evaluate_group_stage(
            stage="final",
            group_name="privileged",
            checkpoints=privileged_ckpts,
            episodes=int(args.num_episodes),
            args=args,
            all_results=all_results,
            results_csv=results_csv,
            summary_json=summary_json,
        )

    resfit_ranked = _rank_group(resfit_rows)
    privileged_ranked = _rank_group(privileged_rows)
    _write_results_csv(all_results, results_csv)
    _write_summary_json(
        path=summary_json,
        args=args,
        resfit_ranked=resfit_ranked,
        privileged_ranked=privileged_ranked,
    )

    print("[DONE] Results written:")
    print(f"  - CSV: {results_csv}")
    print(f"  - JSON: {summary_json}")
    if resfit_ranked:
        best = resfit_ranked[0]
        print(
            "[BEST][resfit-vit] "
            f"sr={best.sr:.3f} ci=[{best.ci_low:.3f},{best.ci_high:.3f}] "
            f"successes={best.successes}/{best.episodes} step={best.step} ckpt={best.checkpoint}"
        )
    if privileged_ranked:
        best = privileged_ranked[0]
        print(
            "[BEST][privileged] "
            f"sr={best.sr:.3f} ci=[{best.ci_low:.3f},{best.ci_high:.3f}] "
            f"successes={best.successes}/{best.episodes} step={best.step} ckpt={best.checkpoint}"
        )


if __name__ == "__main__":
    main()
