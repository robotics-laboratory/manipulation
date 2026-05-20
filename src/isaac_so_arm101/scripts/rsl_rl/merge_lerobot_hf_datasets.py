#!/usr/bin/env python3
"""Merge multiple LeRobot datasets and optionally push to Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-ids",
        type=str,
        required=True,
        help="Comma-separated source LeRobot dataset repo ids.",
    )
    parser.add_argument(
        "--output-repo-id",
        type=str,
        required=True,
        help="Destination LeRobot dataset repo id.",
    )
    parser.add_argument(
        "--roots",
        type=str,
        default=None,
        help="Optional comma-separated local roots for source datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional explicit local output directory for merged dataset.",
    )
    parser.add_argument(
        "--force-cache-sync",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force re-sync source metadata/files from Hub before merging.",
    )
    parser.add_argument(
        "--download-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download source videos when needed (default: true).",
    )
    parser.add_argument(
        "--push-to-hub",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Push merged LeRobot dataset to Hub (default: true).",
    )
    parser.add_argument("--private", action="store_true", help="Push as private dataset.")
    return parser.parse_args()


def _split_csv(value: str) -> list[str]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise ValueError("Expected a non-empty comma-separated list.")
    return items


def _parse_roots(value: str | None, n: int) -> list[Path | None]:
    if value is None:
        return [None] * n
    roots_raw = _split_csv(value)
    if len(roots_raw) != n:
        raise ValueError(
            f"--roots count ({len(roots_raw)}) must match --repo-ids count ({n})."
        )
    return [Path(r).expanduser().resolve() for r in roots_raw]


def main() -> None:
    args = _parse_args()
    repo_ids = _split_csv(args.repo_ids)
    roots = _parse_roots(args.roots, len(repo_ids))

    datasets: list[LeRobotDataset] = []
    for repo_id, root in zip(repo_ids, roots):
        ds = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            force_cache_sync=bool(args.force_cache_sync),
            download_videos=bool(args.download_videos),
        )
        print(
            f"[INFO] Loaded source {repo_id}: episodes={ds.meta.total_episodes} "
            f"frames={ds.meta.total_frames}"
        )
        datasets.append(ds)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    merged = merge_datasets(
        datasets=datasets,
        output_repo_id=args.output_repo_id,
        output_dir=output_dir,
    )
    print(
        f"[INFO] Merged dataset ready: repo={args.output_repo_id} "
        f"episodes={merged.meta.total_episodes} frames={merged.meta.total_frames}"
    )

    if args.push_to_hub:
        merged.push_to_hub(private=bool(args.private))
        print(f"[DONE] Pushed merged LeRobot dataset: https://huggingface.co/datasets/{args.output_repo_id}")
    else:
        print("[DONE] Merged LeRobot dataset created locally (push disabled).")


if __name__ == "__main__":
    main()

