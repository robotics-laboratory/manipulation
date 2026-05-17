#!/usr/bin/env python3
"""Create and push a random episode subset of a LeRobot dataset."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-repo-id", type=str, required=True, help="Source LeRobot repo id, e.g. username/dataset.")
    parser.add_argument("--dst-repo-id", type=str, required=True, help="Destination LeRobot repo id for subset.")
    parser.add_argument("--num-episodes", type=int, default=60, help="Number of episodes to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for episode sampling.")
    parser.add_argument("--revision", type=str, default=None, help="Optional source dataset revision/branch/tag.")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Optional local LeRobot cache root (defaults to ~/.cache/huggingface/lerobot).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional explicit local output directory. Defaults to <root>/<dst-repo-id>.",
    )
    parser.add_argument(
        "--overwrite-local-output",
        action="store_true",
        help="Delete existing local output directory before creating subset.",
    )
    parser.add_argument("--private", action="store_true", help="Create destination Hub dataset as private.")
    parser.add_argument(
        "--push-to-hub",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Push generated subset to Hugging Face Hub (default: true).",
    )
    parser.add_argument(
        "--force-cache-sync",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force re-sync source dataset metadata/files from Hub before processing.",
    )
    parser.add_argument(
        "--download-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download source videos locally if dataset is video-based (default: true).",
    )
    return parser.parse_args()


def _choose_episode_ids(episode_ids: list[int], num_episodes: int, seed: int) -> set[int]:
    if num_episodes <= 0:
        raise ValueError(f"--num-episodes must be > 0, got: {num_episodes}")
    if num_episodes > len(episode_ids):
        raise ValueError(
            f"Requested {num_episodes} episodes but source has only {len(episode_ids)} unique episodes."
        )
    rng = random.Random(seed)
    return set(rng.sample(episode_ids, k=num_episodes))


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir).expanduser().resolve()
    root = Path(args.root).expanduser().resolve() if args.root is not None else HF_LEROBOT_HOME
    return (root / args.dst_repo_id).resolve()


def main() -> None:
    args = _parse_args()
    if args.src_repo_id == args.dst_repo_id:
        raise ValueError("--src-repo-id and --dst-repo-id must differ.")

    src_dataset = LeRobotDataset(
        repo_id=args.src_repo_id,
        root=args.root,
        revision=args.revision,
        force_cache_sync=bool(args.force_cache_sync),
        download_videos=bool(args.download_videos),
    )

    total_episodes = int(src_dataset.meta.total_episodes)
    all_episode_ids = list(range(total_episodes))
    selected_episodes = _choose_episode_ids(all_episode_ids, args.num_episodes, args.seed)
    selected_sorted = sorted(selected_episodes)
    episodes_to_delete = sorted(set(all_episode_ids) - selected_episodes)
    output_dir = _resolve_output_dir(args)

    if output_dir.exists():
        if not args.overwrite_local_output:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Use --overwrite-local-output to replace it."
            )
        shutil.rmtree(output_dir)
        print(f"[INFO] Removed existing output directory: {output_dir}")

    print(
        f"[INFO] Source repo={args.src_repo_id} total_episodes={total_episodes} "
        f"selected={len(selected_episodes)} deleted={len(episodes_to_delete)} seed={args.seed}"
    )
    print(f"[INFO] Writing LeRobot subset locally to: {output_dir}")

    if episodes_to_delete:
        subset_dataset = delete_episodes(
            dataset=src_dataset,
            episode_indices=episodes_to_delete,
            output_dir=output_dir,
            repo_id=args.dst_repo_id,
        )
    else:
        print("[INFO] Selected all source episodes; creating a clean full copy for destination repo.")
        shutil.copytree(src_dataset.root, output_dir, dirs_exist_ok=True)
        subset_dataset = LeRobotDataset(
            repo_id=args.dst_repo_id,
            root=output_dir,
            revision=args.revision,
            force_cache_sync=False,
            download_videos=bool(args.download_videos),
        )

    print(
        f"[INFO] Local subset ready: episodes={subset_dataset.meta.total_episodes} "
        f"frames={subset_dataset.meta.total_frames}"
    )
    if args.push_to_hub:
        subset_dataset.push_to_hub(private=bool(args.private))
        print(f"[DONE] Pushed LeRobot subset: https://huggingface.co/datasets/{args.dst_repo_id}")
    else:
        print("[DONE] Subset created locally (push disabled).")
    print(
        f"[INFO] Sampled episode ids ({len(selected_sorted)}): "
        f"{selected_sorted[:10]}{' ...' if len(selected_sorted) > 10 else ''}"
    )


if __name__ == "__main__":
    main()
