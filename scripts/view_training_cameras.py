#!/usr/bin/env python3
"""
View sample camera frames from the selected SO101 training dataset.
Saves top and wrist images so you can compare with your Isaac camera views.

Usage:
  python view_training_cameras.py
  python view_training_cameras.py --num_frames 5 --episode 0
  python view_training_cameras.py --output_dir ./my_camera_samples

Requires: pip install lerobot (or pip install 'lerobot[smolvla]')
"""

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError as e:
    print("Requires numpy and Pillow: pip install numpy pillow", file=sys.stderr)
    raise SystemExit(1) from e


def main():
    parser = argparse.ArgumentParser(
        description="Save sample top/wrist camera frames from an SO101 LeRobot dataset"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="legalaspro/so101-pick-and-place-cube-lerobot-50hz",
        help="HuggingFace dataset repo (default: selected SO101 pick-place dataset)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="view_training_cameras",
        help="Directory to save PNGs (default: view_training_cameras)",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=3,
        help="Number of frames to sample (default: 3)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to sample from (default: 0)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print Python path, lerobot location, and full traceback on import errors",
    )
    args = parser.parse_args()

    if args.debug:
        print(f"Python: {sys.executable}", file=sys.stderr)
        try:
            import lerobot
            print(f"lerobot: {getattr(lerobot, '__file__', 'no __file__')}", file=sys.stderr)
        except ImportError as e:
            print(f"lerobot import failed: {e}", file=sys.stderr)

    # Try multiple import paths (lerobot package layout varies by version)
    LeRobotDataset = None
    for import_path in (
        "lerobot.common.datasets.lerobot_dataset",
        "lerobot.common.datasets",
        "lerobot.datasets.lerobot_dataset",
        "lerobot.datasets",
    ):
        try:
            mod = __import__(import_path, fromlist=["LeRobotDataset"])
            LeRobotDataset = getattr(mod, "LeRobotDataset", None)
            if LeRobotDataset is not None:
                if args.debug:
                    print(f"Using LeRobotDataset from {import_path}", file=sys.stderr)
                break
        except Exception as e:
            if args.debug:
                import traceback
                print(f"Import {import_path} failed: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
            continue
    if LeRobotDataset is None:
        try:
            import lerobot
            print(
                f"LeRobot is installed (version {getattr(lerobot, '__version__', 'unknown')}) but LeRobotDataset not found.\n"
                "Try: pip install -U 'lerobot[dataset]' or pip install -U lerobot",
                file=sys.stderr,
            )
        except ImportError:
            print(
                "LeRobot is not installed or not on this Python's path.\n"
                f"Python: {sys.executable}\n"
                "Install with: pip install lerobot\n"
                "Run this script with the same Python that has lerobot (e.g. which python; python view_training_cameras.py).",
                file=sys.stderr,
            )
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading dataset: {args.repo_id}")
    try:
        dataset = LeRobotDataset(args.repo_id)
    except Exception as e:
        print(f"Failed to load dataset: {e}", file=sys.stderr)
        print("If using Hugging Face Hub, ensure you can reach it (e.g. huggingface-cli login if needed).", file=sys.stderr)
        sys.exit(1)

    # Find frame indices: from the chosen episode if possible, else first frames
    n_total = len(dataset)
    if hasattr(dataset, "hf_dataset") and "episode_index" in dataset.hf_dataset.column_names:
        ep_col = dataset.hf_dataset["episode_index"]
        indices = [i for i, e in enumerate(ep_col) if e == args.episode]
        if indices:
            start, end = indices[0], indices[-1] + 1
            n_ep = end - start
            step = max(1, (n_ep - 1) // max(1, args.num_frames - 1))
            frame_indices = list(range(start, min(start + step * args.num_frames, end), step))[: args.num_frames]
        else:
            print(f"Episode {args.episode} not found. Using first {args.num_frames} frames.")
            frame_indices = list(range(0, min(args.num_frames, n_total)))
    else:
        step = max(1, (n_total - 1) // max(1, args.num_frames))
        frame_indices = list(range(0, min(step * args.num_frames, n_total), step))[: args.num_frames]

    # Prefer dataset-aligned keys; keep legacy side/up fallback.
    sample = dataset[frame_indices[0]] if frame_indices else {}
    top_key = (
        "observation.images.top"
        if "observation.images.top" in sample
        else ("observation.images_top" if "observation.images_top" in sample else "observation.images.side")
    )
    wrist_key = (
        "observation.images.wrist"
        if "observation.images.wrist" in sample
        else ("observation.images_wrist" if "observation.images_wrist" in sample else "observation.images.up")
    )

    for idx in frame_indices:
        frame = dataset[idx]
        for key, name in [(top_key, "top"), (wrist_key, "wrist")]:
            if key not in frame:
                print(f"Warning: {key} not in frame keys: {list(frame.keys())}")
                continue
            img = frame[key]
            if hasattr(img, "numpy"):
                img = img.numpy()
            else:
                img = np.asarray(img)
            # (C,H,W) -> (H,W,C) for PIL
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = np.transpose(img, (1, 2, 0))
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            out_path = out_dir / f"{name}_ep{args.episode}_frame{idx}.png"
            Image.fromarray(img).save(out_path)
            print(f"Saved {out_path} ({img.shape[0]}x{img.shape[1]})")

    print(f"\nDone. Open images in {out_dir.absolute()} to see how top/wrist cameras looked in the training data.")


if __name__ == "__main__":
    main()
