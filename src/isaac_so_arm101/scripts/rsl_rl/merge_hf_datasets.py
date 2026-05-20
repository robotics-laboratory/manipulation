#!/usr/bin/env python3
"""Merge synthetic and organic Hugging Face datasets into one dataset."""

from __future__ import annotations

import argparse
from typing import Sequence

from datasets import Dataset, DatasetDict, concatenate_datasets, interleave_datasets, load_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-dataset",
        type=str,
        required=True,
        help="Hugging Face dataset id for synthetic data (e.g. user/synth-ds).",
    )
    parser.add_argument(
        "--organic-dataset",
        type=str,
        required=True,
        help="Hugging Face dataset id for organic/real data (e.g. user/organic-ds).",
    )
    parser.add_argument(
        "--synthetic-config",
        type=str,
        default=None,
        help="Optional config/subset name for synthetic dataset.",
    )
    parser.add_argument(
        "--organic-config",
        type=str,
        default=None,
        help="Optional config/subset name for organic dataset.",
    )
    parser.add_argument("--synthetic-revision", type=str, default=None, help="Optional git revision for synthetic dataset.")
    parser.add_argument("--organic-revision", type=str, default=None, help="Optional git revision for organic dataset.")
    parser.add_argument(
        "--splits",
        type=str,
        default="train",
        help="Comma-separated split list to merge (e.g. train or train,validation).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["concatenate", "interleave"],
        default="interleave",
        help="Merge strategy: plain concatenate or probabilistic interleave.",
    )
    parser.add_argument(
        "--synthetic-ratio",
        type=float,
        default=0.7,
        help="Synthetic probability for --strategy=interleave, in [0,1].",
    )
    parser.add_argument(
        "--stopping-strategy",
        type=str,
        choices=["all_exhausted", "first_exhausted"],
        default="all_exhausted",
        help="Interleave stopping strategy.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for interleave/shuffle.")
    parser.add_argument(
        "--shuffle-after-merge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle merged split after merge (default: true).",
    )
    parser.add_argument(
        "--add-source-column",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add source column to track data origin (default: true).",
    )
    parser.add_argument(
        "--source-column-name",
        type=str,
        default="data_source",
        help="Name of source-tracking column.",
    )
    parser.add_argument(
        "--allow-schema-mismatch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow missing columns across datasets (fills missing values with None).",
    )
    parser.add_argument(
        "--output-repo-id",
        type=str,
        default=None,
        help="If set, push merged dataset to this Hub dataset id.",
    )
    parser.add_argument("--private", action="store_true", help="Create output Hub dataset as private.")
    parser.add_argument(
        "--save-to-disk",
        type=str,
        default=None,
        help="Optional local path to save merged dataset via save_to_disk().",
    )
    return parser.parse_args()


def _parse_splits(value: str) -> list[str]:
    splits = [s.strip() for s in value.split(",") if s.strip()]
    if not splits:
        raise ValueError("--splits must contain at least one split.")
    return splits


def _load_split(dataset_id: str, config: str | None, revision: str | None, split: str) -> Dataset:
    return load_dataset(path=dataset_id, name=config, revision=revision, split=split)


def _add_source_column(dataset: Dataset, *, source_column: str, source_value: str) -> Dataset:
    if source_column in dataset.column_names:
        raise ValueError(
            f"Column '{source_column}' already exists in dataset. "
            "Use --source-column-name with another name or disable --add-source-column."
        )
    return dataset.add_column(source_column, [source_value] * len(dataset))


def _ensure_compatible_schema(a: Dataset, b: Dataset, allow_schema_mismatch: bool) -> tuple[Dataset, Dataset]:
    if a.features == b.features:
        return a, b
    if not allow_schema_mismatch:
        raise ValueError(
            "Dataset schemas differ.\n"
            f"Synthetic features: {a.features}\n"
            f"Organic features:   {b.features}\n"
            "Pass --allow-schema-mismatch to fill missing columns with None when types match."
        )

    a_cols = set(a.column_names)
    b_cols = set(b.column_names)
    union_order: list[str] = list(a.column_names) + [c for c in b.column_names if c not in a_cols]

    # For shared columns, require equal feature type to avoid silent corruption.
    shared = a_cols & b_cols
    type_mismatch = [c for c in sorted(shared) if a.features[c] != b.features[c]]
    if type_mismatch:
        detail = ", ".join(type_mismatch)
        raise ValueError(
            "Shared columns have incompatible feature types even with --allow-schema-mismatch.\n"
            f"Mismatched columns: {detail}"
        )

    for col in union_order:
        if col not in a.column_names:
            a = a.add_column(col, [None] * len(a))
        if col not in b.column_names:
            b = b.add_column(col, [None] * len(b))
    return a.select_columns(union_order), b.select_columns(union_order)


def _merge_pair(
    synthetic: Dataset,
    organic: Dataset,
    *,
    strategy: str,
    synthetic_ratio: float,
    stopping_strategy: str,
    seed: int,
    shuffle_after_merge: bool,
) -> Dataset:
    if strategy == "concatenate":
        merged = concatenate_datasets([synthetic, organic])
    elif strategy == "interleave":
        merged = interleave_datasets(
            [synthetic, organic],
            probabilities=[synthetic_ratio, 1.0 - synthetic_ratio],
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    if shuffle_after_merge:
        merged = merged.shuffle(seed=seed)
    return merged


def _validate_ratio(value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"--synthetic-ratio must be in [0,1], got {value}")


def _merge_splits(args: argparse.Namespace, splits: Sequence[str]) -> Dataset | DatasetDict:
    merged_dict: dict[str, Dataset] = {}
    for split in splits:
        synthetic = _load_split(args.synthetic_dataset, args.synthetic_config, args.synthetic_revision, split)
        organic = _load_split(args.organic_dataset, args.organic_config, args.organic_revision, split)

        if args.add_source_column:
            synthetic = _add_source_column(
                synthetic, source_column=args.source_column_name, source_value="synthetic"
            )
            organic = _add_source_column(organic, source_column=args.source_column_name, source_value="organic")

        synthetic, organic = _ensure_compatible_schema(
            synthetic,
            organic,
            allow_schema_mismatch=bool(args.allow_schema_mismatch),
        )

        merged = _merge_pair(
            synthetic,
            organic,
            strategy=args.strategy,
            synthetic_ratio=args.synthetic_ratio,
            stopping_strategy=args.stopping_strategy,
            seed=args.seed,
            shuffle_after_merge=bool(args.shuffle_after_merge),
        )
        merged_dict[split] = merged
        print(
            f"[INFO] split={split} synthetic={len(synthetic)} organic={len(organic)} merged={len(merged)} "
            f"strategy={args.strategy}"
        )

    if len(merged_dict) == 1:
        return next(iter(merged_dict.values()))
    return DatasetDict(merged_dict)


def main() -> None:
    args = _parse_args()
    _validate_ratio(float(args.synthetic_ratio))
    splits = _parse_splits(args.splits)

    merged = _merge_splits(args, splits)

    if args.save_to_disk:
        merged.save_to_disk(args.save_to_disk)
        print(f"[DONE] Saved merged dataset to: {args.save_to_disk}")

    if args.output_repo_id:
        merged.push_to_hub(args.output_repo_id, private=bool(args.private))
        print(f"[DONE] Pushed merged dataset: https://huggingface.co/datasets/{args.output_repo_id}")
    elif not args.save_to_disk:
        print("[DONE] Merge completed. Use --save-to-disk and/or --output-repo-id to persist results.")


if __name__ == "__main__":
    main()

