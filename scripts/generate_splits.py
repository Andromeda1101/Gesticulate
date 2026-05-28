#!/usr/bin/env python3
"""CLI: create train/val/test splits and CV folds from a manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config_loader import load_config
from src.common.logger import get_logger
from src.common.path_manager import resolve_project_root
from src.data.dataset_summary import load_manifest
from src.data.label_mapper import validate_label_coverage
from src.data.split_generator import (
    create_primary_splits,
    create_stratified_folds,
    save_folds,
    save_splits,
    split_distribution,
)

logger = get_logger("scripts.generate_splits")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset splits and CV folds.")
    parser.add_argument("--manifest", required=True, help="Path to manifest Parquet/CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional dataset config for split_strategy ratios",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Prefix for split files under data/splits/ (default: dataset name)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = resolve_project_root() / manifest_path

    samples = load_manifest(manifest_path)
    if not samples:
        logger.error("Manifest is empty: %s", manifest_path)
        return 1

    dataset_name = samples[0].get("dataset_name", "dataset")
    prefix = args.output_prefix or dataset_name

    split_strategy = None
    if args.config:
        config = load_config(args.config)
        split_strategy = config.get("split_strategy")
        reference = config.get("label_vocabulary") or []
        if reference:
            coverage = validate_label_coverage(samples, reference)
            logger.info("Label distribution: %s", coverage.get("observed_labels"))

    splits = create_primary_splits(samples, seed=args.seed, split_strategy=split_strategy)
    folds = create_stratified_folds(
        samples,
        n_folds=args.folds,
        seed=args.seed,
        primary_splits=splits,
    )

    splits_dir = resolve_project_root() / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    tvt_path = save_splits(splits, splits_dir / f"{prefix}_train_val_test.json")
    folds_path = save_folds(folds, splits_dir / f"{prefix}_cv_folds.json")

    distribution = split_distribution(samples, splits)
    logger.info("Split distribution: %s", distribution)
    logger.info("Wrote splits to %s and folds to %s", tvt_path, folds_path)

    result = {
        "train_val_test": str(tvt_path),
        "cv_folds": str(folds_path),
        "distribution": distribution,
        "n_folds": len(folds),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
