#!/usr/bin/env python3
"""CLI: index raw datasets and write canonical manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config_loader import load_config, merge_overrides
from src.common.logger import get_logger
from src.common.path_manager import resolve_project_root
from src.data.dataset_registry import index_dataset, list_supported_datasets
from src.data.dataset_summary import export_summary, save_manifest, summarize_dataset
from src.data.label_mapper import apply_label_normalization, validate_label_coverage

logger = get_logger("scripts.build_dataset_manifests")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset manifest from raw data.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list_supported_datasets(),
        help="Dataset adapter name",
    )
    parser.add_argument("--config", required=True, help="Path to dataset YAML config")
    parser.add_argument(
        "--output",
        required=True,
        help="Output manifest path (Parquet or CSV)",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional JSON summary output path",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)

    if config.get("dataset_name") != args.dataset:
        logger.warning(
            "CLI dataset '%s' differs from config dataset_name '%s'",
            args.dataset,
            config.get("dataset_name"),
        )

    logger.info("Indexing dataset '%s' from %s", args.dataset, config.get("root_path"))
    samples = index_dataset(config)

    reference = config.get("label_vocabulary") or []
    align_to = config.get("align_to")
    align_to_canonical = align_to in ("canonical", "leapgestrecog")
    samples = apply_label_normalization(
        samples,
        args.dataset,
        label_aliases=config.get("label_aliases"),
        canonical_labels=reference or None,
        align_to_canonical=align_to_canonical,
    )

    coverage = validate_label_coverage(samples, reference or None)
    if coverage.get("outside_reference"):
        logger.info(
            "Labels outside canonical reference (retained): %s",
            coverage["outside_reference"],
        )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = resolve_project_root() / output_path

    save_manifest(samples, output_path)
    logger.info("Wrote manifest with %d samples to %s", len(samples), output_path)

    summary = summarize_dataset(samples, reference_labels=reference or None)
    summary["label_coverage"] = coverage

    summary_path = args.summary
    if summary_path is None:
        dataset_slug = args.dataset.replace("_subset", "_subset")
        summary_path = (
            resolve_project_root()
            / "reports"
            / "summaries"
            / f"{dataset_slug}_summary.json"
        )
    else:
        summary_path = Path(summary_path)
        if not summary_path.is_absolute():
            summary_path = resolve_project_root() / summary_path

    export_summary(summary, summary_path)
    logger.info("Wrote summary to %s", summary_path)

    print(json.dumps({"manifest": str(output_path), "summary": str(summary_path), "n_samples": len(samples)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
