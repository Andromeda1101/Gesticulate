#!/usr/bin/env python3
"""CLI: export feature extraction summary report from manifest JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.logger import get_logger
from src.common.path_manager import resolve_project_root
from src.data.dataset_summary import export_summary
from src.features.feature_store import load_feature_matrix
from src.features.quality_checks import evaluate_feature_coverage, flag_low_confidence_samples

logger = get_logger("scripts.export_feature_report")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export feature quality summary JSON.")
    parser.add_argument(
        "--feature-manifest",
        required=True,
        help="Feature manifest JSON path",
    )
    parser.add_argument("--output", required=True, help="Output summary JSON path")
    parser.add_argument(
        "--feature-matrix",
        default=None,
        help="Optional feature matrix path to recompute live coverage",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Threshold for low-confidence sample listing",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = resolve_project_root()

    manifest_path = Path(args.feature_manifest)
    output_path = Path(args.output)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if not output_path.is_absolute():
        output_path = root / output_path

    with manifest_path.open(encoding="utf-8") as fh:
        manifest = json.load(fh)

    report: dict = {
        "feature_family": manifest.get("feature_family"),
        "feature_version": manifest.get("feature_version"),
        "vector_dim": manifest.get("vector_dim"),
        "n_samples": manifest.get("n_samples"),
        "config_fingerprint": manifest.get("config_fingerprint"),
        "config_path": manifest.get("config_path"),
        "source_families": manifest.get("source_families"),
        "extraction_stats": manifest.get("extraction_stats", {}),
        "manifest_path": str(manifest_path),
    }

    matrix_path = args.feature_matrix
    if matrix_path:
        path = Path(matrix_path)
        if not path.is_absolute():
            path = root / path
        table = load_feature_matrix(path, exclude_invalid=False)
        coverage = evaluate_feature_coverage(table.records)
        report["live_coverage"] = coverage
        report["low_confidence_sample_ids"] = flag_low_confidence_samples(
            table.records,
            args.confidence_threshold,
        )
        report["feature_matrix_path"] = str(path)

    export_summary(report, output_path)
    logger.info("Exported feature report to %s", output_path)
    print(json.dumps({"output": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
