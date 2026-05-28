#!/usr/bin/env python3
"""CLI: export dataset summary report from an existing manifest."""

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
from src.data.dataset_summary import export_summary, load_manifest, summarize_dataset
from src.data.label_mapper import validate_label_coverage

logger = get_logger("scripts.export_dataset_report")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export dataset summary JSON from manifest.")
    parser.add_argument("--manifest", required=True, help="Path to manifest Parquet/CSV")
    parser.add_argument("--output", required=True, help="Output JSON summary path")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional dataset config for canonical labels and overlap checks",
    )
    parser.add_argument(
        "--reference-manifest",
        default=None,
        help="Optional second manifest for label overlap analysis",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = resolve_project_root()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path

    samples = load_manifest(manifest_path)
    reference: list[str] | None = None
    if args.config:
        config = load_config(args.config)
        reference = config.get("label_vocabulary")

    reference_labels: list[str] | None = None
    if args.reference_manifest:
        ref_path = Path(args.reference_manifest)
        if not ref_path.is_absolute():
            ref_path = root / ref_path
        ref_samples = load_manifest(ref_path)
        reference_labels = sorted({s["gesture_label"] for s in ref_samples})

    summary = summarize_dataset(
        samples,
        reference_labels=reference_labels or reference,
    )
    if reference:
        summary["label_coverage"] = validate_label_coverage(samples, reference)

    export_summary(summary, output_path)
    logger.info("Exported report for %d samples to %s", len(samples), output_path)
    print(json.dumps({"output": str(output_path), "total_samples": len(samples)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
