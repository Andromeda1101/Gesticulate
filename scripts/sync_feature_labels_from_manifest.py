#!/usr/bin/env python3
"""Rewrite gesture_label in feature Parquet files using a Phase 1 manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.path_manager import resolve_project_root
from src.features.feature_store import (
    build_feature_manifest_from_matrix,
    manifest_path_for_matrix,
    save_feature_manifest,
    sync_gesture_labels_from_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync canonical gesture_label values from manifest into feature matrices"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Phase 1 manifest (e.g. data/interim/leapgestrecog_manifest.parquet)",
    )
    parser.add_argument(
        "--matrix",
        action="append",
        required=True,
        help="Feature matrix to update (repeat for geometric, hog, hybrid)",
    )
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Rebuild sidecar *_manifest.json from updated matrix metadata",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = resolve_project_root()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path

    results: list[dict[str, int | str]] = []
    for matrix_arg in args.matrix:
        matrix_path = Path(matrix_arg)
        if not matrix_path.is_absolute():
            matrix_path = root / matrix_path
        updated = sync_gesture_labels_from_manifest(matrix_path, manifest_path)
        results.append({"matrix": str(matrix_path), "rows_updated": updated})

        if args.refresh_manifest:
            sidecar = manifest_path_for_matrix(matrix_path)
            existing = {}
            if sidecar.is_file():
                existing = json.loads(sidecar.read_text(encoding="utf-8"))
            refreshed = build_feature_manifest_from_matrix(
                matrix_path,
                feature_family=str(existing.get("feature_family", "unknown")),
                feature_version=str(existing.get("feature_version", "v1")),
                config={"_meta": {"config_path": "sync_feature_labels_from_manifest.py"}},
                extraction_stats=existing.get("extraction_stats"),
                source_families=existing.get("source_families"),
            )
            save_feature_manifest(refreshed, sidecar)

    print(json.dumps({"manifest": str(manifest_path), "matrices": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
