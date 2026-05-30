#!/usr/bin/env python3
"""CLI: join feature families into a hybrid feature matrix."""

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
from src.features.feature_combiner import build_feature_record, concatenate_features
from src.features.feature_store import (
    build_feature_manifest,
    load_feature_matrix,
    manifest_path_for_matrix,
    quality_report_path_for_matrix,
    save_feature_manifest,
    save_feature_matrix,
    vector_from_record,
)
from src.features.quality_checks import evaluate_feature_coverage

logger = get_logger("scripts.build_hybrid_features")

HYBRID_FAMILY = "hybrid_keypoints_hog"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hybrid feature matrix from two families.")
    parser.add_argument(
        "--keypoint-features",
        required=True,
        help="Geometric or keypoints feature Parquet path",
    )
    parser.add_argument("--hog-features", required=True, help="HOG feature Parquet path")
    parser.add_argument("--output", required=True, help="Output hybrid Parquet path")
    parser.add_argument(
        "--config",
        default="configs/features/default.yaml",
        help="Feature config (concat order and version)",
    )
    return parser.parse_args()


def _index_by_sample_id(table) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for record in table.records:
        sid = str(record["sample_id"])
        if sid in indexed:
            raise ValueError(f"Duplicate sample_id in feature table: {sid}")
        indexed[sid] = record
    return indexed


def main() -> int:
    args = _parse_args()
    root = resolve_project_root()
    config = load_config(args.config)

    keypoint_path = Path(args.keypoint_features)
    hog_path = Path(args.hog_features)
    output_path = Path(args.output)
    for path in (keypoint_path, hog_path, output_path):
        if not path.is_absolute():
            path = root / path
    keypoint_path, hog_path, output_path = keypoint_path, hog_path, output_path

    keypoint_table = load_feature_matrix(keypoint_path)
    hog_table = load_feature_matrix(hog_path, exclude_invalid=False)

    kp_by_id = _index_by_sample_id(keypoint_table)
    hog_by_id = _index_by_sample_id(hog_table)

    kp_ids = list(kp_by_id.keys())
    missing_in_hog = sorted(set(kp_ids) - set(hog_by_id))
    if missing_in_hog:
        raise ValueError(
            "HOG feature table missing rows for valid geometric samples. "
            f"Missing ({len(missing_in_hog)}): {missing_in_hog[:5]}..."
        )
    skipped_invalid_geom = len(hog_by_id) - len(kp_ids)
    if skipped_invalid_geom:
        logger.info(
            "Skipping %d samples with invalid geometric features when building hybrid",
            skipped_invalid_geom,
        )

    kp_family = str(kp_by_id[kp_ids[0]].get("feature_family", "geometric"))
    hog_family = str(hog_by_id[hog_ids[0]].get("feature_family", "hog"))
    family_order = [kp_family, hog_family]

    feature_version = str(config.get("feature_version", "v1"))
    source_families = list(family_order)

    records: list[dict] = []
    label_mismatches: list[str] = []

    for sample_id in kp_ids:
        kp_rec = kp_by_id[sample_id]
        hog_rec = hog_by_id[sample_id]
        if kp_rec.get("gesture_label") != hog_rec.get("gesture_label"):
            label_mismatches.append(sample_id)

        blocks = {
            kp_family: vector_from_record(kp_rec),
            hog_family: vector_from_record(hog_rec),
        }
        hybrid_vector = concatenate_features(blocks, family_order=family_order)

        quality_flags = {
            **(kp_rec.get("quality_flags") or {}),
            **(hog_rec.get("quality_flags") or {}),
        }
        metadata = {
            "dataset_name": kp_rec.get("dataset_name"),
            "gesture_label": kp_rec.get("gesture_label"),
            "extraction_ok": bool(kp_rec.get("extraction_ok")) and bool(hog_rec.get("extraction_ok")),
            "quality_flags": quality_flags,
            "source_families": source_families,
            "confidence": min(
                float(kp_rec["confidence"]) if kp_rec.get("confidence") is not None else 1.0,
                float(hog_rec["confidence"]) if hog_rec.get("confidence") is not None else 1.0,
            ),
        }
        records.append(
            build_feature_record(
                sample_id,
                HYBRID_FAMILY,
                hybrid_vector,
                metadata,
                feature_version=feature_version,
            )
        )

    if label_mismatches:
        raise ValueError(
            f"Label mismatch for {len(label_mismatches)} samples, e.g. {label_mismatches[:3]}"
        )

    quality_cfg = config.get("quality_flags", {})
    min_class_samples = quality_cfg.get("min_samples_per_class")
    min_class_rate = quality_cfg.get("min_class_success_rate")
    coverage = evaluate_feature_coverage(
        records,
        min_samples_per_class=int(min_class_samples) if min_class_samples is not None else None,
        min_class_success_rate=float(min_class_rate) if min_class_rate is not None else None,
    )
    coverage["label_mismatches"] = label_mismatches

    save_feature_matrix(records, output_path)
    manifest = build_feature_manifest(
        records,
        feature_family=HYBRID_FAMILY,
        feature_version=feature_version,
        config=config,
        extraction_stats=coverage,
        source_families=source_families,
    )
    manifest_path = manifest_path_for_matrix(output_path)
    save_feature_manifest(manifest, manifest_path)

    quality_path = quality_report_path_for_matrix(output_path)
    with quality_path.open("w", encoding="utf-8") as fh:
        json.dump(coverage, fh, indent=2)

    logger.info("Wrote hybrid features for %d samples to %s", len(records), output_path)
    print(
        json.dumps(
            {
                "hybrid_matrix": str(output_path),
                "manifest": str(manifest_path),
                "n_samples": len(records),
                "vector_dim": manifest["vector_dim"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
