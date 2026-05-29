#!/usr/bin/env python3
"""CLI: batch feature extraction from a Phase 1 manifest."""

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
from src.data.dataset_summary import count_manifest_rows, load_manifest
from src.features.batch_extraction import (
    default_worker_count,
    extract_samples_batch,
    resolve_num_workers,
)
from src.features.chunked_extraction import finalize_chunked_artifacts, run_chunked_extraction
from src.features.extraction import SUPPORTED_FAMILIES
from src.features.feature_store import (
    build_feature_manifest,
    manifest_path_for_matrix,
    quality_report_path_for_matrix,
    save_feature_manifest,
    save_feature_matrix,
)
from src.features.quality_checks import evaluate_feature_coverage, flag_low_confidence_samples

logger = get_logger("scripts.extract_features")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract feature family from dataset manifest.")
    parser.add_argument("--manifest", required=True, help="Phase 1 manifest Parquet/CSV path")
    parser.add_argument(
        "--feature-family",
        required=True,
        choices=sorted(SUPPORTED_FAMILIES),
        help="Feature family to extract",
    )
    parser.add_argument("--config", required=True, help="Feature YAML config path")
    parser.add_argument("--output", required=True, help="Output feature matrix Parquet path")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Optional config override (dot paths supported)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Process pool size (default: config extraction.num_workers or "
            f"{default_worker_count()}). Use 1 for serial extraction."
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help="Tasks per worker batch within each manifest chunk (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Manifest rows per extract-and-flush cycle (default: config extraction.batch_size). "
            "Set to 0 to load the full manifest into memory."
        ),
    )
    return parser.parse_args()


def _parse_overrides(pairs: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def _resolve_batch_size(args: argparse.Namespace, extraction_cfg: dict) -> int | None:
    if args.batch_size is not None:
        if args.batch_size <= 0:
            return None
        return int(args.batch_size)
    configured = extraction_cfg.get("batch_size")
    if configured is None:
        return None
    configured = int(configured)
    return configured if configured > 0 else None


def _log_class_coverage(coverage: dict, quality_cfg: dict) -> None:
    min_class_samples = quality_cfg.get("min_samples_per_class")
    min_class_rate = quality_cfg.get("min_class_success_rate")

    for label, stats in sorted(coverage.get("by_gesture_label", {}).items()):
        logger.info(
            "Class '%s': %d/%d successful (%.1f%%)",
            label,
            stats["successful_extractions"],
            stats["total_samples"],
            stats["success_rate"] * 100.0,
        )
    if coverage.get("classes_below_min_samples"):
        logger.warning(
            "Classes below min_samples_per_class (%s): %s",
            min_class_samples,
            [item["gesture_label"] for item in coverage["classes_below_min_samples"]],
        )
    if coverage.get("classes_below_success_rate"):
        logger.warning(
            "Classes below min_class_success_rate (%s): %s",
            min_class_rate,
            [item["gesture_label"] for item in coverage["classes_below_success_rate"]],
        )


def main() -> int:
    args = _parse_args()
    root = resolve_project_root()

    config = load_config(args.config)
    if args.override:
        config = merge_overrides(config, _parse_overrides(args.override))

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path

    quality_cfg = config.get("quality_flags", {})
    min_confidence = float(quality_cfg.get("min_detection_confidence", 0.5))
    min_visible = quality_cfg.get("min_visible_landmarks")
    extraction_cfg = config.get("extraction", {})
    num_workers = args.workers
    if num_workers is None and extraction_cfg.get("num_workers") is not None:
        num_workers = int(extraction_cfg["num_workers"])
    pool_chunksize = args.chunksize
    if pool_chunksize is None and extraction_cfg.get("chunksize") is not None:
        pool_chunksize = int(extraction_cfg["chunksize"])

    batch_size = _resolve_batch_size(args, extraction_cfg)
    n_samples = count_manifest_rows(manifest_path)
    logger.info(
        "Extracting feature family '%s' for %d samples (batch_size=%s)",
        args.feature_family,
        n_samples,
        batch_size if batch_size else "all",
    )

    apply_quality = bool(quality_cfg.get("reject_low_confidence_landmarks"))
    min_class_samples = quality_cfg.get("min_samples_per_class")
    min_class_rate = quality_cfg.get("min_class_success_rate")
    min_visible_int = int(min_visible) if min_visible is not None else None

    if batch_size is not None:
        coverage, n_written = run_chunked_extraction(
            manifest_path,
            output_path,
            args.feature_family,
            config,
            batch_size=batch_size,
            num_workers=num_workers,
            pool_chunksize=pool_chunksize,
            apply_quality=apply_quality,
            min_confidence=min_confidence,
            min_visible_landmarks=min_visible_int,
            min_class_samples=int(min_class_samples) if min_class_samples is not None else None,
            min_class_rate=float(min_class_rate) if min_class_rate is not None else None,
            log_info=logger.info,
        )
        manifest = finalize_chunked_artifacts(
            output_path,
            feature_family=args.feature_family,
            feature_version=str(config.get("feature_version", "v1")),
            config=config,
            coverage=coverage,
        )
        records_count = n_written
    else:
        samples = load_manifest(manifest_path)
        resolved_workers = resolve_num_workers(num_workers, len(samples))
        logger.info("Using %d worker process(es)", resolved_workers)

        records = extract_samples_batch(
            samples,
            args.feature_family,
            config,
            num_workers=num_workers,
            chunksize=pool_chunksize,
            apply_quality=apply_quality,
            min_confidence=min_confidence,
            min_visible_landmarks=min_visible_int,
        )

        coverage = evaluate_feature_coverage(
            records,
            min_samples_per_class=int(min_class_samples) if min_class_samples is not None else None,
            min_class_success_rate=float(min_class_rate) if min_class_rate is not None else None,
        )
        coverage["low_confidence_sample_ids"] = flag_low_confidence_samples(records, min_confidence)

        save_feature_matrix(records, output_path)
        manifest = build_feature_manifest(
            records,
            feature_family=args.feature_family,
            feature_version=str(config.get("feature_version", "v1")),
            config=config,
            extraction_stats=coverage,
        )
        records_count = len(records)

    _log_class_coverage(coverage, quality_cfg)

    manifest_path_out = manifest_path_for_matrix(output_path)
    save_feature_manifest(manifest, manifest_path_out)

    quality_path = quality_report_path_for_matrix(output_path)
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    with quality_path.open("w", encoding="utf-8") as fh:
        json.dump(coverage, fh, indent=2)

    logger.info("Wrote %d records to %s", records_count, output_path)
    logger.info("Manifest: %s", manifest_path_out)
    logger.info("Quality report: %s", quality_path)

    print(
        json.dumps(
            {
                "feature_matrix": str(output_path),
                "manifest": str(manifest_path_out),
                "quality_report": str(quality_path),
                "n_samples": records_count,
                "vector_dim": manifest["vector_dim"],
                "success_rate": coverage["success_rate"],
                "batch_size": batch_size,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
