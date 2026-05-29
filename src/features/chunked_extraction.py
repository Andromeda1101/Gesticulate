"""Memory-bounded feature extraction: manifest batches + incremental Parquet writes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.dataset_summary import count_manifest_rows, iter_manifest_batches
from src.features.batch_extraction import extract_samples_batch, resolve_num_workers
from src.features.feature_store import FeatureMatrixWriter, build_feature_manifest_from_matrix
from src.features.quality_checks import (
    _empty_coverage,
    evaluate_feature_coverage,
    finalize_feature_coverage,
    flag_low_confidence_samples,
    merge_feature_coverage,
)


def run_chunked_extraction(
    manifest_path: Path,
    output_path: Path,
    feature_family: str,
    config: dict[str, Any],
    *,
    batch_size: int,
    num_workers: int | None,
    pool_chunksize: int | None,
    apply_quality: bool,
    min_confidence: float,
    min_visible_landmarks: int | None,
    min_class_samples: int | None,
    min_class_rate: float | None,
    log_info,
) -> tuple[dict[str, Any], int]:
    """
    Extract features in manifest batches and append each batch to *output_path*.

    Returns merged coverage stats and total rows written.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    total_rows = count_manifest_rows(manifest_path)
    resolved_workers = resolve_num_workers(num_workers, min(batch_size, total_rows or 1))
    log_info(
        "Chunked extraction: %d samples, batch_size=%d, workers=%d",
        total_rows,
        batch_size,
        resolved_workers,
    )

    coverage = _empty_coverage()
    low_confidence_ids: list[str] = []
    written = 0
    batch_index = 0

    writer = FeatureMatrixWriter(output_path)
    try:
        for batch in iter_manifest_batches(manifest_path, batch_size):
            batch_index += 1
            records = extract_samples_batch(
                batch,
                feature_family,
                config,
                num_workers=num_workers,
                chunksize=pool_chunksize,
                apply_quality=apply_quality,
                min_confidence=min_confidence,
                min_visible_landmarks=min_visible_landmarks,
            )
            writer.append(records)

            chunk_coverage = evaluate_feature_coverage(records)
            coverage = merge_feature_coverage(coverage, chunk_coverage)
            low_confidence_ids.extend(flag_low_confidence_samples(records, min_confidence))

            written += len(records)
            log_info(
                "Batch %d: wrote %d rows (%d/%d, %.1f%%)",
                batch_index,
                len(records),
                written,
                total_rows,
                100.0 * written / total_rows if total_rows else 100.0,
            )
    finally:
        writer.close()

    coverage = finalize_feature_coverage(
        coverage,
        min_samples_per_class=min_class_samples,
        min_class_success_rate=min_class_rate,
    )
    coverage["low_confidence_sample_ids"] = low_confidence_ids
    return coverage, written


def finalize_chunked_artifacts(
    output_path: Path,
    *,
    feature_family: str,
    feature_version: str,
    config: dict[str, Any],
    coverage: dict[str, Any],
) -> dict[str, Any]:
    """Build manifest metadata after chunked extraction completes."""
    return build_feature_manifest_from_matrix(
        output_path,
        feature_family=feature_family,
        feature_version=feature_version,
        config=config,
        extraction_stats=coverage,
    )
