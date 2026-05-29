"""Dataset statistics and quality checks."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def summarize_dataset(
    samples: list[dict[str, Any]],
    *,
    canonical_labels: list[str] | None = None,
    reference_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Compute sample counts, subject distribution, and file integrity flags."""
    if not samples:
        return {
            "total_samples": 0,
            "class_counts": {},
            "subject_counts": {},
            "missing_files": [],
            "duplicate_sample_ids": [],
            "label_overlap_with_reference": {},
        }

    df = pd.DataFrame(samples)
    sample_ids = df["sample_id"].tolist()
    id_counts = Counter(sample_ids)
    duplicates = sorted([sid for sid, count in id_counts.items() if count > 1])

    missing_files: list[str] = []
    for _, row in df.iterrows():
        path = Path(str(row["image_path"]))
        if not path.is_file():
            missing_files.append(str(row["sample_id"]))

    class_counts = dict(Counter(df["gesture_label"].astype(str)))
    subject_series = df.get("subject_id")
    if subject_series is not None:
        subject_counts = dict(Counter(subject_series.fillna("unknown").astype(str)))
    else:
        subject_counts = {}

    summary: dict[str, Any] = {
        "total_samples": len(samples),
        "dataset_name": df["dataset_name"].iloc[0] if "dataset_name" in df.columns else None,
        "class_counts": class_counts,
        "subject_counts": subject_counts,
        "missing_files": missing_files,
        "missing_file_count": len(missing_files),
        "duplicate_sample_ids": duplicates,
        "has_duplicate_sample_ids": bool(duplicates),
    }

    if canonical_labels:
        observed = set(class_counts)
        summary["canonical_labels"] = list(canonical_labels)
        summary["unknown_labels"] = sorted(observed - set(canonical_labels))
        summary["missing_canonical_labels"] = sorted(set(canonical_labels) - observed)

    if reference_labels is not None:
        observed = set(class_counts)
        ref = set(reference_labels)
        summary["label_overlap_with_reference"] = {
            "shared": sorted(observed & ref),
            "only_in_dataset": sorted(observed - ref),
            "only_in_reference": sorted(ref - observed),
        }

    return summary


def export_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return path


def save_manifest(samples: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Persist canonical manifest as Parquet (fallback CSV if pyarrow missing)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(samples)

    if path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(path, index=False)
        except ImportError as exc:
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            raise ImportError(
                "pyarrow is required for Parquet output. "
                f"Wrote CSV fallback to {csv_path}"
            ) from exc
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported manifest format: {path.suffix}")

    return path


def iter_manifest_batches(
    manifest_path: str | Path,
    batch_size: int,
):
    """
    Yield manifest rows in fixed-size batches without loading the full table.

    Falls back to slicing an in-memory manifest for non-Parquet inputs.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError:
            records = load_manifest(path)
            for start in range(0, len(records), batch_size):
                yield records[start : start + batch_size]
            return

        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            yield batch.to_pandas().to_dict(orient="records")
        return

    records = load_manifest(path)
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def count_manifest_rows(manifest_path: str | Path) -> int:
    """Return row count without loading all manifest columns into memory."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq

            return int(pq.ParquetFile(path).metadata.num_rows)
        except ImportError:
            return len(load_manifest(path))

    return len(load_manifest(path))


def load_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported manifest format: {path.suffix}")

    return df.to_dict(orient="records")
