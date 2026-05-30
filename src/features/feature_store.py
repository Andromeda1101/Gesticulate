"""Persist and load feature matrices and manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.logger import get_logger
from src.features.quality_checks import filter_invalid_geometric_feature_records

_logger = get_logger(__name__)


@dataclass
class FeatureTable:
    """In-memory feature table loaded from disk."""

    records: list[dict[str, Any]]
    path: Path | None = None

    @property
    def sample_ids(self) -> list[str]:
        return [str(r["sample_id"]) for r in self.records]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


def config_fingerprint(config: dict[str, Any]) -> str:
    """Stable hash of configuration for reproducibility metadata."""
    payload = {k: v for k, v in config.items() if k != "_meta"}
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _vector_dim(records: list[dict[str, Any]]) -> int:
    for record in records:
        inline = record.get("vector_inline")
        if inline is not None and len(inline) > 0:
            return len(inline)
    return 0


def _serialize_for_storage(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize nested fields for Parquet compatibility."""
    serialized: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        flags = row.get("quality_flags")
        if isinstance(flags, dict):
            row["quality_flags"] = json.dumps(flags)
        serialized.append(row)
    return serialized


def _deserialize_from_storage(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deserialized: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        flags = row.get("quality_flags")
        if isinstance(flags, str):
            row["quality_flags"] = json.loads(flags)
        deserialized.append(row)
    return deserialized


def save_feature_matrix(records: list[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(_serialize_for_storage(records))
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported feature matrix format: {path.suffix}")
    return path


class FeatureMatrixWriter:
    """Append feature batches to a single Parquet file with bounded memory."""

    def __init__(self, output_path: str | Path) -> None:
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._parquet_writer: Any = None
        self._part_paths: list[Path] = []

    def append(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return

        df = pd.DataFrame(_serialize_for_storage(records))
        if self.path.suffix.lower() != ".parquet":
            raise ValueError(f"Chunked writes require Parquet output, got {self.path.suffix}")

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            part_path = self.path.with_name(f"{self.path.stem}.part{len(self._part_paths):05d}.parquet")
            df.to_parquet(part_path, index=False)
            self._part_paths.append(part_path)
            return

        table = pa.Table.from_pandas(df, preserve_index=False)
        if self._parquet_writer is None:
            if self.path.exists():
                self.path.unlink()
            self._parquet_writer = pq.ParquetWriter(self.path, table.schema)
        self._parquet_writer.write_table(table)

    def close(self) -> Path:
        if self._parquet_writer is not None:
            self._parquet_writer.close()
            return self.path

        if self._part_paths:
            frames = [pd.read_parquet(part_path) for part_path in self._part_paths]
            pd.concat(frames, ignore_index=True).to_parquet(self.path, index=False)
            for part_path in self._part_paths:
                part_path.unlink(missing_ok=True)
        return self.path


def read_feature_matrix_metadata(matrix_path: str | Path) -> tuple[int, int, list[str]]:
    """Read sample count, vector dimension, and ordered sample IDs from a matrix file."""
    path = Path(matrix_path)
    if not path.is_file():
        raise FileNotFoundError(f"Feature matrix not found: {path}")

    id_df = pd.read_parquet(path, columns=["sample_id"])
    sample_ids = id_df["sample_id"].astype(str).tolist()
    n_samples = len(sample_ids)

    vector_df = pd.read_parquet(path, columns=["vector_inline"])
    first_vector = vector_df["vector_inline"].iloc[0]
    if isinstance(first_vector, np.ndarray):
        vector_dim = int(first_vector.shape[0])
    else:
        vector_dim = len(list(first_vector))

    return n_samples, vector_dim, sample_ids


def build_feature_manifest_from_matrix(
    matrix_path: str | Path,
    *,
    feature_family: str,
    feature_version: str,
    config: dict[str, Any],
    extraction_stats: dict[str, Any] | None = None,
    source_families: list[str] | None = None,
) -> dict[str, Any]:
    """Build manifest metadata by reading matrix dimensions from disk."""
    n_samples, vector_dim, sample_ids = read_feature_matrix_metadata(matrix_path)
    return {
        "feature_family": feature_family,
        "feature_version": feature_version,
        "vector_dim": vector_dim,
        "n_samples": n_samples,
        "sample_ids": sample_ids,
        "config_fingerprint": config_fingerprint(config),
        "config_path": (config.get("_meta") or {}).get("config_path"),
        "extraction_stats": extraction_stats or {},
        "source_families": source_families,
        "schema": {
            "sample_id": "str",
            "dataset_name": "str",
            "gesture_label": "str",
            "feature_family": "str",
            "feature_version": "str",
            "vector_inline": f"list[float] length {vector_dim}",
            "quality_flags": "dict",
            "extraction_ok": "bool",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def load_feature_matrix(
    path: str | Path,
    *,
    exclude_invalid: bool = True,
) -> FeatureTable:
    """Load a feature matrix from disk.

    When *exclude_invalid* is True (default), rows with failed geometric extraction
    (all-zero placeholder vectors, ``extraction_ok=False``, or ``detection_failed``)
    are removed so training and hybrid merges do not learn from empty hand features.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Feature matrix not found: {file_path}")

    if file_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported feature matrix format: {file_path.suffix}")

    records = _deserialize_from_storage(df.to_dict(orient="records"))
    for record in records:
        inline = record.get("vector_inline")
        if isinstance(inline, np.ndarray):
            record["vector_inline"] = inline.tolist()

    if exclude_invalid and records:
        family = str(records[0].get("feature_family", ""))
        should_filter = family in ("geometric", "keypoints_raw") or "hybrid" in family
        if should_filter:
            before = len(records)
            records, dropped = filter_invalid_geometric_feature_records(records)
            if dropped:
                _logger.info(
                    "Excluded %d / %d invalid %s feature rows from %s",
                    dropped,
                    before,
                    family,
                    file_path,
                )

    return FeatureTable(records=records, path=file_path)


def build_feature_manifest(
    records: list[dict[str, Any]],
    *,
    feature_family: str,
    feature_version: str,
    config: dict[str, Any],
    extraction_stats: dict[str, Any] | None = None,
    source_families: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble manifest metadata for a feature artifact."""
    dim = _vector_dim(records)
    sample_ids = [str(r["sample_id"]) for r in records]
    return {
        "feature_family": feature_family,
        "feature_version": feature_version,
        "vector_dim": dim,
        "n_samples": len(records),
        "sample_ids": sample_ids,
        "config_fingerprint": config_fingerprint(config),
        "config_path": (config.get("_meta") or {}).get("config_path"),
        "extraction_stats": extraction_stats or {},
        "source_families": source_families,
        "schema": {
            "sample_id": "str",
            "dataset_name": "str",
            "gesture_label": "str",
            "feature_family": "str",
            "feature_version": "str",
            "vector_inline": f"list[float] length {dim}",
            "quality_flags": "dict",
            "extraction_ok": "bool",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def save_feature_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return path


def manifest_path_for_matrix(matrix_path: str | Path) -> Path:
    """Derive default manifest path from feature matrix path."""
    path = Path(matrix_path)
    stem = path.stem
    return path.with_name(f"{stem}_manifest.json")


def quality_report_path_for_matrix(matrix_path: str | Path) -> Path:
    path = Path(matrix_path)
    stem = path.stem
    return path.with_name(f"{stem}_quality.json")


def vector_from_record(record: dict[str, Any]) -> np.ndarray:
    inline = record.get("vector_inline")
    if inline is None:
        raise ValueError(f"Record {record.get('sample_id')} has no vector_inline")
    return np.asarray(inline, dtype=np.float64).reshape(-1)
