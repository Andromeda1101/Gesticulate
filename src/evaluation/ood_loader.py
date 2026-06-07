"""Load OOD evaluation feature sets and validate schema compatibility."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.label_mapper import LEAPGEST_MAPPED_HAGRID_LABELS, observed_labels
from src.features.feature_store import (
    FeatureTable,
    load_feature_matrix,
    manifest_path_for_matrix,
)
from src.models.feature_resolver import manifest_feature_family, resolve_feature_matrix_path


def load_feature_manifest(matrix_path: str | Path) -> dict[str, Any]:
    """Load sidecar manifest for a feature matrix."""
    path = Path(matrix_path)
    manifest_path = manifest_path_for_matrix(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Feature manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_ood_feature_set(
    dataset_name: str,
    feature_family: str,
    version: str = "v1",
    *,
    project_root: Path | str | None = None,
    matrix_path: str | Path | None = None,
    exclude_invalid: bool = True,
) -> tuple[FeatureTable, dict[str, Any]]:
    """
    Load a feature matrix and manifest for cross-dataset evaluation.

    When *matrix_path* is provided it is used directly; otherwise the canonical
    artifact path is resolved from dataset/family/version.
    """
    if matrix_path is None:
        matrix_path = resolve_feature_matrix_path(
            feature_family,
            dataset_name=dataset_name,
            feature_version=version,
            project_root=project_root,
        )
    matrix_path = Path(matrix_path)
    table = load_feature_matrix(matrix_path, exclude_invalid=exclude_invalid)
    manifest = load_feature_manifest(matrix_path)
    return table, manifest


def _labels_from_records(records: list[dict[str, Any]]) -> list[str]:
    return observed_labels(records)


def validate_schema_compatibility(
    train_manifest: dict[str, Any],
    test_manifest: dict[str, Any],
    *,
    train_records: list[dict[str, Any]] | None = None,
    test_records: list[dict[str, Any]] | None = None,
    train_labels: list[str] | set[str] | None = None,
    test_labels: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    """
    Confirm matching feature dimensions and overlapping label vocabulary.

    Returns a structured report with ``compatible`` bool and issue details.
    """
    issues: list[str] = []
    train_dim = int(train_manifest.get("vector_dim", -1))
    test_dim = int(test_manifest.get("vector_dim", -1))
    if train_dim != test_dim:
        issues.append(f"vector_dim mismatch: train={train_dim} test={test_dim}")

    train_family = train_manifest.get("feature_family")
    test_family = test_manifest.get("feature_family")
    if train_family != test_family:
        issues.append(f"feature_family mismatch: train={train_family} test={test_family}")

    train_version = train_manifest.get("feature_version")
    test_version = test_manifest.get("feature_version")
    if train_version != test_version:
        issues.append(f"feature_version mismatch: train={train_version} test={test_version}")

    train_label_set: set[str] = set()
    test_label_set: set[str] = set()
    if train_records is not None:
        train_label_set = set(_labels_from_records(train_records))
    elif train_labels is not None:
        train_label_set = {str(label) for label in train_labels}
    if test_records is not None:
        test_label_set = set(_labels_from_records(test_records))
    elif test_labels is not None:
        test_label_set = {str(label) for label in test_labels}

    shared_labels = (
        sorted(train_label_set & test_label_set) if train_label_set and test_label_set else []
    )
    train_only = sorted(train_label_set - test_label_set) if train_label_set else []
    test_only = sorted(test_label_set - train_label_set) if test_label_set else []

    if train_label_set and test_label_set and not shared_labels:
        issues.append("no overlapping gesture labels between train and test domains")

    reference = set(LEAPGEST_MAPPED_HAGRID_LABELS)
    outside_reference_train = sorted(train_label_set - reference) if train_label_set else []
    outside_reference_test = sorted(test_label_set - reference) if test_label_set else []

    return {
        "compatible": len(issues) == 0,
        "issues": issues,
        "vector_dim": {"train": train_dim, "test": test_dim},
        "feature_family": {"train": train_family, "test": test_family},
        "feature_version": {"train": train_version, "test": test_version},
        "label_overlap": {
            "shared": shared_labels,
            "train_only": train_only,
            "test_only": test_only,
            "n_shared": len(shared_labels),
        },
        "outside_reference": {
            "train": outside_reference_train,
            "test": outside_reference_test,
        },
    }


def filter_records_by_split(
    records: list[dict[str, Any]],
    sample_ids: set[str] | list[str],
) -> list[dict[str, Any]]:
    """Return records whose sample_id is in *sample_ids*."""
    id_set = set(sample_ids)
    return [r for r in records if str(r["sample_id"]) in id_set]


def filter_records_by_feature_family(
    records: list[dict[str, Any]],
    feature_family: str,
) -> list[dict[str, Any]]:
    """Filter rows to the manifest column value for an experiment feature family alias."""
    expected = manifest_feature_family(feature_family)
    filtered = [r for r in records if r.get("feature_family") in (expected, feature_family)]
    return filtered if filtered else list(records)


_MANIFEST_INDEX_COLUMNS = (
    "sample_id",
    "dataset_name",
    "image_path",
    "subject_id",
    "capture_context",
)


def build_sample_metadata_index(
    manifest_path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Map sample_id -> manifest row (image_path, capture_context, subject_id)."""
    path = Path(manifest_path)
    if not path.is_file():
        return {}

    index: dict[str, dict[str, Any]] = {}
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(path)
            available = set(parquet_file.schema_arrow.names)
            columns = [name for name in _MANIFEST_INDEX_COLUMNS if name in available]
            for batch in parquet_file.iter_batches(batch_size=8192, columns=columns):
                for row in batch.to_pandas().to_dict(orient="records"):
                    sid = str(row.get("sample_id", ""))
                    if sid:
                        index[sid] = row
            return index
        except ImportError:
            pass

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        columns = [name for name in _MANIFEST_INDEX_COLUMNS if name in df.columns]
        df = df[columns]
    else:
        df = pd.read_csv(path)
    for row in df.to_dict(orient="records"):
        sid = str(row.get("sample_id", ""))
        if sid:
            index[sid] = row
    return index
