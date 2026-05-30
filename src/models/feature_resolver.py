"""Resolve experiment feature-family names to Phase 2 artifact paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.path_manager import resolve_project_root

# Experiment alias -> (manifest feature_family column, default parquet stem)
FEATURE_FAMILY_MAP: dict[str, tuple[str, str]] = {
    "keypoints_only": ("geometric", "hagrid_subset_geometric_v1"),
    "geometric": ("geometric", "hagrid_subset_geometric_v1"),
    "hog_only": ("hog", "hagrid_subset_hog_v1"),
    "hog": ("hog", "hagrid_subset_hog_v1"),
    "hybrid": ("hybrid_keypoints_hog", "hagrid_subset_hybrid_v1"),
    "hybrid_keypoints_hog": ("hybrid_keypoints_hog", "hagrid_subset_hybrid_v1"),
}

# Backward-compatible algorithm aliases from older configs
ALGORITHM_ALIASES: dict[str, str] = {
    "svm_rbf": "svm",
    "naive_bayes_gaussian": "naive_bayes",
    "decision_tree_cart": "decision_tree",
}


def normalize_algorithm_name(name: str) -> str:
    return ALGORITHM_ALIASES.get(name, name)


def resolve_feature_matrix_path(
    feature_family: str,
    *,
    dataset_name: str = "hagrid_subset",
    feature_version: str = "v1",
    project_root: Path | str | None = None,
) -> Path:
    """Return path to feature matrix parquet for *feature_family*."""
    root = Path(project_root) if project_root else resolve_project_root()
    if feature_family in FEATURE_FAMILY_MAP:
        _, stem = FEATURE_FAMILY_MAP[feature_family]
        # Allow dataset override by replacing prefix
        if not stem.startswith(dataset_name):
            parts = stem.split("_", 1)
            if len(parts) == 2 and parts[0] != dataset_name:
                stem = f"{dataset_name}_{parts[1]}"
        return root / "artifacts" / "features" / f"{stem}.parquet"

    column_family = feature_family
    stem = f"{dataset_name}_{feature_family}_{feature_version}"
    return root / "artifacts" / "features" / f"{stem}.parquet"


def resolve_split_path(
    dataset_name: str = "hagrid_subset",
    *,
    split_name: str = "train_val_test",
    project_root: Path | str | None = None,
) -> Path:
    root = Path(project_root) if project_root else resolve_project_root()
    return root / "data" / "splits" / f"{dataset_name}_{split_name}.json"


def manifest_feature_family(feature_family: str) -> str:
    """Column value stored in feature matrix records."""
    if feature_family in FEATURE_FAMILY_MAP:
        return FEATURE_FAMILY_MAP[feature_family][0]
    return feature_family
