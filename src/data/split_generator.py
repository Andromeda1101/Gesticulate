"""Generate reproducible train/validation/test splits and CV folds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def _train_test_split(
    ids: list[str],
    labels: list[str],
    *,
    test_size: float,
    seed: int,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Stratified split when possible; otherwise shuffle without stratification."""
    stratify = labels
    try:
        return train_test_split(
            ids,
            labels,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            ids,
            labels,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )


def _split_ratios(config: dict[str, Any] | None) -> tuple[float, float, float]:
    strategy = config or {}
    train_ratio = float(strategy.get("train_ratio", 0.7))
    val_ratio = float(strategy.get("val_ratio", 0.15))
    test_ratio = float(strategy.get("test_ratio", 0.15))
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    return train_ratio, val_ratio, test_ratio


def create_primary_splits(
    samples: list[dict[str, Any]],
    seed: int = 42,
    *,
    split_strategy: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    """
    Stratified train/val/test split returning sample_id lists per split.

    Default ratios: 70/15/15.
    """
    if not samples:
        return {"train": [], "val": [], "test": []}

    df = pd.DataFrame(samples)
    if "sample_id" not in df.columns or "gesture_label" not in df.columns:
        raise ValueError("Samples must include sample_id and gesture_label")

    train_ratio, val_ratio, test_ratio = _split_ratios(split_strategy)
    ids = df["sample_id"].tolist()
    labels = df["gesture_label"].tolist()

    holdout_size = 1.0 - train_ratio
    if len(ids) < 3 or holdout_size <= 0:
        return {"train": list(ids), "val": [], "test": []}

    train_ids, holdout_ids, train_labels, holdout_labels = _train_test_split(
        ids,
        labels,
        test_size=holdout_size,
        seed=seed,
    )

    if len(holdout_ids) < 2:
        return {
            "train": list(train_ids),
            "val": list(holdout_ids),
            "test": [],
        }

    relative_val = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids, _, _ = _train_test_split(
        holdout_ids,
        holdout_labels,
        test_size=(1.0 - relative_val),
        seed=seed,
    )

    return {
        "train": list(train_ids),
        "val": list(val_ids),
        "test": list(test_ids),
    }


def create_stratified_folds(
    samples: list[dict[str, Any]],
    n_folds: int = 5,
    seed: int = 42,
    *,
    train_split_name: str = "train",
    primary_splits: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """
    Build stratified K-fold memberships for training records.

    Returns a list of fold dicts: ``{"fold": i, "train": [...], "val": [...]}``.
    """
    df = pd.DataFrame(samples)
    if primary_splits:
        train_ids = set(primary_splits.get(train_split_name, []))
        df = df[df["sample_id"].isin(train_ids)]

    if df.empty:
        return []

    n_classes = df["gesture_label"].nunique()
    effective_folds = min(n_folds, len(df), n_classes)
    if effective_folds < 2:
        return []

    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    folds: list[dict[str, Any]] = []
    ids = df["sample_id"].values
    labels = df["gesture_label"].values

    try:
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(ids, labels)):
            folds.append(
                {
                    "fold": fold_idx,
                    "train": ids[train_idx].tolist(),
                    "val": ids[val_idx].tolist(),
                }
            )
    except ValueError:
        return []
    return folds


def split_distribution(
    samples: list[dict[str, Any]],
    splits: dict[str, list[str]],
) -> dict[str, dict[str, int]]:
    """Count gesture labels per split."""
    id_to_label = {s["sample_id"]: s["gesture_label"] for s in samples}
    dist: dict[str, dict[str, int]] = {}
    for split_name, split_ids in splits.items():
        counter: dict[str, int] = {}
        for sid in split_ids:
            label = id_to_label.get(sid, "unknown")
            counter[label] = counter.get(label, 0) + 1
        dist[split_name] = counter
    return dist


def save_splits(
    splits: dict[str, list[str]],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(splits, fh, indent=2)
    return path


def save_folds(
    folds: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump({"folds": folds}, fh, indent=2)
    return path
