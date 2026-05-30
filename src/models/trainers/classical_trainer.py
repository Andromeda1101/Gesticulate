"""Train custom classical models on feature matrices."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.features.feature_store import vector_from_record
from src.models.classical.base import BaseClassifier
from src.models.classical.scaler import StandardScaler
from src.models.model_registry import SCALE_ALGORITHMS, build_model, normalize_algorithm_name


def _subset_by_ids(
    records: list[dict[str, Any]],
    sample_ids: set[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_rows: list[np.ndarray] = []
    y_rows: list[Any] = []
    ids: list[str] = []
    for record in records:
        sid = str(record["sample_id"])
        if sid not in sample_ids:
            continue
        X_rows.append(vector_from_record(record))
        y_rows.append(record["gesture_label"])
        ids.append(sid)
    if not X_rows:
        raise ValueError("No samples matched the provided split IDs")
    return np.vstack(X_rows), np.array(y_rows), ids


def train_model(
    records: list[dict[str, Any]],
    train_ids: list[str],
    val_ids: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Train a classical model and return predictions plus timing metadata.

    *config* keys: ``algorithm``, ``hyperparameters``, optional ``scale_features``.
    """
    algorithm = normalize_algorithm_name(config["algorithm"])
    hyperparameters = dict(config.get("hyperparameters", {}))
    scale = config.get("scale_features", algorithm in SCALE_ALGORITHMS)

    train_set = set(train_ids)
    val_set = set(val_ids)
    X_train, y_train, _ = _subset_by_ids(records, train_set)
    X_val, y_val, val_sample_ids = _subset_by_ids(records, val_set)

    scaler: StandardScaler | None = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    model: BaseClassifier = build_model(algorithm, hyperparameters)

    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    infer_start = time.perf_counter()
    y_pred = model.predict(X_val)
    infer_seconds = time.perf_counter() - infer_start
    per_sample_ms = (infer_seconds / max(len(y_val), 1)) * 1000.0

    y_proba: np.ndarray | None = None
    try:
        y_proba = model.predict_proba(X_val)
    except NotImplementedError:
        pass

    return {
        "model": model,
        "scaler": scaler,
        "algorithm": algorithm,
        "y_true": y_val,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "val_sample_ids": val_sample_ids,
        "timing": {
            "fit_seconds": fit_seconds,
            "inference_seconds": infer_seconds,
            "per_sample_inference_ms": per_sample_ms,
        },
        "n_train": len(y_train),
        "n_val": len(y_val),
    }
