"""Classification and efficiency metrics for experiment runs."""

from __future__ import annotations

from typing import Any

import numpy as np

# sklearn allowed only for metric utilities per project guardrails
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: list[Any] | None = None,
) -> dict[str, Any]:
    """Compute accuracy, macro/micro precision/recall/F1, and confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist(), key=str)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
        ),
        "precision_micro": float(
            precision_score(y_true, y_pred, average="micro", zero_division=0, labels=labels)
        ),
        "recall_micro": float(
            recall_score(y_true, y_pred, average="micro", zero_division=0, labels=labels)
        ),
        "f1_micro": float(
            f1_score(y_true, y_pred, average="micro", zero_division=0, labels=labels)
        ),
        "confusion_matrix": cm.tolist(),
        "labels": [str(l) for l in labels],
    }


def compute_efficiency_metrics(timing_info: dict[str, Any]) -> dict[str, Any]:
    """Normalize timing fields from trainer output."""
    return {
        "fit_seconds": float(timing_info.get("fit_seconds", 0.0)),
        "inference_seconds": float(timing_info.get("inference_seconds", 0.0)),
        "per_sample_inference_ms": float(timing_info.get("per_sample_inference_ms", 0.0)),
    }
