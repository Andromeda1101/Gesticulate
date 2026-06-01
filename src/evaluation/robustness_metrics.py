"""Robustness-specific metrics for cross-dataset evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

from src.evaluation.metrics import compute_classification_metrics

_OOD_PRED_OTHER = "_other_"


def compute_ood_drop(id_metrics: dict[str, Any], ood_metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Compare in-domain and OOD metrics.

    Returns absolute accuracy drop and relative performance retention.
    """
    id_acc = float(id_metrics.get("accuracy", 0.0))
    ood_acc = float(ood_metrics.get("accuracy", 0.0))
    abs_drop = id_acc - ood_acc
    retention = (ood_acc / id_acc) if id_acc > 0 else 0.0

    id_f1 = float(id_metrics.get("f1_macro", 0.0))
    ood_f1 = float(ood_metrics.get("f1_macro", 0.0))

    return {
        "in_domain_accuracy": id_acc,
        "ood_accuracy": ood_acc,
        "absolute_accuracy_drop": abs_drop,
        "relative_performance_retention": retention,
        "in_domain_f1_macro": id_f1,
        "ood_f1_macro": ood_f1,
        "absolute_f1_macro_drop": id_f1 - ood_f1,
    }


def _per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc_by_class: dict[str, float] = {}
    for label in labels:
        mask = y_true == label
        if mask.sum() == 0:
            acc_by_class[label] = float("nan")
        else:
            acc_by_class[label] = float(np.mean(y_pred[mask] == y_true[mask]))
    return acc_by_class


def compute_per_class_shift(
    id_y_true: np.ndarray,
    id_y_pred: np.ndarray,
    ood_y_true: np.ndarray,
    ood_y_pred: np.ndarray,
    *,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Per-class accuracy in each domain and absolute drop."""
    if labels is None:
        labels = sorted(
            np.unique(
                np.concatenate([id_y_true, id_y_pred, ood_y_true, ood_y_pred])
            ).tolist(),
            key=str,
        )
    labels = [str(l) for l in labels]

    id_acc = _per_class_accuracy(id_y_true, id_y_pred, labels)
    ood_acc = _per_class_accuracy(ood_y_true, ood_y_pred, labels)

    rows: list[dict[str, Any]] = []
    for label in labels:
        id_a = id_acc.get(label, float("nan"))
        ood_a = ood_acc.get(label, float("nan"))
        drop = (
            float(id_a - ood_a)
            if not (np.isnan(id_a) or np.isnan(ood_a))
            else float("nan")
        )
        rows.append(
            {
                "gesture_label": label,
                "in_domain_accuracy": id_a,
                "ood_accuracy": ood_a,
                "absolute_drop": drop,
            }
        )

    return {
        "labels": labels,
        "per_class": rows,
        "dataframe": pd.DataFrame(rows),
    }


def compute_misclassification_concentration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """
    Summarize which true classes are confused with which predicted classes (OOD errors).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    errors = y_true != y_pred
    if not np.any(errors):
        return {"total_errors": 0, "confusion_pairs": [], "by_true_class": {}}

    pair_counts: Counter[tuple[str, str]] = Counter()
    by_true: dict[str, Counter[str]] = defaultdict(Counter)
    for t, p in zip(y_true[errors], y_pred[errors]):
        t_s, p_s = str(t), str(p)
        pair_counts[(t_s, p_s)] += 1
        by_true[t_s][p_s] += 1

    confusion_pairs = [
        {"true_label": t, "predicted_label": p, "count": c}
        for (t, p), c in pair_counts.most_common()
    ]
    by_true_class = {
        label: dict(counter.most_common()) for label, counter in by_true.items()
    }

    return {
        "total_errors": int(errors.sum()),
        "confusion_pairs": confusion_pairs,
        "by_true_class": by_true_class,
    }


def compute_ood_domain_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    ood_label_vocab: list[str] | None = None,
    map_out_of_vocab_predictions: bool = True,
) -> dict[str, Any]:
    """
    Per-class OOD accuracy and confusion matrix on the LeapGestRecog label vocabulary.

    Predictions outside *ood_label_vocab* are mapped to ``_other_`` so the matrix
    remains readable when a HaGRID-trained classifier emits train-only class names.
    """
    y_true = np.asarray([str(y) for y in y_true])
    y_pred = np.asarray([str(y) for y in y_pred])
    if ood_label_vocab is None:
        ood_label_vocab = sorted(np.unique(y_true).tolist(), key=str)
    ood_label_vocab = [str(label) for label in ood_label_vocab]
    vocab_set = set(ood_label_vocab)

    if map_out_of_vocab_predictions:
        y_pred_cm = np.array([p if p in vocab_set else _OOD_PRED_OTHER for p in y_pred])
        cm_labels = ood_label_vocab + [_OOD_PRED_OTHER]
    else:
        y_pred_cm = y_pred
        cm_labels = ood_label_vocab

    per_class_rows: list[dict[str, Any]] = []
    for label in ood_label_vocab:
        mask = y_true == label
        n_samples = int(mask.sum())
        if n_samples == 0:
            accuracy = float("nan")
            n_correct = 0
        else:
            n_correct = int(np.sum(y_pred[mask] == y_true[mask]))
            accuracy = float(n_correct / n_samples)
        per_class_rows.append(
            {
                "gesture_label": label,
                "n_samples": n_samples,
                "n_correct": n_correct,
                "accuracy": accuracy,
            }
        )

    cm = confusion_matrix(y_true, y_pred_cm, labels=cm_labels)
    metrics = compute_classification_metrics(y_true, y_pred_cm, labels=cm_labels)
    n_other_pred = int(np.sum(y_pred_cm == _OOD_PRED_OTHER)) if map_out_of_vocab_predictions else 0

    return {
        "ood_label_vocab": ood_label_vocab,
        "per_class": per_class_rows,
        "dataframe": pd.DataFrame(per_class_rows),
        "confusion_matrix": cm.tolist(),
        "labels": cm_labels,
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "n_samples": int(len(y_true)),
        "n_out_of_vocab_predictions": n_other_pred,
        "out_of_vocab_prediction_rate": n_other_pred / max(len(y_true), 1),
        "metrics": metrics,
    }


def evaluate_domain(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: list[str] | None = None,
    timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classification metrics for one domain."""
    metrics = compute_classification_metrics(y_true, y_pred, labels=labels)
    if timing:
        metrics["inference_seconds"] = float(timing.get("inference_seconds", 0.0))
        metrics["per_sample_inference_ms"] = float(
            timing.get("per_sample_inference_ms", 0.0)
        )
    return metrics


_UNKNOWN_LABEL = "unknown"


def ood_label_vocab_from_schema(schema_validation: dict[str, Any]) -> list[str]:
    """OOD canonical vocabulary = shared train/OOD labels plus OOD-only classes."""
    overlap = schema_validation.get("label_overlap", {})
    shared = [str(l) for l in overlap.get("shared", [])]
    test_only = [str(l) for l in overlap.get("test_only", [])]
    return sorted(set(shared) | set(test_only))


def mask_predictions_unknown(
    y_pred: np.ndarray,
    allowed_labels: list[str] | set[str],
) -> np.ndarray:
    """Map predictions outside *allowed_labels* to ``unknown``."""
    allowed = {str(label) for label in allowed_labels}
    out = np.asarray([str(p) for p in y_pred], dtype=object)
    for i, pred in enumerate(out):
        if pred not in allowed:
            out[i] = _UNKNOWN_LABEL
    return out.astype(str)


def mask_predictions_shared_argmax(
    y_proba: np.ndarray,
    class_names: list[str],
    shared_labels: list[str],
) -> np.ndarray:
    """Restrict decisions to *shared_labels* by masked argmax over ``predict_proba``."""
    if y_proba is None or len(y_proba) == 0:
        raise ValueError("y_proba required for shared-class argmax masking")

    class_names = [str(c) for c in class_names]
    shared_set = {str(label) for label in shared_labels}
    shared_indices = [i for i, name in enumerate(class_names) if name in shared_set]
    if not shared_indices:
        raise ValueError("No shared labels found in model class list")

    proba = np.asarray(y_proba, dtype=np.float64)
    if proba.ndim != 2 or proba.shape[1] != len(class_names):
        raise ValueError(
            f"predict_proba shape {proba.shape} incompatible with {len(class_names)} classes"
        )

    out = np.empty(proba.shape[0], dtype=object)
    for row_idx in range(proba.shape[0]):
        scores = proba[row_idx, shared_indices]
        best_local = int(np.argmax(scores))
        out[row_idx] = class_names[shared_indices[best_local]]
    return out.astype(str)


def _subset_by_true_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    allowed_true: list[str] | set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep rows whose ground-truth label is in *allowed_true*."""
    allowed = {str(label) for label in allowed_true}
    y_true = np.asarray(y_true)
    mask = np.array([str(t) in allowed for t in y_true], dtype=bool)
    return y_true[mask], np.asarray(y_pred)[mask], mask


def compute_ood_eval_protocols(
    ood_y_true: np.ndarray,
    ood_y_pred: np.ndarray,
    schema_validation: dict[str, Any],
    *,
    ood_y_proba: np.ndarray | None = None,
    model_class_names: list[str] | None = None,
    id_y_true: np.ndarray | None = None,
    id_y_pred: np.ndarray | None = None,
    ood_timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute supplementary OOD metrics for cross-dataset evaluation.

    Protocols:
    - ``shared_subset``: accuracy only on samples whose true label is in the
      train/OOD shared vocabulary (e.g. 7 classes, excluding L/Down/Palm_Moved).
    - ``masked_unknown``: predictions outside the OOD canonical vocabulary are
      mapped to ``unknown`` before scoring.
    - ``masked_shared_argmax``: decisions are forced to the shared label set via
      masked argmax over ``predict_proba`` (requires probabilities).
    """
    overlap = schema_validation.get("label_overlap", {})
    shared_labels = [str(l) for l in overlap.get("shared", [])]
    ood_vocab = ood_label_vocab_from_schema(schema_validation)

    protocols: dict[str, Any] = {
        "shared_labels": shared_labels,
        "ood_label_vocab": ood_vocab,
        "ood_only_labels": [str(l) for l in overlap.get("test_only", [])],
    }

    if shared_labels:
        ood_true_sub, ood_pred_sub, subset_mask = _subset_by_true_labels(
            ood_y_true, ood_y_pred, shared_labels
        )
        ood_subset_metrics = evaluate_domain(
            ood_true_sub,
            ood_pred_sub,
            labels=shared_labels,
        )
        ood_subset_metrics["n_samples"] = int(subset_mask.sum())
        ood_subset_metrics["n_excluded"] = int(len(subset_mask) - subset_mask.sum())
        protocols["shared_subset"] = {
            "description": (
                "OOD accuracy on samples whose true label is in the shared "
                "train/OOD vocabulary (excludes OOD-only classes)."
            ),
            "ood": ood_subset_metrics,
        }
        if id_y_true is not None and id_y_pred is not None:
            id_true_sub, id_pred_sub, id_mask = _subset_by_true_labels(
                id_y_true, id_y_pred, shared_labels
            )
            id_subset_metrics = evaluate_domain(
                id_true_sub,
                id_pred_sub,
                labels=shared_labels,
            )
            id_subset_metrics["n_samples"] = int(id_mask.sum())
            protocols["shared_subset"]["in_domain"] = id_subset_metrics
            protocols["shared_subset"]["robustness"] = compute_ood_drop(
                id_subset_metrics,
                ood_subset_metrics,
            )

    y_pred_unknown = mask_predictions_unknown(ood_y_pred, ood_vocab)
    unknown_metrics = evaluate_domain(ood_y_true, y_pred_unknown, labels=ood_vocab + [_UNKNOWN_LABEL])
    unknown_metrics["n_unknown_predictions"] = int(np.sum(y_pred_unknown == _UNKNOWN_LABEL))
    unknown_metrics["unknown_rate"] = (
        unknown_metrics["n_unknown_predictions"] / max(len(y_pred_unknown), 1)
    )
    protocols["masked_unknown"] = {
        "description": (
            "Predictions outside the OOD canonical vocabulary are mapped to "
            "'unknown' before accuracy is computed."
        ),
        "ood": unknown_metrics,
    }

    if ood_y_proba is not None and model_class_names:
        try:
            y_pred_shared = mask_predictions_shared_argmax(
                ood_y_proba,
                model_class_names,
                shared_labels,
            )
            shared_argmax_metrics = evaluate_domain(
                ood_y_true,
                y_pred_shared,
                labels=shared_labels,
            )
            shared_argmax_metrics["n_samples"] = int(len(ood_y_true))
            protocols["masked_shared_argmax"] = {
                "description": (
                    "Each prediction is the argmax over predict_proba restricted "
                    "to shared train/OOD classes only."
                ),
                "ood": shared_argmax_metrics,
                "available": True,
            }
            if shared_labels and id_y_true is not None and id_y_pred is not None:
                id_true_sub, _, id_mask = _subset_by_true_labels(
                    id_y_true, id_y_pred, shared_labels
                )
                ood_true_sub, ood_pred_sub, _ = _subset_by_true_labels(
                    ood_y_true, y_pred_shared, shared_labels
                )
                id_pred_sub = np.asarray(id_y_pred)[id_mask]
                id_sub = evaluate_domain(id_true_sub, id_pred_sub, labels=shared_labels)
                ood_sub = evaluate_domain(ood_true_sub, ood_pred_sub, labels=shared_labels)
                protocols["masked_shared_argmax"]["shared_subset"] = {
                    "in_domain": id_sub,
                    "ood": ood_sub,
                    "robustness": compute_ood_drop(id_sub, ood_sub),
                }
        except (ValueError, TypeError) as exc:
            protocols["masked_shared_argmax"] = {
                "available": False,
                "reason": str(exc),
            }
    else:
        protocols["masked_shared_argmax"] = {
            "available": False,
            "reason": (
                "predict_proba not available; re-run with --include-proba for "
                "masked shared-class argmax metrics."
            ),
        }

    if ood_timing:
        for key in ("shared_subset", "masked_unknown", "masked_shared_argmax"):
            block = protocols.get(key)
            if isinstance(block, dict) and isinstance(block.get("ood"), dict):
                block["ood"]["inference_seconds"] = float(
                    ood_timing.get("inference_seconds", 0.0)
                )

    return protocols
