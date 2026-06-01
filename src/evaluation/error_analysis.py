"""OOD error pattern analysis and failure-case sampling."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


def _parse_capture_context(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
    return {}


def group_errors_by_context(predictions: pd.DataFrame) -> dict[str, Any]:
    """
    Summarize OOD misclassifications by capture context dimensions.

    Expected columns: true_label, predicted_label, and optional capture_context
    or flattened context fields (source, subject_id).
    """
    if predictions.empty:
        return {"n_errors": 0, "by_source": {}, "by_subject": {}, "confusion_pairs": []}

    df = predictions.copy()
    if "correct" not in df.columns:
        df["correct"] = df["true_label"].astype(str) == df["predicted_label"].astype(str)
    errors = df[~df["correct"]]

    if "capture_context" in errors.columns:
        contexts = errors["capture_context"].apply(_parse_capture_context)
        errors = errors.copy()
        errors["context_source"] = contexts.apply(lambda c: c.get("source", "unknown"))
        errors["context_subject"] = errors.get("subject_id").fillna(
            contexts.apply(lambda c: c.get("subject_id"))
        )
    else:
        errors = errors.copy()
        errors["context_source"] = errors.get("domain", "ood")
        errors["context_subject"] = errors.get("subject_id", "unknown")

    by_source = (
        errors.groupby("context_source", dropna=False).size().to_dict()
        if len(errors)
        else {}
    )
    by_subject = (
        errors.groupby("context_subject", dropna=False).size().to_dict()
        if len(errors) and "context_subject" in errors.columns
        else {}
    )

    pair_counts = (
        errors.groupby(["true_label", "predicted_label"]).size().reset_index(name="count")
        if len(errors)
        else pd.DataFrame(columns=["true_label", "predicted_label", "count"])
    )
    confusion_pairs = pair_counts.sort_values("count", ascending=False).to_dict(
        orient="records"
    )

    return {
        "n_errors": int(len(errors)),
        "by_source": {str(k): int(v) for k, v in by_source.items()},
        "by_subject": {str(k): int(v) for k, v in by_subject.items()},
        "confusion_pairs": confusion_pairs,
    }


def sample_failure_cases(
    predictions: pd.DataFrame,
    n_per_class: int = 3,
    *,
    domain: str | None = "ood",
) -> pd.DataFrame:
    """
    Sample representative misclassifications, balanced across true classes.

    Prefers higher-confidence wrong predictions when a confidence column exists.
    """
    df = predictions.copy()
    if domain is not None and "domain" in df.columns:
        df = df[df["domain"] == domain]
    if "correct" not in df.columns:
        df["correct"] = df["true_label"].astype(str) == df["predicted_label"].astype(str)
    errors = df[~df["correct"]]
    if errors.empty:
        return errors

    sort_cols: list[str] = []
    ascending: list[bool] = []
    if "confidence" in errors.columns:
        sort_cols.append("confidence")
        ascending.append(False)
    if sort_cols:
        errors = errors.sort_values(sort_cols, ascending=ascending)

    samples: list[pd.DataFrame] = []
    for label, group in errors.groupby("true_label", sort=False):
        samples.append(group.head(n_per_class))
    return pd.concat(samples, ignore_index=True) if samples else errors.iloc[:0]
