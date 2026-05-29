"""Merge feature families into hybrid vectors with stable ordering."""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_FAMILY_ORDER: tuple[str, ...] = ("keypoints_raw", "geometric", "hog")


def concatenate_features(
    feature_blocks: dict[str, np.ndarray],
    *,
    family_order: tuple[str, ...] | list[str] | None = None,
) -> np.ndarray:
    """Concatenate feature blocks in deterministic family order."""
    order = tuple(family_order or DEFAULT_FAMILY_ORDER)
    missing = [name for name in order if name not in feature_blocks]
    if missing:
        raise KeyError(f"Missing feature blocks for families: {missing}")

    vectors: list[np.ndarray] = []
    for name in order:
        block = np.asarray(feature_blocks[name], dtype=np.float64).reshape(-1)
        if block.size == 0:
            raise ValueError(f"Empty feature block for family '{name}'")
        vectors.append(block)

    extra = sorted(set(feature_blocks) - set(order))
    if extra:
        raise ValueError(f"Unexpected feature families (not in order): {extra}")

    return np.concatenate(vectors)


def build_feature_record(
    sample_id: str,
    feature_family: str,
    vector: np.ndarray,
    metadata: dict[str, Any],
    *,
    dataset_name: str | None = None,
    gesture_label: str | None = None,
    feature_version: str = "v1",
) -> dict[str, Any]:
    """Build a feature record conforming to the project schema."""
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    quality_flags = dict(metadata.get("quality_flags", {}))
    if metadata.get("detection_failed"):
        quality_flags["detection_failed"] = True
    if metadata.get("low_confidence"):
        quality_flags["low_confidence"] = True

    record: dict[str, Any] = {
        "sample_id": sample_id,
        "dataset_name": dataset_name or metadata.get("dataset_name"),
        "gesture_label": gesture_label or metadata.get("gesture_label"),
        "feature_family": feature_family,
        "feature_version": feature_version,
        "vector_path": metadata.get("vector_path"),
        "vector_inline": vec.tolist(),
        "quality_flags": quality_flags,
        "extraction_ok": bool(metadata.get("extraction_ok", True)),
        "confidence": metadata.get("confidence"),
    }
    if metadata.get("source_families"):
        record["source_families"] = list(metadata["source_families"])
    return record
