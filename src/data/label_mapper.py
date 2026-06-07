"""Convert dataset-native labels to HaGRID-native gesture names."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

# HaGRID-native labels observed after mapping LeapGestRecog OOD folders.
LEAPGEST_MAPPED_HAGRID_LABELS: tuple[str, ...] = (
    "stop",
    "palm",
    "fist",
    "like",
    "one",
    "ok",
    "grip",
    "thumb_index",
)

# Backward-compatible aliases for modules that imported the old canonical names.
CANONICAL_GESTURE_LABELS = LEAPGEST_MAPPED_HAGRID_LABELS
LEAPGESTRECOG_LABELS = LEAPGEST_MAPPED_HAGRID_LABELS

# LeapGestRecog folder / alias keys -> HaGRID training labels.
_LEAPGEST_TO_HAGRID: dict[str, str] = {
    "01_palm": "stop",
    "02_l": "thumb_index",
    "l": "thumb_index",
    "l_shape": "thumb_index",
    "03_fist": "fist",
    "fist": "fist",
    "04_fist_moved": "fist",
    "fist_moved": "fist",
    "05_thumb": "like",
    "thumb": "like",
    "thumb_up": "like",
    "thumbup": "like",
    "06_index": "one",
    "index": "one",
    "07_ok": "ok",
    "ok": "ok",
    "08_palm_moved": "palm",
    "palm_moved": "palm",
    "09_c": "grip",
    "c": "grip",
    "10_down": "palm",
}

_DATASET_ALIASES: dict[str, dict[str, str]] = {
    "leapgestrecog": dict(_LEAPGEST_TO_HAGRID),
    "hagrid_subset": {},
}


def _normalize_key(raw_label: str) -> str:
    return raw_label.strip().lower().replace("-", "_").replace(" ", "_")


def _format_unmapped_label(raw_label: str) -> str:
    """Normalize labels with no explicit alias to a stable lowercase form."""
    return _normalize_key(raw_label)


def normalize_label(
    raw_label: str,
    dataset_name: str,
    *,
    label_aliases: dict[str, str] | None = None,
    canonical_labels: list[str] | None = None,
    align_to_canonical: bool = False,
) -> str:
    """Map a raw label string to a HaGRID-native gesture label."""
    if not raw_label or not str(raw_label).strip():
        raise ValueError("Empty raw label cannot be normalized")

    stripped = str(raw_label).strip()
    reference_set = set(canonical_labels or [])

    if reference_set and stripped in reference_set:
        return stripped

    merged_aliases: dict[str, str] = {}
    if align_to_canonical:
        merged_aliases.update(_LEAPGEST_TO_HAGRID)
    merged_aliases.update(_DATASET_ALIASES.get(dataset_name, {}))
    if label_aliases:
        for key, value in label_aliases.items():
            merged_aliases[_normalize_key(key)] = str(value)

    key = _normalize_key(stripped)
    if key in merged_aliases:
        return merged_aliases[key]

    if reference_set and stripped in reference_set:
        return stripped

    if reference_set:
        for reference in reference_set:
            if _normalize_key(reference) == key:
                return reference

    return _format_unmapped_label(stripped)


def apply_label_normalization(
    samples: list[dict[str, Any]],
    dataset_name: str,
    *,
    label_aliases: dict[str, str] | None = None,
    canonical_labels: list[str] | None = None,
    align_to_canonical: bool = False,
    raw_label_field: str = "raw_gesture_label",
) -> list[dict[str, Any]]:
    """Normalize gesture labels on sample dict copies."""
    normalized: list[dict[str, Any]] = []
    for sample in samples:
        row = dict(sample)
        raw = row.get(raw_label_field) or row.get("gesture_label", "")
        row["raw_gesture_label"] = str(raw)
        row["gesture_label"] = normalize_label(
            str(raw),
            dataset_name,
            label_aliases=label_aliases,
            canonical_labels=canonical_labels,
            align_to_canonical=align_to_canonical,
        )
        normalized.append(row)
    return normalized


def normalize_sample_gesture_label(
    sample: dict[str, Any],
    *,
    label_aliases: dict[str, str] | None = None,
    canonical_labels: list[str] | None = None,
    align_to_canonical: bool = False,
) -> dict[str, Any]:
    """
    Ensure *gesture_label* on a manifest or feature row uses HaGRID-native names.

    Uses ``raw_gesture_label`` when present; otherwise infers LeapGestRecog gesture
    folders from ``capture_context.relative_path`` or ``image_path``.
    """
    row = dict(sample)
    dataset_name = str(row.get("dataset_name", ""))
    raw = row.get("raw_gesture_label") or row.get("gesture_label", "")
    raw_str = str(raw).strip()

    if dataset_name == "leapgestrecog" and (
        not _looks_like_leapgest_gesture_token(raw_str) or _LEAPGEST_SUBJECT_ID.match(raw_str)
    ):
        inferred = _infer_leapgest_gesture_from_paths(row)
        if inferred:
            raw_str = inferred
            row["raw_gesture_label"] = inferred

    row["raw_gesture_label"] = raw_str
    row["gesture_label"] = normalize_label(
        raw_str,
        dataset_name or "leapgestrecog",
        label_aliases=label_aliases,
        canonical_labels=canonical_labels,
        align_to_canonical=align_to_canonical,
    )
    return row


_LEAPGEST_GESTURE_TOKEN = re.compile(r"^\d{2}_[a-z0-9_]+$", re.IGNORECASE)
_LEAPGEST_SUBJECT_ID = re.compile(r"^\d{2}$")
_LEAPGEST_GESTURE_IN_PATH = re.compile(r"/(\d{2}_[a-z0-9_]+)/", re.IGNORECASE)


def _looks_like_leapgest_gesture_token(token: str) -> bool:
    stripped = token.strip()
    return bool(_LEAPGEST_GESTURE_TOKEN.match(stripped)) or stripped.lower() in _LEAPGEST_TO_HAGRID


def _infer_leapgest_gesture_from_paths(sample: dict[str, Any]) -> str | None:
    """Read ``01_palm``-style folder names from image or capture_context paths."""
    candidates: list[str] = []
    capture = sample.get("capture_context") or {}
    if isinstance(capture, str):
        try:
            capture = json.loads(capture)
        except json.JSONDecodeError:
            capture = {}
    rel = capture.get("relative_path") if isinstance(capture, dict) else None
    if rel:
        candidates.append(str(rel))
    image_path = sample.get("image_path")
    if image_path:
        candidates.append(str(image_path))

    for path in candidates:
        match = _LEAPGEST_GESTURE_IN_PATH.search(path.replace("\\", "/"))
        if match:
            return match.group(1)
    return None


def observed_labels(samples: list[dict[str, Any]]) -> list[str]:
    """Return sorted unique gesture labels present in *samples*."""
    return sorted({str(s.get("gesture_label", "")) for s in samples if s.get("gesture_label")})


def validate_label_coverage(
    samples: list[dict[str, Any]],
    reference_labels: list[str] | None = None,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """
    Return label statistics.

    When *reference_labels* is provided, reports overlap with that vocabulary.
    Labels outside the reference set are listed as *outside_reference* (not dropped).
    """
    labels = [s.get("gesture_label", "") for s in samples]
    counter = Counter(labels)
    ref_set = set(reference_labels or [])

    outside_reference = sorted({lbl for lbl in counter if ref_set and lbl not in ref_set})
    missing_reference = sorted(ref_set - set(counter)) if ref_set else []

    aligned = sum(counter[l] for l in counter if l in ref_set) if ref_set else len(samples)

    return {
        "total_samples": len(samples),
        "observed_labels": dict(counter),
        "reference_labels": list(reference_labels) if reference_labels else [],
        "outside_reference": outside_reference,
        "missing_reference_labels": missing_reference,
        "aligned_with_reference_ratio": (aligned / len(samples) if samples else 0.0),
        "unknown_labels": outside_reference if strict else [],
    }
