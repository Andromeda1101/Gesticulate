"""Convert dataset-native labels to a shared canonical gesture vocabulary."""

from __future__ import annotations

from collections import Counter
from typing import Any

# Canonical gesture names (10-class shared vocabulary)
CANONICAL_GESTURE_LABELS: tuple[str, ...] = (
    "Palm",
    "L",
    "Fist",
    "Fist_Moved",
    "Thumb",
    "Index",
    "OK",
    "Palm_Moved",
    "C",
    "Down",
)

# Backward-compatible alias
LEAPGESTRECOG_LABELS = CANONICAL_GESTURE_LABELS

# Native folder / alias keys -> canonical names (LeapGestRecog)
_LEAPGEST_TO_CANONICAL: dict[str, str] = {
    "01_palm": "Palm",
    "palm": "Palm",
    "02_l": "L",
    "l": "L",
    "l_shape": "L",
    "03_fist": "Fist",
    "fist": "Fist",
    "04_fist_moved": "Fist_Moved",
    "fist_moved": "Fist_Moved",
    "05_thumb": "Thumb",
    "thumb": "Thumb",
    "thumb_up": "Thumb",
    "thumbup": "Thumb",
    "06_index": "Index",
    "index": "Index",
    "07_ok": "OK",
    "ok": "OK",
    "08_palm_moved": "Palm_Moved",
    "palm_moved": "Palm_Moved",
    "09_c": "C",
    "c": "C",
    "10_down": "Down",
    "down": "Down",
}

# HaGRID native labels mapped to the closest canonical class where applicable
_HAGRID_TO_CANONICAL: dict[str, str] = {
    "palm": "Palm",
    "fist": "Fist",
    "like": "Thumb",
    "ok": "OK",
    "one": "Index",
    "rock": "C",
    "two_up": "Thumb",
    "two_up_inverted": "Thumb",
}

_DATASET_ALIASES: dict[str, dict[str, str]] = {
    "leapgestrecog": dict(_LEAPGEST_TO_CANONICAL),
    "hagrid_subset": dict(_HAGRID_TO_CANONICAL),
}


def _normalize_key(raw_label: str) -> str:
    return raw_label.strip().lower().replace("-", "_").replace(" ", "_")


def _format_unmapped_label(raw_label: str) -> str:
    """Normalize labels with no explicit alias to a stable Title_Case form."""
    key = _normalize_key(raw_label)
    parts = key.split("_")
    return "_".join(p.capitalize() for p in parts if p)


def normalize_label(
    raw_label: str,
    dataset_name: str,
    *,
    label_aliases: dict[str, str] | None = None,
    canonical_labels: list[str] | None = None,
    align_to_canonical: bool = False,
) -> str:
    """Map a raw label string to a canonical gesture label."""
    if not raw_label or not str(raw_label).strip():
        raise ValueError("Empty raw label cannot be normalized")

    stripped = str(raw_label).strip()
    canonical_set = set(canonical_labels or [])

    if canonical_set and stripped in canonical_set:
        return stripped

    merged_aliases: dict[str, str] = {}
    if align_to_canonical:
        merged_aliases.update(_HAGRID_TO_CANONICAL)
        merged_aliases.update(_LEAPGEST_TO_CANONICAL)
    merged_aliases.update(_DATASET_ALIASES.get(dataset_name, {}))
    if label_aliases:
        for key, value in label_aliases.items():
            merged_aliases[_normalize_key(key)] = value

    key = _normalize_key(stripped)
    if key in merged_aliases:
        return merged_aliases[key]

    title = _format_unmapped_label(stripped)
    if canonical_set and title in canonical_set:
        return title

    if canonical_set:
        for canonical in canonical_set:
            if _normalize_key(canonical) == key:
                return canonical

    return title if align_to_canonical else stripped


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
