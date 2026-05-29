"""Feature extraction coverage and quality flagging."""

from __future__ import annotations

from collections import Counter
from typing import Any

_UNKNOWN_LABEL = "_unknown_"


def _gesture_label(record: dict[str, Any]) -> str:
    label = record.get("gesture_label")
    if label is None or str(label).strip() == "":
        return _UNKNOWN_LABEL
    return str(label)


def _record_extraction_successful(record: dict[str, Any]) -> bool:
    ok = bool(record.get("extraction_ok", True))
    inline = record.get("vector_inline")
    has_vector = inline is not None and len(inline) > 0
    return ok and has_vector


def _empty_coverage() -> dict[str, Any]:
    return {
        "total_samples": 0,
        "successful_extractions": 0,
        "failed_extractions": 0,
        "success_rate": 0.0,
        "detection_failures": 0,
        "low_confidence_count": 0,
        "missing_vectors": 0,
        "quality_flag_counts": {},
        "by_gesture_label": {},
        "classes_below_min_samples": [],
        "classes_below_success_rate": [],
    }


def merge_feature_coverage(
    accumulated: dict[str, Any],
    chunk: dict[str, Any],
) -> dict[str, Any]:
    """Merge two coverage dicts produced by :func:`evaluate_feature_coverage`."""
    if not accumulated.get("total_samples"):
        return dict(chunk)
    if not chunk.get("total_samples"):
        return dict(accumulated)

    flag_counter = Counter(accumulated.get("quality_flag_counts", {}))
    flag_counter.update(chunk.get("quality_flag_counts", {}))

    labels = set(accumulated.get("by_gesture_label", {})) | set(chunk.get("by_gesture_label", {}))
    by_gesture_label: dict[str, dict[str, Any]] = {}
    for label in labels:
        left = accumulated["by_gesture_label"].get(label, {})
        right = chunk["by_gesture_label"].get(label, {})
        total = int(left.get("total_samples", 0)) + int(right.get("total_samples", 0))
        success = int(left.get("successful_extractions", 0)) + int(
            right.get("successful_extractions", 0)
        )
        failed = total - success
        by_gesture_label[label] = {
            "total_samples": total,
            "successful_extractions": success,
            "failed_extractions": failed,
            "success_rate": success / total if total else 0.0,
        }

    total = int(accumulated["total_samples"]) + int(chunk["total_samples"])
    successful = int(accumulated["successful_extractions"]) + int(chunk["successful_extractions"])
    failed = total - successful
    return {
        "total_samples": total,
        "successful_extractions": successful,
        "failed_extractions": failed,
        "success_rate": successful / total if total else 0.0,
        "detection_failures": int(accumulated["detection_failures"])
        + int(chunk["detection_failures"]),
        "low_confidence_count": int(accumulated["low_confidence_count"])
        + int(chunk["low_confidence_count"]),
        "missing_vectors": int(accumulated["missing_vectors"]) + int(chunk["missing_vectors"]),
        "quality_flag_counts": dict(flag_counter),
        "by_gesture_label": by_gesture_label,
        "classes_below_min_samples": [],
        "classes_below_success_rate": [],
    }


def finalize_feature_coverage(
    coverage: dict[str, Any],
    *,
    min_samples_per_class: int | None = None,
    min_class_success_rate: float | None = None,
) -> dict[str, Any]:
    """Apply per-class threshold checks after merging chunked coverage stats."""
    if not coverage.get("total_samples"):
        return coverage

    by_gesture_label = coverage.get("by_gesture_label", {})
    classes_below_min_samples: list[dict[str, Any]] = []
    if min_samples_per_class is not None and min_samples_per_class > 0:
        for label, stats in by_gesture_label.items():
            if stats["total_samples"] < min_samples_per_class:
                classes_below_min_samples.append(
                    {
                        "gesture_label": label,
                        "total_samples": stats["total_samples"],
                        "min_required": min_samples_per_class,
                    }
                )
        classes_below_min_samples.sort(key=lambda item: item["total_samples"])

    classes_below_success_rate: list[dict[str, Any]] = []
    if min_class_success_rate is not None:
        for label, stats in by_gesture_label.items():
            if stats["success_rate"] < min_class_success_rate:
                classes_below_success_rate.append(
                    {
                        "gesture_label": label,
                        "success_rate": stats["success_rate"],
                        "successful_extractions": stats["successful_extractions"],
                        "total_samples": stats["total_samples"],
                        "threshold": min_class_success_rate,
                    }
                )
        classes_below_success_rate.sort(key=lambda item: item["success_rate"])

    finalized = dict(coverage)
    finalized["classes_below_min_samples"] = classes_below_min_samples
    finalized["classes_below_success_rate"] = classes_below_success_rate
    return finalized


def evaluate_feature_coverage(
    records: list[dict[str, Any]],
    *,
    min_samples_per_class: int | None = None,
    min_class_success_rate: float | None = None,
) -> dict[str, Any]:
    """Aggregate extraction success, failures, and per-gesture-label coverage."""
    if not records:
        return _empty_coverage()

    total = len(records)
    successful = 0
    detection_failures = 0
    low_confidence = 0
    missing_vectors = 0
    flag_counter: Counter[str] = Counter()

    class_total: Counter[str] = Counter()
    class_success: Counter[str] = Counter()

    for record in records:
        label = _gesture_label(record)
        class_total[label] += 1

        if _record_extraction_successful(record):
            successful += 1
            class_success[label] += 1

        inline = record.get("vector_inline")
        has_vector = inline is not None and len(inline) > 0
        if not has_vector:
            missing_vectors += 1

        flags = record.get("quality_flags") or {}
        if flags.get("detection_failed"):
            detection_failures += 1
        if flags.get("low_confidence"):
            low_confidence += 1
        for key, value in flags.items():
            if value:
                flag_counter[key] += 1

    by_gesture_label: dict[str, dict[str, Any]] = {}
    for label in sorted(class_total):
        class_count = class_total[label]
        class_ok = class_success[label]
        class_failed = class_count - class_ok
        by_gesture_label[label] = {
            "total_samples": class_count,
            "successful_extractions": class_ok,
            "failed_extractions": class_failed,
            "success_rate": class_ok / class_count if class_count else 0.0,
        }

    failed = total - successful
    partial = {
        "total_samples": total,
        "successful_extractions": successful,
        "failed_extractions": failed,
        "success_rate": successful / total if total else 0.0,
        "detection_failures": detection_failures,
        "low_confidence_count": low_confidence,
        "missing_vectors": missing_vectors,
        "quality_flag_counts": dict(flag_counter),
        "by_gesture_label": by_gesture_label,
        "classes_below_min_samples": [],
        "classes_below_success_rate": [],
    }
    return finalize_feature_coverage(
        partial,
        min_samples_per_class=min_samples_per_class,
        min_class_success_rate=min_class_success_rate,
    )


def flag_low_confidence_samples(
    records: list[dict[str, Any]],
    threshold: float,
) -> list[str]:
    """Return sample IDs whose landmark confidence is below *threshold*."""
    flagged: list[str] = []
    for record in records:
        confidence = record.get("confidence")
        if confidence is None:
            flags = record.get("quality_flags") or {}
            if flags.get("low_confidence"):
                flagged.append(str(record["sample_id"]))
            continue
        if float(confidence) < threshold:
            flagged.append(str(record["sample_id"]))
    return flagged


def apply_quality_flags(
    record: dict[str, Any],
    *,
    min_confidence: float,
    min_visible_landmarks: int | None = None,
) -> dict[str, Any]:
    """Augment record quality_flags based on thresholds."""
    flags = dict(record.get("quality_flags") or {})
    confidence = record.get("confidence")
    if confidence is not None and float(confidence) < min_confidence:
        flags["low_confidence"] = True

    visible = (record.get("quality_flags") or {}).get("visible_landmarks")
    if (
        min_visible_landmarks is not None
        and visible is not None
        and int(visible) < min_visible_landmarks
    ):
        flags["insufficient_landmarks"] = True

    record["quality_flags"] = flags
    return record
