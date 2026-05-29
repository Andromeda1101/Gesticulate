"""Parse HaGRID subset (annotation JSON or folder-by-label) into sample records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_ANNOTATION_CANDIDATES = (
    "annotations.json",
    "annotations_train.json",
    "annotations_val.json",
    "annotations_test.json",
    "subset_annotations.json",
)


def _stable_sample_id(dataset_name: str, rel_path: str) -> str:
    payload = f"{dataset_name}:{rel_path}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _detect_format(root: Path) -> str:
    for name in _ANNOTATION_CANDIDATES:
        if (root / name).is_file():
            return "annotations"
    for name in _ANNOTATION_CANDIDATES:
        matches = list(root.rglob(name))
        if matches:
            return "annotations"
    return "folder_by_label"


def _find_annotation_file(root: Path) -> Path | None:
    for name in _ANNOTATION_CANDIDATES:
        direct = root / name
        if direct.is_file():
            return direct
    matches = sorted(root.rglob(name))
    return matches[0] if matches else None


def _resolve_image_path(root: Path, record: dict[str, Any]) -> Path | None:
    for key in ("image_path", "file_name", "filename", "path", "img"):
        if key not in record:
            continue
        raw = Path(str(record[key]))
        if raw.is_file():
            return raw
        candidate = root / raw
        if candidate.is_file():
            return candidate
        for sub in ("images", "imgs", "data"):
            nested = root / sub / raw
            if nested.is_file():
                return nested
    return None


def _extract_label(record: dict[str, Any]) -> str:
    for key in ("label", "gesture", "class", "category", "gesture_label"):
        if key in record and record[key] is not None:
            return str(record[key])
    return "unknown"


def _index_from_annotations(
    root: Path,
    *,
    dataset_name: str,
    subset_spec: dict[str, Any],
    capture_context: dict[str, Any],
) -> list[dict[str, Any]]:
    ann_path = _find_annotation_file(root)
    if ann_path is None:
        raise FileNotFoundError(f"No HaGRID annotation file found under {root}")

    with ann_path.open(encoding="utf-8") as fh:
        payload = json.load(fh)

    if isinstance(payload, dict):
        records = list(payload.values()) if all(isinstance(v, dict) for v in payload.values()) else [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(f"Unsupported annotation JSON structure: {ann_path}")

    # Only explicit target_labels / label_filter restrict classes; label_vocabulary
    # is a post-index reference vocabulary (e.g. canonical vocabulary overlap), not a filter.
    target_labels = set(subset_spec.get("target_labels") or [])
    max_per_class = subset_spec.get("max_samples_per_class")
    if max_per_class is None:
        sampling = subset_spec.get("sampling") or {}
        max_per_class = sampling.get("max_samples_per_class")

    per_class_counts: dict[str, int] = {}
    samples: list[dict[str, Any]] = []

    for record in records:
        if not isinstance(record, dict):
            continue
        raw_label = _extract_label(record)
        if target_labels:
            normalized_key = raw_label.lower().replace("-", "_").replace(" ", "_")
            allowed = {t.lower().replace("-", "_").replace(" ", "_") for t in target_labels}
            if normalized_key not in allowed:
                continue

        if max_per_class is not None:
            key = raw_label
            if per_class_counts.get(key, 0) >= int(max_per_class):
                continue

        image_path = _resolve_image_path(root, record)
        if image_path is None:
            continue

        rel = image_path.relative_to(root).as_posix()
        background = record.get("background") or record.get("bg") or record.get("scene")
        sample = {
            "sample_id": _stable_sample_id(dataset_name, rel),
            "dataset_name": dataset_name,
            "subject_id": str(record.get("user_id") or record.get("subject_id") or ""),
            "raw_gesture_label": raw_label,
            "gesture_label": raw_label,
            "image_path": str(image_path),
            "split": record.get("split"),
            "capture_context": {
                **capture_context,
                "relative_path": rel,
                "annotation_file": str(ann_path.relative_to(root)),
                "background": background,
                "format": "annotations",
            },
        }
        samples.append(sample)
        per_class_counts[raw_label] = per_class_counts.get(raw_label, 0) + 1

    return samples


def _index_from_folders(
    root: Path,
    *,
    dataset_name: str,
    subset_spec: dict[str, Any],
    capture_context: dict[str, Any],
) -> list[dict[str, Any]]:
    # Only explicit target_labels / label_filter restrict classes; label_vocabulary
    # is a post-index reference vocabulary (e.g. canonical vocabulary overlap), not a filter.
    target_labels = subset_spec.get("target_labels") or []
    max_per_class = subset_spec.get("max_samples_per_class")
    if max_per_class is None:
        sampling = subset_spec.get("sampling") or {}
        max_per_class = sampling.get("max_samples_per_class")

    per_class_counts: dict[str, int] = {}
    samples: list[dict[str, Any]] = []

    label_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if not label_dirs:
        label_dirs = [p for p in sorted(root.rglob("*")) if p.is_dir() and any(p.glob("*.*"))]

    for label_dir in label_dirs:
        raw_label = label_dir.name
        if target_labels and raw_label not in target_labels:
            allowed = {str(t) for t in target_labels}
            if raw_label not in allowed:
                continue

        for image_path in sorted(label_dir.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue

            if max_per_class is not None and per_class_counts.get(raw_label, 0) >= int(max_per_class):
                break

            rel = image_path.relative_to(root).as_posix()
            samples.append(
                {
                    "sample_id": _stable_sample_id(dataset_name, rel),
                    "dataset_name": dataset_name,
                    "subject_id": None,
                    "raw_gesture_label": raw_label,
                    "gesture_label": raw_label,
                    "image_path": str(image_path),
                    "split": None,
                    "capture_context": {
                        **capture_context,
                        "relative_path": rel,
                        "format": "folder_by_label",
                    },
                }
            )
            per_class_counts[raw_label] = per_class_counts.get(raw_label, 0) + 1

    return samples


def index_samples(
    root_dir: str,
    subset_spec: dict[str, Any] | None = None,
    *,
    dataset_name: str = "hagrid_subset",
) -> list[dict[str, Any]]:
    """Index HaGRID subset samples using auto-detected layout."""
    root = Path(root_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"HaGRID root not found: {root}")

    spec = dict(subset_spec or {})
    capture_context = dict(spec.get("capture_context") or {"source": "in_the_wild"})
    spec.setdefault("label_vocabulary", spec.get("label_vocabulary"))

    fmt = spec.get("format") or _detect_format(root)
    if fmt == "annotations" or (fmt != "folder_by_label" and _find_annotation_file(root)):
        samples = _index_from_annotations(
            root,
            dataset_name=dataset_name,
            subset_spec=spec,
            capture_context=capture_context,
        )
    else:
        samples = _index_from_folders(
            root,
            dataset_name=dataset_name,
            subset_spec=spec,
            capture_context=capture_context,
        )

    if spec.get("shuffle") and samples:
        df = pd.DataFrame(samples)
        seed = int((spec.get("split_strategy") or {}).get("seed", 42))
        samples = df.sample(frac=1.0, random_state=seed).to_dict(orient="records")

    return samples
