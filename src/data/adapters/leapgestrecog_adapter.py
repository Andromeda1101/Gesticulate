"""Parse LeapGestRecog folder structure into standardized sample records."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_LEAPGEST_GESTURE_DIR = re.compile(r"^\d{2}_[a-z0-9_]+$", re.IGNORECASE)


def _stable_sample_id(dataset_name: str, rel_path: str) -> str:
    payload = f"{dataset_name}:{rel_path}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _looks_like_leapgest_gesture_dir(name: str) -> bool:
    """True for LeapGestRecog folders such as ``01_palm`` or ``10_down``."""
    return bool(_LEAPGEST_GESTURE_DIR.match(name.strip()))


def _infer_subject_and_label(parts: tuple[str, ...]) -> tuple[str | None, str]:
    """
    Infer subject ID and gesture label from path segments.

    Common layouts:
      subject_XX/Gesture/image.jpg
      leapGestRecog/<subject_id>/01_palm/image.jpg
      Gesture/subject_XX/image.jpg
      Gesture/image.jpg
    """
    if not parts:
        return None, "unknown"
    if len(parts) == 1:
        return None, parts[0]

    lower_parts = [p.lower() for p in parts]
    subject_idx = next(
        (i for i, p in enumerate(lower_parts) if p.startswith("subject") or p.startswith("s_")),
        None,
    )

    if subject_idx is not None:
        subject_id = parts[subject_idx]
        label_candidates = [p for i, p in enumerate(parts) if i != subject_idx]
        gesture = label_candidates[-1] if label_candidates else "unknown"
        return subject_id, gesture

    # Official layout: .../<subject_id>/<NN_gesture>/frame.jpg
    if _looks_like_leapgest_gesture_dir(parts[-1]):
        return (parts[-2] if len(parts) >= 2 else None), parts[-1]

    # Single gesture folder above the image
    return (parts[-2] if len(parts) >= 2 else None), parts[-1]


def index_samples(
    root_dir: str,
    label_map: dict[str, Any] | None = None,
    *,
    dataset_name: str = "leapgestrecog",
    label_vocabulary: list[str] | None = None,
    capture_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Traverse subject and gesture folders and emit canonical sample metadata.

    Parameters
    ----------
    root_dir:
        Dataset root directory.
    label_map:
        Optional alias map (raw -> canonical); applied later by label_mapper.
    """
    root = Path(root_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LeapGestRecog root not found: {root}")

    base_context = dict(capture_context or {"source": "static_images"})
    samples: list[dict[str, Any]] = []

    for image_path in sorted(root.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue

        rel = image_path.relative_to(root)
        parts = rel.parts
        if len(parts) < 2:
            continue

        subject_id, raw_label = _infer_subject_and_label(parts[:-1])

        rel_posix = rel.as_posix()
        sample: dict[str, Any] = {
            "sample_id": _stable_sample_id(dataset_name, rel_posix),
            "dataset_name": dataset_name,
            "subject_id": subject_id,
            "raw_gesture_label": raw_label,
            "gesture_label": raw_label,
            "image_path": str(image_path),
            "split": None,
            "capture_context": {
                **base_context,
                "relative_path": rel_posix,
            },
        }
        if label_map:
            sample["capture_context"]["label_map"] = label_map
        samples.append(sample)

    return samples
