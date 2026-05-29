"""Batch feature extraction helpers used by Phase 2 scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.features.geometric_features import build_geometric_from_config, geometric_vector_dim
from src.features.hand_detector import (
    HAND_LANDMARK_COUNT,
    detect_hand_landmarks,
    landmarks_to_raw_vector,
)
from src.features.hog_features import extract_hog_from_image, hog_descriptor_dim
from src.features.feature_combiner import build_feature_record

KEYPOINTS_RAW_DIM = HAND_LANDMARK_COUNT * 3
SUPPORTED_FAMILIES = frozenset({"keypoints_raw", "geometric", "hog"})


def family_config(config: dict[str, Any], feature_family: str) -> dict[str, Any]:
    families = config.get("feature_families", {})
    if feature_family in ("keypoints_raw", "geometric"):
        return dict(families.get("keypoints_only", {}))
    if feature_family == "hog":
        return dict(families.get("hog_only", {}))
    raise ValueError(f"Unsupported feature family: {feature_family}")


def expected_vector_dim(config: dict[str, Any], feature_family: str) -> int:
    fam_cfg = family_config(config, feature_family)
    if feature_family == "keypoints_raw":
        return KEYPOINTS_RAW_DIM
    if feature_family == "geometric":
        geometric_cfg = fam_cfg.get("geometric", {})
        return geometric_vector_dim(include_angles=bool(geometric_cfg.get("include_angles", True)))
    if feature_family == "hog":
        hog_cfg = fam_cfg.get("hog", fam_cfg)
        crop_size = tuple(hog_cfg.get("crop_size", (64, 64)))
        return hog_descriptor_dim(crop_size, hog_cfg)
    raise ValueError(feature_family)


def _zero_vector(dim: int) -> np.ndarray:
    return np.zeros(dim, dtype=np.float64)


def extract_sample_features(
    sample: dict[str, Any],
    feature_family: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Extract one feature vector for a manifest sample row."""
    if feature_family not in SUPPORTED_FAMILIES:
        raise ValueError(f"Unsupported feature family: {feature_family}")

    feature_version = str(config.get("feature_version", "v1"))
    fam_cfg = family_config(config, feature_family)
    sample_id = str(sample["sample_id"])
    image_path = Path(str(sample["image_path"]))

    metadata: dict[str, Any] = {
        "dataset_name": sample.get("dataset_name"),
        "gesture_label": sample.get("gesture_label"),
        "extraction_ok": False,
        "detection_failed": False,
        "low_confidence": False,
        "quality_flags": {},
    }

    dim = expected_vector_dim(config, feature_family)
    vector = _zero_vector(dim)
    detection = None

    if image_path.is_file():
        image = cv2.imread(str(image_path))
        if image is not None:
            mp_cfg = fam_cfg.get("mediapipe", fam_cfg)
            detection = detect_hand_landmarks(image, mp_cfg)

            if feature_family == "keypoints_raw":
                if detection is not None:
                    vector = landmarks_to_raw_vector(detection)
                    metadata["extraction_ok"] = True
                    metadata["confidence"] = detection.get("confidence")
                else:
                    metadata["detection_failed"] = True
                    metadata["quality_flags"]["detection_failed"] = True

            elif feature_family == "geometric":
                if detection is not None:
                    landmarks = detection["landmarks_pixel"]
                    vector = build_geometric_from_config(landmarks, fam_cfg)
                    metadata["extraction_ok"] = True
                    metadata["confidence"] = detection.get("confidence")
                else:
                    metadata["detection_failed"] = True
                    metadata["quality_flags"]["detection_failed"] = True

            elif feature_family == "hog":
                landmarks = detection["landmarks_pixel"] if detection else None
                vector, hog_meta = extract_hog_from_image(image, landmarks, fam_cfg)
                metadata.update(hog_meta)
                metadata["extraction_ok"] = True
                if detection is None:
                    metadata["quality_flags"]["landmark_crop_fallback"] = True
                else:
                    metadata["confidence"] = detection.get("confidence")
    else:
        metadata["quality_flags"]["missing_image"] = True

    min_confidence = float(
        config.get("quality_flags", {}).get(
            "min_detection_confidence",
            fam_cfg.get("mediapipe", {}).get("min_detection_confidence", 0.5),
        )
    )
    if metadata.get("confidence") is not None and float(metadata["confidence"]) < min_confidence:
        metadata["low_confidence"] = True
        metadata["quality_flags"]["low_confidence"] = True

    return build_feature_record(
        sample_id,
        feature_family,
        vector,
        metadata,
        dataset_name=str(sample.get("dataset_name", "")),
        gesture_label=str(sample.get("gesture_label", "")),
        feature_version=feature_version,
    )
