"""Live frame preprocessing and online feature extraction."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.features.extraction import (
    SUPPORTED_FAMILIES,
    expected_vector_dim,
    family_config,
)
from src.features.feature_combiner import build_feature_record, concatenate_features
from src.features.geometric_features import build_geometric_from_config
from src.features.hand_detector import detect_hand_landmarks
from src.features.hog_features import extract_hog_from_image
from src.models.feature_resolver import manifest_feature_family

HYBRID_FAMILY = "hybrid_keypoints_hog"
HYBRID_SOURCE_ORDER = ("geometric", "hog")


def resolve_runtime_feature_family(feature_family: str) -> str:
    """Map runtime config aliases to manifest column values."""
    return manifest_feature_family(feature_family)


def prepare_frame(frame: np.ndarray, runtime_config: dict[str, Any]) -> np.ndarray:
    """Resize or pass through a live frame according to camera settings."""
    image = np.asarray(frame)
    camera_cfg = runtime_config.get("camera", {})
    target_w = camera_cfg.get("width")
    target_h = camera_cfg.get("height")
    if target_w and target_h:
        current_h, current_w = image.shape[:2]
        if current_w != int(target_w) or current_h != int(target_h):
            image = cv2.resize(
                image,
                (int(target_w), int(target_h)),
                interpolation=cv2.INTER_AREA,
            )
    return image


def _min_confidence(feature_config: dict[str, Any], fam_cfg: dict[str, Any]) -> float:
    return float(
        feature_config.get("quality_flags", {}).get(
            "min_detection_confidence",
            fam_cfg.get("mediapipe", {}).get("min_detection_confidence", 0.5),
        )
    )


def _apply_quality_flags(metadata: dict[str, Any], min_confidence: float) -> None:
    confidence = metadata.get("confidence")
    if confidence is not None and float(confidence) < min_confidence:
        metadata["low_confidence"] = True
        metadata.setdefault("quality_flags", {})["low_confidence"] = True


def _extract_geometric_block(
    image: np.ndarray,
    detection: dict[str, Any] | None,
    fam_cfg: dict[str, Any],
    *,
    dim: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "extraction_ok": False,
        "detection_failed": False,
        "quality_flags": {},
    }
    vector = np.zeros(dim, dtype=np.float64)
    if detection is not None:
        landmarks = detection["landmarks_pixel"]
        vector = build_geometric_from_config(landmarks, fam_cfg)
        metadata["extraction_ok"] = True
        metadata["confidence"] = detection.get("confidence")
    else:
        metadata["detection_failed"] = True
        metadata["quality_flags"]["detection_failed"] = True
    return vector, metadata


def _extract_hog_block(
    image: np.ndarray,
    detection: dict[str, Any] | None,
    fam_cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    landmarks = detection["landmarks_pixel"] if detection else None
    vector, hog_meta = extract_hog_from_image(image, landmarks, fam_cfg)
    metadata: dict[str, Any] = {
        "extraction_ok": True,
        "quality_flags": {},
        **hog_meta,
    }
    if detection is None:
        metadata["quality_flags"]["landmark_crop_fallback"] = True
    else:
        metadata["confidence"] = detection.get("confidence")
    return vector, metadata


def extract_runtime_features(
    frame: np.ndarray,
    feature_config: dict[str, Any],
    *,
    feature_family: str = "hybrid",
) -> dict[str, Any]:
    """
    Extract a feature vector from a live frame with quality flags.

    Returns a feature-record-like dict compatible with ``vector_from_record``.
    """
    image = np.asarray(frame)
    feature_version = str(feature_config.get("feature_version", "v1"))
    resolved_family = resolve_runtime_feature_family(feature_family)

    if resolved_family == HYBRID_FAMILY or feature_family in ("hybrid", HYBRID_FAMILY):
        geom_cfg = family_config(feature_config, "geometric")
        hog_cfg = family_config(feature_config, "hog")
        mp_cfg = geom_cfg.get("mediapipe", geom_cfg)
        detection = detect_hand_landmarks(image, mp_cfg)

        geom_dim = expected_vector_dim(feature_config, "geometric")
        geom_vector, geom_meta = _extract_geometric_block(
            image, detection, geom_cfg, dim=geom_dim
        )
        hog_vector, hog_meta = _extract_hog_block(image, detection, hog_cfg)

        blocks = {"geometric": geom_vector, "hog": hog_vector}
        vector = concatenate_features(blocks, family_order=HYBRID_SOURCE_ORDER)

        quality_flags = {
            **(geom_meta.get("quality_flags") or {}),
            **(hog_meta.get("quality_flags") or {}),
        }
        metadata: dict[str, Any] = {
            "extraction_ok": bool(geom_meta.get("extraction_ok")) and bool(hog_meta.get("extraction_ok")),
            "detection_failed": bool(geom_meta.get("detection_failed")),
            "quality_flags": quality_flags,
            "source_families": list(HYBRID_SOURCE_ORDER),
            "confidence": min(
                float(geom_meta["confidence"]) if geom_meta.get("confidence") is not None else 1.0,
                float(hog_meta["confidence"]) if hog_meta.get("confidence") is not None else 1.0,
            ),
        }
        _apply_quality_flags(metadata, _min_confidence(feature_config, geom_cfg))
        return build_feature_record(
            "runtime_frame",
            HYBRID_FAMILY,
            vector,
            metadata,
            feature_version=feature_version,
        )

    if feature_family not in SUPPORTED_FAMILIES:
        raise ValueError(
            f"Unsupported runtime feature family: {feature_family}. "
            f"Supported: {sorted(SUPPORTED_FAMILIES)} or hybrid"
        )

    fam_cfg = family_config(feature_config, feature_family)
    mp_cfg = fam_cfg.get("mediapipe", fam_cfg)
    detection = detect_hand_landmarks(image, mp_cfg)
    dim = expected_vector_dim(feature_config, feature_family)

    metadata = {
        "extraction_ok": False,
        "detection_failed": False,
        "quality_flags": {},
    }
    vector = np.zeros(dim, dtype=np.float64)

    if feature_family == "geometric":
        vector, metadata = _extract_geometric_block(image, detection, fam_cfg, dim=dim)
    elif feature_family == "hog":
        vector, metadata_dict = _extract_hog_block(image, detection, fam_cfg)
        metadata.update(metadata_dict)
    elif feature_family == "keypoints_raw":
        from src.features.hand_detector import landmarks_to_raw_vector

        if detection is not None:
            vector = landmarks_to_raw_vector(detection)
            metadata["extraction_ok"] = True
            metadata["confidence"] = detection.get("confidence")
        else:
            metadata["detection_failed"] = True
            metadata["quality_flags"]["detection_failed"] = True

    _apply_quality_flags(metadata, _min_confidence(feature_config, fam_cfg))
    return build_feature_record(
        "runtime_frame",
        resolved_family,
        vector,
        metadata,
        feature_version=feature_version,
    )
