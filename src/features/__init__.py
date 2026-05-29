"""Feature extraction pipeline (Phase 2+)."""

from src.features.feature_combiner import (
    DEFAULT_FAMILY_ORDER,
    build_feature_record,
    concatenate_features,
)
from src.features.feature_store import (
    FeatureTable,
    build_feature_manifest,
    config_fingerprint,
    load_feature_matrix,
    manifest_path_for_matrix,
    save_feature_manifest,
    save_feature_matrix,
    vector_from_record,
)
from src.features.geometric_features import (
    GEOMETRIC_VECTOR_DIM,
    build_geometric_vector,
    compute_joint_angles,
    compute_pairwise_distances,
    normalize_landmarks,
)
from src.features.hand_detector import (
    HAND_LANDMARK_COUNT,
    detect_hand_landmarks,
    landmarks_to_raw_vector,
    reset_detector,
)
from src.features.hog_features import crop_hand_region, extract_hog_descriptor, extract_hog_from_image
from src.features.quality_checks import (
    apply_quality_flags,
    evaluate_feature_coverage,
    flag_low_confidence_samples,
)

__all__ = [
    "DEFAULT_FAMILY_ORDER",
    "FeatureTable",
    "GEOMETRIC_VECTOR_DIM",
    "HAND_LANDMARK_COUNT",
    "apply_quality_flags",
    "build_feature_manifest",
    "build_feature_record",
    "build_geometric_vector",
    "compute_joint_angles",
    "compute_pairwise_distances",
    "concatenate_features",
    "config_fingerprint",
    "crop_hand_region",
    "detect_hand_landmarks",
    "evaluate_feature_coverage",
    "extract_hog_descriptor",
    "extract_hog_from_image",
    "flag_low_confidence_samples",
    "landmarks_to_raw_vector",
    "load_feature_matrix",
    "manifest_path_for_matrix",
    "normalize_landmarks",
    "reset_detector",
    "save_feature_manifest",
    "save_feature_matrix",
    "vector_from_record",
]
