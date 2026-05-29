"""Translation- and scale-invariant geometric hand descriptors."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.features.hand_detector import HAND_LANDMARK_COUNT, WRIST_INDEX

# Finger chains for angle computation (MediaPipe hand topology).
_ANGLE_TRIPLETS: tuple[tuple[int, int, int], ...] = (
    (0, 5, 9),
    (0, 9, 13),
    (0, 13, 17),
    (5, 6, 8),
    (9, 10, 12),
    (13, 14, 16),
    (17, 18, 20),
    (0, 1, 5),
)

NUM_PAIRWISE_DISTANCES = HAND_LANDMARK_COUNT * (HAND_LANDMARK_COUNT - 1) // 2
NUM_JOINT_ANGLES = len(_ANGLE_TRIPLETS)
GEOMETRIC_VECTOR_DIM = HAND_LANDMARK_COUNT * 2 + NUM_PAIRWISE_DISTANCES + NUM_JOINT_ANGLES


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Wrist-relative centering and scale normalization (returns xy, shape 21x2)."""
    coords = np.asarray(landmarks, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[0] != HAND_LANDMARK_COUNT:
        raise ValueError(f"Expected landmarks shape ({HAND_LANDMARK_COUNT}, D), got {coords.shape}")

    xy = coords[:, :2].copy()
    wrist = xy[WRIST_INDEX]
    centered = xy - wrist
    bbox = xy.max(axis=0) - xy.min(axis=0)
    scale = float(np.linalg.norm(bbox))
    if scale < 1e-6:
        scale = 1.0
    return centered / scale


def compute_pairwise_distances(landmarks: np.ndarray) -> np.ndarray:
    """Upper-triangle pairwise Euclidean distances in landmark space."""
    xy = np.asarray(landmarks, dtype=np.float64)
    if xy.ndim == 2 and xy.shape[1] >= 2:
        xy = xy[:, :2]
    n = xy.shape[0]
    distances: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(float(np.linalg.norm(xy[i] - xy[j])))
    return np.asarray(distances, dtype=np.float64)


def _angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8:
        return 0.0
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.arccos(cosine))


def compute_joint_angles(landmarks: np.ndarray) -> np.ndarray:
    """Selected joint angles (radians) from landmark xy coordinates."""
    xy = np.asarray(landmarks, dtype=np.float64)
    if xy.ndim == 2 and xy.shape[1] >= 2:
        xy = xy[:, :2]
    angles = [_angle_at(xy[i], xy[j], xy[k]) for i, j, k in _ANGLE_TRIPLETS]
    return np.asarray(angles, dtype=np.float64)


def build_geometric_vector(
    landmarks: np.ndarray,
    *,
    include_angles: bool = True,
) -> np.ndarray:
    """Concatenate normalized coords, pairwise distances, and optional angles."""
    normalized = normalize_landmarks(landmarks)
    blocks = [normalized.reshape(-1), compute_pairwise_distances(normalized)]
    if include_angles:
        blocks.append(compute_joint_angles(normalized))
    vector = np.concatenate(blocks)
    if vector.shape[0] != GEOMETRIC_VECTOR_DIM:
        raise ValueError(f"Unexpected geometric dim {vector.shape[0]}, expected {GEOMETRIC_VECTOR_DIM}")
    return vector


def geometric_vector_dim(*, include_angles: bool = True) -> int:
    base = HAND_LANDMARK_COUNT * 2 + NUM_PAIRWISE_DISTANCES
    return base + (NUM_JOINT_ANGLES if include_angles else 0)


def build_geometric_from_config(landmarks: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    geometric_cfg = config.get("geometric", config)
    include_angles = bool(geometric_cfg.get("include_angles", True))
    return build_geometric_vector(landmarks, include_angles=include_angles)
