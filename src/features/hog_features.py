"""HOG appearance descriptors from cropped hand regions."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog

from src.features.hand_detector import HAND_LANDMARK_COUNT
from src.features.hog_layout import DEFAULT_CROP_SIZE, hog_block_grid_shape
DEFAULT_PADDING = 0.15


def crop_hand_region(
    image: np.ndarray,
    landmarks: np.ndarray | None,
    *,
    crop_size: tuple[int, int] = DEFAULT_CROP_SIZE,
    padding: float = DEFAULT_PADDING,
) -> np.ndarray:
    """Crop and resize a hand region; falls back to centered square crop."""
    img = np.asarray(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[-1] == 4:
        img = img[..., :3]

    height, width = img.shape[:2]

    if landmarks is not None and len(landmarks) >= HAND_LANDMARK_COUNT:
        xy = np.asarray(landmarks, dtype=np.float64)
        if xy.ndim == 2 and xy.shape[1] >= 2:
            xy = xy[:, :2]
        if xy.max() <= 1.0 and xy.min() >= 0.0:
            xy = xy * np.array([width, height], dtype=np.float64)

        x_min, y_min = xy.min(axis=0)
        x_max, y_max = xy.max(axis=0)
        pad_x = (x_max - x_min) * padding
        pad_y = (y_max - y_min) * padding
        x0 = int(max(0, x_min - pad_x))
        y0 = int(max(0, y_min - pad_y))
        x1 = int(min(width, x_max + pad_x))
        y1 = int(min(height, y_max + pad_y))
    else:
        side = min(height, width)
        x0 = (width - side) // 2
        y0 = (height - side) // 2
        x1 = x0 + side
        y1 = y0 + side

    if x1 <= x0 or y1 <= y0:
        crop = img
    else:
        crop = img[y0:y1, x0:x1]

    target_w, target_h = crop_size
    return cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)


def hog_descriptor_dim(image_shape: tuple[int, int], hog_config: dict[str, Any]) -> int:
    """Return the HOG vector length for a given crop size and parameters."""
    dummy = np.zeros(image_shape, dtype=np.uint8)
    return int(extract_hog_descriptor(dummy, hog_config).shape[0])


def extract_hog_descriptor(image_crop: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Compute a fixed-length HOG feature vector from a grayscale crop."""
    hog_cfg = config.get("hog", config)
    crop = np.asarray(image_crop)
    if crop.ndim == 3:
        gray = rgb2gray(crop)
    else:
        gray = crop.astype(np.float64) / 255.0 if crop.max() > 1.0 else crop.astype(np.float64)

    orientations = int(hog_cfg.get("orientations", 9))
    pixels_per_cell = tuple(hog_cfg.get("pixels_per_cell", (8, 8)))
    cells_per_block = tuple(hog_cfg.get("cells_per_block", (2, 2)))
    block_norm = str(hog_cfg.get("block_norm", "L2-Hys"))
    transform_sqrt = bool(hog_cfg.get("transform_sqrt", True))

    descriptor = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        transform_sqrt=transform_sqrt,
        feature_vector=True,
    )
    return np.asarray(descriptor, dtype=np.float64)


def extract_hog_from_image(
    image: np.ndarray,
    landmarks: np.ndarray | None,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Crop hand region and return HOG vector plus crop metadata."""
    hog_cfg = config.get("hog", config)
    crop_size = tuple(hog_cfg.get("crop_size", DEFAULT_CROP_SIZE))
    padding = float(hog_cfg.get("crop_padding", DEFAULT_PADDING))
    crop = crop_hand_region(image, landmarks, crop_size=crop_size, padding=padding)
    vector = extract_hog_descriptor(crop, hog_cfg)
    metadata = {
        "crop_size": list(crop_size),
        "used_landmark_crop": landmarks is not None,
        "hog_dim": int(vector.shape[0]),
    }
    return vector, metadata
