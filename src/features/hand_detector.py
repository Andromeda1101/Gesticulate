"""MediaPipe hand landmark detection with normalized outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.common.path_manager import resolve_project_root

HAND_LANDMARK_COUNT = 21
WRIST_INDEX = 0

_DEFAULT_MODEL_REL = Path("data/models/hand_landmarker.task")
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

_landmarker = None
_legacy_hands = None


def _mediapipe_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("mediapipe", config))


def resolve_model_path(config: dict[str, Any]) -> Path:
    """Resolve Hand Landmarker ``.task`` model path (download target if missing)."""
    mp_config = _mediapipe_config(config)
    raw = mp_config.get("model_path", _DEFAULT_MODEL_REL)
    path = Path(str(raw))
    if not path.is_absolute():
        path = resolve_project_root() / path
    return path


def ensure_hand_landmarker_model(config: dict[str, Any]) -> Path:
    """Return model path, downloading the bundled task file when absent."""
    path = resolve_model_path(config)
    if path.is_file():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from urllib.request import urlretrieve

        urlretrieve(_MODEL_URL, path)  # noqa: S310 — trusted Google static URL
    except Exception as exc:
        raise FileNotFoundError(
            f"Hand landmarker model not found at {path}. "
            f"Download manually from {_MODEL_URL}"
        ) from exc
    return path


def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.shape[-1] == 4:
        return image[..., :3]
    if image.shape[-1] == 3:
        # OpenCV loads BGR; MediaPipe expects RGB.
        return image[..., ::-1]
    return image


def _apply_clahe(gray: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    import cv2

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(gray)


def prepare_image_for_detection(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """
    Convert grayscale/BGR inputs to RGB for MediaPipe.

    LeapGestRecog (OOD) uses full-frame near-infrared images (often 240x640). Hands
    are typically near the image centre; no automatic panel cropping is applied.
    """
    mp_config = _mediapipe_config(config)
    prep = dict(mp_config.get("preprocessing", {}))
    rgb = _to_rgb(np.asarray(image))

    crop_mode = str(prep.get("crop_region", "none")).lower()
    if crop_mode == "right_half":
        w = rgb.shape[1]
        rgb = rgb[:, w // 2 :]
    elif crop_mode == "left_half":
        w = rgb.shape[1]
        rgb = rgb[:, : w // 2]

    scale = float(prep.get("scale", 1.0))
    if scale > 1.0:
        import cv2

        rgb = cv2.resize(
            rgb,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    if prep.get("apply_clahe", False):
        import cv2

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        rgb = np.stack([_apply_clahe(gray)] * 3, axis=-1)

    return rgb


def _get_tasks_landmarker(config: dict[str, Any]):
    global _landmarker
    import mediapipe as mp

    mp_config = _mediapipe_config(config)
    if _landmarker is None:
        model_path = ensure_hand_landmarker_model(config)
        min_conf = float(mp_config.get("min_detection_confidence", 0.5))
        min_track = float(mp_config.get("min_tracking_confidence", 0.5))
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=int(mp_config.get("max_num_hands", 1)),
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_track,
        )
        _landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
    return _landmarker


def _get_legacy_hands_solution(config: dict[str, Any]):
    global _legacy_hands
    import mediapipe as mp

    if not hasattr(mp, "solutions"):
        return None

    mp_config = _mediapipe_config(config)
    if _legacy_hands is None:
        _legacy_hands = mp.solutions.hands.Hands(
            static_image_mode=bool(mp_config.get("static_image_mode", True)),
            max_num_hands=int(mp_config.get("max_num_hands", 1)),
            min_detection_confidence=float(mp_config.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(mp_config.get("min_tracking_confidence", 0.5)),
        )
    return _legacy_hands


def reset_detector() -> None:
    """Release cached detector instances (for tests)."""
    global _landmarker, _legacy_hands
    if _landmarker is not None:
        _landmarker.close()
        _landmarker = None
    if _legacy_hands is not None:
        _legacy_hands.close()
        _legacy_hands = None


def _landmarks_from_tasks_result(
    result: Any,
    width: int,
    height: int,
) -> dict[str, Any] | None:
    if not result.hand_landmarks:
        return None

    hand_landmarks = result.hand_landmarks[0]
    normalized = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
        dtype=np.float64,
    )
    pixel_xy = normalized[:, :2] * np.array([width, height], dtype=np.float64)

    handedness = "Unknown"
    confidence = 0.0
    if result.handedness and result.handedness[0]:
        classification = result.handedness[0][0]
        handedness = classification.category_name
        confidence = float(classification.score)

    return {
        "landmarks_normalized": normalized,
        "landmarks_pixel": np.column_stack([pixel_xy, normalized[:, 2]]),
        "handedness": handedness,
        "confidence": confidence,
        "image_shape": (height, width),
        "success": True,
    }


def _detect_with_tasks(rgb: np.ndarray, config: dict[str, Any]) -> dict[str, Any] | None:
    import mediapipe as mp

    height, width = rgb.shape[:2]
    landmarker = _get_tasks_landmarker(config)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    return _landmarks_from_tasks_result(result, width, height)


def _detect_with_legacy(rgb: np.ndarray, config: dict[str, Any]) -> dict[str, Any] | None:
    hands = _get_legacy_hands_solution(config)
    if hands is None:
        return None

    height, width = rgb.shape[:2]
    results = hands.process(rgb)
    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0]
    normalized = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float64,
    )
    pixel_xy = normalized[:, :2] * np.array([width, height], dtype=np.float64)

    handedness = "Unknown"
    confidence = 0.0
    if results.multi_handedness:
        classification = results.multi_handedness[0].classification[0]
        handedness = classification.label
        confidence = float(classification.score)

    return {
        "landmarks_normalized": normalized,
        "landmarks_pixel": np.column_stack([pixel_xy, normalized[:, 2]]),
        "handedness": handedness,
        "confidence": confidence,
        "image_shape": (height, width),
        "success": True,
    }


def detect_hand_landmarks(image: np.ndarray, config: dict[str, Any]) -> dict[str, Any] | None:
    """
    Detect hand landmarks in an image.

    Returns a dict with normalized (0–1) and pixel-space landmarks, handedness,
    and confidence, or ``None`` when no hand is detected.
    """
    if image is None or image.size == 0:
        return None

    rgb = prepare_image_for_detection(image, config)
    mp_config = _mediapipe_config(config)

    try:
        import mediapipe as mp

        if hasattr(mp, "tasks"):
            return _detect_with_tasks(rgb, mp_config)
        return _detect_with_legacy(rgb, mp_config)
    except Exception:
        return None


def landmarks_to_raw_vector(detection: dict[str, Any]) -> np.ndarray:
    """Flatten normalized x,y,z landmarks into a fixed-length keypoints vector."""
    normalized = np.asarray(detection["landmarks_normalized"], dtype=np.float64)
    if normalized.shape != (HAND_LANDMARK_COUNT, 3):
        raise ValueError(f"Expected {(HAND_LANDMARK_COUNT, 3)} landmarks, got {normalized.shape}")
    return normalized.reshape(-1)
