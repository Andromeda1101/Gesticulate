"""Webcam initialization, frame capture, and cleanup."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

_cap: cv2.VideoCapture | None = None


def open_camera(
    camera_index: int,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> cv2.VideoCapture:
    """Open webcam with OpenCV and apply optional resolution hints."""
    global _cap
    close_camera()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera at index {camera_index}")

    if frame_width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(frame_width))
    if frame_height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(frame_height))

    _cap = cap
    return cap


def read_frame() -> tuple[bool, np.ndarray | None]:
    """Read one frame from the active camera stream."""
    if _cap is None or not _cap.isOpened():
        raise RuntimeError("Camera is not open. Call open_camera() first.")

    ok, frame = _cap.read()
    if not ok:
        return False, None
    return True, frame


def get_camera_properties() -> dict[str, Any]:
    """Return width/height/fps for the active capture device."""
    if _cap is None or not _cap.isOpened():
        return {}
    return {
        "width": int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(_cap.get(cv2.CAP_PROP_FPS)),
    }


def close_camera() -> None:
    """Release the active camera resource."""
    global _cap
    if _cap is not None:
        _cap.release()
        _cap = None
