"""Webcam initialization, frame capture, and cleanup."""

from __future__ import annotations

import platform
import sys
from typing import Any

import cv2
import numpy as np

from src.common.logger import get_logger

logger = get_logger("src.runtime.camera_stream")

_cap: cv2.VideoCapture | None = None
_active_backend: str | None = None
_open_params: dict[str, Any] = {}

_BACKEND_ALIASES: dict[str, int | None] = {
    "default": None,
    "auto": None,
    "dshow": getattr(cv2, "CAP_DSHOW", None),
    "msmf": getattr(cv2, "CAP_MSMF", None),
    "v4l2": getattr(cv2, "CAP_V4L2", None),
}


def _default_fallback_backends() -> list[str]:
    if sys.platform == "win32":
        return ["dshow", "msmf", "default"]
    if sys.platform == "linux":
        return ["v4l2", "default"]
    return ["default"]


def _backend_label(api: int | None) -> str:
    if api is None:
        return "default"
    for name, value in _BACKEND_ALIASES.items():
        if name in {"default", "auto"}:
            continue
        if value == api:
            return name
    return f"api_{api}"


def _resolve_backend_token(token: str) -> int | None:
    key = str(token).strip().lower()
    if key not in _BACKEND_ALIASES:
        supported = ", ".join(sorted(k for k in _BACKEND_ALIASES if k != "auto"))
        raise ValueError(f"Unsupported camera backend '{token}'. Supported: {supported}")
    api = _BACKEND_ALIASES[key]
    if api is None and key not in {"default", "auto"}:
        raise ValueError(f"Camera backend '{token}' is unavailable in this OpenCV build")
    return api


def _resolve_backend_chain(
    backend: str | None,
    fallback_backends: list[str] | None,
) -> list[int | None]:
    """Build ordered OpenCV backend API list to try when opening the camera."""
    if fallback_backends:
        chain = [_resolve_backend_token(name) for name in fallback_backends]
    elif backend and str(backend).lower() not in {"", "auto"}:
        chain = [_resolve_backend_token(str(backend))]
    else:
        chain = [_resolve_backend_token(name) for name in _default_fallback_backends()]

    deduped: list[int | None] = []
    seen: set[int | None] = set()
    for api in chain:
        if api in seen:
            continue
        seen.add(api)
        deduped.append(api)
    return deduped or [None]


def _create_capture(camera_index: int, api: int | None) -> cv2.VideoCapture:
    if api is None:
        return cv2.VideoCapture(camera_index)
    return cv2.VideoCapture(camera_index, api)


def _apply_resolution(cap: cv2.VideoCapture, frame_width: int | None, frame_height: int | None) -> None:
    if frame_width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(frame_width))
    if frame_height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(frame_height))


def get_active_backend() -> str | None:
    """Return the backend label used for the active capture device."""
    return _active_backend


def open_camera(
    camera_index: int,
    frame_width: int | None = None,
    frame_height: int | None = None,
    *,
    backend: str | None = "auto",
    fallback_backends: list[str] | None = None,
) -> cv2.VideoCapture:
    """Open webcam with OpenCV, trying configured backends until one succeeds."""
    global _cap, _active_backend, _open_params
    close_camera()

    backend_chain = _resolve_backend_chain(backend, fallback_backends)
    failures: list[str] = []

    for api in backend_chain:
        label = _backend_label(api)
        cap = _create_capture(camera_index, api)
        if not cap.isOpened():
            failures.append(f"{label}: failed to open")
            cap.release()
            continue

        _apply_resolution(cap, frame_width, frame_height)
        ok, frame = cap.read()
        if not ok or frame is None:
            failures.append(f"{label}: opened but failed initial frame read")
            cap.release()
            continue

        _cap = cap
        _active_backend = label
        _open_params = {
            "camera_index": camera_index,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "backend": backend,
            "fallback_backends": fallback_backends,
        }
        props = get_camera_properties()
        logger.info(
            "Opened camera index %s via %s backend (%sx%s @ %.1f fps)",
            camera_index,
            label,
            props.get("width", "?"),
            props.get("height", "?"),
            props.get("fps", 0.0),
        )
        if failures:
            logger.info("Camera backend fallback attempts before success: %s", "; ".join(failures))
        return cap

    detail = "; ".join(failures) if failures else "no backends attempted"
    raise RuntimeError(
        f"Failed to open camera at index {camera_index} on {platform.system()}. Attempts: {detail}"
    )


def reopen_camera() -> cv2.VideoCapture:
    """Reopen the camera using the most recent open parameters."""
    if not _open_params:
        raise RuntimeError("Camera reopen requested before open_camera() was called")
    return open_camera(
        int(_open_params["camera_index"]),
        _open_params.get("frame_width"),
        _open_params.get("frame_height"),
        backend=_open_params.get("backend"),
        fallback_backends=_open_params.get("fallback_backends"),
    )


def read_frame() -> tuple[bool, np.ndarray | None]:
    """Read one frame from the active camera stream."""
    if _cap is None or not _cap.isOpened():
        raise RuntimeError("Camera is not open. Call open_camera() first.")

    ok, frame = _cap.read()
    if not ok:
        return False, None
    return True, frame


def get_camera_properties() -> dict[str, Any]:
    """Return width/height/fps/backend for the active capture device."""
    if _cap is None or not _cap.isOpened():
        return {}
    return {
        "width": int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(_cap.get(cv2.CAP_PROP_FPS)),
        "backend": _active_backend,
    }


def close_camera() -> None:
    """Release the active camera resource."""
    global _cap, _active_backend
    if _cap is not None:
        _cap.release()
        _cap = None
    _active_backend = None
