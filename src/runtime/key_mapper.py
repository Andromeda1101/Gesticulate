"""Map gesture labels to OS keyboard actions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.common.config_loader import load_config

try:
    from pynput.keyboard import Controller, Key
except ImportError:  # pragma: no cover
    Controller = None  # type: ignore[assignment,misc]
    Key = None  # type: ignore[assignment,misc]

_keyboard: Any | None = None

# HaGRID-native gesture labels expected from trained classifiers.
DEFAULT_KEYMAP: dict[str, str] = {
    "palm": "space",
    "fist": "enter",
    "like": "up",
    "peace": "down",
}

# Runtime aliases before keymap lookup (legacy / alternate model outputs).
_RUNTIME_GESTURE_ALIASES: dict[str, str] = {
    "thumb": "like",
    "thumb_up": "like",
    "thumbup": "like",
}


def _normalize_gesture_key(gesture_label: str) -> str:
    return gesture_label.strip().lower().replace("-", "_").replace(" ", "_")


def normalize_runtime_gesture_label(gesture_label: str) -> str:
    """Map model output labels to HaGRID-native names used by the runtime keymap."""
    key = _normalize_gesture_key(gesture_label)
    return _RUNTIME_GESTURE_ALIASES.get(key, key)

_NAMED_KEYS: dict[str, Any] = {}


def _ensure_keyboard() -> Any:
    global _keyboard, _NAMED_KEYS
    if Controller is None:
        raise ImportError("pynput is required for live keyboard dispatch")
    if _keyboard is None:
        _keyboard = Controller()
        _NAMED_KEYS = {
            "space": Key.space,
            "enter": Key.enter,
            "up": Key.up,
            "down": Key.down,
            "left": Key.left,
            "right": Key.right,
            "tab": Key.tab,
            "esc": Key.esc,
        }
    return _keyboard


def load_keymap(config_path: str | None = None, runtime_config: dict[str, Any] | None = None) -> dict[str, str]:
    """Load gesture-to-key mapping from runtime config or YAML path."""
    if runtime_config is not None:
        mapping = runtime_config.get("gesture_mapping")
        if isinstance(mapping, dict):
            return {str(k): str(v) for k, v in mapping.items()}

    if config_path:
        config = load_config(config_path)
        mapping = config.get("gesture_mapping", {})
        return {str(k): str(v) for k, v in mapping.items()}

    return dict(DEFAULT_KEYMAP)


def _resolve_key(key_name: str) -> Any:
    lower = key_name.lower()
    if lower in _NAMED_KEYS or (Key is not None and hasattr(Key, lower)):
        _ensure_keyboard()
        return _NAMED_KEYS.get(lower, getattr(Key, lower, lower))
    return key_name


def dispatch_key_action(
    gesture_label: str,
    keymap: dict[str, str],
    *,
    dry_run: bool = True,
    enable_dispatch: bool = False,
) -> dict[str, Any]:
    """
    Dispatch a keyboard action for *gesture_label*.

    Real OS events require ``enable_dispatch=True`` and ``dry_run=False``.
    """
    normalized_label = normalize_runtime_gesture_label(gesture_label)
    mapped_key = keymap.get(normalized_label) or keymap.get(gesture_label)
    if mapped_key is None:
        return {
            "emitted": False,
            "reason": "unmapped_gesture",
            "gesture_label": gesture_label,
            "normalized_gesture_label": normalized_label,
            "mapped_key": None,
            "dry_run": dry_run,
        }

    if dry_run or not enable_dispatch:
        return {
            "emitted": False,
            "reason": "dry_run" if dry_run else "dispatch_disabled",
            "gesture_label": gesture_label,
            "normalized_gesture_label": normalized_label,
            "mapped_key": mapped_key,
            "dry_run": True,
            "would_emit": True,
        }

    keyboard = _ensure_keyboard()
    resolved = _resolve_key(mapped_key)
    if isinstance(resolved, str) and len(resolved) == 1:
        keyboard.press(resolved)
        keyboard.release(resolved)
    else:
        keyboard.press(resolved)
        keyboard.release(resolved)

    return {
        "emitted": True,
        "reason": "dispatched",
        "gesture_label": gesture_label,
        "normalized_gesture_label": normalized_label,
        "mapped_key": mapped_key,
        "dry_run": False,
    }


def load_keymap_from_json(path: str | Path) -> dict[str, str]:
    """Load a standalone JSON keymap file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Keymap JSON must be an object")
    return {str(k): str(v) for k, v in data.items()}
