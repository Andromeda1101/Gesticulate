"""Load and validate YAML/JSON configuration files."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

# Required top-level keys per config scope (inferred from path segment)
_REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "datasets": ("dataset_name", "root_path"),
    "features": ("feature_version",),
    "models": ("algorithm_registry",),
    "experiments": ("experiment_id",),
    "runtime": ("camera", "gesture_mapping"),
}


def _infer_config_type(config_path: Path) -> str | None:
    parts = config_path.parts
    if "configs" not in parts:
        return None
    idx = parts.index("configs")
    if idx + 1 < len(parts):
        return parts[idx + 1]
    return None


def _load_raw(config_path: Path) -> dict[str, Any]:
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")

    if suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config format: {config_path}")

    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")

    return data


def _validate_config(config: dict[str, Any], config_type: str | None) -> None:
    if config_type is None or config_type not in _REQUIRED_KEYS:
        return

    missing = [k for k in _REQUIRED_KEYS[config_type] if k not in config]
    if missing:
        raise ValueError(
            f"Config type '{config_type}' missing required keys: {missing}"
        )


def _parse_scalar(value: str) -> Any:
    lower = value.lower()
    if lower in ("true", "false"):
        return lower == "true"
    if lower == "null" or lower == "none":
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _set_nested(config: dict[str, Any], key_path: str, value: Any) -> None:
    keys = key_path.split(".")
    current = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def load_config(config_path: str) -> dict[str, Any]:
    """Read, validate, and return a normalized configuration dictionary."""
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    config = _load_raw(path)
    config_type = _infer_config_type(path)
    _validate_config(config, config_type)

    normalized = copy.deepcopy(config)
    normalized["_meta"] = {
        "config_path": str(path),
        "config_type": config_type,
    }
    return normalized


def merge_overrides(base_config: dict[str, Any], cli_args: dict[str, Any]) -> dict[str, Any]:
    """
    Apply CLI overrides to *base_config*.

    *cli_args* values may be plain keys or dot-separated paths (``a.b.c``).
    String values are coerced to bool/int/float when unambiguous.
    """
    merged = copy.deepcopy(base_config)

    for key, raw_value in cli_args.items():
        value = _parse_scalar(raw_value) if isinstance(raw_value, str) else raw_value
        if "." in key:
            _set_nested(merged, key, value)
        else:
            merged[key] = value

    return merged
