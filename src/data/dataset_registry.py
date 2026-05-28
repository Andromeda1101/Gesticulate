"""Register dataset-specific adapters under a shared API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

from src.data.adapters import hagrid_adapter, leapgestrecog_adapter

_SUPPORTED = ("leapgestrecog", "hagrid_subset")


class DatasetAdapter(Protocol):
    def index_samples(self, root_dir: str, **kwargs: Any) -> list[dict[str, Any]]: ...


def list_supported_datasets() -> list[str]:
    return list(_SUPPORTED)


def _validate_config_availability(config: dict[str, Any]) -> None:
    root = config.get("root_path")
    if not root:
        raise ValueError("Dataset config missing 'root_path'")
    root_path = Path(root)
    if not root_path.is_absolute():
        from src.common.path_manager import resolve_project_root

        root_path = resolve_project_root() / root_path
    if not root_path.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {root_path}. "
            "Place raw data under data/raw/ before indexing."
        )


def get_dataset_adapter(dataset_name: str) -> Callable[..., list[dict[str, Any]]]:
    """Return the indexing callable for *dataset_name*."""
    name = dataset_name.strip().lower()
    if name == "leapgestrecog":
        return leapgestrecog_adapter.index_samples
    if name == "hagrid_subset":
        return hagrid_adapter.index_samples
    raise ValueError(
        f"Unsupported dataset '{dataset_name}'. "
        f"Supported: {', '.join(_SUPPORTED)}"
    )


def index_dataset(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Index samples for a loaded dataset configuration."""
    dataset_name = config["dataset_name"]
    _validate_config_availability(config)

    adapter = get_dataset_adapter(dataset_name)
    root_path = config["root_path"]
    from src.common.path_manager import resolve_project_root

    root = Path(root_path)
    if not root.is_absolute():
        root = resolve_project_root() / root

    if dataset_name == "leapgestrecog":
        return adapter(
            str(root),
            config.get("label_aliases"),
            dataset_name=dataset_name,
            label_vocabulary=config.get("label_vocabulary"),
            capture_context=config.get("capture_context"),
        )

    subset_spec = {
        "label_vocabulary": config.get("label_vocabulary"),
        "target_labels": config.get("label_filter"),
        "label_aliases": config.get("label_aliases"),
        "align_to": config.get("align_to"),
        "sampling": config.get("sampling"),
        "split_strategy": config.get("split_strategy"),
        "capture_context": config.get("capture_context"),
        "format": config.get("layout"),
        "shuffle": (config.get("sampling") or {}).get("shuffle", False),
    }
    if config.get("sampling", {}).get("max_samples_per_class") is not None:
        subset_spec["max_samples_per_class"] = config["sampling"]["max_samples_per_class"]

    return adapter(str(root), subset_spec, dataset_name=dataset_name)
