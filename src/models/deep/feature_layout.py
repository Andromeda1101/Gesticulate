"""Map flat feature vectors to tensor layouts for CNN/LSTM baselines."""

from __future__ import annotations

from typing import Any, Literal

from src.common.config_loader import load_config
from src.features.geometric_features import GEOMETRIC_VECTOR_DIM
from src.features.hog_layout import DEFAULT_CROP_SIZE, hog_block_grid_shape, hog_descriptor_dim_from_grid

LayoutKind = Literal["flat", "hog", "hybrid"]


def default_hog_config() -> dict[str, Any]:
    cfg = load_config("configs/features/default.yaml")
    return dict(cfg["feature_families"]["hog_only"]["hog"])


def resolve_feature_layout(
    input_dim: int,
    records: list[dict[str, Any]],
    *,
    hog_config: dict[str, Any] | None = None,
) -> tuple[LayoutKind, int, tuple[int, int, int] | None]:
    """
    Infer how to interpret *input_dim* for deep models.

    Returns:
        kind: ``hybrid``, ``hog``, or ``flat`` (legacy sqrt reshape).
        geom_dim: leading geometric sub-vector length (0 if none).
        hog_grid: ``(block_rows, block_cols, features_per_block)`` or None.
    """
    hog_cfg = hog_config or default_hog_config()
    crop_size = tuple(hog_cfg.get("crop_size", DEFAULT_CROP_SIZE))
    grid = hog_block_grid_shape(crop_size, hog_cfg)
    br, bc, bpf = grid
    hog_dim = hog_descriptor_dim_from_grid(grid)
    hybrid_dim = GEOMETRIC_VECTOR_DIM + hog_dim

    family = str(records[0].get("feature_family", "")) if records else ""

    if input_dim == hybrid_dim or "hybrid" in family:
        return "hybrid", GEOMETRIC_VECTOR_DIM, grid
    if input_dim == hog_dim or family == "hog":
        return "hog", 0, grid
    return "flat", 0, None
