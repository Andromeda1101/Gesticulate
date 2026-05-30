"""HOG vector layout helpers (no OpenCV dependency)."""

from __future__ import annotations

from typing import Any

DEFAULT_CROP_SIZE = (64, 64)


def hog_block_grid_shape(
    crop_size: tuple[int, int],
    hog_config: dict[str, Any],
) -> tuple[int, int, int]:
    """
    Spatial layout of a flattened HOG vector: (block_rows, block_cols, features_per_block).

    Matches scikit-image ``feature_vector=True`` block raster order.
    """
    hog_cfg = hog_config.get("hog", hog_config)
    height, width = int(crop_size[0]), int(crop_size[1])
    ppc = hog_cfg.get("pixels_per_cell", (8, 8))
    cpb = hog_cfg.get("cells_per_block", (2, 2))
    ppc_y, ppc_x = int(ppc[0]), int(ppc[1])
    cpb_y, cpb_x = int(cpb[0]), int(cpb[1])
    orientations = int(hog_cfg.get("orientations", 9))
    n_cells_y = 1 + (height - ppc_y) // ppc_y
    n_cells_x = 1 + (width - ppc_x) // ppc_x
    block_rows = n_cells_y - cpb_y + 1
    block_cols = n_cells_x - cpb_x + 1
    features_per_block = cpb_y * cpb_x * orientations
    return (block_rows, block_cols, features_per_block)


def hog_descriptor_dim_from_grid(grid: tuple[int, int, int]) -> int:
    br, bc, bpf = grid
    return br * bc * bpf
