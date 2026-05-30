"""Deep models should respect HOG block structure for hybrid vectors."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.geometric_features import GEOMETRIC_VECTOR_DIM
from src.models.deep.feature_layout import default_hog_config, resolve_feature_layout
from src.models.deep.cnn import HybridCNNClassifier, build_cnn
from src.models.deep.lstm import HybridLSTMClassifier, build_lstm

pytest.importorskip("torch")
import torch


def test_hybrid_vector_dim_matches_project_default() -> None:
    from src.features.hog_layout import hog_block_grid_shape, hog_descriptor_dim_from_grid

    hog_cfg = default_hog_config()
    crop = tuple(hog_cfg["crop_size"])
    grid = hog_block_grid_shape(crop, hog_cfg)
    dim = hog_descriptor_dim_from_grid(grid)
    assert dim + GEOMETRIC_VECTOR_DIM == 2024


def test_resolve_hybrid_layout() -> None:
    records = [{"feature_family": "hybrid_keypoints_hog"}]
    kind, geom, grid = resolve_feature_layout(2024, records)
    assert kind == "hybrid"
    assert geom == GEOMETRIC_VECTOR_DIM
    assert grid == (7, 7, 36)


def test_hybrid_cnn_forward_shape() -> None:
    grid = (7, 7, 36)
    model = HybridCNNClassifier(GEOMETRIC_VECTOR_DIM, grid, n_classes=5)
    x = torch.randn(4, GEOMETRIC_VECTOR_DIM + 7 * 7 * 36)
    logits = model(x)
    assert logits.shape == (4, 5)


def test_build_cnn_hybrid_returns_no_dataset_reshape() -> None:
    model, reshape = build_cnn(
        2024,
        10,
        geom_dim=GEOMETRIC_VECTOR_DIM,
        hog_grid=(7, 7, 36),
        layout="hybrid",
    )
    assert isinstance(model, HybridCNNClassifier)
    assert reshape is None
