"""MLP baseline on flat feature vectors."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def build_mlp(
    input_dim: int,
    n_classes: int,
    hidden_dims: list[int] | None = None,
    dropout: float = 0.2,
) -> nn.Module:
    if torch is None:
        raise ImportError("PyTorch is required. Install with: pip install torch")
    hidden_dims = hidden_dims or [256, 128]
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    return nn.Sequential(*layers)
