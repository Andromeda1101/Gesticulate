"""PyTorch datasets for feature vectors."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[misc, assignment]


class FeatureVectorDataset(Dataset):
    """Wrap (X, y) arrays for deep baseline training."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        label_to_idx: dict[Any, int],
        *,
        reshape: tuple[int, ...] | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for deep baselines. Install with: pip install torch")
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.array([label_to_idx[label] for label in y], dtype=np.int64)
        self.reshape = reshape

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        x = self.X[idx]
        if self.reshape is not None:
            x = x.reshape(self.reshape)
        return torch.from_numpy(x.copy()), torch.tensor(self.y[idx], dtype=torch.long)
