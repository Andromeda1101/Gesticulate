"""Unified classifier interface for custom classical models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseClassifier(ABC):
    """Common fit/predict API used by trainers and the model registry."""

    def __init__(self, random_state: int = 42, **hyperparameters: Any) -> None:
        self.random_state = int(random_state)
        self.hyperparameters = dict(hyperparameters)
        self.classes_: np.ndarray | None = None
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseClassifier:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement predict_proba"
        )

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before predict")

    def _encode_labels(self, y: np.ndarray) -> tuple[np.ndarray, dict[Any, int]]:
        classes = np.unique(y)
        self.classes_ = classes
        label_to_idx = {label: idx for idx, label in enumerate(classes)}
        encoded = np.array([label_to_idx[label] for label in y], dtype=np.int64)
        return encoded, label_to_idx

    def _decode_labels(self, indices: np.ndarray) -> np.ndarray:
        self._check_fitted()
        assert self.classes_ is not None
        return self.classes_[indices.astype(int)]
