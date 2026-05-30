"""K-Nearest Neighbors classifier (NumPy, memory-efficient distances)."""

from __future__ import annotations

import numpy as np

from src.models.classical.base import BaseClassifier

# Cap peak RAM for distance blocks: batch_queries * n_train * 8 bytes.
_DEFAULT_QUERY_BATCH_SIZE = 256


class KNNClassifier(BaseClassifier):
    def __init__(self, n_neighbors: int = 5, query_batch_size: int = _DEFAULT_QUERY_BATCH_SIZE, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.n_neighbors = int(n_neighbors)
        self.query_batch_size = max(1, int(query_batch_size))
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._train_norm_sq: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> KNNClassifier:
        self._X_train = np.asarray(X, dtype=np.float64)
        self._train_norm_sq = np.sum(self._X_train * self._X_train, axis=1)
        y_enc, _ = self._encode_labels(y)
        self._y_train = y_enc
        self._is_fitted = True
        return self

    def _distances(self, X: np.ndarray) -> np.ndarray:
        """
        Pairwise Euclidean distances, shape (n_query, n_train).

        Uses ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x·y with query batching so we never
        materialize a (n_query, n_train, n_features) tensor.
        """
        assert self._X_train is not None and self._train_norm_sq is not None
        X = np.asarray(X, dtype=np.float64)
        train = self._X_train
        n_query, n_train = X.shape[0], train.shape[0]
        out = np.empty((n_query, n_train), dtype=np.float64)
        batch_size = self.query_batch_size

        for start in range(0, n_query, batch_size):
            end = min(start + batch_size, n_query)
            chunk = X[start:end]
            chunk_norm_sq = np.sum(chunk * chunk, axis=1)
            dists_sq = chunk_norm_sq[:, None] + self._train_norm_sq[None, :] - 2.0 * (chunk @ train.T)
            np.maximum(dists_sq, 0.0, out=dists_sq)
            out[start:end] = np.sqrt(dists_sq, dtype=np.float64)

        return out

    def _neighbor_indices(self, X: np.ndarray) -> np.ndarray:
        dists = self._distances(X)
        k = min(self.n_neighbors, dists.shape[1])
        return np.argpartition(dists, kth=k - 1, axis=1)[:, :k]

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        assert self._y_train is not None
        nn_idx = self._neighbor_indices(np.asarray(X, dtype=np.float64))
        votes = self._y_train[nn_idx]
        preds = np.apply_along_axis(
            lambda row: np.bincount(row.astype(int)).argmax(), 1, votes
        )
        return self._decode_labels(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        assert self._y_train is not None and self.classes_ is not None
        n_classes = len(self.classes_)
        nn_idx = self._neighbor_indices(np.asarray(X, dtype=np.float64))
        votes = self._y_train[nn_idx]
        proba = np.zeros((votes.shape[0], n_classes), dtype=np.float64)
        for i in range(votes.shape[0]):
            counts = np.bincount(votes[i].astype(int), minlength=n_classes)
            proba[i] = counts / counts.sum()
        return proba
