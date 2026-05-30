"""CART decision tree with Gini impurity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.models.classical.base import BaseClassifier


@dataclass
class _TreeNode:
    is_leaf: bool
    prediction: int | None = None
    feature: int | None = None
    threshold: float | None = None
    left: _TreeNode | None = None
    right: _TreeNode | None = None


class DecisionTreeClassifier(BaseClassifier):
    def __init__(
        self,
        max_depth: int | None = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.root_: _TreeNode | None = None
        self._rng = np.random.default_rng(self.random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier:
        X = np.asarray(X, dtype=np.float64)
        y_enc, _ = self._encode_labels(y)
        self.root_ = self._build_tree(X, y_enc, depth=0)
        self._is_fitted = True
        return self

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y.astype(int))
        probs = counts[counts > 0] / len(y)
        return float(1.0 - np.sum(probs * probs))

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float] | None:
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None
        parent_gini = self._gini(y)
        best_gain = 0.0
        best_feature = 0
        best_threshold = 0.0
        for feat in range(n_features):
            values = X[:, feat]
            thresholds = np.unique(values)
            if len(thresholds) > 32:
                thresholds = np.quantile(values, np.linspace(0.1, 0.9, 20))
            for thr in thresholds:
                left_mask = values <= thr
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue
                gain = parent_gini - (
                    left_mask.sum() / n_samples * self._gini(y[left_mask])
                    + right_mask.sum() / n_samples * self._gini(y[right_mask])
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = float(thr)
        if best_gain <= 0:
            return None
        return best_feature, best_threshold

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        if len(np.unique(y)) == 1:
            return _TreeNode(is_leaf=True, prediction=int(y[0]))
        if self.max_depth is not None and depth >= self.max_depth:
            majority = int(np.bincount(y.astype(int)).argmax())
            return _TreeNode(is_leaf=True, prediction=majority)
        split = self._best_split(X, y)
        if split is None:
            majority = int(np.bincount(y.astype(int)).argmax())
            return _TreeNode(is_leaf=True, prediction=majority)
        feat, thr = split
        left_mask = X[:, feat] <= thr
        return _TreeNode(
            is_leaf=False,
            feature=feat,
            threshold=thr,
            left=self._build_tree(X[left_mask], y[left_mask], depth + 1),
            right=self._build_tree(X[~left_mask], y[~left_mask], depth + 1),
        )

    def _predict_row(self, node: _TreeNode, row: np.ndarray) -> int:
        if node.is_leaf:
            assert node.prediction is not None
            return node.prediction
        assert node.feature is not None and node.threshold is not None
        if row[node.feature] <= node.threshold:
            assert node.left is not None
            return self._predict_row(node.left, row)
        assert node.right is not None
        return self._predict_row(node.right, row)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        assert self.root_ is not None
        X = np.asarray(X, dtype=np.float64)
        preds = np.array([self._predict_row(self.root_, row) for row in X])
        return self._decode_labels(preds)
