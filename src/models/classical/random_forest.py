"""Random Forest over custom decision trees with bootstrap aggregation."""

from __future__ import annotations

import numpy as np

from src.models.classical.base import BaseClassifier
from src.models.classical.decision_tree import DecisionTreeClassifier


class RandomForestClassifier(BaseClassifier):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 10,
        max_features: str | float = "sqrt",
        min_samples_leaf: int = 1,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = int(min_samples_leaf)
        self.trees_: list[DecisionTreeClassifier] = []
        self._rng = np.random.default_rng(self.random_state)

    def _feature_count(self, n_features: int) -> int:
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                return max(1, int(np.sqrt(n_features)))
            if self.max_features == "log2":
                return max(1, int(np.log2(n_features)))
            raise ValueError(f"Unknown max_features: {self.max_features}")
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        return int(self.max_features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        X = np.asarray(X, dtype=np.float64)
        self._encode_labels(y)
        n_samples, n_features = X.shape
        n_feat_subset = self._feature_count(n_features)
        self.trees_ = []
        y_arr = np.asarray(y)
        for i in range(self.n_estimators):
            rng = np.random.default_rng(self.random_state + i)
            boot_idx = rng.integers(0, n_samples, size=n_samples)
            feat_idx = rng.choice(n_features, size=n_feat_subset, replace=False)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + i,
            )
            tree.fit(X[boot_idx][:, feat_idx], y_arr[boot_idx])
            tree._feature_indices = feat_idx  # type: ignore[attr-defined]
            self.trees_.append(tree)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        n_classes = len(self.classes_)  # type: ignore[arg-type]
        votes = np.zeros((X.shape[0], n_classes), dtype=np.int64)
        for tree in self.trees_:
            feat_idx = tree._feature_indices  # type: ignore[attr-defined]
            preds = tree.predict(X[:, feat_idx])
            for i, label in enumerate(preds):
                class_idx = int(np.where(self.classes_ == label)[0][0])
                votes[i, class_idx] += 1
        best = np.argmax(votes, axis=1)
        return self._decode_labels(best)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        n_classes = len(self.classes_)  # type: ignore[arg-type]
        votes = np.zeros((X.shape[0], n_classes), dtype=np.float64)
        for tree in self.trees_:
            feat_idx = tree._feature_indices  # type: ignore[attr-defined]
            preds = tree.predict(X[:, feat_idx])
            for i, label in enumerate(preds):
                class_idx = int(np.where(self.classes_ == label)[0][0])
                votes[i, class_idx] += 1
        row_sums = votes.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return votes / row_sums
