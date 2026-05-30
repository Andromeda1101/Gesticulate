"""Gaussian Naive Bayes with log-probability inference."""

from __future__ import annotations

import numpy as np

from src.models.classical.base import BaseClassifier


class GaussianNBClassifier(BaseClassifier):
    def __init__(self, var_smoothing: float = 1e-9, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.var_smoothing = float(var_smoothing)
        self.class_count_: np.ndarray | None = None
        self.class_log_prior_: np.ndarray | None = None
        self.theta_: np.ndarray | None = None
        self.var_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianNBClassifier:
        X = np.asarray(X, dtype=np.float64)
        y_enc, _ = self._encode_labels(y)
        n_classes = len(self.classes_)  # type: ignore[arg-type]
        n_features = X.shape[1]
        self.class_count_ = np.zeros(n_classes, dtype=np.int64)
        self.theta_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var_ = np.zeros((n_classes, n_features), dtype=np.float64)

        for c in range(n_classes):
            mask = y_enc == c
            X_c = X[mask]
            self.class_count_[c] = X_c.shape[0]
            self.theta_[c] = X_c.mean(axis=0)
            self.var_[c] = X_c.var(axis=0) + self.var_smoothing

        self.class_log_prior_ = np.log(self.class_count_ / self.class_count_.sum())
        self._is_fitted = True
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        assert self.theta_ is not None and self.var_ is not None
        assert self.class_log_prior_ is not None
        n_classes = self.theta_.shape[0]
        log_probs = np.zeros((X.shape[0], n_classes), dtype=np.float64)
        for c in range(n_classes):
            diff = X - self.theta_[c]
            log_probs[:, c] = (
                -0.5 * np.sum(np.log(2.0 * np.pi * self.var_[c]) + (diff * diff) / self.var_[c], axis=1)
                + self.class_log_prior_[c]
            )
        return log_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_probs = self._joint_log_likelihood(np.asarray(X, dtype=np.float64))
        return self._decode_labels(np.argmax(log_probs, axis=1))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_probs = self._joint_log_likelihood(np.asarray(X, dtype=np.float64))
        log_probs -= np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs)
        return probs / probs.sum(axis=1, keepdims=True)
