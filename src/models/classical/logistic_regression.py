"""Multiclass logistic regression via one-vs-rest gradient descent."""

from __future__ import annotations

import numpy as np

from src.models.classical.base import BaseClassifier


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegressionClassifier(BaseClassifier):
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 500,
        learning_rate: float = 0.1,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.learning_rate = float(learning_rate)
        self.weights_: np.ndarray | None = None
        self.bias_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegressionClassifier:
        X = np.asarray(X, dtype=np.float64)
        y_enc, _ = self._encode_labels(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)  # type: ignore[arg-type]
        reg = 1.0 / max(self.C, 1e-8)

        W = np.zeros((n_classes, n_features), dtype=np.float64)
        b = np.zeros(n_classes, dtype=np.float64)
        Y = np.zeros((n_samples, n_classes), dtype=np.float64)
        Y[np.arange(n_samples), y_enc] = 1.0

        lr = self.learning_rate
        for _ in range(self.max_iter):
            logits = X @ W.T + b
            probs = _sigmoid(logits)
            error = probs - Y
            grad_w = (error.T @ X) / n_samples + reg * W
            grad_b = error.mean(axis=0)
            W -= lr * grad_w
            b -= lr * grad_b

        self.weights_ = W
        self.bias_ = b
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        assert self.weights_ is not None and self.bias_ is not None
        X = np.asarray(X, dtype=np.float64)
        logits = X @ self.weights_.T + self.bias_
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self._decode_labels(np.argmax(proba, axis=1))
