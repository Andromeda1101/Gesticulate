"""One-vs-rest SVM with Platt SMO (custom implementation, no scikit-learn solver)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, TypeVar

import numpy as np

from src.models.classical.base import BaseClassifier

_DEFAULT_ROW_BATCH_SIZE = 256
_PROGRESS_MIN_ROWS = 512
_DEFAULT_TOL = 1e-3
_DEFAULT_MAX_PASSES = 100
_LARGE_TRAIN_THRESHOLD = 8000
_LARGE_TRAIN_SAMPLES_PER_PASS = 4096

_T = TypeVar("_T")


def _progress_iter(
    iterable: Iterable[_T],
    *,
    disable: bool = False,
    **kwargs: Any,
) -> Iterator[_T] | Iterable[_T]:
    if disable:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


def _squared_euclidean_distances(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    row_batch_size: int = _DEFAULT_ROW_BATCH_SIZE,
    progress: bool = False,
    desc: str | None = None,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    y_norm_sq = np.sum(Y * Y, axis=1)
    n_rows = X.shape[0]
    out = np.empty((n_rows, Y.shape[0]), dtype=np.float64)
    batch = max(1, int(row_batch_size))
    n_batches = (n_rows + batch - 1) // batch
    starts = range(0, n_rows, batch)
    if progress and n_batches > 1:
        starts = _progress_iter(
            starts,
            total=n_batches,
            desc=desc or "RBF kernel distances",
            unit="batch",
        )

    for start in starts:
        end = min(start + batch, n_rows)
        chunk = X[start:end]
        chunk_norm_sq = np.sum(chunk * chunk, axis=1)
        dists_sq = chunk_norm_sq[:, None] + y_norm_sq[None, :] - 2.0 * (chunk @ Y.T)
        np.maximum(dists_sq, 0.0, out=dists_sq)
        out[start:end] = dists_sq

    return out


def _rbf_kernel(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: float,
    *,
    row_batch_size: int = _DEFAULT_ROW_BATCH_SIZE,
    progress: bool = False,
    desc: str | None = None,
) -> np.ndarray:
    sq_dist = _squared_euclidean_distances(
        X,
        Y,
        row_batch_size=row_batch_size,
        progress=progress,
        desc=desc,
    )
    return np.exp(-gamma * sq_dist)


def _linear_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64) @ np.asarray(Y, dtype=np.float64).T


def _gram_matrix(
    X: np.ndarray,
    *,
    kernel: str,
    gamma: float,
    row_batch_size: int,
    progress: bool,
    desc: str,
) -> np.ndarray:
    if kernel == "linear":
        return _linear_kernel(X, X)
    return _rbf_kernel(
        X,
        X,
        gamma,
        row_batch_size=row_batch_size,
        progress=progress,
        desc=desc,
    )


def _kernel_matrix(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    kernel: str,
    gamma: float,
    row_batch_size: int,
    progress: bool,
    desc: str,
) -> np.ndarray:
    if kernel == "linear":
        return _linear_kernel(X, Y)
    return _rbf_kernel(
        X,
        Y,
        gamma,
        row_batch_size=row_batch_size,
        progress=progress,
        desc=desc,
    )


@dataclass
class _BinarySMOModel:
    """Dual coefficients for f(x) = K(x, X_train) @ dual_coef + bias."""

    dual_coef: np.ndarray
    bias: float


def _smo(
    K: np.ndarray,
    y: np.ndarray,
    C: float,
    *,
    tol: float = _DEFAULT_TOL,
    max_passes: int = _DEFAULT_MAX_PASSES,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """
    Platt SMO with labels y in {-1, +1} and Gram matrix K.

    Decision function: f(x_i) = sum_j alpha_j y_j K_ij + b; cache errors E_i = f(x_i) - y_i.
    """
    n = y.shape[0]
    alpha = np.zeros(n, dtype=np.float64)
    b = 0.0
    passes = 0

    def dual_coef() -> np.ndarray:
        return alpha * y

    def errors_vec() -> np.ndarray:
        return K @ dual_coef() + b - y

    while passes < max_passes:
        num_changed = 0
        errors = errors_vec()
        if n > _LARGE_TRAIN_THRESHOLD:
            scan = rng.permutation(n)[:_LARGE_TRAIN_SAMPLES_PER_PASS]
        else:
            scan = np.arange(n)

        for i in scan:
            e_i = float(errors[i])
            violates = (y[i] * e_i < -tol and alpha[i] < C) or (y[i] * e_i > tol and alpha[i] > 0)
            if not violates:
                continue

            j_candidates = np.concatenate([np.arange(0, i), np.arange(i + 1, n)])
            if j_candidates.size == 0:
                continue
            j = int(j_candidates[np.argmax(np.abs(errors[i] - errors[j_candidates]))])

            e_j = float(errors[j])
            alpha_i_old = alpha[i]
            alpha_j_old = alpha[j]

            if y[i] != y[j]:
                lower = max(0.0, alpha_j_old - alpha_i_old)
                upper = min(C, C + alpha_j_old - alpha_i_old)
            else:
                lower = max(0.0, alpha_i_old + alpha_j_old - C)
                upper = min(C, alpha_i_old + alpha_j_old)

            if lower >= upper:
                continue

            k_ii = K[i, i]
            k_ij = K[i, j]
            k_jj = K[j, j]
            eta = 2.0 * k_ij - k_ii - k_jj
            if eta >= -1e-12:
                continue

            alpha_j_new = alpha_j_old - y[j] * (e_i - e_j) / eta
            alpha_j_new = float(np.clip(alpha_j_new, lower, upper))
            if abs(alpha_j_new - alpha_j_old) < 1e-5:
                continue

            alpha_i_new = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha_j_new)

            delta_i = alpha_i_new - alpha_i_old
            delta_j = alpha_j_new - alpha_j_old

            b1 = b - e_i - y[i] * delta_i * k_ii - y[j] * delta_j * k_ij
            b2 = b - e_j - y[i] * delta_i * k_ij - y[j] * delta_j * k_jj
            b_old = b
            if 0.0 < alpha_i_new < C:
                b = b1
            elif 0.0 < alpha_j_new < C:
                b = b2
            else:
                b = 0.5 * (b1 + b2)

            alpha[i] = alpha_i_new
            alpha[j] = alpha_j_new
            errors += y[i] * delta_i * K[:, i] + y[j] * delta_j * K[:, j] + (b - b_old)
            num_changed += 1

        if num_changed == 0:
            passes += 1
        else:
            passes = 0

    return alpha, b


class SVMClassifier(BaseClassifier):
    """
    Multiclass SVM (one-vs-rest) trained with custom Platt SMO on the kernel matrix.

    No scikit-learn solver is used. RBF and linear kernels are supported.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        max_iter: int = _DEFAULT_MAX_PASSES,
        tol: float = _DEFAULT_TOL,
        row_batch_size: int = _DEFAULT_ROW_BATCH_SIZE,
        show_progress: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.C = float(C)
        self.kernel = str(kernel).lower()
        self.gamma = gamma
        self.max_passes = int(max_iter) if int(max_iter) > 0 else _DEFAULT_MAX_PASSES
        self.tol = float(tol)
        self.row_batch_size = max(1, int(row_batch_size))
        self.show_progress = bool(show_progress)
        self._X_train: np.ndarray | None = None
        self._gamma_val: float = 1.0
        self._K_train: np.ndarray | None = None
        self._binary_models: list[_BinarySMOModel] = []
        self._rng = np.random.default_rng(self.random_state)

    def _resolve_gamma(self, X: np.ndarray) -> float:
        if isinstance(self.gamma, (int, float)):
            return float(self.gamma)
        if self.gamma == "scale":
            return float(1.0 / (X.shape[1] * np.var(X)))
        if self.gamma == "auto":
            return float(1.0 / X.shape[1])
        raise ValueError(f"Unknown gamma: {self.gamma}")

    def _use_kernel_progress(self, n_rows: int) -> bool:
        return self.show_progress and self.kernel == "rbf" and n_rows >= _PROGRESS_MIN_ROWS

    def fit(self, X: np.ndarray, y: np.ndarray) -> SVMClassifier:
        X = np.asarray(X, dtype=np.float64)
        y_enc, _ = self._encode_labels(y)
        self._X_train = X
        self._gamma_val = self._resolve_gamma(X)

        if self.show_progress:
            print(
                f"SVM SMO ({self.kernel}): {X.shape[0]} samples, {X.shape[1]} features, "
                f"{len(self.classes_)} classes"
            )

        self._K_train = _gram_matrix(
            X,
            kernel=self.kernel,
            gamma=self._gamma_val,
            row_batch_size=self.row_batch_size,
            progress=self._use_kernel_progress(X.shape[0]),
            desc=f"SVM Gram ({self.kernel})",
        )

        n_classes = len(self.classes_)  # type: ignore[arg-type]
        self._binary_models = []
        for c in _progress_iter(
            range(n_classes),
            disable=not self.show_progress,
            desc="SVM SMO one-vs-rest",
            unit="class",
        ):
            y_binary = np.where(y_enc == c, 1.0, -1.0)
            alpha, b = _smo(
                self._K_train,
                y_binary,
                self.C,
                tol=self.tol,
                max_passes=self.max_passes,
                rng=self._rng,
            )
            dual_coef = alpha * y_binary
            self._binary_models.append(_BinarySMOModel(dual_coef=dual_coef, bias=b))

        self._K_train = None
        self._is_fitted = True
        return self

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        assert self._X_train is not None
        X = np.asarray(X, dtype=np.float64)
        K_cross = _kernel_matrix(
            X,
            self._X_train,
            kernel=self.kernel,
            gamma=self._gamma_val,
            row_batch_size=self.row_batch_size,
            progress=self._use_kernel_progress(X.shape[0]),
            desc=f"SVM kernel ({self.kernel}, predict)",
        )
        scores = np.column_stack(
            [K_cross @ model.dual_coef + model.bias for model in self._binary_models]
        )
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self._decision_function(X)
        return self._decode_labels(np.argmax(scores, axis=1))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self._decision_function(X)
        exp_s = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_s / exp_s.sum(axis=1, keepdims=True)
