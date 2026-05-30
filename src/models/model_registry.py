"""Algorithm registry and model factory for Phase 3 benchmarks."""

from __future__ import annotations

from typing import Any, Callable

from src.models.classical import (
    DecisionTreeClassifier,
    GaussianNBClassifier,
    KNNClassifier,
    LogisticRegressionClassifier,
    RandomForestClassifier,
    SVMClassifier,
)
from src.models.classical.base import BaseClassifier
from src.models.feature_resolver import normalize_algorithm_name  # re-exported

__all__ = [
    "ALL_ALGORITHMS",
    "CLASSICAL_ALGORITHMS",
    "DEEP_ALGORITHMS",
    "SCALE_ALGORITHMS",
    "build_model",
    "get_model_builder",
    "is_deep_algorithm",
    "list_supported_algorithms",
    "normalize_algorithm_name",
]

ModelBuilder = Callable[..., Any]

CLASSICAL_ALGORITHMS: frozenset[str] = frozenset(
    {
        "knn",
        "svm",
        "decision_tree",
        "random_forest",
        "naive_bayes",
        "logistic_regression",
    }
)

DEEP_ALGORITHMS: frozenset[str] = frozenset({"mlp", "cnn", "lstm"})

ALL_ALGORITHMS: frozenset[str] = CLASSICAL_ALGORITHMS | DEEP_ALGORITHMS

# Algorithms that benefit from feature scaling before training
SCALE_ALGORITHMS: frozenset[str] = frozenset(
    {"knn", "svm", "logistic_regression", "mlp", "cnn", "lstm"}
)


def list_supported_algorithms() -> list[str]:
    return sorted(ALL_ALGORITHMS)


def is_deep_algorithm(name: str) -> bool:
    return normalize_algorithm_name(name) in DEEP_ALGORITHMS


def get_model_builder(algorithm_name: str) -> ModelBuilder:
    """Return a callable that constructs a model from hyperparameters."""
    name = normalize_algorithm_name(algorithm_name)
    if name not in ALL_ALGORITHMS:
        raise KeyError(
            f"Unknown algorithm '{algorithm_name}'. "
            f"Supported: {', '.join(list_supported_algorithms())}"
        )

    builders: dict[str, ModelBuilder] = {
        "knn": lambda **hp: KNNClassifier(**hp),
        "svm": lambda **hp: SVMClassifier(**hp),
        "decision_tree": lambda **hp: DecisionTreeClassifier(**hp),
        "random_forest": lambda **hp: RandomForestClassifier(**hp),
        "naive_bayes": lambda **hp: GaussianNBClassifier(**hp),
        "logistic_regression": lambda **hp: LogisticRegressionClassifier(**hp),
        "mlp": _build_deep_placeholder,
        "cnn": _build_deep_placeholder,
        "lstm": _build_deep_placeholder,
    }
    return builders[name]


def build_model(algorithm_name: str, hyperparameters: dict[str, Any] | None = None) -> Any:
    """Instantiate a model for *algorithm_name*."""
    name = normalize_algorithm_name(algorithm_name)
    hp = dict(hyperparameters or {})
    if name in DEEP_ALGORITHMS:
        return {"algorithm": name, "hyperparameters": hp}
    builder = get_model_builder(name)
    model: BaseClassifier = builder(**hp)
    return model


def _build_deep_placeholder(**hp: Any) -> dict[str, Any]:
    raise RuntimeError("Deep models are built via deep_baseline_trainer, not classical builder")
