"""Hyperparameter search spaces and tuning plan builders."""

from __future__ import annotations

import copy
import itertools
from typing import Any

from src.models.feature_resolver import normalize_algorithm_name

DEFAULT_SEARCH_SPACES: dict[str, dict[str, list[Any]]] = {
    "knn": {"n_neighbors": [3, 5, 7, 11]},
    "svm": {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"], "gamma": ["scale"]},
    "decision_tree": {"max_depth": [5, 10, 20, None], "min_samples_leaf": [1, 5]},
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "max_features": ["sqrt"],
    },
    "naive_bayes": {"var_smoothing": [1e-9, 1e-7]},
    "logistic_regression": {"C": [0.1, 1.0, 10.0], "max_iter": [500]},
    "mlp": {"hidden_dims": [[128, 64], [256, 128]], "dropout": [0.1, 0.2]},
    "cnn": {"hidden_channels": [16, 32]},
    "lstm": {"hidden_size": [64, 128], "num_layers": [1, 2]},
}


def get_search_space(algorithm_name: str) -> dict[str, list[Any]]:
    name = normalize_algorithm_name(algorithm_name)
    if name not in DEFAULT_SEARCH_SPACES:
        raise KeyError(f"No search space defined for algorithm '{algorithm_name}'")
    return copy.deepcopy(DEFAULT_SEARCH_SPACES[name])


def _grid_from_space(space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(space.keys())
    if not keys:
        return [{}]
    values = [space[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def build_search_plan(experiment_config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Emit candidate hyperparameter sets for algorithms listed in the experiment.

    Uses registry entries from ``models.config`` when present; otherwise defaults.
    """
    models_cfg = experiment_config.get("models", {})
    algorithms = models_cfg.get("algorithms", [])
    registry = models_cfg.get("algorithm_registry", {})
    plans: list[dict[str, Any]] = []

    for algo in algorithms:
        name = normalize_algorithm_name(algo)
        entry = registry.get(algo) or registry.get(name) or {}
        hp = dict(entry.get("hyperparameters", {}))
        tuning = entry.get("tuning")
        if tuning and tuning.get("enabled"):
            space = tuning.get("search_space") or get_search_space(name)
            candidates = _grid_from_space(space)
        else:
            candidates = [hp]

        for idx, params in enumerate(candidates):
            plans.append(
                {
                    "algorithm": name,
                    "hyperparameters": params,
                    "plan_index": idx,
                }
            )
    return plans
