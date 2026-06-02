"""Helpers for EXP-03 batch robustness (feature × model) suites."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from src.common.path_manager import resolve_project_root
from src.models.feature_resolver import (
    normalize_algorithm_name,
    resolve_feature_matrix_path,
)


@dataclass(frozen=True)
class RobustnessRunSpec:
    """One feature-family × algorithm OOD evaluation."""

    algorithm: str
    feature_family: str
    model_artifact: Path
    in_domain_features: Path
    ood_features: Path


def _resolve_model_artifact(
    root: Path,
    *,
    train_experiment_id: str,
    algorithm: str,
    feature_family: str,
    models_dir: Path | None = None,
) -> Path:
    base_dir = models_dir or (root / "artifacts" / "models")
    stem = f"{train_experiment_id}_{algorithm}_{feature_family}"
    for ext in (".joblib", ".pt"):
        candidate = base_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return base_dir / f"{stem}.joblib"


def iter_robustness_suite_specs(
    experiment_config: dict[str, Any],
    *,
    project_root: Path | str | None = None,
    algorithms: list[str] | None = None,
    feature_families: list[str] | None = None,
) -> Iterator[RobustnessRunSpec]:
    """
    Yield run specs from ``robustness_suite`` in the experiment YAML.

    Uses explicit ``pairs`` when present; otherwise the Cartesian product of
    ``feature_families`` × ``algorithms``.
    """
    root = Path(project_root) if project_root else resolve_project_root()
    suite_cfg = experiment_config.get("robustness_suite") or {}
    if not suite_cfg:
        raise ValueError(
            "Experiment config missing 'robustness_suite'. "
            "Add robustness_suite to exp03_robustness.yaml or pass explicit CLI overrides."
        )

    dataset_name = suite_cfg.get("dataset_name", "hagrid_subset")
    ood_dataset_name = suite_cfg.get("ood_dataset_name", "leapgestrecog")
    feature_version = str(
        suite_cfg.get("feature_version")
        or experiment_config.get("features", {}).get("feature_version", "v1")
    )
    train_experiment_id = suite_cfg.get(
        "train_experiment_id",
        experiment_config.get("models", {}).get("train_experiment_id", "EXP-01"),
    )
    models_dir_rel = suite_cfg.get("models_dir") or experiment_config.get("outputs", {}).get(
        "models_dir", "artifacts/models"
    )
    models_dir = root / models_dir_rel

    default_algorithms = experiment_config.get("models", {}).get("algorithms", [])
    default_families = suite_cfg.get("feature_families") or [
        experiment_config.get("features", {}).get("feature_family", "hybrid")
    ]
    algo_list = algorithms or suite_cfg.get("algorithms") or default_algorithms
    family_list = feature_families or default_families

    pairs: list[dict[str, str]] = list(suite_cfg.get("pairs") or [])
    if not pairs:
        for family in family_list:
            for algo in algo_list:
                pairs.append({"algorithm": algo, "feature_family": family})

    seen: set[tuple[str, str]] = set()
    for pair in pairs:
        algo = normalize_algorithm_name(str(pair["algorithm"]))
        family = str(pair["feature_family"])
        key = (algo, family)
        if key in seen:
            continue
        seen.add(key)

        model_path = _resolve_model_artifact(
            root,
            train_experiment_id=str(train_experiment_id),
            algorithm=algo,
            feature_family=family,
            models_dir=models_dir,
        )
        in_path = resolve_feature_matrix_path(
            family,
            dataset_name=dataset_name,
            feature_version=feature_version,
            project_root=root,
        )
        ood_path = resolve_feature_matrix_path(
            family,
            dataset_name=ood_dataset_name,
            feature_version=feature_version,
            project_root=root,
        )
        yield RobustnessRunSpec(
            algorithm=algo,
            feature_family=family,
            model_artifact=model_path,
            in_domain_features=in_path,
            ood_features=ood_path,
        )


def build_robustness_suite_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate EXP-03 run records into a comparison table."""
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("status") != "completed":
            continue
        metrics = record.get("metrics") or {}
        in_domain = metrics.get("in_domain") or {}
        ood = metrics.get("ood") or {}
        robustness = record.get("robustness") or metrics.get("robustness") or {}
        protocols = record.get("ood_eval_protocols") or metrics.get("ood_eval_protocols") or {}
        shared = (protocols.get("shared_subset") or {}).get("ood") or {}

        rows.append(
            {
                "run_id": record.get("run_id"),
                "algorithm": record.get("algorithm"),
                "feature_family": record.get("feature_family"),
                "in_domain_accuracy": in_domain.get("accuracy"),
                "ood_accuracy": ood.get("accuracy"),
                "absolute_accuracy_drop": robustness.get("absolute_accuracy_drop"),
                "relative_performance_retention": robustness.get(
                    "relative_performance_retention"
                ),
                "ood_shared_subset_accuracy": shared.get("accuracy"),
                "metrics_path": record.get("outputs", {}).get("metrics_path"),
            }
        )

    import pandas as pd

    df = pd.DataFrame(rows)
    if not df.empty and "ood_accuracy" in df.columns:
        df = df.sort_values(
            by=["ood_accuracy", "in_domain_accuracy"],
            ascending=[False, False],
            na_position="last",
        )
    return {"n_runs": len(rows), "leaderboard": df.to_dict(orient="records"), "dataframe": df}
