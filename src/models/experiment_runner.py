"""Shared experiment execution logic for Phase 3 scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.common.path_manager import build_metrics_record_path, resolve_project_root
from src.common.run_registry import create_run_record, save_run_record
from src.evaluation.metrics import compute_classification_metrics, compute_efficiency_metrics
from src.evaluation.report_builder import (
    export_confusion_matrix_csv,
    plot_confusion_matrix_from_csv,
)
from src.features.feature_store import FeatureTable, load_feature_matrix
from src.models.exporter import export_model
from src.models.feature_resolver import (
    manifest_feature_family,
    normalize_algorithm_name,
    resolve_feature_matrix_path,
    resolve_split_path,
)
from src.models.model_registry import is_deep_algorithm
from src.models.trainers.classical_trainer import train_model


def _load_splits(split_path: Path) -> tuple[list[str], list[str], list[str]]:
    data = json.loads(split_path.read_text(encoding="utf-8"))
    return list(data["train"]), list(data["val"]), list(data.get("test", []))


def _algorithm_hyperparameters(
    experiment_config: dict[str, Any],
    algorithm: str,
    baselines_config: dict[str, Any] | None,
) -> dict[str, Any]:
    algo = normalize_algorithm_name(algorithm)
    registry = (baselines_config or {}).get("algorithm_registry", {})
    for key in (algorithm, algo):
        if key in registry:
            entry = registry[key]
            hp = dict(entry.get("hyperparameters", {}))
            hp.setdefault("random_state", 42)
            return hp
    return {"random_state": 42}


def run_single_experiment(
    experiment_config: dict[str, Any],
    *,
    feature_family: str,
    algorithm: str,
    baselines_config: dict[str, Any] | None = None,
    dataset_name: str = "hagrid_subset",
    feature_version: str = "v1",
    dry_run: bool = False,
    preloaded_table: FeatureTable | None = None,
) -> dict[str, Any]:
    """Execute one train/eval run and persist artifacts."""
    root = resolve_project_root()
    experiment_id = experiment_config["experiment_id"]
    algo = normalize_algorithm_name(algorithm)

    matrix_path = resolve_feature_matrix_path(
        feature_family,
        dataset_name=dataset_name,
        feature_version=feature_version,
        project_root=root,
    )
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {matrix_path}. "
            "Run Phase 2 extraction before Phase 3."
        )

    split_path = resolve_split_path(dataset_name, project_root=root)
    train_ids, val_ids, _ = _load_splits(split_path)

    table = preloaded_table if preloaded_table is not None else load_feature_matrix(matrix_path)
    expected_family = manifest_feature_family(feature_family)
    records = [
        r
        for r in table.records
        if r.get("feature_family") in (expected_family, feature_family)
    ]
    if not records:
        records = table.records

    hp = _algorithm_hyperparameters(experiment_config, algo, baselines_config)
    train_config = {
        "algorithm": algo,
        "hyperparameters": hp,
        "scale_features": hp.get("scale_features", algo in {"knn", "svm", "logistic_regression", "mlp", "cnn", "lstm"}),
    }

    run_record = create_run_record(experiment_id, experiment_config, status="running")
    run_record["algorithm"] = algo
    run_record["feature_family"] = feature_family
    run_record["feature_matrix_path"] = str(matrix_path)
    run_record["split_path"] = str(split_path)

    if dry_run:
        run_record["status"] = "dry_run"
        return run_record

    if is_deep_algorithm(algo):
        from src.models.trainers.deep_baseline_trainer import train_deep_baseline

        train_result = train_deep_baseline(records, train_ids, val_ids, train_config)
        export_format = "torch"
    else:
        train_result = train_model(records, train_ids, val_ids, train_config)
        export_format = "joblib"

    class_metrics = compute_classification_metrics(
        train_result["y_true"],
        train_result["y_pred"],
    )
    eff_metrics = compute_efficiency_metrics(train_result["timing"])
    metrics = {**class_metrics, **eff_metrics}

    reports_dir = Path(experiment_config.get("outputs", {}).get("reports_dir", "reports/tables"))
    figures_dir = root / "reports" / "figures"
    cm_base = f"{experiment_id}_{algo}_{feature_family}"
    cm_csv = export_confusion_matrix_csv(
        class_metrics["confusion_matrix"],
        class_metrics["labels"],
        root / reports_dir / f"{cm_base}_confusion.csv",
    )
    cm_png = plot_confusion_matrix_from_csv(
        cm_csv,
        figures_dir / f"{cm_base}_confusion.png",
        title=f"{experiment_id} {algo} ({feature_family})",
    )

    export_info = export_model(
        train_result["model"],
        metadata={
            "algorithm_name": algo,
            "feature_family": feature_family,
            "feature_version": feature_version,
            "train_split_id": split_path.stem,
            "validation_strategy": "holdout",
            "hyperparameters": hp,
            "metrics_summary": metrics,
            "scaler": train_result.get("scaler"),
            "label_to_idx": train_result.get("label_to_idx"),
            "classes": (
                train_result.get("classes").tolist()
                if train_result.get("classes") is not None
                else None
            ),
            "reshape": train_result.get("reshape"),
            "scale_features": train_config["scale_features"],
            "vector_dim": len(records[0].get("vector_inline", [])) if records else None,
        },
        experiment_id=experiment_id,
        algorithm_name=algo,
        feature_family=feature_family,
        export_format=export_format,
    )

    run_record["status"] = "completed"
    run_record["metrics"] = metrics
    run_record["artifacts"] = {
        "model_path": export_info["artifact_path"],
        "model_metadata_path": export_info["sidecar_path"],
        "confusion_matrix_csv": str(cm_csv),
        "confusion_matrix_png": str(cm_png),
    }
    metrics_path = save_run_record(
        run_record,
        output_path=str(
            build_metrics_record_path(
                experiment_id,
                run_record["run_id"],
                config=experiment_config,
                project_root=root,
            )
        ),
    )
    run_record["outputs"]["metrics_path"] = str(metrics_path)
    return run_record
