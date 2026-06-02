"""Orchestrate EXP-03 robustness evaluation runs."""

from __future__ import annotations

import json
import gc
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.path_manager import resolve_metrics_dir, resolve_project_root
from src.common.run_registry import create_run_record, save_run_record
from src.evaluation.domain_report import (
    build_domain_shift_report,
    export_domain_shift_json,
    export_domain_shift_report,
    export_ood_per_class_accuracy_figure,
    export_per_class_drop_figure,
)
from src.evaluation.error_analysis import group_errors_by_context, sample_failure_cases
from src.evaluation.metrics import compute_efficiency_metrics
from src.evaluation.ood_loader import (
    build_sample_metadata_index,
    filter_records_by_feature_family,
    filter_records_by_split,
    load_feature_manifest,
    validate_schema_compatibility,
)
from src.evaluation.report_builder import (
    export_confusion_matrix_csv,
    plot_confusion_matrix_from_csv,
)
from src.evaluation.robustness_metrics import (
    compute_misclassification_concentration,
    compute_ood_domain_report,
    compute_ood_drop,
    compute_ood_eval_protocols,
    ood_label_vocab_from_schema,
    compute_per_class_shift,
    evaluate_domain,
)
from src.features.feature_store import (
    collect_gesture_labels_from_matrix,
    iter_feature_matrix_batches,
)
from src.models.feature_resolver import resolve_split_path
from src.models.inference import load_exported_bundle, predict_on_record_batches


def _load_splits(split_path: Path) -> tuple[list[str], list[str], list[str]]:
    data = json.loads(split_path.read_text(encoding="utf-8"))
    return list(data["train"]), list(data["val"]), list(data.get("test", []))


def _confidence_from_proba(
    y_proba: np.ndarray | None,
    y_pred: np.ndarray,
    classes: list[str] | None,
) -> list[float | None]:
    if y_proba is None:
        return [None] * len(y_pred)
    if classes is None:
        return [float(np.max(row)) for row in y_proba]
    class_to_idx = {str(c): i for i, c in enumerate(classes)}
    confidences: list[float | None] = []
    for pred, row in zip(y_pred, y_proba):
        idx = class_to_idx.get(str(pred))
        if idx is None or idx >= len(row):
            confidences.append(float(np.max(row)))
        else:
            confidences.append(float(row[idx]))
    return confidences


def _build_predictions_dataframe(
    inference_result: dict[str, Any],
    *,
    domain: str,
    metadata_index: dict[str, dict[str, Any]],
    dataset_name: str,
) -> pd.DataFrame:
    y_true = inference_result["y_true"]
    y_pred = inference_result["y_pred"]
    sample_ids = inference_result["sample_ids"]
    y_proba = inference_result.get("y_proba")

    classes: list[str] | None = None
    if y_proba is not None and y_proba.ndim == 2:
        n_classes = y_proba.shape[1]
        classes = [str(i) for i in range(n_classes)]

    confidences = _confidence_from_proba(y_proba, y_pred, classes)

    rows: list[dict[str, Any]] = []
    for i, sid in enumerate(sample_ids):
        meta = metadata_index.get(sid, {})
        capture = meta.get("capture_context", {})
        if isinstance(capture, str):
            try:
                capture = json.loads(capture)
            except json.JSONDecodeError:
                capture = {"raw": capture}
        rows.append(
            {
                "sample_id": sid,
                "dataset_name": meta.get("dataset_name", dataset_name),
                "image_path": meta.get("image_path", ""),
                "subject_id": meta.get("subject_id"),
                "true_label": str(y_true[i]),
                "predicted_label": str(y_pred[i]),
                "correct": str(y_true[i]) == str(y_pred[i]),
                "confidence": confidences[i],
                "domain": domain,
                "capture_context": capture,
            }
        )
    return pd.DataFrame(rows)


def _batched_records_for_matrix(
    matrix_path: Path,
    *,
    feature_family: str,
    sample_ids: set[str] | None,
    batch_size: int,
) -> Iterator[list[dict[str, Any]]]:
    for batch in iter_feature_matrix_batches(
        matrix_path,
        batch_size=batch_size,
        sample_ids=sample_ids,
        exclude_invalid=True,
    ):
        filtered = filter_records_by_feature_family(batch, feature_family)
        if filtered:
            yield filtered


def _predict_matrix(
    bundle: dict[str, Any],
    matrix_path: Path,
    *,
    feature_family: str,
    sample_ids: set[str] | None,
    batch_size: int,
    include_proba: bool,
) -> tuple[dict[str, Any], int]:
    batches = _batched_records_for_matrix(
        matrix_path,
        feature_family=feature_family,
        sample_ids=sample_ids,
        batch_size=batch_size,
    )
    result = predict_on_record_batches(
        bundle,
        batches,
        include_proba=include_proba,
        batch_size=batch_size,
    )
    return result, len(result["sample_ids"])


def run_robustness_eval(
    experiment_config: dict[str, Any],
    *,
    model_artifact: str | Path,
    in_domain_features: str | Path,
    ood_features: str | Path,
    in_domain_manifest: str | Path | None = None,
    ood_manifest: str | Path | None = None,
    dataset_name: str = "hagrid_subset",
    ood_dataset_name: str = "leapgestrecog",
    feature_family: str | None = None,
    batch_size: int = 256,
    include_proba: bool = False,
) -> dict[str, Any]:
    """Execute EXP-03 and persist metrics, predictions, and reports."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    root = resolve_project_root()
    experiment_id = experiment_config.get("experiment_id", "EXP-03")
    features_cfg = experiment_config.get("features", {})
    feature_family = feature_family or features_cfg.get("feature_family", "hybrid")
    feature_version = str(features_cfg.get("feature_version", "v1"))

    bundle, sidecar = load_exported_bundle(model_artifact)
    model_path = Path(model_artifact)

    in_path = Path(in_domain_features)
    ood_path = Path(ood_features)
    if not in_path.is_absolute():
        in_path = root / in_path
    if not ood_path.is_absolute():
        ood_path = root / ood_path

    train_manifest = load_feature_manifest(in_path)
    test_manifest = load_feature_manifest(ood_path)

    split_path = resolve_split_path(dataset_name, project_root=root)
    _, _, test_ids = _load_splits(split_path)
    test_id_set = set(test_ids) if test_ids else None

    if in_domain_manifest is None:
        in_domain_manifest = root / "data" / "interim" / f"{dataset_name}_manifest.parquet"
    if ood_manifest is None:
        ood_manifest = root / "data" / "interim" / f"{ood_dataset_name}_manifest.parquet"

    train_labels = collect_gesture_labels_from_matrix(
        in_path,
        exclude_invalid=True,
        batch_size=max(batch_size * 4, 512),
    )
    test_labels = collect_gesture_labels_from_matrix(
        ood_path,
        exclude_invalid=True,
        batch_size=max(batch_size * 4, 512),
    )
    schema_validation = validate_schema_compatibility(
        train_manifest,
        test_manifest,
        train_labels=train_labels,
        test_labels=test_labels,
    )

    in_meta_index = build_sample_metadata_index(in_domain_manifest)
    id_result, n_in_domain_eval = _predict_matrix(
        bundle,
        in_path,
        feature_family=feature_family,
        sample_ids=test_id_set,
        batch_size=batch_size,
        include_proba=include_proba,
    )
    if n_in_domain_eval == 0:
        id_result, n_in_domain_eval = _predict_matrix(
            bundle,
            in_path,
            feature_family=feature_family,
            sample_ids=None,
            batch_size=batch_size,
            include_proba=include_proba,
        )

    gc.collect()

    ood_meta_index = build_sample_metadata_index(ood_manifest)
    ood_result, n_ood_eval = _predict_matrix(
        bundle,
        ood_path,
        feature_family=feature_family,
        sample_ids=None,
        batch_size=batch_size,
        include_proba=include_proba,
    )
    gc.collect()

    train_classes = bundle.get("classes")
    if train_classes is not None:
        label_list = [str(c) for c in train_classes]
    elif sidecar and sidecar.get("algorithm_name"):
        label_list = None
    else:
        estimator = bundle.get("estimator")
        if hasattr(estimator, "classes_"):
            label_list = [str(c) for c in estimator.classes_]
        else:
            label_list = None

    if label_list is None:
        label_list = sorted(
            np.unique(
                np.concatenate([id_result["y_true"], id_result["y_pred"], ood_result["y_true"], ood_result["y_pred"]])
            ).tolist(),
            key=str,
        )

    id_metrics = evaluate_domain(
        id_result["y_true"],
        id_result["y_pred"],
        labels=label_list,
        timing=id_result["timing"],
    )
    ood_metrics = evaluate_domain(
        ood_result["y_true"],
        ood_result["y_pred"],
        labels=label_list,
        timing=ood_result["timing"],
    )
    robustness = compute_ood_drop(id_metrics, ood_metrics)
    per_class_shift = compute_per_class_shift(
        id_result["y_true"],
        id_result["y_pred"],
        ood_result["y_true"],
        ood_result["y_pred"],
        labels=label_list,
    )
    misclass = compute_misclassification_concentration(
        ood_result["y_true"],
        ood_result["y_pred"],
    )

    model_class_names = label_list
    if model_class_names is None:
        estimator = bundle.get("estimator")
        if estimator is not None and hasattr(estimator, "classes_"):
            model_class_names = [str(c) for c in estimator.classes_]

    ood_eval_protocols = compute_ood_eval_protocols(
        ood_result["y_true"],
        ood_result["y_pred"],
        schema_validation,
        ood_y_proba=ood_result.get("y_proba"),
        model_class_names=model_class_names,
        id_y_true=id_result["y_true"],
        id_y_pred=id_result["y_pred"],
        ood_timing=ood_result["timing"],
    )

    ood_canonical_vocab = ood_label_vocab_from_schema(schema_validation)
    ood_domain_report = compute_ood_domain_report(
        ood_result["y_true"],
        ood_result["y_pred"],
        ood_label_vocab=ood_canonical_vocab,
    )

    id_predictions = _build_predictions_dataframe(
        id_result,
        domain="in_domain",
        metadata_index=in_meta_index,
        dataset_name=dataset_name,
    )
    ood_predictions = _build_predictions_dataframe(
        ood_result,
        domain="ood",
        metadata_index=ood_meta_index,
        dataset_name=ood_dataset_name,
    )
    id_timing = id_result["timing"]
    ood_timing = ood_result["timing"]
    del id_result
    del ood_result
    del in_meta_index
    del ood_meta_index
    gc.collect()
    all_predictions = pd.concat([id_predictions, ood_predictions], ignore_index=True)

    error_analysis = group_errors_by_context(ood_predictions)
    failure_samples = sample_failure_cases(ood_predictions, n_per_class=3)

    run_record = create_run_record(experiment_id, experiment_config, status="running")
    run_id = run_record["run_id"]

    outputs_cfg = experiment_config.get("outputs", {})
    reports_dir = Path(outputs_cfg.get("reports_dir", "reports/tables"))
    metrics_dir = resolve_metrics_dir(
        experiment_id,
        config=experiment_config,
        project_root=root,
        create=True,
    )
    reports_path = root / reports_dir
    reports_path.mkdir(parents=True, exist_ok=True)
    figures_dir = root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir = root / "reports" / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{experiment_id}_{run_id}"
    id_pred_csv = reports_path / f"{prefix}_in_domain_predictions.csv"
    ood_pred_csv = reports_path / f"{prefix}_ood_predictions.csv"
    per_class_csv = reports_path / f"{prefix}_per_class_drop.csv"
    ood_per_class_csv = reports_path / f"{prefix}_ood_per_class_accuracy.csv"
    combined_pred_csv = metrics_dir / f"{prefix}_predictions.csv"
    per_class_png = figures_dir / "exp03_per_class_drop.png"
    ood_per_class_png = figures_dir / f"{prefix}_ood_per_class_accuracy.png"
    failure_gallery_csv = summaries_dir / f"exp03_{run_id}_failure_samples.csv"

    id_predictions.to_csv(id_pred_csv, index=False)
    ood_predictions.to_csv(ood_pred_csv, index=False)
    all_predictions.to_csv(combined_pred_csv, index=False)
    per_class_shift["dataframe"].to_csv(per_class_csv, index=False)
    export_per_class_drop_figure(per_class_shift["dataframe"], per_class_png)
    ood_domain_report["dataframe"].to_csv(ood_per_class_csv, index=False)
    export_ood_per_class_accuracy_figure(
        ood_domain_report["dataframe"],
        ood_per_class_png,
        title=f"{experiment_id} OOD per-class accuracy",
    )
    if not failure_samples.empty:
        failure_samples.to_csv(failure_gallery_csv, index=False)

    algo = sidecar.get("algorithm_name") if sidecar else bundle.get("algorithm", "model")
    cm_base = f"{prefix}_{algo}_{feature_family}"
    ood_cm_csv = export_confusion_matrix_csv(
        ood_domain_report["confusion_matrix"],
        ood_domain_report["labels"],
        reports_path / f"{prefix}_ood_confusion.csv",
    )
    ood_cm_png = plot_confusion_matrix_from_csv(
        ood_cm_csv,
        figures_dir / f"{prefix}_ood_confusion.png",
        title=f"{experiment_id} OOD confusion ({algo}, canonical vocab)",
    )
    ood_model_cm_csv = export_confusion_matrix_csv(
        ood_metrics["confusion_matrix"],
        ood_metrics["labels"],
        reports_path / f"{cm_base}_ood_model_label_confusion.csv",
    )
    plot_confusion_matrix_from_csv(
        ood_model_cm_csv,
        figures_dir / f"{cm_base}_ood_model_label_confusion.png",
        title=f"{experiment_id} OOD confusion ({algo}, full model labels)",
    )

    report_inputs = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "model_artifact": str(model_path),
        "model_metadata": sidecar or {},
        "schema_validation": schema_validation,
        "in_domain_metrics": id_metrics,
        "ood_metrics": ood_metrics,
        "robustness": robustness,
        "ood_eval_protocols": ood_eval_protocols,
        "ood_domain_report": ood_domain_report,
        "per_class_shift": per_class_shift,
        "misclassification_concentration": misclass,
        "error_analysis": error_analysis,
    }
    domain_report = build_domain_shift_report(report_inputs)
    summary_md = summaries_dir / "robustness_summary.md"
    export_domain_shift_report(domain_report, summary_md)
    report_json = summaries_dir / f"exp03_{run_id}_domain_report.json"
    export_domain_shift_json(domain_report, report_json)

    eff = compute_efficiency_metrics(
        {
            "fit_seconds": 0.0,
            "inference_seconds": id_timing["inference_seconds"] + ood_timing["inference_seconds"],
            "per_sample_inference_ms": (
                id_timing["per_sample_inference_ms"] + ood_timing["per_sample_inference_ms"]
            )
            / 2.0,
        }
    )

    run_record["status"] = "completed"
    run_record["feature_family"] = feature_family
    run_record["algorithm"] = algo
    run_record["metrics"] = {
        **eff,
        "in_domain": id_metrics,
        "ood": ood_metrics,
        "robustness": robustness,
        "ood_eval_protocols": ood_eval_protocols,
    }
    run_record["robustness"] = robustness
    run_record["ood_eval_protocols"] = ood_eval_protocols
    run_record["schema_validation"] = schema_validation
    run_record["per_class_shift"] = per_class_shift["per_class"]
    run_record["ood_per_class"] = ood_domain_report["per_class"]
    run_record["ood_canonical_metrics"] = {
        "accuracy": ood_domain_report["accuracy"],
        "f1_macro": ood_domain_report["f1_macro"],
        "n_samples": ood_domain_report["n_samples"],
        "n_out_of_vocab_predictions": ood_domain_report["n_out_of_vocab_predictions"],
        "out_of_vocab_prediction_rate": ood_domain_report["out_of_vocab_prediction_rate"],
        "labels": ood_domain_report["labels"],
    }
    run_record["artifacts"] = {
        "model_path": str(model_path),
        "model_metadata_path": str(model_path.with_name(model_path.stem + ".meta.json")),
        "in_domain_predictions_csv": str(id_pred_csv),
        "ood_predictions_csv": str(ood_pred_csv),
        "predictions_csv": str(combined_pred_csv),
        "per_class_drop_csv": str(per_class_csv),
        "per_class_drop_png": str(per_class_png),
        "ood_per_class_accuracy_csv": str(ood_per_class_csv),
        "ood_per_class_accuracy_png": str(ood_per_class_png),
        "ood_confusion_matrix_csv": str(ood_cm_csv),
        "ood_confusion_matrix_png": str(ood_cm_png),
        "ood_model_label_confusion_matrix_csv": str(ood_model_cm_csv),
        "ood_model_label_confusion_matrix_png": str(
            figures_dir / f"{cm_base}_ood_model_label_confusion.png"
        ),
        "failure_samples_csv": str(failure_gallery_csv) if not failure_samples.empty else None,
        "domain_report_json": str(report_json),
        "robustness_summary_md": str(summary_md),
    }
    run_record["inputs"] = {
        **run_record.get("inputs", {}),
        "in_domain_features": str(in_path),
        "ood_features": str(ood_path),
        "in_domain_manifest": str(in_domain_manifest),
        "ood_manifest": str(ood_manifest),
        "split_path": str(split_path),
        "n_in_domain_eval": n_in_domain_eval,
        "n_ood_eval": n_ood_eval,
        "batch_size": batch_size,
        "include_proba": include_proba,
    }

    metrics_path = save_run_record(run_record, output_path=str(metrics_dir / f"{prefix}.json"))
    run_record["outputs"]["metrics_path"] = str(metrics_path)
    run_record["domain_report"] = domain_report
    run_record["failure_samples"] = failure_samples
    return run_record
