"""Smoke checks for Phase 4 robustness evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.error_analysis import group_errors_by_context, sample_failure_cases
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.ood_loader import validate_schema_compatibility
from src.evaluation.robustness_metrics import (
    compute_misclassification_concentration,
    compute_ood_domain_report,
    compute_ood_drop,
    compute_ood_eval_protocols,
    compute_per_class_shift,
    mask_predictions_shared_argmax,
    mask_predictions_unknown,
)
from src.evaluation.domain_report import build_domain_shift_report, format_domain_shift_markdown
from src.features.feature_store import save_feature_matrix
from src.models.classical.logistic_regression import LogisticRegressionClassifier
from src.models.exporter import export_model
from src.models.inference import load_exported_bundle, predict_on_records


@pytest.fixture
def tiny_robustness_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Minimal in-domain + OOD feature matrices and a trained model."""
    rng = np.random.default_rng(7)
    labels = ["Palm", "Fist", "Thumb"]

    def _records(dataset_name: str, prefix: str, n: int = 30) -> list[dict]:
        rows = []
        for i in range(n):
            label = labels[i % 3]
            vec = rng.normal(size=8)
            if label == "Palm":
                vec[0] += 2.0
            elif label == "Fist":
                vec[0] -= 2.0
            else:
                vec[1] += 2.0
            rows.append(
                {
                    "sample_id": f"{prefix}_{i:03d}",
                    "dataset_name": dataset_name,
                    "gesture_label": label,
                    "feature_family": "hybrid_keypoints_hog",
                    "feature_version": "v1",
                    "vector_inline": vec.tolist(),
                    "quality_flags": {},
                    "extraction_ok": True,
                }
            )
        return rows

    in_records = _records("hagrid_subset", "h", 45)
    ood_records = _records("leapgestrecog", "l", 30)

    features_dir = tmp_path / "artifacts" / "features"
    features_dir.mkdir(parents=True)
    in_matrix = features_dir / "hagrid_subset_hybrid_v1.parquet"
    ood_matrix = features_dir / "leapgestrecog_hybrid_v1.parquet"
    save_feature_matrix(in_records, in_matrix)
    save_feature_matrix(ood_records, ood_matrix)

    for matrix_path, records in ((in_matrix, in_records), (ood_matrix, ood_records)):
        manifest = {
            "feature_family": "hybrid_keypoints_hog",
            "feature_version": "v1",
            "vector_dim": 8,
            "n_samples": len(records),
            "sample_ids": [r["sample_id"] for r in records],
        }
        manifest_path = matrix_path.with_name(f"{matrix_path.stem}_manifest.json")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    splits_dir = tmp_path / "data" / "splits"
    splits_dir.mkdir(parents=True)
    split_ids = [r["sample_id"] for r in in_records]
    splits = {"train": split_ids[:30], "val": split_ids[30:40], "test": split_ids[40:]}
    (splits_dir / "hagrid_subset_train_val_test.json").write_text(
        json.dumps(splits), encoding="utf-8"
    )

    interim = tmp_path / "data" / "interim"
    interim.mkdir(parents=True)
    for dataset, records in (("hagrid_subset", in_records), ("leapgestrecog", ood_records)):
        manifest_rows = [
            {
                "sample_id": r["sample_id"],
                "dataset_name": r["dataset_name"],
                "gesture_label": r["gesture_label"],
                "image_path": f"/tmp/{r['sample_id']}.jpg",
                "subject_id": "s1",
                "capture_context": {"source": "test"},
            }
            for r in records
        ]
        pd.DataFrame(manifest_rows).to_parquet(
            interim / f"{dataset}_manifest.parquet", index=False
        )

    train_ids = split_ids[:30]
    train_set = set(train_ids)
    X_train = np.vstack(
        [np.asarray(r["vector_inline"], dtype=np.float64) for r in in_records if r["sample_id"] in train_set]
    )
    y_train = np.array([r["gesture_label"] for r in in_records if r["sample_id"] in train_set])
    model = LogisticRegressionClassifier(max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    models_dir = tmp_path / "artifacts" / "models"
    models_dir.mkdir(parents=True)
    export_info = export_model(
        model,
        metadata={
            "algorithm_name": "logistic_regression",
            "feature_family": "hybrid",
            "feature_version": "v1",
            "vector_dim": 8,
            "scale_features": False,
        },
        output_dir=models_dir,
        experiment_id="EXP-01",
        algorithm_name="logistic_regression",
        feature_family="hybrid",
        export_format="joblib",
    )

    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("# test\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    (tmp_path / "configs").mkdir(exist_ok=True)

    return {
        "root": tmp_path,
        "in_matrix": in_matrix,
        "ood_matrix": ood_matrix,
        "model_path": Path(export_info["artifact_path"]),
    }


def test_ood_domain_report_per_class_and_confusion() -> None:
    y_true = np.array(["Palm", "Palm", "Fist", "Down"])
    y_pred = np.array(["Palm", "dislike", "Fist", "dislike"])
    vocab = ["Palm", "Fist", "Down", "Thumb"]
    report = compute_ood_domain_report(y_true, y_pred, ood_label_vocab=vocab)
    assert len(report["per_class"]) == 4
    palm_row = next(r for r in report["per_class"] if r["gesture_label"] == "Palm")
    assert palm_row["n_correct"] == 1
    assert palm_row["accuracy"] == 0.5
    assert "_other_" in report["labels"]
    assert report["n_out_of_vocab_predictions"] == 2
    assert len(report["confusion_matrix"]) == len(report["labels"])


def test_ood_eval_protocols() -> None:
    schema = {
        "label_overlap": {
            "shared": ["Palm", "Fist", "Thumb"],
            "test_only": ["Down"],
            "n_shared": 3,
        }
    }
    y_true = np.array(["Palm", "Palm", "Down", "Fist"])
    y_pred = np.array(["Palm", "dislike", "Down", "Fist"])
    y_proba = np.array(
        [
            [0.1, 0.7, 0.2],
            [0.6, 0.3, 0.1],
            [0.2, 0.2, 0.6],
            [0.1, 0.1, 0.8],
        ]
    )
    classes = ["dislike", "Palm", "Fist"]

    unknown = mask_predictions_unknown(y_pred, ["Palm", "Fist", "Down", "Thumb"])
    assert unknown[1] == "unknown"
    assert unknown[0] == "Palm"

    shared_pred = mask_predictions_shared_argmax(y_proba, classes, ["Palm", "Fist", "Thumb"])
    assert shared_pred[0] == "Palm"
    assert shared_pred[1] == "Palm"

    protocols = compute_ood_eval_protocols(
        y_true,
        y_pred,
        schema,
        ood_y_proba=y_proba,
        model_class_names=classes,
        id_y_true=y_true,
        id_y_pred=y_pred,
    )
    assert protocols["shared_subset"]["ood"]["n_samples"] == 3
    assert protocols["shared_subset"]["ood"]["n_excluded"] == 1
    assert protocols["masked_unknown"]["ood"]["n_unknown_predictions"] == 1
    assert protocols["masked_shared_argmax"]["available"] is True


def test_compute_ood_drop() -> None:
    drop = compute_ood_drop({"accuracy": 0.9}, {"accuracy": 0.7})
    assert drop["absolute_accuracy_drop"] == pytest.approx(0.2)
    assert drop["relative_performance_retention"] == pytest.approx(0.7 / 0.9)


def test_validate_schema_compatibility_mismatch() -> None:
    train_m = {"vector_dim": 64, "feature_family": "hybrid_keypoints_hog", "feature_version": "v1"}
    test_m = {"vector_dim": 32, "feature_family": "hybrid_keypoints_hog", "feature_version": "v1"}
    report = validate_schema_compatibility(train_m, test_m)
    assert report["compatible"] is False
    assert any("vector_dim" in issue for issue in report["issues"])


def test_per_class_shift_and_misclassification() -> None:
    y_true = np.array(["A", "A", "B", "B"])
    y_pred_id = np.array(["A", "A", "B", "B"])
    y_pred_ood = np.array(["A", "B", "B", "A"])
    shift = compute_per_class_shift(y_true, y_pred_id, y_true, y_pred_ood, labels=["A", "B"])
    assert len(shift["per_class"]) == 2
    misclass = compute_misclassification_concentration(y_true, y_pred_ood)
    assert misclass["total_errors"] == 2


def test_error_analysis_sampling() -> None:
    df = pd.DataFrame(
        [
            {"true_label": "A", "predicted_label": "B", "domain": "ood", "confidence": 0.9},
            {"true_label": "A", "predicted_label": "A", "domain": "ood", "confidence": 0.8},
            {"true_label": "B", "predicted_label": "A", "domain": "ood", "confidence": 0.7},
        ]
    )
    summary = group_errors_by_context(df)
    assert summary["n_errors"] == 2
    samples = sample_failure_cases(df, n_per_class=1, domain="ood")
    assert len(samples) == 2


def test_inference_roundtrip(tiny_robustness_artifacts: dict) -> None:
    from src.features.feature_store import load_feature_matrix

    bundle, _ = load_exported_bundle(tiny_robustness_artifacts["model_path"])
    table = load_feature_matrix(tiny_robustness_artifacts["ood_matrix"])
    result = predict_on_records(bundle, table.records)
    assert len(result["y_pred"]) == len(table.records)


def test_resolve_metrics_dir_uses_experiment_slug() -> None:
    from src.common.path_manager import resolve_metrics_dir

    path = resolve_metrics_dir("EXP-03", config=None, create=False)
    assert path.as_posix().endswith("artifacts/metrics/exp03_robustness")


def test_load_run_records_recursive(tmp_path: Path) -> None:
    from src.evaluation.report_builder import load_run_records

    sub = tmp_path / "exp03_robustness"
    sub.mkdir(parents=True)
    record = {
        "run_id": "abc",
        "experiment_id": "EXP-03",
        "status": "completed",
        "metrics": {},
    }
    (sub / "EXP-03_abc.json").write_text(json.dumps(record), encoding="utf-8")
    (tmp_path / "noise.json").write_text(json.dumps({"foo": 1}), encoding="utf-8")

    flat = load_run_records(tmp_path, recursive=False)
    assert flat == []

    nested = load_run_records(tmp_path, recursive=True)
    assert len(nested) == 1
    assert nested[0]["run_id"] == "abc"


def test_iter_robustness_suite_specs_explicit_pair() -> None:
    from src.evaluation.robustness_suite import iter_robustness_suite_specs

    config = {
        "features": {"feature_version": "v1"},
        "robustness_suite": {
            "dataset_name": "hagrid_subset",
            "ood_dataset_name": "leapgestrecog",
            "train_experiment_id": "EXP-01",
            "pairs": [{"algorithm": "svm", "feature_family": "hybrid"}],
        },
    }
    specs = list(iter_robustness_suite_specs(config, project_root=PROJECT_ROOT))
    assert len(specs) == 1
    assert specs[0].algorithm == "svm"
    assert specs[0].feature_family == "hybrid"
    assert specs[0].in_domain_features.name == "hagrid_subset_hybrid_v1.parquet"
    assert specs[0].ood_features.name == "leapgestrecog_hybrid_v1.parquet"


def test_end_to_end_robustness_runner(tiny_robustness_artifacts: dict) -> None:
    from src.evaluation.robustness_runner import run_robustness_eval

    exp_config = {
        "experiment_id": "EXP-03",
        "features": {"feature_family": "hybrid", "feature_version": "v1"},
        "outputs": {
            "metrics_dir": "artifacts/metrics/exp03_robustness",
            "reports_dir": "reports/tables",
        },
    }
    record = run_robustness_eval(
        exp_config,
        model_artifact=tiny_robustness_artifacts["model_path"],
        in_domain_features=tiny_robustness_artifacts["in_matrix"],
        ood_features=tiny_robustness_artifacts["ood_matrix"],
    )
    assert record["status"] == "completed"
    assert "robustness" in record["metrics"]
    assert "ood_eval_protocols" in record
    assert "shared_subset" in record["ood_eval_protocols"]
    metrics_path = Path(record["outputs"]["metrics_path"])
    assert metrics_path.exists()
    assert "exp03_robustness" in metrics_path.as_posix()
    assert Path(record["artifacts"]["ood_predictions_csv"]).exists()
    assert Path(record["artifacts"]["ood_per_class_accuracy_csv"]).exists()
    assert Path(record["artifacts"]["ood_confusion_matrix_csv"]).exists()
