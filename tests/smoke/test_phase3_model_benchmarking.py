"""Smoke checks for Phase 3 model benchmarking."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config_loader import load_config
from src.evaluation.metrics import compute_classification_metrics, compute_efficiency_metrics
from src.evaluation.report_builder import build_experiment_summary, resolve_report_metrics
from src.features.feature_store import save_feature_matrix
from src.models.experiment_runner import run_single_experiment
from src.models.model_registry import build_model, list_supported_algorithms
from src.models.trainers.classical_trainer import train_model


@pytest.fixture
def synthetic_feature_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create minimal feature matrix and splits under tmp_path."""
    rng = np.random.default_rng(42)
    labels = ["A", "B", "C"]
    records = []
    for i in range(60):
        records.append(
            {
                "sample_id": f"s{i:03d}",
                "dataset_name": "hagrid_subset",
                "gesture_label": labels[i % 3],
                "feature_family": "hybrid_keypoints_hog",
                "feature_version": "v1",
                "vector_inline": rng.normal(size=32).tolist(),
                "quality_flags": {},
                "extraction_ok": True,
            }
        )

    features_dir = tmp_path / "artifacts" / "features"
    features_dir.mkdir(parents=True)
    matrix_path = features_dir / "hagrid_subset_hybrid_v1.parquet"
    save_feature_matrix(records, matrix_path)

    splits_dir = tmp_path / "data" / "splits"
    splits_dir.mkdir(parents=True)
    split_ids = [r["sample_id"] for r in records]
    splits = {
        "train": split_ids[:40],
        "val": split_ids[40:50],
        "test": split_ids[50:],
    }
    (splits_dir / "hagrid_subset_train_val_test.json").write_text(
        json.dumps(splits), encoding="utf-8"
    )

    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("# test root\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    (tmp_path / "configs").mkdir(exist_ok=True)
    return tmp_path


def test_smo_separable_binary_accuracy() -> None:
    from src.models.classical.svm import SVMClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(160, 8))
    y = np.array(["neg"] * 80 + ["pos"] * 80)
    X[:80, 0] -= 2.5
    X[80:, 0] += 2.5
    model = SVMClassifier(kernel="rbf", C=1.0, max_iter=80, tol=1e-3, show_progress=False)
    model.fit(X, y)
    acc = float(np.mean(model.predict(X) == y))
    assert acc >= 0.95


def test_svm_rbf_fit_without_broadcast_oom() -> None:
    from src.models.classical.svm import SVMClassifier

    rng = np.random.default_rng(1)
    n_train, n_features = 400, 32
    X = rng.normal(size=(n_train, n_features))
    y = np.array([f"c{i % 4}" for i in range(n_train)])

    model = SVMClassifier(kernel="rbf", max_iter=30, tol=1e-2, show_progress=False)
    model.fit(X, y)
    preds = model.predict(X[:50])
    assert preds.shape == (50,)


def test_knn_predict_large_query_train_without_broadcast_oom() -> None:
    """Regression: avoid (n_query, n_train, n_features) distance tensor."""
    from src.models.classical.knn import KNNClassifier

    rng = np.random.default_rng(0)
    n_train, n_query, n_features = 800, 200, 64
    X_train = rng.normal(size=(n_train, n_features))
    y_train = np.array([f"c{i % 5}" for i in range(n_train)])
    X_val = rng.normal(size=(n_query, n_features))

    model = KNNClassifier(n_neighbors=5, query_batch_size=64)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    assert preds.shape == (n_query,)
    proba = model.predict_proba(X_val)
    assert proba.shape == (n_query, 5)


@pytest.mark.parametrize("algorithm", list_supported_algorithms())
def test_registry_builds_all_algorithms(algorithm: str) -> None:
    if algorithm in {"mlp", "cnn", "lstm"}:
        pytest.importorskip("torch")
        spec = build_model(algorithm, {"random_state": 42})
        assert spec["algorithm"] == algorithm
        return
    model = build_model(algorithm, {"random_state": 42})
    X = np.random.randn(20, 8)
    y = np.array(["A", "B"] * 10)
    model.fit(X, y)
    preds = model.predict(X[:5])
    assert len(preds) == 5


def test_classical_trainer_contract() -> None:
    rng = np.random.default_rng(0)
    records = []
    for i in range(30):
        records.append(
            {
                "sample_id": f"x{i}",
                "gesture_label": "A" if i % 2 == 0 else "B",
                "vector_inline": rng.normal(size=10).tolist(),
            }
        )
    train_ids = [f"x{i}" for i in range(20)]
    val_ids = [f"x{i}" for i in range(20, 30)]
    result = train_model(
        records,
        train_ids,
        val_ids,
        {"algorithm": "logistic_regression", "hyperparameters": {"max_iter": 100}},
    )
    assert "y_pred" in result
    assert len(result["y_pred"]) == len(val_ids)
    metrics = compute_classification_metrics(result["y_true"], result["y_pred"])
    assert "accuracy" in metrics


def test_experiment_summary_includes_configured_metrics() -> None:
    metrics_config = {
        "primary": "acc",
        "report": [
            "acc",
            "f1_macro",
            "recall_macro",
            "inference_seconds",
            "confusion_matrix",
        ],
    }
    report_keys = resolve_report_metrics(metrics_config)
    assert report_keys == ["accuracy", "f1_macro", "recall_macro", "inference_seconds"]

    timing = {"fit_seconds": 1.0, "inference_seconds": 0.2, "per_sample_inference_ms": 0.5}
    class_m = compute_classification_metrics(
        np.array(["A", "B", "A"]),
        np.array(["A", "B", "B"]),
    )
    records = [
        {
            "run_id": "r1",
            "experiment_id": "EXP-01",
            "algorithm": "svm",
            "feature_family": "hybrid",
            "status": "completed",
            "metrics": {**class_m, **compute_efficiency_metrics(timing)},
        }
    ]
    summary = build_experiment_summary(records, metrics_config=metrics_config)
    row = summary["leaderboard"][0]
    assert row["accuracy"] == class_m["accuracy"]
    assert row["inference_seconds"] == 0.2
    assert "confusion_matrix" not in row


def test_format_leaderboard_markdown_table() -> None:
    from src.evaluation.report_builder import format_leaderboard_markdown_table

    summary = {
        "report_metrics": ["accuracy", "f1_macro"],
        "leaderboard": [
            {
                "algorithm": "svm",
                "feature_family": "hybrid",
                "accuracy": 0.8123456789,
                "f1_macro": 0.8012345678,
            }
        ],
    }
    md = format_leaderboard_markdown_table(summary)
    assert "| algorithm | feature_family | accuracy | f1_macro |" in md
    assert "| svm | hybrid | 0.8123 | 0.8012 |" in md


def test_end_to_end_runner(synthetic_feature_artifacts: Path, tmp_path: Path) -> None:
    exp_config = {
        "experiment_id": "EXP-01",
        "outputs": {"reports_dir": "reports/tables"},
        "models": {"config": "configs/models/baselines.yaml"},
    }
    baselines = load_config(str(PROJECT_ROOT / "configs/models/baselines.yaml"))
    record = run_single_experiment(
        exp_config,
        feature_family="hybrid",
        algorithm="naive_bayes",
        baselines_config=baselines,
        dataset_name="hagrid_subset",
    )
    assert record["status"] == "completed"
    assert Path(record["artifacts"]["model_path"]).exists()
    metrics_path = Path(record["outputs"]["metrics_path"])
    assert metrics_path.exists()
    loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert loaded["metrics"]["accuracy"] >= 0.0
