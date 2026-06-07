"""Smoke checks for Phase 5 real-time runtime modules."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config_loader import load_config
from src.features.extraction import expected_vector_dim
from src.features.feature_combiner import concatenate_features
from src.runtime.gesture_filter import GestureFilter
from src.runtime.key_mapper import DEFAULT_KEYMAP, dispatch_key_action, load_keymap
from src.runtime.model_runner import RuntimeModel
from src.runtime.preprocess import extract_runtime_features, resolve_runtime_feature_family
from src.runtime.session_logger import SessionLogger, log_runtime_event
from src.runtime.telemetry import RuntimeTelemetry, compute_runtime_summary, record_stage_timing


@pytest.fixture
def feature_config(project_root: Path) -> dict:
    return load_config(str(project_root / "configs/features/default.yaml"))


@pytest.fixture
def runtime_config(project_root: Path) -> dict:
    return load_config(str(project_root / "configs/runtime/default.yaml"))


@pytest.fixture
def project_root() -> Path:
    from src.common.path_manager import resolve_project_root

    return resolve_project_root()


def test_runtime_config_loads(project_root: Path) -> None:
    config = load_config(str(project_root / "configs/runtime/default.yaml"))
    assert "gesture_mapping" in config
    assert config["dispatch"]["dry_run"] is True
    assert config["dispatch"]["enable_key_dispatch"] is False


def test_resolve_runtime_feature_family_hybrid() -> None:
    assert resolve_runtime_feature_family("hybrid") == "hybrid_keypoints_hog"


def test_hybrid_vector_dim_matches_offline(feature_config: dict) -> None:
    geom_dim = expected_vector_dim(feature_config, "geometric")
    hog_dim = expected_vector_dim(feature_config, "hog")
    hybrid = concatenate_features(
        {"geometric": np.zeros(geom_dim), "hog": np.zeros(hog_dim)},
        family_order=("geometric", "hog"),
    )
    assert hybrid.shape[0] == geom_dim + hog_dim


@patch("src.runtime.preprocess.detect_hand_landmarks")
def test_extract_runtime_features_hybrid_shape(
    mock_detect: MagicMock,
    feature_config: dict,
) -> None:
    mock_detect.return_value = None
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    record = extract_runtime_features(frame, feature_config, feature_family="hybrid")
    assert record["feature_family"] == "hybrid_keypoints_hog"
    geom_dim = expected_vector_dim(feature_config, "geometric")
    hog_dim = expected_vector_dim(feature_config, "hog")
    assert len(record["vector_inline"]) == geom_dim + hog_dim
    assert record["quality_flags"].get("detection_failed") is True


def test_gesture_filter_consensus_and_debounce() -> None:
    filt = GestureFilter(
        window_size=3,
        min_consensus_ratio=0.67,
        debounce_ms=500.0,
        confidence_threshold=0.5,
    )
    t0 = 1000.0
    for i, label in enumerate(["Palm", "Palm", "Palm"]):
        state = filt.update_prediction({"label": label, "confidence": 0.9}, t0 + i * 0.05)
    assert state["stable_label"] == "Palm"
    assert state["meets_consensus"] is True
    assert filt.should_emit_action(state) is True

    filt.mark_emitted(state)
    assert filt.should_emit_action(state) is False

    later = filt.update_prediction({"label": "Palm", "confidence": 0.9}, t0 + 0.2)
    assert filt.should_emit_action(later) is False

    much_later = filt.update_prediction({"label": "Palm", "confidence": 0.9}, t0 + 1.0)
    assert filt.should_emit_action(much_later) is True


def test_dispatch_dry_run_never_emits() -> None:
    result = dispatch_key_action(
        "Palm",
        DEFAULT_KEYMAP,
        dry_run=True,
        enable_dispatch=True,
    )
    assert result["emitted"] is False
    assert result["reason"] == "dry_run"
    assert result.get("would_emit") is True


def test_dispatch_disabled_without_enable_flag() -> None:
    result = dispatch_key_action(
        "Fist",
        DEFAULT_KEYMAP,
        dry_run=False,
        enable_dispatch=False,
    )
    assert result["emitted"] is False
    assert result["reason"] == "dispatch_disabled"


def test_load_keymap_from_runtime_config(runtime_config: dict) -> None:
    keymap = load_keymap(runtime_config=runtime_config)
    assert keymap["Palm"] == "space"
    assert keymap["Peace"] == "down"


def test_telemetry_summary_p95() -> None:
    samples = [
        {"end_to_end_ms": 10.0, "fps": 30.0, "frame_interval_ms": 33.0},
        {"end_to_end_ms": 20.0, "fps": 25.0, "frame_interval_ms": 40.0},
        {"end_to_end_ms": 30.0, "fps": 20.0, "frame_interval_ms": 50.0},
        {"end_to_end_ms": 40.0, "fps": 15.0, "frame_interval_ms": 66.0},
    ]
    summary = compute_runtime_summary(samples)
    assert summary["frame_count"] == 4
    assert summary["latency_ms"]["end_to_end_p95"] >= summary["latency_ms"]["end_to_end_avg"]
    assert summary["avg_fps"] > 0


def test_record_stage_timing() -> None:
    samples: list[dict] = []
    record_stage_timing(samples, "preprocess", 0.0, 0.01)
    assert samples[0]["stage"] == "preprocess"
    assert samples[0]["duration_ms"] == pytest.approx(10.0, rel=0.01)


def test_runtime_model_predict_gesture() -> None:
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 8))
    y = np.array(["Palm"] * 10 + ["Fist"] * 10 + ["Peace"] * 10)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = SVC(kernel="linear", probability=True)
    clf.fit(Xs, y)

    bundle = {
        "estimator": clf,
        "scaler": scaler,
        "algorithm": "svm",
        "classes": list(clf.classes_),
    }
    model = RuntimeModel(bundle=bundle, metadata={"feature_family": "hybrid_keypoints_hog"})
    vec = X[0]
    pred = model.predict_gesture(vec)
    assert pred["label"] in {"Palm", "Fist", "Peace"}
    assert 0.0 <= pred["confidence"] <= 1.0
    assert pred["scores"] is not None


def test_session_logger_jsonl_and_summary(tmp_path: Path) -> None:
    log_path = tmp_path / "runtime_session_test.jsonl"
    logger = SessionLogger(log_path)
    logger.log_event({"predicted_gesture": "Palm", "confidence": 0.91})
    logger.log_event({"predicted_gesture": "Fist", "confidence": 0.88})

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["predicted_gesture"] == "Palm"
    assert "timestamp_iso" in first

    summary_path = logger.write_summary({"avg_fps": 28.5, "frame_count": 2})
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["event_count"] == 2
    assert payload["summary"]["avg_fps"] == 28.5


def test_log_runtime_event_append(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log_runtime_event({"event": "test"}, path)
    log_runtime_event({"event": "test2"}, path)
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_runtime_telemetry_end_frame() -> None:
    telemetry = RuntimeTelemetry()
    start = time.perf_counter()
    preprocess_end = start + 0.01
    predict_end = preprocess_end + 0.005
    dispatch_end = predict_end + 0.001

    sample = telemetry.end_frame(
        frame_start=start,
        preprocess_end=preprocess_end,
        predict_end=predict_end,
        dispatch_end=dispatch_end,
        prediction={"label": "Palm", "confidence": 0.8},
        action_result={"emitted": False, "reason": "dry_run"},
        detection_ok=True,
    )
    assert sample["capture_to_prediction_ms"] > 0
    assert sample["end_to_end_ms"] > 0
    summary = telemetry.summary()
    assert summary["frame_count"] == 1
