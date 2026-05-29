"""Smoke checks for Phase 2 feature extraction and storage."""

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
from src.features.feature_combiner import build_feature_record, concatenate_features
from src.features.chunked_extraction import run_chunked_extraction
from src.features.feature_store import (
    FeatureMatrixWriter,
    build_feature_manifest,
    build_feature_manifest_from_matrix,
    load_feature_matrix,
    save_feature_manifest,
    save_feature_matrix,
    vector_from_record,
)
from src.features.geometric_features import GEOMETRIC_VECTOR_DIM, build_geometric_vector
from src.features.hand_detector import (
    HAND_LANDMARK_COUNT,
    landmarks_to_raw_vector,
    prepare_image_for_detection,
)
from src.features.hog_features import extract_hog_descriptor, hog_descriptor_dim
from src.features.batch_extraction import extract_samples_batch, resolve_num_workers
from src.features.quality_checks import (
    evaluate_feature_coverage,
    flag_low_confidence_samples,
    merge_feature_coverage,
)
from src.data.dataset_summary import save_manifest


def _synthetic_landmarks() -> np.ndarray:
    rng = np.random.default_rng(42)
    xy = rng.uniform(0.2, 0.8, size=(HAND_LANDMARK_COUNT, 2))
    z = rng.uniform(-0.05, 0.05, size=(HAND_LANDMARK_COUNT, 1))
    return np.hstack([xy, z])


@pytest.fixture
def feature_config() -> dict:
    return load_config(str(PROJECT_ROOT / "configs/features/default.yaml"))


@pytest.fixture
def synthetic_image() -> np.ndarray:
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[32:96, 32:96] = 200
    return img


def test_resolve_num_workers() -> None:
    assert resolve_num_workers(1, 100) == 1
    assert resolve_num_workers(8, 3) == 3
    assert resolve_num_workers(None, 1) == 1


def test_batch_extraction_serial_matches_length(feature_config: dict) -> None:
    samples = [
        {
            "sample_id": f"s{i}",
            "image_path": "/nonexistent/path.png",
            "dataset_name": "test",
            "gesture_label": "Palm",
        }
        for i in range(3)
    ]
    records = extract_samples_batch(
        samples,
        "geometric",
        feature_config,
        num_workers=1,
    )
    assert len(records) == 3
    assert all(record["feature_family"] == "geometric" for record in records)


def test_prepare_image_keeps_full_frame(feature_config: dict) -> None:
    """LeapGestRecog (OOD) IR frames should not be auto-cropped before detection."""
    gray = np.zeros((240, 640), dtype=np.uint8)
    gray[60:180, 220:420] = 200
    mp_cfg = feature_config["feature_families"]["keypoints_only"]
    rgb = prepare_image_for_detection(gray, mp_cfg)
    assert rgb.shape == (240, 640, 3)


def test_geometric_vector_fixed_dimensionality() -> None:
    landmarks = _synthetic_landmarks()
    vector = build_geometric_vector(landmarks)
    assert vector.shape == (GEOMETRIC_VECTOR_DIM,)
    assert np.isfinite(vector).all()


def test_keypoints_raw_vector_dimensionality() -> None:
    detection = {"landmarks_normalized": _synthetic_landmarks()}
    vector = landmarks_to_raw_vector(detection)
    assert vector.shape == (HAND_LANDMARK_COUNT * 3,)


def test_hog_fixed_dimensionality(feature_config: dict) -> None:
    hog_cfg = feature_config["feature_families"]["hog_only"]["hog"]
    crop_size = tuple(hog_cfg["crop_size"])
    crop = np.zeros((*crop_size[::-1], 3), dtype=np.uint8)
    crop[8:56, 8:56] = 200
    expected = hog_descriptor_dim(crop_size, hog_cfg)
    vector = extract_hog_descriptor(crop, hog_cfg)
    assert vector.shape == (expected,)
    assert expected > 0


def test_feature_matrix_persistence_roundtrip(tmp_path: Path, feature_config: dict) -> None:
    landmarks = _synthetic_landmarks()
    vector = build_geometric_vector(landmarks)
    records = [
        build_feature_record(
            f"sample_{i}",
            "geometric",
            vector,
            {"extraction_ok": True, "confidence": 0.9},
            dataset_name="test",
            gesture_label="Palm",
            feature_version="v1",
        )
        for i in range(3)
    ]
    matrix_path = tmp_path / "test_geometric_v1.parquet"
    save_feature_matrix(records, matrix_path)

    loaded = load_feature_matrix(matrix_path)
    assert len(loaded.records) == 3
    for original, restored in zip(records, loaded.records, strict=True):
        assert original["sample_id"] == restored["sample_id"]
        np.testing.assert_allclose(
            original["vector_inline"],
            restored["vector_inline"],
            rtol=1e-6,
        )

    manifest = build_feature_manifest(
        loaded.records,
        feature_family="geometric",
        feature_version="v1",
        config=feature_config,
        extraction_stats=evaluate_feature_coverage(loaded.records),
    )
    manifest_path = tmp_path / "test_geometric_v1_manifest.json"
    save_feature_manifest(manifest, manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["vector_dim"] == GEOMETRIC_VECTOR_DIM
    assert payload["config_fingerprint"]


def test_hybrid_sample_id_alignment_and_concat(tmp_path: Path, feature_config: dict) -> None:
    landmarks = _synthetic_landmarks()
    geom = build_geometric_vector(landmarks)
    hog_cfg = feature_config["feature_families"]["hog_only"]["hog"]
    hog = extract_hog_descriptor(np.zeros((64, 64, 3), dtype=np.uint8), hog_cfg)

    sample_ids = ["s_a", "s_b", "s_c"]
    geom_records = [
        build_feature_record(
            sid,
            "geometric",
            geom,
            {"extraction_ok": True, "gesture_label": "Palm"},
            feature_version="v1",
        )
        for sid in sample_ids
    ]
    hog_records = [
        build_feature_record(
            sid,
            "hog",
            hog,
            {"extraction_ok": True, "gesture_label": "Palm"},
            feature_version="v1",
        )
        for sid in reversed(sample_ids)
    ]
    hog_records.sort(key=lambda r: r["sample_id"])

    geom_path = tmp_path / "geom.parquet"
    hog_path = tmp_path / "hog.parquet"
    save_feature_matrix(geom_records, geom_path)
    save_feature_matrix(hog_records, hog_path)

    geom_table = load_feature_matrix(geom_path)
    hog_table = load_feature_matrix(hog_path)
    assert geom_table.sample_ids == hog_table.sample_ids

    hybrid_records = []
    for g_rec, h_rec in zip(geom_table.records, hog_table.records, strict=True):
        assert g_rec["gesture_label"] == h_rec["gesture_label"]
        hybrid_vector = concatenate_features(
            {"geometric": vector_from_record(g_rec), "hog": vector_from_record(h_rec)},
            family_order=["geometric", "hog"],
        )
        hybrid_records.append(
            build_feature_record(
                g_rec["sample_id"],
                "hybrid_keypoints_hog",
                hybrid_vector,
                {"extraction_ok": True, "source_families": ["geometric", "hog"]},
                feature_version="v1",
            )
        )

    hybrid_path = tmp_path / "hybrid.parquet"
    save_feature_matrix(hybrid_records, hybrid_path)
    hybrid_loaded = load_feature_matrix(hybrid_path)
    assert hybrid_loaded.sample_ids == sample_ids
    assert len(hybrid_loaded.records[0]["vector_inline"]) == geom.size + hog.size


def test_feature_matrix_writer_appends_batches(tmp_path: Path, feature_config: dict) -> None:
    landmarks = _synthetic_landmarks()
    vector = build_geometric_vector(landmarks)
    matrix_path = tmp_path / "chunked_geometric_v1.parquet"

    writer = FeatureMatrixWriter(matrix_path)
    for start in range(0, 6, 2):
        writer.append(
            [
                build_feature_record(
                    f"sample_{i}",
                    "geometric",
                    vector,
                    {"extraction_ok": True, "confidence": 0.9},
                    dataset_name="test",
                    gesture_label="Palm",
                    feature_version="v1",
                )
                for i in range(start, start + 2)
            ]
        )
    writer.close()

    loaded = load_feature_matrix(matrix_path)
    assert len(loaded.records) == 6
    assert loaded.sample_ids == [f"sample_{i}" for i in range(6)]


def test_merge_feature_coverage() -> None:
    chunk_a = evaluate_feature_coverage(
        [
            {
                "sample_id": "a",
                "gesture_label": "Palm",
                "vector_inline": [1.0],
                "extraction_ok": True,
                "quality_flags": {},
            }
        ]
    )
    chunk_b = evaluate_feature_coverage(
        [
            {
                "sample_id": "b",
                "gesture_label": "Fist",
                "vector_inline": [],
                "extraction_ok": False,
                "quality_flags": {"detection_failed": True},
            }
        ]
    )
    merged = merge_feature_coverage(chunk_a, chunk_b)
    assert merged["total_samples"] == 2
    assert merged["failed_extractions"] == 1
    assert merged["by_gesture_label"]["Palm"]["successful_extractions"] == 1
    assert merged["by_gesture_label"]["Fist"]["successful_extractions"] == 0


def test_chunked_extraction_from_manifest(tmp_path: Path, feature_config: dict) -> None:
    samples = [
        {
            "sample_id": f"s{i}",
            "image_path": "/nonexistent/path.png",
            "dataset_name": "test",
            "gesture_label": "Palm",
        }
        for i in range(5)
    ]
    manifest_path = tmp_path / "tiny_manifest.parquet"
    save_manifest(samples, manifest_path)
    output_path = tmp_path / "chunked_out.parquet"

    coverage, written = run_chunked_extraction(
        manifest_path,
        output_path,
        "geometric",
        feature_config,
        batch_size=2,
        num_workers=1,
        pool_chunksize=None,
        apply_quality=False,
        min_confidence=0.5,
        min_visible_landmarks=None,
        min_class_samples=None,
        min_class_rate=None,
        log_info=lambda *args, **kwargs: None,
    )
    assert written == 5
    assert coverage["total_samples"] == 5

    loaded = load_feature_matrix(output_path)
    assert len(loaded.records) == 5

    manifest = build_feature_manifest_from_matrix(
        output_path,
        feature_family="geometric",
        feature_version="v1",
        config=feature_config,
        extraction_stats=coverage,
    )
    assert manifest["n_samples"] == 5
    assert manifest["vector_dim"] == GEOMETRIC_VECTOR_DIM


def test_quality_coverage_and_low_confidence_flagging() -> None:
    records = [
        {
            "sample_id": "a",
            "gesture_label": "Palm",
            "vector_inline": [1.0],
            "extraction_ok": True,
            "confidence": 0.9,
            "quality_flags": {},
        },
        {
            "sample_id": "b",
            "gesture_label": "Fist",
            "vector_inline": [],
            "extraction_ok": False,
            "confidence": 0.2,
            "quality_flags": {"detection_failed": True, "low_confidence": True},
        },
        {
            "sample_id": "c",
            "gesture_label": "Palm",
            "vector_inline": [2.0, 3.0],
            "extraction_ok": True,
            "confidence": 0.8,
            "quality_flags": {},
        },
    ]
    coverage = evaluate_feature_coverage(
        records,
        min_samples_per_class=2,
        min_class_success_rate=0.75,
    )
    assert coverage["total_samples"] == 3
    assert coverage["failed_extractions"] == 1
    assert coverage["by_gesture_label"]["Palm"]["successful_extractions"] == 2
    assert coverage["by_gesture_label"]["Palm"]["success_rate"] == 1.0
    assert coverage["by_gesture_label"]["Fist"]["successful_extractions"] == 0
    assert coverage["by_gesture_label"]["Fist"]["success_rate"] == 0.0
    assert any(item["gesture_label"] == "Fist" for item in coverage["classes_below_success_rate"])
    flagged = flag_low_confidence_samples(records, threshold=0.5)
    assert "b" in flagged
