"""Smoke checks for Phase 1 dataset ingestion modules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.adapters.hagrid_adapter import index_samples as index_hagrid
from src.data.adapters.leapgestrecog_adapter import index_samples as index_leap
from src.data.dataset_summary import load_manifest, save_manifest, summarize_dataset
from src.data.label_mapper import (
    LEAPGESTRECOG_LABELS,
    apply_label_normalization,
    normalize_label,
    validate_label_coverage,
)
from src.data.split_generator import (
    create_primary_splits,
    create_stratified_folds,
    save_folds,
    save_splits,
)


@pytest.fixture
def leapgest_tree(tmp_path: Path) -> Path:
    root = tmp_path / "leapgestrecog"
    layout = [
        ("subject_01", "01_palm", 2),
        ("subject_01", "03_fist", 2),
        ("subject_02", "05_thumb", 2),
        ("subject_02", "07_ok", 2),
        ("subject_02", "10_down", 2),
    ]
    for subject, gesture, count in layout:
        folder = root / subject / gesture
        folder.mkdir(parents=True)
        for i in range(count):
            (folder / f"{gesture}_{i}.jpg").write_bytes(b"fake")
    return root


@pytest.fixture
def hagrid_folder_tree(tmp_path: Path) -> Path:
    root = tmp_path / "hagrid"
    for gesture in ("palm", "fist", "like", "peace", "mute"):
        folder = root / gesture
        folder.mkdir(parents=True)
        (folder / f"{gesture}_1.jpg").write_bytes(b"fake")
    return root


@pytest.fixture
def hagrid_annotation_tree(tmp_path: Path) -> Path:
    root = tmp_path / "hagrid_ann"
    images = root / "images"
    images.mkdir(parents=True)
    records = []
    for i, gesture in enumerate(("palm", "fist", "like", "peace")):
        fname = f"img_{i}.jpg"
        (images / fname).write_bytes(b"fake")
        records.append({"file_name": f"images/{fname}", "label": gesture, "user_id": f"u{i}"})
    (root / "annotations.json").write_text(json.dumps(records), encoding="utf-8")
    return root


def test_leapgestrecog_indexing(leapgest_tree: Path) -> None:
    samples = index_leap(str(leapgest_tree))
    assert len(samples) == 10
    ids = [s["sample_id"] for s in samples]
    assert len(ids) == len(set(ids))
    assert all(Path(s["image_path"]).is_file() for s in samples)


def test_leapgestrecog_label_normalization(leapgest_tree: Path) -> None:
    raw = index_leap(str(leapgest_tree))
    normalized = apply_label_normalization(raw, "leapgestrecog")
    labels = {s["gesture_label"] for s in normalized}
    assert "Palm" in labels
    assert "Fist" in labels
    assert "Thumb" in labels
    assert "OK" in labels
    assert "Down" in labels
    assert "01_palm" not in labels


def test_hagrid_folder_indexing_retains_all_classes(hagrid_folder_tree: Path) -> None:
    samples = index_hagrid(str(hagrid_folder_tree), {"max_samples_per_class": 10})
    assert len(samples) == 5
    raw_labels = {s["raw_gesture_label"] for s in samples}
    assert raw_labels == {"palm", "fist", "like", "peace", "mute"}


def test_hagrid_alignment_to_leapgestrecog(hagrid_folder_tree: Path) -> None:
    raw = index_hagrid(str(hagrid_folder_tree), {"max_samples_per_class": 10})
    normalized = apply_label_normalization(
        raw,
        "hagrid_subset",
        align_to_leapgestrecog=True,
    )
    by_raw = {s["raw_gesture_label"]: s["gesture_label"] for s in normalized}
    assert by_raw["palm"] == "Palm"
    assert by_raw["fist"] == "Fist"
    assert by_raw["like"] == "Thumb"
    assert by_raw["peace"] == "Peace"
    assert by_raw["mute"] == "Mute"


def test_hagrid_annotation_indexing(hagrid_annotation_tree: Path) -> None:
    samples = index_hagrid(str(hagrid_annotation_tree), {})
    assert len(samples) == 4
    assert samples[0]["capture_context"]["format"] == "annotations"


def test_label_normalization_aliases() -> None:
    assert normalize_label("05_thumb", "leapgestrecog") == "Thumb"
    assert normalize_label("like", "hagrid_subset", align_to_leapgestrecog=True) == "Thumb"


def test_validate_label_coverage_reference_only(leapgest_tree: Path) -> None:
    samples = apply_label_normalization(index_leap(str(leapgest_tree)), "leapgestrecog")
    coverage = validate_label_coverage(samples, list(LEAPGESTRECOG_LABELS))
    assert coverage["total_samples"] == 10
    assert "outside_reference" in coverage


def test_split_reproducibility(leapgest_tree: Path) -> None:
    samples = apply_label_normalization(index_leap(str(leapgest_tree)), "leapgestrecog")
    split_a = create_primary_splits(samples, seed=42)
    split_b = create_primary_splits(samples, seed=42)
    assert split_a == split_b

    folds = create_stratified_folds(samples, n_folds=2, seed=42, primary_splits=split_a)
    assert len(folds) >= 2
    assert len(folds[0]["train"]) + len(folds[0]["val"]) == len(split_a["train"])


def test_manifest_roundtrip(tmp_path: Path, leapgest_tree: Path) -> None:
    samples = apply_label_normalization(index_leap(str(leapgest_tree)), "leapgestrecog")
    manifest_path = tmp_path / "manifest.parquet"
    save_manifest(samples, manifest_path)
    loaded = load_manifest(manifest_path)
    assert len(loaded) == len(samples)
    summary = summarize_dataset(loaded, reference_labels=list(LEAPGESTRECOG_LABELS))
    assert summary["has_duplicate_sample_ids"] is False
    assert summary["missing_file_count"] == 0


def test_save_split_artifacts(tmp_path: Path, leapgest_tree: Path) -> None:
    samples = apply_label_normalization(index_leap(str(leapgest_tree)), "leapgestrecog")
    splits = create_primary_splits(samples, seed=7)
    folds = create_stratified_folds(samples, n_folds=2, seed=7, primary_splits=splits)
    tvt = save_splits(splits, tmp_path / "leapgestrecog_train_val_test.json")
    cv = save_folds(folds, tmp_path / "leapgestrecog_cv_folds.json")
    assert tvt.exists()
    assert cv.exists()
    payload = json.loads(tvt.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {"train", "val", "test"}
