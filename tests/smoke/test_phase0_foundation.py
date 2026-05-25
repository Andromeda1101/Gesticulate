"""Lightweight smoke checks for Phase 0 foundation modules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config_loader import load_config, merge_overrides
from src.common.path_manager import build_artifact_path, resolve_project_root
from src.common.run_registry import create_run_record, save_run_record


@pytest.fixture
def project_root() -> Path:
    return resolve_project_root()


def test_resolve_project_root(project_root: Path) -> None:
    assert (project_root / "README.md").exists()


@pytest.mark.parametrize(
    "rel_path",
    [
        "configs/datasets/leapgestrecog.yaml",
        "configs/features/default.yaml",
        "configs/models/baselines.yaml",
        "configs/experiments/exp01_model_comparison.yaml",
        "configs/runtime/default.yaml",
    ],
)
def test_load_starter_configs(project_root: Path, rel_path: str) -> None:
    config = load_config(str(project_root / rel_path))
    assert "_meta" in config
    assert config["_meta"]["config_type"] is not None


def test_merge_overrides_dot_keys() -> None:
    base = {"camera": {"index": 0}, "enabled": True}
    merged = merge_overrides(base, {"camera.index": "1", "enabled": "false"})
    assert merged["camera"]["index"] == 1
    assert merged["enabled"] is False


def test_build_artifact_path(project_root: Path, tmp_path: Path) -> None:
    path = build_artifact_path(
        "metrics",
        "EXP-01_test-run",
        "json",
        project_root=project_root,
    )
    assert path.parent.name == "metrics"
    assert path.suffix == ".json"


def test_run_record_roundtrip(project_root: Path) -> None:
    config = load_config(
        str(project_root / "configs/experiments/exp01_model_comparison.yaml")
    )
    record = create_run_record("EXP-01", config)
    out = save_run_record(record)
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["experiment_id"] == "EXP-01"
    assert "run_id" in loaded
