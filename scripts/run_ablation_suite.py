#!/usr/bin/env python3
"""Run EXP-02 feature-family ablation with fixed algorithm settings."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import gc
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config_loader import load_config
from src.evaluation.report_builder import build_experiment_summary, export_leaderboard
from src.features.feature_store import load_feature_matrix
from src.models.experiment_runner import run_single_experiment
from src.models.feature_resolver import resolve_feature_matrix_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature-family ablation suite (EXP-02)")
    parser.add_argument("--experiment-id", default="EXP-02")
    parser.add_argument(
        "--config",
        default="configs/experiments/exp02_feature_ablation.yaml",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help="Override algorithms; defaults to experiment config",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset stem for splits/features (default: datasets.name in experiment YAML)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_config = load_config(str(PROJECT_ROOT / args.config))
    experiment_config["experiment_id"] = args.experiment_id

    baselines_path = experiment_config.get("models", {}).get("config", "configs/models/baselines.yaml")
    baselines_config = load_config(str(PROJECT_ROOT / baselines_path))

    features_cfg = experiment_config.get("features", {})
    datasets_cfg = experiment_config.get("datasets", {})
    dataset_name = (
        args.dataset_name
        or datasets_cfg.get("name")
        or "hagrid_subset"
    )
    feature_version = features_cfg.get("feature_version", "v1")
    feature_families = features_cfg.get("feature_families", [])
    algorithms = args.algorithms or experiment_config.get("models", {}).get("algorithms", ["random_forest"])

    records = []
    for family in feature_families:
        matrix_path = resolve_feature_matrix_path(
            family,
            dataset_name=dataset_name,
            feature_version=feature_version,
            project_root=PROJECT_ROOT,
        )
        family_table = load_feature_matrix(matrix_path)
        for algo in algorithms:
            print(f"Ablation: {family} x {algo}")
            record = run_single_experiment(
                experiment_config,
                feature_family=family,
                algorithm=algo,
                baselines_config=baselines_config,
                dataset_name=dataset_name,
                feature_version=feature_version,
                preloaded_table=family_table,
            )
            records.append(record)
            gc.collect()
        del family_table
        gc.collect()

    summary = build_experiment_summary(
        records,
        metrics_config=experiment_config.get("metrics"),
    )
    out_csv = PROJECT_ROOT / "reports" / "tables" / f"{args.experiment_id.lower()}_ablation_leaderboard.csv"
    export_leaderboard(summary, out_csv)
    print(f"Ablation leaderboard: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
