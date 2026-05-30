#!/usr/bin/env python3
"""Run multiple algorithms on the same feature family and split (EXP-01 sweep)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config_loader import load_config
from src.evaluation.report_builder import build_experiment_summary, export_leaderboard
from src.models.experiment_runner import run_single_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark multiple algorithms")
    parser.add_argument("--experiment-id", default="EXP-01")
    parser.add_argument("--feature-family", default="hybrid")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=[
            "knn",
            "svm",
            "decision_tree",
            "random_forest",
            "naive_bayes",
            "logistic_regression",
            "mlp",
            "cnn",
            "lstm",
        ],
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/exp01_model_comparison.yaml",
    )
    parser.add_argument("--dataset-name", default="hagrid_subset")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    experiment_config = load_config(str(config_path))
    experiment_config["experiment_id"] = args.experiment_id

    baselines_path = experiment_config.get("models", {}).get("config", "configs/models/baselines.yaml")
    baselines_config = load_config(str(PROJECT_ROOT / baselines_path))

    records = []
    for algo in args.algorithms:
        print(f"Running {algo} on {args.feature_family}...")
        record = run_single_experiment(
            experiment_config,
            feature_family=args.feature_family,
            algorithm=algo,
            baselines_config=baselines_config,
            dataset_name=args.dataset_name,
        )
        records.append(record)

    summary = build_experiment_summary(
        records,
        metrics_config=experiment_config.get("metrics"),
    )
    out_csv = PROJECT_ROOT / "reports" / "tables" / f"{args.experiment_id.lower()}_leaderboard.csv"
    export_leaderboard(summary, out_csv)
    print(f"Leaderboard: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
