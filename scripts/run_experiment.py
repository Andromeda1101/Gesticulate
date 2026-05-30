#!/usr/bin/env python3
"""Run a single Phase 3 experiment (EXP-01 / EXP-02)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config_loader import load_config, merge_overrides
from src.models.experiment_runner import run_single_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one model benchmarking experiment")
    parser.add_argument("--experiment-id", required=True, help="e.g. EXP-01")
    parser.add_argument("--feature-family", required=True, help="e.g. hybrid, geometric")
    parser.add_argument("--algorithm", required=True, help="e.g. svm, random_forest, mlp")
    parser.add_argument(
        "--config",
        default="configs/experiments/exp01_model_comparison.yaml",
        help="Experiment YAML path",
    )
    parser.add_argument("--dataset-name", default="hagrid_subset")
    parser.add_argument("--feature-version", default="v1")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    experiment_config = load_config(str(config_path))
    experiment_config["experiment_id"] = args.experiment_id

    baselines_path = experiment_config.get("models", {}).get("config", "configs/models/baselines.yaml")
    baselines_config = load_config(str(PROJECT_ROOT / baselines_path))

    record = run_single_experiment(
        experiment_config,
        feature_family=args.feature_family,
        algorithm=args.algorithm,
        baselines_config=baselines_config,
        dataset_name=args.dataset_name,
        feature_version=args.feature_version,
        dry_run=args.dry_run,
    )
    print(f"Run {record.get('run_id')} status={record.get('status')}")
    if record.get("artifacts"):
        for key, path in record["artifacts"].items():
            print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
