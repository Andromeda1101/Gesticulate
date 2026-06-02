#!/usr/bin/env python3
"""Run EXP-03 OOD evaluation for all configured feature × model combinations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config_loader import load_config
from src.common.path_manager import resolve_project_root
from src.evaluation.robustness_runner import run_robustness_eval
from src.evaluation.robustness_suite import (
    build_robustness_suite_summary,
    iter_robustness_suite_specs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch EXP-03 robustness evaluation (feature family × model)"
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/exp03_robustness.yaml",
        help="Experiment YAML path",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help="Override algorithms from robustness_suite / models config",
    )
    parser.add_argument(
        "--feature-families",
        nargs="+",
        default=None,
        help="Override feature families from robustness_suite config",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="In-domain dataset stem (default: robustness_suite.dataset_name)",
    )
    parser.add_argument(
        "--ood-dataset-name",
        default=None,
        help="OOD dataset stem (default: robustness_suite.ood_dataset_name)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Parquet read / inference batch size",
    )
    parser.add_argument(
        "--include-proba",
        action="store_true",
        help="Compute predict_proba for masked shared-class argmax protocol",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip combinations whose model or feature matrix is missing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = resolve_project_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    experiment_config = load_config(str(config_path))
    suite_cfg = experiment_config.setdefault("robustness_suite", {})
    if args.dataset_name:
        suite_cfg["dataset_name"] = args.dataset_name
    if args.ood_dataset_name:
        suite_cfg["ood_dataset_name"] = args.ood_dataset_name

    records: list[dict] = []
    errors: list[str] = []

    for spec in iter_robustness_suite_specs(
        experiment_config,
        project_root=root,
        algorithms=args.algorithms,
        feature_families=args.feature_families,
    ):
        label = f"{spec.feature_family} x {spec.algorithm}"
        missing = [
            p
            for p in (spec.model_artifact, spec.in_domain_features, spec.ood_features)
            if not p.exists()
        ]
        if missing:
            msg = f"{label}: missing {', '.join(str(p) for p in missing)}"
            if args.skip_missing:
                print(f"SKIP {msg}")
                errors.append(msg)
                continue
            raise FileNotFoundError(msg)

        print(f"OOD eval: {label}")
        record = run_robustness_eval(
            experiment_config,
            model_artifact=spec.model_artifact,
            in_domain_features=spec.in_domain_features,
            ood_features=spec.ood_features,
            dataset_name=suite_cfg.get("dataset_name", "hagrid_subset"),
            ood_dataset_name=suite_cfg.get("ood_dataset_name", "leapgestrecog"),
            feature_family=spec.feature_family,
            batch_size=args.batch_size,
            include_proba=args.include_proba,
        )
        records.append(record)

    summary = build_robustness_suite_summary(records)
    out_csv = root / "reports" / "tables" / "exp03_robustness_suite_leaderboard.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if summary.get("dataframe") is not None and not summary["dataframe"].empty:
        summary["dataframe"].to_csv(out_csv, index=False)
        print(f"Suite leaderboard: {out_csv}")

    manifest = {
        "n_completed": summary.get("n_runs", 0),
        "n_skipped_or_failed": len(errors),
        "leaderboard_csv": str(out_csv) if out_csv.exists() else None,
        "runs": [
            {
                "run_id": r.get("run_id"),
                "algorithm": r.get("algorithm"),
                "feature_family": r.get("feature_family"),
                "metrics_path": r.get("outputs", {}).get("metrics_path"),
            }
            for r in records
        ],
    }
    print(json.dumps(manifest, indent=2))
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
