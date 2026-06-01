#!/usr/bin/env python3
"""Run EXP-03 cross-dataset robustness evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config_loader import load_config
from src.common.path_manager import resolve_project_root
from src.evaluation.robustness_runner import run_robustness_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a champion model on in-domain test and OOD features (EXP-03)"
    )
    parser.add_argument(
        "--model-artifact",
        required=True,
        help="Path to exported model .joblib or .pt",
    )
    parser.add_argument(
        "--in-domain-features",
        required=True,
        help="In-domain feature matrix (e.g. hagrid_subset_hybrid_v1.parquet)",
    )
    parser.add_argument(
        "--ood-features",
        required=True,
        help="OOD feature matrix (e.g. leapgestrecog_hybrid_v1.parquet)",
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/exp03_robustness.yaml",
        help="Experiment YAML path",
    )
    parser.add_argument(
        "--in-domain-manifest",
        default=None,
        help="Phase 1 manifest for in-domain image paths",
    )
    parser.add_argument(
        "--ood-manifest",
        default=None,
        help="Phase 1 manifest for OOD image paths",
    )
    parser.add_argument("--dataset-name", default="hagrid_subset")
    parser.add_argument("--ood-dataset-name", default="leapgestrecog")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Parquet read / inference batch size (lower if WSL runs out of memory)",
    )
    parser.add_argument(
        "--include-proba",
        action="store_true",
        help="Compute predict_proba (uses much more memory on large SVM models)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = resolve_project_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    experiment_config = load_config(str(config_path))

    record = run_robustness_eval(
        experiment_config,
        model_artifact=args.model_artifact,
        in_domain_features=args.in_domain_features,
        ood_features=args.ood_features,
        in_domain_manifest=args.in_domain_manifest,
        ood_manifest=args.ood_manifest,
        dataset_name=args.dataset_name,
        ood_dataset_name=args.ood_dataset_name,
        batch_size=args.batch_size,
        include_proba=args.include_proba,
    )

    protocols = record.get("ood_eval_protocols") or {}
    protocol_summary: dict[str, Any] = {}
    if protocols.get("shared_subset", {}).get("ood"):
        protocol_summary["ood_shared_subset_accuracy"] = protocols["shared_subset"]["ood"].get(
            "accuracy"
        )
    if protocols.get("masked_unknown", {}).get("ood"):
        protocol_summary["ood_masked_unknown_accuracy"] = protocols["masked_unknown"]["ood"].get(
            "accuracy"
        )
    masked_argmax = protocols.get("masked_shared_argmax", {})
    if masked_argmax.get("available") and masked_argmax.get("ood"):
        protocol_summary["ood_masked_shared_argmax_accuracy"] = masked_argmax["ood"].get(
            "accuracy"
        )

    manifest = {
        "run_id": record.get("run_id"),
        "status": record.get("status"),
        "metrics_path": record.get("outputs", {}).get("metrics_path"),
        "artifacts": record.get("artifacts"),
        "robustness": record.get("robustness"),
        "ood_eval_protocols_summary": protocol_summary,
        "schema_compatible": record.get("schema_validation", {}).get("compatible"),
    }
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
