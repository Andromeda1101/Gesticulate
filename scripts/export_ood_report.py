#!/usr/bin/env python3
"""Export Phase 4 domain-shift summary from EXP-03 metrics JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.domain_report import (
    build_domain_shift_report,
    export_domain_shift_report,
    export_per_class_drop_figure,
)
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Phase 4 robustness summary markdown")
    parser.add_argument(
        "--metrics",
        required=True,
        help="EXP-03 run metrics JSON path",
    )
    parser.add_argument(
        "--output",
        default="reports/summaries/robustness_summary.md",
    )
    parser.add_argument(
        "--figure",
        default="reports/figures/exp03_per_class_drop.png",
        help="Optional per-class drop figure path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = PROJECT_ROOT
    metrics_path = Path(args.metrics)
    if not metrics_path.is_absolute():
        metrics_path = root / metrics_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path

    record = json.loads(metrics_path.read_text(encoding="utf-8"))
    report_inputs = {
        "experiment_id": record.get("experiment_id", "EXP-03"),
        "run_id": record.get("run_id"),
        "model_artifact": record.get("artifacts", {}).get("model_path"),
        "model_metadata": {},
        "schema_validation": record.get("schema_validation", {}),
        "in_domain_metrics": record.get("metrics", {}).get("in_domain", {}),
        "ood_metrics": record.get("metrics", {}).get("ood", {}),
        "robustness": record.get("robustness") or record.get("metrics", {}).get("robustness", {}),
        "ood_eval_protocols": record.get("ood_eval_protocols")
        or record.get("metrics", {}).get("ood_eval_protocols", {}),
        "ood_domain_report": record.get("ood_domain_report")
        or {
            "per_class": record.get("ood_per_class", []),
            **(record.get("ood_canonical_metrics") or {}),
        },
        "per_class_shift": {"per_class": record.get("per_class_shift", [])},
        "error_analysis": {},
    }
    report = build_domain_shift_report(report_inputs)
    export_domain_shift_report(report, output_path)

    per_class = record.get("per_class_shift", [])
    if per_class and args.figure:
        figure_path = Path(args.figure)
        if not figure_path.is_absolute():
            figure_path = root / figure_path
        export_per_class_drop_figure(pd.DataFrame(per_class), figure_path)

    print(
        json.dumps(
            {
                "summary_path": str(output_path),
                "run_id": record.get("run_id"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
