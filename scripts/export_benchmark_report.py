#!/usr/bin/env python3
"""Export benchmark summary markdown from metrics artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.report_builder import (
    build_experiment_summary,
    export_leaderboard,
    format_leaderboard_markdown_table,
    load_run_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export benchmark summary report")
    parser.add_argument("--input-dir", default="artifacts/metrics")
    parser.add_argument(
        "--output",
        default="reports/summaries/benchmark_summary.md",
    )
    parser.add_argument("--experiment-id", default=None, help="Filter by experiment ID")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = PROJECT_ROOT / args.input_dir
    records = load_run_records(input_dir)
    if args.experiment_id:
        records = [r for r in records if r.get("experiment_id") == args.experiment_id]

    metrics_config = None
    if records and records[0].get("config_snapshot"):
        metrics_config = records[0]["config_snapshot"].get("metrics")

    summary = build_experiment_summary(records, metrics_config=metrics_config)
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Benchmark Summary",
        "",
        f"Completed runs: {summary.get('n_runs', 0)}",
        f"Primary metric: `{summary.get('primary_metric', 'accuracy')}`",
        "",
        "## Leaderboard",
        "",
        format_leaderboard_markdown_table(summary).rstrip(),
        "",
    ]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    csv_path = output_path.with_suffix(".csv")
    export_leaderboard(summary, csv_path)

    manifest = {
        "summary_path": str(output_path),
        "leaderboard_csv": str(csv_path),
        "n_runs": summary.get("n_runs"),
    }
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
