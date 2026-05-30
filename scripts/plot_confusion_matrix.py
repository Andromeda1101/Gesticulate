#!/usr/bin/env python3
"""Render a confusion-matrix heatmap from an exported CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.report_builder import (
    default_confusion_matrix_figure_path,
    plot_confusion_matrix_from_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a confusion matrix heatmap from a CSV file",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Confusion matrix CSV (true_label index, predicted class columns)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path (default: reports/figures/<stem>.png)",
    )
    parser.add_argument("--title", default="Confusion Matrix", help="Figure title")
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Figure resolution for the saved PNG",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not input_path.is_file():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else None
    if output_path is not None and not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    saved = plot_confusion_matrix_from_csv(
        input_path,
        output_path,
        title=args.title,
        dpi=args.dpi,
    )

    manifest = {
        "input_csv": str(input_path),
        "output_png": str(saved),
        "default_output": str(default_confusion_matrix_figure_path(input_path)),
    }
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
