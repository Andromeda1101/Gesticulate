#!/usr/bin/env python3
"""Plot experiment leaderboard tables as bar charts, heatmaps, and line charts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.report_builder import (
    build_experiment_summary,
    export_leaderboard_grouped_bar_figure,
    export_leaderboard_heatmap_figure,
    export_multi_metric_line_figure,
    export_robustness_comparison_figure,
    load_run_records,
)

CHART_EXPORTERS = {
    "grouped_bar": export_leaderboard_grouped_bar_figure,
    "heatmap": export_leaderboard_heatmap_figure,
    "multi_metric_line": export_multi_metric_line_figure,
    "robustness": export_robustness_comparison_figure,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize experiment leaderboard CSV files",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Leaderboard CSV path (or metrics directory when --from-metrics is set)",
    )
    parser.add_argument(
        "--chart",
        choices=sorted(CHART_EXPORTERS),
        default="grouped_bar",
        help="Chart type to render",
    )
    parser.add_argument("--x", default="algorithm", help="X-axis / index column")
    parser.add_argument("--y", default="accuracy", help="Y-axis / values column")
    parser.add_argument("--hue", default="feature_family", help="Grouping / columns column")
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path (default: reports/figures/<stem>_<chart>.png)",
    )
    parser.add_argument("--title", default=None, help="Figure title")
    parser.add_argument("--dpi", type=int, default=120, help="Figure DPI")
    parser.add_argument(
        "--from-metrics",
        action="store_true",
        help="Build leaderboard from artifacts/metrics JSON instead of CSV",
    )
    parser.add_argument(
        "--preset",
        choices=("exp01_exp02", "exp03", "all"),
        default=None,
        help="Generate standard figure set for bundled experiment tables",
    )
    return parser.parse_args()


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def _default_output(input_path: Path, chart: str) -> Path:
    stem = input_path.stem
    return PROJECT_ROOT / "reports" / "figures" / f"{stem}_{chart}.png"


def _load_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    input_path = _resolve_path(args.input)
    if args.from_metrics:
        records = load_run_records(input_path)
        summary = build_experiment_summary(records)
        return pd.DataFrame(summary.get("leaderboard", []))
    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")
    return pd.read_csv(input_path)


def _plot_single(args: argparse.Namespace) -> Path:
    df = _load_dataframe(args)
    input_path = _resolve_path(args.input)
    output_path = _resolve_path(args.output) if args.output else _default_output(input_path, args.chart)
    title = args.title or f"{args.y} by {args.x} ({args.chart})"

    if args.chart == "grouped_bar":
        return export_leaderboard_grouped_bar_figure(
            df,
            output_path,
            x=args.x,
            y=args.y,
            hue=args.hue,
            title=title,
            dpi=args.dpi,
        )
    if args.chart == "heatmap":
        return export_leaderboard_heatmap_figure(
            df,
            output_path,
            index=args.x,
            columns=args.hue,
            values=args.y,
            title=title,
            dpi=args.dpi,
        )
    if args.chart == "multi_metric_line":
        metrics = tuple(
            m.strip()
            for m in args.y.split(",")
            if m.strip()
        ) or ("accuracy", "f1_macro", "per_sample_inference_ms")
        return export_multi_metric_line_figure(
            df,
            output_path,
            group_col=args.x,
            hue_col=args.hue,
            metrics=metrics,
            title=title,
            dpi=args.dpi,
        )
    return export_robustness_comparison_figure(
        df,
        output_path,
        group_col=args.x,
        hue_col=args.hue,
        title=title,
        dpi=args.dpi,
    )


def _run_presets(preset: str, dpi: int) -> list[dict[str, str]]:
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    exp12_csv = PROJECT_ROOT / "reports" / "tables" / "exp01_exp02_leaderboard.csv"
    exp03_csv = PROJECT_ROOT / "reports" / "tables" / "exp03_robustness_suite_leaderboard.csv"
    outputs: list[dict[str, str]] = []

    if preset in ("exp01_exp02", "all"):
        df12 = pd.read_csv(exp12_csv)
        outputs.extend(
            [
                {
                    "chart": "grouped_bar",
                    "output": str(
                        export_leaderboard_grouped_bar_figure(
                            df12,
                            figures_dir / "exp01_exp02_accuracy_grouped_bar.png",
                            title="EXP-01×EXP-02: Validation accuracy by algorithm and feature family",
                            dpi=dpi,
                        )
                    ),
                },
                {
                    "chart": "heatmap",
                    "output": str(
                        export_leaderboard_heatmap_figure(
                            df12,
                            figures_dir / "exp01_exp02_accuracy_heatmap.png",
                            title="EXP-01×EXP-02: Accuracy heatmap (algorithm × feature family)",
                            dpi=dpi,
                        )
                    ),
                },
                {
                    "chart": "multi_metric_line",
                    "output": str(
                        export_multi_metric_line_figure(
                            df12,
                            figures_dir / "exp01_exp02_top12_multi_metric_line.png",
                            metrics=("accuracy", "f1_macro", "per_sample_inference_ms"),
                            title="EXP-01×EXP-02: Top-12 configs multi-metric profile (normalized)",
                            dpi=dpi,
                        )
                    ),
                },
            ]
        )

    if preset in ("exp03", "all"):
        df03 = pd.read_csv(exp03_csv)
        outputs.extend(
            [
                {
                    "chart": "robustness",
                    "output": str(
                        export_robustness_comparison_figure(
                            df03,
                            figures_dir / "exp03_robustness_comparison.png",
                            title="EXP-03: In-domain vs OOD accuracy (top-15 by OOD)",
                            dpi=dpi,
                        )
                    ),
                },
                {
                    "chart": "grouped_bar",
                    "output": str(
                        export_leaderboard_grouped_bar_figure(
                            df03,
                            figures_dir / "exp03_ood_accuracy_grouped_bar.png",
                            y="ood_accuracy",
                            title="EXP-03: OOD accuracy by algorithm and feature family",
                            dpi=dpi,
                        )
                    ),
                },
            ]
        )

    return outputs


def main() -> int:
    args = parse_args()
    try:
        if args.preset:
            manifest = {"preset": args.preset, "figures": _run_presets(args.preset, args.dpi)}
            print(json.dumps(manifest, indent=2))
            return 0

        saved = _plot_single(args)
        print(
            json.dumps(
                {
                    "input": str(_resolve_path(args.input)),
                    "chart": args.chart,
                    "output_png": str(saved),
                },
                indent=2,
            )
        )
        return 0
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
