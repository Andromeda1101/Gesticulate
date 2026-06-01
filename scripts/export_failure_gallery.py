#!/usr/bin/env python3
"""Export a human-review index of OOD misclassifications."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.evaluation.error_analysis import group_errors_by_context, sample_failure_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export OOD failure-case gallery")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Predictions CSV (combined or OOD-only)",
    )
    parser.add_argument(
        "--output",
        default="reports/summaries/exp03_failure_gallery.md",
        help="Markdown output path",
    )
    parser.add_argument("--n-per-class", type=int, default=3)
    parser.add_argument(
        "--domain",
        default="ood",
        help="Filter predictions to this domain (default: ood)",
    )
    return parser.parse_args()


def format_failure_gallery_markdown(
    samples: pd.DataFrame,
    error_summary: dict,
) -> str:
    lines = [
        "# EXP-03 Failure Gallery",
        "",
        f"Total OOD errors: **{error_summary.get('n_errors', 0)}**",
        "",
    ]
    if error_summary.get("confusion_pairs"):
        lines.extend(["## Top confusion pairs", ""])
        lines.append("| True | Predicted | Count |")
        lines.append("| --- | --- | ---: |")
        for row in error_summary["confusion_pairs"][:15]:
            lines.append(
                f"| {row.get('true_label')} | {row.get('predicted_label')} | {row.get('count')} |"
            )
        lines.append("")

    if samples.empty:
        lines.append("_No failure samples to display._")
        return "\n".join(lines) + "\n"

    lines.extend(["## Representative failures", ""])
    lines.append("| Sample ID | True | Predicted | Confidence | Image path |")
    lines.append("| --- | --- | --- | ---: | --- |")
    for _, row in samples.iterrows():
        conf = row.get("confidence")
        conf_s = f"{float(conf):.4f}" if conf is not None and pd.notna(conf) else ""
        img = str(row.get("image_path", "")).replace("|", "\\|")
        lines.append(
            f"| {row.get('sample_id')} | {row.get('true_label')} | "
            f"{row.get('predicted_label')} | {conf_s} | `{img}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = PROJECT_ROOT
    pred_path = Path(args.predictions)
    if not pred_path.is_absolute():
        pred_path = root / pred_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path

    df = pd.read_csv(pred_path)
    if args.domain and "domain" in df.columns:
        domain_df = df[df["domain"] == args.domain]
    else:
        domain_df = df

    error_summary = group_errors_by_context(domain_df)
    samples = sample_failure_cases(domain_df, n_per_class=args.n_per_class, domain=None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = format_failure_gallery_markdown(samples, error_summary)
    output_path.write_text(markdown, encoding="utf-8")

    csv_path = output_path.with_suffix(".csv")
    if not samples.empty:
        samples.to_csv(csv_path, index=False)

    print(
        json.dumps(
            {
                "gallery_md": str(output_path),
                "gallery_csv": str(csv_path) if not samples.empty else None,
                "n_samples": len(samples),
                "n_errors": error_summary.get("n_errors"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
