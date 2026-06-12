"""Aggregate experiment runs into leaderboards and summary reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

METRIC_ALIASES: dict[str, str] = {
    "acc": "accuracy",
    "f1": "f1_macro",
    "recall": "recall_macro",
    "precision": "precision_macro",
}

NON_SCALAR_METRICS = frozenset({"confusion_matrix", "labels"})

DEFAULT_REPORT_METRICS: tuple[str, ...] = (
    "accuracy",
    "f1_macro",
    "recall_macro",
    "precision_macro",
    "f1_micro",
    "recall_micro",
    "precision_micro",
    "fit_seconds",
    "inference_seconds",
    "per_sample_inference_ms",
)


def normalize_metric_name(name: str) -> str:
    """Map config aliases (e.g. acc) to keys stored in run metrics JSON."""
    key = str(name).strip()
    return METRIC_ALIASES.get(key, key)


def resolve_report_metrics(metrics_config: dict[str, Any] | None) -> list[str]:
    """Return ordered scalar metric keys for leaderboard / summary tables."""
    if not metrics_config:
        return list(DEFAULT_REPORT_METRICS)

    raw: list[str] = []
    report = metrics_config.get("report")
    if report:
        raw.extend(str(m) for m in report)
    else:
        primary = metrics_config.get("primary")
        if primary:
            raw.append(str(primary))
        secondary = metrics_config.get("secondary") or []
        raw.extend(str(m) for m in secondary)

    seen: set[str] = set()
    resolved: list[str] = []
    for name in raw:
        canonical = normalize_metric_name(name)
        if canonical in NON_SCALAR_METRICS or canonical in seen:
            continue
        seen.add(canonical)
        resolved.append(canonical)
    return resolved or list(DEFAULT_REPORT_METRICS)


def resolve_primary_metric(metrics_config: dict[str, Any] | None) -> str:
    if metrics_config and metrics_config.get("primary"):
        return normalize_metric_name(str(metrics_config["primary"]))
    report = resolve_report_metrics(metrics_config)
    return report[0] if report else "accuracy"


def build_experiment_summary(
    run_records: list[dict[str, Any]],
    *,
    metrics_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate per-run metrics into a ranked summary."""
    report_metrics = resolve_report_metrics(metrics_config)
    primary = resolve_primary_metric(metrics_config)

    rows: list[dict[str, Any]] = []
    for record in run_records:
        metrics = record.get("metrics", {})
        row: dict[str, Any] = {
            "run_id": record.get("run_id"),
            "experiment_id": record.get("experiment_id"),
            "algorithm": record.get("algorithm") or record.get("config_snapshot", {}).get(
                "algorithm"
            ),
            "feature_family": record.get("feature_family"),
            "status": record.get("status"),
        }
        for key in report_metrics:
            row[key] = metrics.get(key)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return {
            "experiment_id": None,
            "n_runs": 0,
            "primary_metric": primary,
            "report_metrics": report_metrics,
            "runs": [],
            "leaderboard": [],
        }

    if primary in df.columns:
        df = df.sort_values(primary, ascending=False, na_position="last")

    return {
        "experiment_id": run_records[0].get("experiment_id") if run_records else None,
        "n_runs": len(run_records),
        "primary_metric": primary,
        "report_metrics": report_metrics,
        "runs": rows,
        "leaderboard": df.to_dict(orient="records"),
    }


def export_leaderboard(summary: dict[str, Any], output_path: str | Path) -> Path:
    """Write leaderboard CSV from *summary*."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summary.get("leaderboard", []))
    df.to_csv(path, index=False)
    return path


def format_leaderboard_markdown_table(
    summary: dict[str, Any],
    *,
    id_columns: tuple[str, ...] = ("algorithm", "feature_family"),
    float_precision: int = 4,
) -> str:
    """Render leaderboard rows as a GitHub-flavored markdown table."""
    leaderboard = summary.get("leaderboard", [])
    if not leaderboard:
        return "_No completed runs._\n"

    report_metrics = list(summary.get("report_metrics") or [])
    columns = [*id_columns, *report_metrics]

    def _cell(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.{float_precision}f}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_cell(row.get(col)) for col in columns) + " |"
        for row in leaderboard
    ]
    return "\n".join([header, separator, *body]) + "\n"


def load_confusion_matrix_csv(path: str | Path) -> tuple[list[list[int]], list[str]]:
    """Load matrix and class labels from an exported confusion CSV."""
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    labels = df.index.tolist()
    if list(df.columns) != labels:
        raise ValueError(
            "Column labels must match row labels; "
            f"rows={labels[:3]}... cols={df.columns.tolist()[:3]}..."
        )
    matrix = df.astype(int).values.tolist()
    return matrix, labels


def default_confusion_matrix_figure_path(csv_path: str | Path) -> Path:
    """Default PNG path for a confusion matrix CSV under reports/tables/."""
    csv_path = Path(csv_path)
    stem = csv_path.name
    if stem.endswith("_confusion.csv"):
        stem = stem[: -len("_confusion.csv")] + "_confusion"
    else:
        stem = csv_path.stem
    if csv_path.parent.name == "tables":
        return csv_path.parent.parent / "figures" / f"{stem}.png"
    return csv_path.with_name(f"{stem}.png")


def plot_confusion_matrix_from_csv(
    csv_path: str | Path,
    output_path: str | Path | None = None,
    *,
    title: str = "Confusion Matrix",
    dpi: int = 120,
) -> Path:
    """Render a confusion-matrix heatmap from an exported CSV."""
    csv_path = Path(csv_path)
    if output_path is None:
        output_path = default_confusion_matrix_figure_path(csv_path)
    matrix, labels = load_confusion_matrix_csv(csv_path)
    return export_confusion_matrix_figure(
        matrix,
        labels,
        output_path,
        title=title,
        dpi=dpi,
    )


def export_confusion_matrix_figure(
    confusion_matrix: list[list[int]],
    labels: list[str],
    output_path: str | Path,
    *,
    title: str = "Confusion Matrix",
    figsize: tuple[float, float] | None = None,
    dpi: int = 120,
) -> Path:
    """Save confusion matrix heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_classes = len(labels)
    if figsize is None:
        figsize = (max(8.0, n_classes * 0.35), max(6.0, n_classes * 0.3))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def export_confusion_matrix_csv(
    confusion_matrix: list[list[int]],
    labels: list[str],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    df.index.name = "true_label"
    df.to_csv(path)
    return path


def export_leaderboard_grouped_bar_figure(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    x: str = "algorithm",
    y: str = "accuracy",
    hue: str = "feature_family",
    title: str = "Leaderboard comparison",
    dpi: int = 120,
) -> Path:
    """Grouped bar chart for algorithm × feature-family metrics."""
    import matplotlib.pyplot as plt
    import numpy as np

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty or x not in df.columns or y not in df.columns or hue not in df.columns:
        path.write_text("", encoding="utf-8")
        return path

    plot_df = df.copy()
    plot_df[x] = plot_df[x].astype(str)
    plot_df[hue] = plot_df[hue].astype(str)
    categories = sorted(plot_df[x].unique())
    groups = sorted(plot_df[hue].unique())
    n_groups = len(groups)
    width = 0.8 / max(n_groups, 1)
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    fig, ax = plt.subplots(figsize=(max(10.0, len(categories) * 1.1), 6.0))
    x_pos = np.arange(len(categories))
    for idx, group in enumerate(groups):
        subset = plot_df[plot_df[hue] == group].set_index(x)
        values = [float(subset.loc[cat, y]) if cat in subset.index else 0.0 for cat in categories]
        offset = (idx - (n_groups - 1) / 2) * width
        bars = ax.bar(x_pos + offset, values, width, label=group, color=palette[idx % len(palette)])
        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.set_ylim(0.0, min(1.05, plot_df[y].max() * 1.15 + 0.05))
    ax.legend(title=hue, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def export_leaderboard_heatmap_figure(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    index: str = "algorithm",
    columns: str = "feature_family",
    values: str = "accuracy",
    title: str = "Accuracy heatmap",
    dpi: int = 120,
) -> Path:
    """Pivot-table heatmap for algorithm × feature-family."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        path.write_text("", encoding="utf-8")
        return path

    pivot = df.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(8.0, pivot.shape[1] * 1.5), max(6.0, pivot.shape[0] * 0.5)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        ax=ax,
        cbar_kws={"label": values},
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def export_multi_metric_line_figure(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    group_col: str = "algorithm",
    hue_col: str = "feature_family",
    metrics: tuple[str, ...] = ("accuracy", "f1_macro", "per_sample_inference_ms"),
    title: str = "Multi-metric profile",
    dpi: int = 120,
) -> Path:
    """Line chart comparing normalized metric profiles across configurations."""
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        path.write_text("", encoding="utf-8")
        return path

    available = [m for m in metrics if m in df.columns]
    if not available:
        path.write_text("", encoding="utf-8")
        return path

    plot_df = df.copy()
    plot_df["_label"] = (
        plot_df[group_col].astype(str) + " / " + plot_df[hue_col].astype(str)
    )
    # Rank by primary metric (accuracy if present) and keep top-N for readability.
    sort_key = "accuracy" if "accuracy" in plot_df.columns else available[0]
    top_n = min(12, len(plot_df))
    plot_df = plot_df.sort_values(sort_key, ascending=False).head(top_n)

    normalized = plot_df[available].copy()
    for col in available:
        col_min = normalized[col].min()
        col_max = normalized[col].max()
        if col_max > col_min:
            if col == "per_sample_inference_ms":
                normalized[col] = 1.0 - (normalized[col] - col_min) / (col_max - col_min)
            else:
                normalized[col] = (normalized[col] - col_min) / (col_max - col_min)
        else:
            normalized[col] = 0.5

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    x = list(range(len(available)))
    for idx, row in plot_df.iterrows():
        label = row["_label"]
        y_vals = normalized.loc[idx, available].tolist()
        ax.plot(x, y_vals, marker="o", linewidth=1.5, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(available, rotation=20, ha="right")
    ax.set_ylabel("Normalized score (higher is better)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def export_robustness_comparison_figure(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    group_col: str = "algorithm",
    hue_col: str = "feature_family",
    in_domain_col: str = "in_domain_accuracy",
    ood_col: str = "ood_accuracy",
    drop_col: str = "absolute_accuracy_drop",
    title: str = "In-domain vs OOD accuracy",
    top_n: int = 15,
    dpi: int = 120,
) -> Path:
    """Dual bar chart for in-domain/OOD accuracy with accuracy-drop overlay."""
    import matplotlib.pyplot as plt
    import numpy as np

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    required = {in_domain_col, ood_col, drop_col, group_col, hue_col}
    if df.empty or not required.issubset(df.columns):
        path.write_text("", encoding="utf-8")
        return path

    plot_df = df.copy()
    plot_df["_label"] = (
        plot_df[group_col].astype(str) + "\n" + plot_df[hue_col].astype(str)
    )
    plot_df = plot_df.sort_values(ood_col, ascending=False).head(top_n)
    labels = plot_df["_label"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(max(12.0, len(labels) * 0.75), 6.5))
    ax1.bar(
        x - width / 2,
        plot_df[in_domain_col],
        width,
        label="in-domain",
        color="#4C72B0",
    )
    ax1.bar(
        x + width / 2,
        plot_df[ood_col],
        width,
        label="OOD",
        color="#DD8452",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_title(title)
    ax1.legend(loc="upper right")

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        plot_df[drop_col],
        color="#C44E52",
        marker="D",
        linewidth=1.5,
        label="absolute drop",
    )
    ax2.set_ylabel("Absolute accuracy drop")
    ax2.set_ylim(0.0, max(0.85, float(plot_df[drop_col].max()) * 1.1))
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def load_run_records(
    metrics_dir: str | Path,
    *,
    recursive: bool = True,
) -> list[dict[str, Any]]:
    """Load completed run JSON files from a metrics directory (optionally recursive)."""
    metrics_dir = Path(metrics_dir)
    if not metrics_dir.exists():
        return []

    pattern = "**/*.json" if recursive else "*.json"
    records: list[dict[str, Any]] = []
    for path in sorted(metrics_dir.glob(pattern)):
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        if "run_id" not in data or "experiment_id" not in data:
            continue
        if data.get("status") == "completed":
            records.append(data)
    return records
