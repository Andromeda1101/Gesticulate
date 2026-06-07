"""Deployment-oriented domain-shift report assembly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _deployment_recommendation(robustness: dict[str, Any], schema: dict[str, Any]) -> str:
    retention = float(robustness.get("relative_performance_retention", 0.0))
    abs_drop = float(robustness.get("absolute_accuracy_drop", 0.0))
    compatible = schema.get("compatible", True)

    if not compatible:
        return (
            "Do not deploy without resolving schema or label mismatches between "
            "training and OOD feature stores."
        )
    if retention >= 0.85 and abs_drop <= 0.10:
        return (
            "Champion model shows acceptable cross-dataset retention for a first "
            "deployment; monitor live webcam performance and failure classes in production."
        )
    if retention >= 0.70:
        return (
            "Moderate domain shift detected. Deploy with caution: add confidence "
            "thresholding, gesture debouncing, and targeted mitigation for worst per-class drops."
        )
    return (
        "Large OOD accuracy drop. Not recommended for deployment without mitigation "
        "(retraining with OOD samples, domain adaptation, or restricting to stable gesture classes)."
    )


def build_domain_shift_report(run_inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Assemble a domain-shift report from robustness run inputs.

    *run_inputs* should include in_domain_metrics, ood_metrics, robustness,
    per_class_shift, schema_validation, model metadata, and optional error_analysis.
    """
    id_metrics = run_inputs.get("in_domain_metrics", {})
    ood_metrics = run_inputs.get("ood_metrics", {})
    robustness = run_inputs.get("robustness", {})
    schema = run_inputs.get("schema_validation", {})
    per_class = run_inputs.get("per_class_shift", {})
    model_meta = run_inputs.get("model_metadata", {})

    report = {
        "experiment_id": run_inputs.get("experiment_id", "EXP-03"),
        "run_id": run_inputs.get("run_id"),
        "model_artifact": run_inputs.get("model_artifact"),
        "model_metadata": model_meta,
        "assumptions": [
            "Zero-shot transfer: model trained on HaGRID subset, evaluated on LeapGestRecog without fine-tuning.",
            "Same hybrid feature family and extraction config required for both domains.",
            "Gesture labels normalized to HaGRID-native names at manifest time.",
        ],
        "schema_validation": schema,
        "in_domain_metrics": id_metrics,
        "ood_metrics": ood_metrics,
        "robustness": robustness,
        "ood_eval_protocols": run_inputs.get("ood_eval_protocols", {}),
        "ood_domain_report": run_inputs.get("ood_domain_report", {}),
        "per_class_shift": per_class.get("per_class", []),
        "misclassification_concentration": run_inputs.get("misclassification_concentration", {}),
        "error_analysis": run_inputs.get("error_analysis", {}),
        "deployment_recommendation": _deployment_recommendation(robustness, schema),
    }
    return report


def export_ood_per_class_accuracy_figure(
    per_class_df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "OOD per-class accuracy",
    dpi: int = 120,
) -> Path:
    """Bar chart of per-class accuracy on the OOD HaGRID label vocabulary."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if per_class_df.empty:
        path.write_text("", encoding="utf-8")
        return path

    df = per_class_df.copy()
    df = df.sort_values("accuracy", ascending=True, na_position="first")
    labels = df["gesture_label"].astype(str).tolist()
    accuracies = df["accuracy"].fillna(0.0).tolist()

    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 0.55), 6.0))
    ax.barh(labels, accuracies, color="#4C72B0")
    ax.set_xlabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.05)
    for i, (acc, n) in enumerate(zip(accuracies, df["n_samples"].tolist())):
        ax.text(min(acc + 0.02, 1.0), i, f"n={n}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def export_per_class_drop_figure(
    per_class_df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "Per-class accuracy drop (in-domain vs OOD)",
    dpi: int = 120,
) -> Path:
    """Bar chart of per-class in-domain vs OOD accuracy."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if per_class_df.empty:
        path.write_text("", encoding="utf-8")
        return path

    df = per_class_df.copy()
    df = df.sort_values("absolute_drop", ascending=False, na_position="last")
    labels = df["gesture_label"].astype(str).tolist()
    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 0.5), 6.0))
    ax.bar(
        [i - width / 2 for i in x],
        df["in_domain_accuracy"],
        width,
        label="in-domain",
    )
    ax.bar(
        [i + width / 2 for i in x],
        df["ood_accuracy"],
        width,
        label="OOD",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def format_domain_shift_markdown(report: dict[str, Any]) -> str:
    """Render domain-shift report as markdown."""
    robustness = report.get("robustness", {})
    id_m = report.get("in_domain_metrics", {})
    ood_m = report.get("ood_metrics", {})
    schema = report.get("schema_validation", {})

    lines = [
        "# Phase 4 Robustness Summary (EXP-03)",
        "",
        f"**Run ID:** `{report.get('run_id', '')}`",
        f"**Model artifact:** `{report.get('model_artifact', '')}`",
        "",
        "## Domain metrics",
        "",
        "| Domain | Accuracy | F1 macro |",
        "| --- | ---: | ---: |",
        f"| In-domain (test) | {id_m.get('accuracy', 0):.4f} | {id_m.get('f1_macro', 0):.4f} |",
        f"| OOD (LeapGestRecog) | {ood_m.get('accuracy', 0):.4f} | {ood_m.get('f1_macro', 0):.4f} |",
        "",
        "## Robustness",
        "",
        f"- Absolute accuracy drop: **{robustness.get('absolute_accuracy_drop', 0):.4f}**",
        f"- Relative performance retention: **{robustness.get('relative_performance_retention', 0):.4f}**",
        "",
        "## Schema compatibility",
        "",
        f"- Compatible: **{schema.get('compatible', False)}**",
    ]
    if schema.get("issues"):
        lines.append("- Issues:")
        for issue in schema["issues"]:
            lines.append(f"  - {issue}")
    overlap = schema.get("label_overlap", {})
    if overlap.get("shared"):
        lines.append(f"- Shared labels ({overlap.get('n_shared', 0)}): {', '.join(overlap['shared'][:15])}")
    if overlap.get("test_only"):
        lines.append(f"- OOD-only labels: {', '.join(overlap['test_only'][:10])}")

    ood_domain = report.get("ood_domain_report") or {}
    per_class_ood = ood_domain.get("per_class") or []
    if per_class_ood:
        lines.extend(
            [
                "",
                "## OOD per-class accuracy (HaGRID label vocabulary)",
                "",
                "| Gesture | N | Correct | Accuracy |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in sorted(per_class_ood, key=lambda r: float(r.get("accuracy") or 0.0)):
            lines.append(
                f"| {row.get('gesture_label')} | {row.get('n_samples', 0)} | "
                f"{row.get('n_correct', 0)} | {float(row.get('accuracy', 0.0)):.4f} |"
            )
        if ood_domain.get("n_out_of_vocab_predictions") is not None:
            lines.append(
                f"\nPredictions mapped to `_other_` in the confusion matrix: "
                f"**{ood_domain.get('n_out_of_vocab_predictions')}** "
                f"({float(ood_domain.get('out_of_vocab_prediction_rate', 0.0)):.1%} of OOD samples)."
            )

    protocols = report.get("ood_eval_protocols") or {}
    if protocols:
        lines.extend(
            [
                "",
                "## OOD evaluation protocols (restricted label space)",
                "",
                "These metrics address label-space mismatch when the HaGRID classifier "
                "predicts classes outside the OOD evaluation vocabulary.",
                "",
            ]
        )
        shared_block = protocols.get("shared_subset", {})
        if shared_block.get("ood"):
            ood_s = shared_block["ood"]
            lines.append(
                f"- **Shared-class subset** (true label in 7 shared classes, n={ood_s.get('n_samples', 0)}): "
                f"OOD accuracy **{ood_s.get('accuracy', 0):.4f}**, F1 macro **{ood_s.get('f1_macro', 0):.4f}**"
            )
            rob = shared_block.get("robustness", {})
            if rob:
                lines.append(
                    f"  - vs in-domain on same subset: retention **{rob.get('relative_performance_retention', 0):.4f}**"
                )
        masked_u = protocols.get("masked_unknown", {}).get("ood", {})
        if masked_u:
            lines.append(
                f"- **Masked unknown** (out-of-vocab predictions → `unknown`): "
                f"OOD accuracy **{masked_u.get('accuracy', 0):.4f}**, "
                f"unknown rate **{masked_u.get('unknown_rate', 0):.4f}**"
            )
        masked_a = protocols.get("masked_shared_argmax", {})
        if masked_a.get("available") and masked_a.get("ood"):
            ood_a = masked_a["ood"]
            lines.append(
                f"- **Masked shared argmax** (decision only among shared classes): "
                f"OOD accuracy **{ood_a.get('accuracy', 0):.4f}**, F1 macro **{ood_a.get('f1_macro', 0):.4f}**"
            )
            sub_a = masked_a.get("shared_subset", {})
            if sub_a.get("ood"):
                lines.append(
                    f"  - shared subset: OOD accuracy **{sub_a['ood'].get('accuracy', 0):.4f}**"
                )
        elif masked_a.get("available") is False:
            lines.append(f"- **Masked shared argmax**: not available ({masked_a.get('reason', '')})")

    lines.extend(
        [
            "",
            "## Assumptions",
            "",
        ]
    )
    for assumption in report.get("assumptions", []):
        lines.append(f"- {assumption}")

    lines.extend(
        [
            "",
            "## Deployment recommendation",
            "",
            report.get("deployment_recommendation", ""),
            "",
            "## Per-class drop (top losses)",
            "",
        ]
    )
    per_class = report.get("per_class_shift", [])
    if per_class:
        sorted_pc = sorted(
            per_class,
            key=lambda r: float(r.get("absolute_drop") or 0),
            reverse=True,
        )[:10]
        lines.append("| Gesture | In-domain | OOD | Drop |")
        lines.append("| --- | ---: | ---: | ---: |")
        for row in sorted_pc:
            lines.append(
                f"| {row.get('gesture_label')} | "
                f"{row.get('in_domain_accuracy', float('nan')):.4f} | "
                f"{row.get('ood_accuracy', float('nan')):.4f} | "
                f"{row.get('absolute_drop', float('nan')):.4f} |"
            )
    else:
        lines.append("_No per-class data._")

    lines.append("")
    return "\n".join(lines)


def export_domain_shift_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write markdown summary for Phase 4."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_domain_shift_markdown(report), encoding="utf-8")
    return path


def export_domain_shift_json(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    return path
