"""Serialize trained models and metadata sidecars."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from src.common.path_manager import build_artifact_path


def export_model(
    estimator: Any,
    metadata: dict[str, Any],
    output_dir: str | Path | None = None,
    *,
    experiment_id: str,
    algorithm_name: str,
    feature_family: str,
    export_format: str = "joblib",
) -> dict[str, Any]:
    """
    Persist model artifact and JSON sidecar per project overview schema.

    Deep torch models are saved as ``.pt``; classical models as ``.joblib`` or pickle.
    """
    algorithm_name = metadata.get("algorithm_name", algorithm_name)
    feature_family = metadata.get("feature_family", feature_family)

    if output_dir is None:
        artifact_path = build_artifact_path(
            "models",
            f"{experiment_id}_{algorithm_name}_{feature_family}",
            "pt" if export_format == "torch" else "joblib",
        )
    else:
        ext = "pt" if export_format == "torch" else "joblib"
        artifact_path = Path(output_dir) / f"{experiment_id}_{algorithm_name}_{feature_family}.{ext}"

    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "estimator": estimator,
        "scaler": metadata.get("scaler"),
        "label_to_idx": metadata.get("label_to_idx"),
        "classes": metadata.get("classes"),
        "reshape": metadata.get("reshape"),
        "algorithm": algorithm_name,
    }

    if export_format == "torch":
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch required for torch export") from exc
        torch.save(bundle, artifact_path)
    elif export_format == "joblib":
        joblib.dump(bundle, artifact_path)
    else:
        with artifact_path.open("wb") as fh:
            pickle.dump(bundle, fh)

    sidecar = {
        "model_id": metadata.get(
            "model_id", f"{experiment_id}_{algorithm_name}_{feature_family}"
        ),
        "experiment_id": experiment_id,
        "algorithm_name": algorithm_name,
        "feature_family": feature_family,
        "feature_version": metadata.get("feature_version", "v1"),
        "train_split_id": metadata.get("train_split_id", "train_val_test"),
        "validation_strategy": metadata.get("validation_strategy", "holdout"),
        "hyperparameters": metadata.get("hyperparameters", {}),
        "metrics_summary": metadata.get("metrics_summary", {}),
        "artifact_path": str(artifact_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "vector_dim": metadata.get("vector_dim"),
        "preprocessing": {
            "scale_features": metadata.get("scale_features", False),
            "reshape": metadata.get("reshape"),
        },
    }

    sidecar_path = artifact_path.with_name(artifact_path.stem + ".meta.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")

    return {
        "artifact_path": str(artifact_path),
        "sidecar_path": str(sidecar_path),
        "metadata": sidecar,
    }
