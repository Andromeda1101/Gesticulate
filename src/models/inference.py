"""Load exported model bundles and run inference on feature records."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.features.feature_store import vector_from_record
from src.models.model_registry import is_deep_algorithm

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]


def load_exported_bundle(artifact_path: str | Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """
    Load serialized model bundle and optional sidecar metadata.

    Returns (bundle, sidecar_metadata).
    """
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    sidecar_path = path.with_name(path.stem + ".meta.json")
    sidecar: dict[str, Any] | None = None
    if sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

    if path.suffix.lower() == ".pt":
        if torch is None:
            raise ImportError("PyTorch required to load .pt model artifacts")
        bundle = torch.load(path, map_location="cpu", weights_only=False)
    else:
        bundle = joblib.load(path)

    if not isinstance(bundle, dict) or "estimator" not in bundle:
        raise ValueError(f"Invalid model bundle format: {path}")
    return bundle, sidecar


def _records_to_arrays(
    records: list[dict[str, Any]],
    *,
    sample_ids: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_rows: list[np.ndarray] = []
    y_rows: list[Any] = []
    ids: list[str] = []
    for record in records:
        sid = str(record["sample_id"])
        if sample_ids is not None and sid not in sample_ids:
            continue
        X_rows.append(vector_from_record(record))
        y_rows.append(record["gesture_label"])
        ids.append(sid)
    if not X_rows:
        raise ValueError("No samples matched the provided records or split IDs")
    return np.vstack(X_rows), np.array(y_rows), ids


class _PredictOnlyDataset:
    """Torch dataset for inference without label encoding."""

    def __init__(self, X: np.ndarray, *, reshape: tuple[int, ...] | None = None) -> None:
        if torch is None:
            raise ImportError("PyTorch required")
        self.X = np.asarray(X, dtype=np.float32)
        self.reshape = reshape

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        x = self.X[idx]
        if self.reshape is not None:
            x = x.reshape(self.reshape)
        return torch.from_numpy(x.copy()), idx


def predict_on_records(
    bundle: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    sample_ids: set[str] | list[str] | None = None,
    batch_size: int = 64,
    include_proba: bool = True,
) -> dict[str, Any]:
    """
    Run inference on feature records using an exported bundle.

    Returns y_true, y_pred, y_proba (optional), sample_ids, and timing info.
    """
    id_set = set(sample_ids) if sample_ids is not None else None
    X, y_true, out_ids = _records_to_arrays(records, sample_ids=id_set)

    scaler = bundle.get("scaler")
    if scaler is not None:
        X = scaler.transform(X)

    algorithm = str(bundle.get("algorithm", ""))
    infer_start = time.perf_counter()

    if is_deep_algorithm(algorithm):
        if torch is None:
            raise ImportError("PyTorch required for deep model inference")
        from src.models.deep.trainer_utils import predict_torch_classifier

        model = bundle["estimator"]
        model.eval()
        reshape = bundle.get("reshape")
        dataset = _PredictOnlyDataset(X, reshape=reshape)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        pred_idx, y_proba = predict_torch_classifier(model, loader)
        label_to_idx = bundle.get("label_to_idx") or {}
        if label_to_idx:
            idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        else:
            classes = bundle.get("classes")
            if classes is None:
                raise ValueError("Deep bundle missing label_to_idx and classes")
            idx_to_label = {idx: label for idx, label in enumerate(classes)}
        y_pred = np.array([idx_to_label[int(i)] for i in pred_idx])
    else:
        estimator = bundle["estimator"]
        y_pred = estimator.predict(X)
        y_proba = None
        if include_proba:
            try:
                y_proba = estimator.predict_proba(X)
            except (AttributeError, NotImplementedError):
                pass

    infer_seconds = time.perf_counter() - infer_start
    n_samples = len(y_true)

    result: dict[str, Any] = {
        "y_true": y_true,
        "y_pred": y_pred,
        "sample_ids": out_ids,
        "timing": {
            "inference_seconds": infer_seconds,
            "per_sample_inference_ms": (infer_seconds / max(n_samples, 1)) * 1000.0,
        },
        "algorithm": algorithm,
    }
    if is_deep_algorithm(algorithm):
        result["y_proba"] = y_proba
    elif y_proba is not None:
        result["y_proba"] = y_proba
    return result


def predict_on_record_batches(
    bundle: dict[str, Any],
    record_batches: Iterable[list[dict[str, Any]]],
    *,
    sample_ids: set[str] | list[str] | None = None,
    batch_size: int = 256,
    include_proba: bool = False,
) -> dict[str, Any]:
    """
    Run inference over streamed record batches with bounded peak memory.

    Probability outputs are disabled by default because ``predict_proba`` on
    large SVM models can allocate multi-gigabyte arrays.
    """
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    id_parts: list[list[str]] = []
    proba_parts: list[np.ndarray] = []
    infer_start = time.perf_counter()
    algorithm = str(bundle.get("algorithm", ""))

    for records in record_batches:
        if not records:
            continue
        partial = predict_on_records(
            bundle,
            records,
            sample_ids=sample_ids,
            batch_size=batch_size,
            include_proba=include_proba,
        )
        y_true_parts.append(np.asarray(partial["y_true"]))
        y_pred_parts.append(np.asarray(partial["y_pred"]))
        id_parts.append(list(partial["sample_ids"]))
        if partial.get("y_proba") is not None:
            proba_parts.append(np.asarray(partial["y_proba"]))

    if not y_true_parts:
        raise ValueError("No samples matched the provided records or split IDs")

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    out_ids = [sid for chunk in id_parts for sid in chunk]
    infer_seconds = time.perf_counter() - infer_start
    n_samples = len(y_true)

    result: dict[str, Any] = {
        "y_true": y_true,
        "y_pred": y_pred,
        "sample_ids": out_ids,
        "timing": {
            "inference_seconds": infer_seconds,
            "per_sample_inference_ms": (infer_seconds / max(n_samples, 1)) * 1000.0,
        },
        "algorithm": algorithm,
    }
    if proba_parts:
        result["y_proba"] = np.concatenate(proba_parts)
    return result
