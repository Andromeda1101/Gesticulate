"""Train PyTorch MLP/CNN/LSTM baselines with a unified output bundle."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.features.feature_store import vector_from_record
from src.models.classical.scaler import StandardScaler
from src.models.deep.cnn import build_cnn
from src.models.deep.dataset import FeatureVectorDataset
from src.models.deep.feature_layout import resolve_feature_layout
from src.models.deep.lstm import build_lstm
from src.models.deep.mlp import build_mlp
from src.models.deep.trainer_utils import predict_torch_classifier, train_torch_classifier
from src.models.feature_resolver import normalize_algorithm_name
from src.models.model_registry import SCALE_ALGORITHMS

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]

_MODEL_HYPERPARAM_KEYS: dict[str, frozenset[str]] = {
    "mlp": frozenset({"hidden_dims", "dropout"}),
    "cnn": frozenset({"channels", "hidden_channels"}),
    "lstm": frozenset({"seq_len", "hidden_size", "num_layers"}),
}


def _set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_hyperparameters(algorithm: str, hyperparameters: dict[str, Any]) -> dict[str, Any]:
    """Keep only architecture kwargs accepted by build_mlp / build_cnn / build_lstm."""
    allowed = _MODEL_HYPERPARAM_KEYS[algorithm]
    return {key: value for key, value in hyperparameters.items() if key in allowed}


def _prepare_arrays(
    records: list[dict[str, Any]],
    train_ids: list[str],
    val_ids: list[str],
    *,
    scale: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[Any, int],
    list[str],
    StandardScaler | None,
]:
    train_set = set(train_ids)
    val_set = set(val_ids)
    X_train, y_train, X_val, y_val = [], [], [], []
    val_ids_out: list[str] = []
    for record in records:
        sid = str(record["sample_id"])
        vec = vector_from_record(record)
        label = record["gesture_label"]
        if sid in train_set:
            X_train.append(vec)
            y_train.append(label)
        elif sid in val_set:
            X_val.append(vec)
            y_val.append(label)
            val_ids_out.append(sid)

    if not X_train or not X_val:
        raise ValueError("Train or validation split produced zero samples")

    X_train_arr = np.vstack(X_train)
    y_train_arr = np.array(y_train)
    X_val_arr = np.vstack(X_val)
    y_val_arr = np.array(y_val)

    classes = np.unique(np.concatenate([y_train_arr, y_val_arr]))
    label_to_idx = {label: idx for idx, label in enumerate(classes)}

    scaler: StandardScaler | None = None
    if scale:
        scaler = StandardScaler()
        X_train_arr = scaler.fit_transform(X_train_arr)
        X_val_arr = scaler.transform(X_val_arr)

    return X_train_arr, y_train_arr, X_val_arr, y_val_arr, label_to_idx, val_ids_out, scaler


def train_deep_baseline(
    records: list[dict[str, Any]],
    train_ids: list[str],
    val_ids: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Train a deep baseline; output bundle matches classical_trainer."""
    if torch is None:
        raise ImportError(
            "PyTorch is required for deep baselines. Install with: pip install torch"
        )

    algorithm = normalize_algorithm_name(config["algorithm"])
    if algorithm not in {"mlp", "cnn", "lstm"}:
        raise ValueError(f"Not a deep algorithm: {algorithm}")

    hyperparameters = dict(config.get("hyperparameters", {}))
    scale = config.get("scale_features", algorithm in SCALE_ALGORITHMS)
    batch_size = int(config.get("batch_size", 64))
    epochs = int(config.get("epochs", hyperparameters.pop("epochs", 30)))
    learning_rate = float(config.get("learning_rate", hyperparameters.pop("learning_rate", 1e-3)))
    patience = int(config.get("patience", hyperparameters.pop("patience", 5)))
    random_state = int(hyperparameters.pop("random_state", 42))
    _set_random_seed(random_state)
    model_hp = _model_hyperparameters(algorithm, hyperparameters)

    X_train, y_train, X_val, y_val, label_to_idx, val_sample_ids, scaler = _prepare_arrays(
        records, train_ids, val_ids, scale=scale
    )
    classes = np.array(list(label_to_idx.keys()))
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    input_dim = X_train.shape[1]
    n_classes = len(label_to_idx)

    layout_kind, geom_dim, hog_grid = resolve_feature_layout(input_dim, records)
    reshape: tuple[int, ...] | None = None
    if algorithm == "mlp":
        model = build_mlp(input_dim, n_classes, **model_hp)
    elif algorithm == "cnn":
        model, reshape = build_cnn(
            input_dim,
            n_classes,
            geom_dim=geom_dim,
            hog_grid=hog_grid,
            layout=layout_kind,
            **model_hp,
        )
    else:
        model, reshape = build_lstm(
            input_dim,
            n_classes,
            geom_dim=geom_dim,
            hog_grid=hog_grid,
            layout=layout_kind,
            **model_hp,
        )

    train_ds = FeatureVectorDataset(X_train, y_train, label_to_idx, reshape=reshape)
    val_ds = FeatureVectorDataset(X_val, y_val, label_to_idx, reshape=reshape)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    train_info = train_torch_classifier(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        patience=patience,
    )

    infer_start = time.perf_counter()
    pred_idx, y_proba = predict_torch_classifier(model, val_loader)
    infer_seconds = time.perf_counter() - infer_start
    y_pred = np.array([idx_to_label[int(i)] for i in pred_idx])

    return {
        "model": model,
        "scaler": scaler,
        "algorithm": algorithm,
        "reshape": reshape,
        "label_to_idx": label_to_idx,
        "classes": classes,
        "y_true": y_val,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "val_sample_ids": val_sample_ids,
        "timing": {
            "fit_seconds": train_info["fit_seconds"],
            "inference_seconds": infer_seconds,
            "per_sample_inference_ms": (infer_seconds / max(len(y_val), 1)) * 1000.0,
        },
        "n_train": len(y_train),
        "n_val": len(y_val),
        "deep_train_info": train_info,
    }
