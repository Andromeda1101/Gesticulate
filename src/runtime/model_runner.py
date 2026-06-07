"""Load champion model artifacts and run single-vector inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.features.feature_store import vector_from_record
from src.models.inference import load_exported_bundle
from src.models.model_registry import is_deep_algorithm

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass
class RuntimeModel:
    """Wrapper around an exported model bundle for frame-level inference."""

    bundle: dict[str, Any]
    metadata: dict[str, Any] | None = None
    artifact_path: str = ""
    _idx_to_label: dict[int, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        label_to_idx = self.bundle.get("label_to_idx") or {}
        if label_to_idx:
            self._idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        else:
            classes = self.bundle.get("classes")
            if classes is not None:
                self._idx_to_label = {idx: str(label) for idx, label in enumerate(classes)}

    @property
    def algorithm(self) -> str:
        return str(self.bundle.get("algorithm", ""))

    @property
    def feature_family(self) -> str | None:
        if self.metadata:
            return self.metadata.get("feature_family")
        return None

    def _prepare_vector(self, feature_vector: np.ndarray) -> np.ndarray:
        x = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
        scaler = self.bundle.get("scaler")
        if scaler is not None:
            x = scaler.transform(x)
        return x

    def predict_gesture(self, feature_vector: np.ndarray) -> dict[str, Any]:
        """Predict gesture label and confidence for one feature vector."""
        x = self._prepare_vector(feature_vector)
        algorithm = self.algorithm

        if is_deep_algorithm(algorithm):
            if torch is None:
                raise ImportError("PyTorch required for deep model runtime inference")
            model = self.bundle["estimator"]
            model.eval()
            reshape = self.bundle.get("reshape")
            tensor_x = x.astype(np.float32)
            if reshape is not None:
                tensor_x = tensor_x.reshape(1, *reshape)
            with torch.no_grad():
                logits = model(torch.from_numpy(tensor_x))
                if hasattr(logits, "cpu"):
                    logits = logits.cpu().numpy()
                else:
                    logits = np.asarray(logits)
            if logits.ndim == 2:
                scores = logits[0]
            else:
                scores = logits.reshape(-1)
            if scores.min() < 0 or np.abs(scores.sum() - 1.0) > 0.01:
                exp_scores = np.exp(scores - scores.max())
                proba = exp_scores / exp_scores.sum()
            else:
                proba = scores
            pred_idx = int(np.argmax(proba))
            label = self._idx_to_label.get(pred_idx, str(pred_idx))
            confidence = float(proba[pred_idx])
            return {
                "label": label,
                "confidence": confidence,
                "scores": proba.tolist(),
                "model_metadata": self._summary_metadata(),
            }

        estimator = self.bundle["estimator"]
        pred = estimator.predict(x)[0]
        label = str(pred)
        confidence = 1.0
        scores: list[float] | None = None

        try:
            proba = estimator.predict_proba(x)[0]
            scores = proba.tolist()
            classes = list(estimator.classes_)
            if label in classes:
                confidence = float(proba[classes.index(label)])
            else:
                confidence = float(np.max(proba))
        except (AttributeError, NotImplementedError):
            pass

        return {
            "label": label,
            "confidence": confidence,
            "scores": scores,
            "model_metadata": self._summary_metadata(),
        }

    def predict_from_record(self, feature_record: dict[str, Any]) -> dict[str, Any]:
        """Predict from a runtime feature record dict."""
        vector = vector_from_record(feature_record)
        result = self.predict_gesture(vector)
        result["extraction_ok"] = bool(feature_record.get("extraction_ok", True))
        result["quality_flags"] = dict(feature_record.get("quality_flags") or {})
        return result

    def _summary_metadata(self) -> dict[str, Any]:
        if not self.metadata:
            return {"artifact_path": self.artifact_path, "algorithm": self.algorithm}
        return {
            "model_id": self.metadata.get("model_id"),
            "experiment_id": self.metadata.get("experiment_id"),
            "algorithm_name": self.metadata.get("algorithm_name"),
            "feature_family": self.metadata.get("feature_family"),
            "feature_version": self.metadata.get("feature_version"),
            "artifact_path": self.artifact_path,
        }


def load_runtime_model(
    model_path: str | Path,
    metadata_path: str | Path | None = None,
) -> RuntimeModel:
    """Load serialized champion model and optional sidecar metadata."""
    path = Path(model_path)
    bundle, sidecar = load_exported_bundle(path)

    if metadata_path is not None:
        meta_file = Path(metadata_path)
        if meta_file.is_file():
            import json

            sidecar = json.loads(meta_file.read_text(encoding="utf-8"))

    return RuntimeModel(bundle=bundle, metadata=sidecar, artifact_path=str(path.resolve()))


def predict_gesture(runtime_model: RuntimeModel, feature_vector: np.ndarray) -> dict[str, Any]:
    """Module-level alias for ``RuntimeModel.predict_gesture``."""
    return runtime_model.predict_gesture(feature_vector)
