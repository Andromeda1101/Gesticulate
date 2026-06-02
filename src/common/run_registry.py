"""Experiment run metadata and manifest persistence."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.path_manager import build_metrics_record_path


def _config_fingerprint(config: dict[str, Any]) -> str:
    """Stable SHA-256 fingerprint of configuration (excluding _meta)."""
    payload = {k: v for k, v in config.items() if k != "_meta"}
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def create_run_record(
    experiment_id: str,
    config: dict[str, Any],
    *,
    status: str = "initialized",
    artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Generate a run manifest with unique run ID and config snapshot."""
    run_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    record: dict[str, Any] = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "status": status,
        "created_at": now,
        "updated_at": now,
        "config_fingerprint": _config_fingerprint(config),
        "config_snapshot": config,
        "artifacts": artifacts or {},
        "inputs": {
            "datasets": config.get("datasets"),
            "features": config.get("features"),
            "models": config.get("models"),
        },
        "outputs": {
            "metrics_path": str(
                build_metrics_record_path(
                    experiment_id,
                    run_id,
                    config=config,
                    create_parents=False,
                )
            ),
        },
    }
    return record


def save_run_record(record: dict[str, Any], output_path: str | None = None) -> Path:
    """Persist *record* as JSON; default path follows artifact naming convention."""
    if output_path is None:
        experiment_id = record["experiment_id"]
        run_id = record["run_id"]
        config = record.get("config_snapshot")
        path = build_metrics_record_path(
            experiment_id,
            run_id,
            config=config,
            create_parents=True,
        )
    else:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

    record["updated_at"] = datetime.now(timezone.utc).isoformat()

    with path.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, default=str)

    return path
