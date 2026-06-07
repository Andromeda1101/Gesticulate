"""Persist runtime events and session summaries."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log_runtime_event(event: dict[str, Any], output_path: str | Path) -> None:
    """Append one JSON event line to a session log file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    if "timestamp_iso" not in payload:
        payload["timestamp_iso"] = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")


class SessionLogger:
    """Buffered runtime event logger with summary export."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.events: list[dict[str, Any]] = []

    def log_event(self, event: dict[str, Any]) -> None:
        """Append event to memory and JSONL file."""
        payload = dict(event)
        payload.setdefault("timestamp_iso", datetime.now(timezone.utc).isoformat())
        self.events.append(payload)
        log_runtime_event(payload, self.output_path)

    def write_summary(self, summary: dict[str, Any], summary_path: str | Path | None = None) -> Path:
        """Write session summary JSON alongside the event log."""
        out = Path(summary_path) if summary_path else self.output_path.with_suffix(".summary.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_log": str(self.output_path),
            "event_count": len(self.events),
            "summary": summary,
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return out
