"""Stabilize frame-level predictions before keyboard dispatch."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GestureFilter:
    """Rolling-window majority vote with debounce cooldown."""

    window_size: int = 5
    min_consensus_ratio: float = 0.6
    debounce_ms: float = 300.0
    confidence_threshold: float = 0.6

    _history: deque[tuple[str, float, float]] = field(default_factory=deque, init=False)
    _last_emitted_label: str | None = field(default=None, init=False)
    _last_emit_time: float = field(default=0.0, init=False)

    def update_prediction(self, prediction: dict[str, Any], timestamp: float) -> dict[str, Any]:
        """Ingest a prediction and return smoothed filter state."""
        label = str(prediction.get("label", ""))
        confidence = float(prediction.get("confidence", 0.0))

        self._history.append((label, confidence, timestamp))
        while len(self._history) > self.window_size:
            self._history.popleft()

        labels = [entry[0] for entry in self._history]
        confidences = [entry[1] for entry in self._history]
        counts: dict[str, int] = {}
        for lbl in labels:
            counts[lbl] = counts.get(lbl, 0) + 1

        stable_label = max(counts, key=counts.get) if counts else label
        consensus_ratio = counts.get(stable_label, 0) / max(len(labels), 1)
        avg_confidence = sum(confidences) / max(len(confidences), 1)

        return {
            "raw_label": label,
            "raw_confidence": confidence,
            "stable_label": stable_label,
            "consensus_ratio": consensus_ratio,
            "avg_confidence": avg_confidence,
            "window_size": len(self._history),
            "timestamp": timestamp,
            "meets_confidence": avg_confidence >= self.confidence_threshold,
            "meets_consensus": consensus_ratio >= self.min_consensus_ratio,
        }

    def should_emit_action(self, filtered_state: dict[str, Any]) -> bool:
        """Return True when a stable gesture should trigger a key action."""
        if not filtered_state.get("meets_confidence"):
            return False
        if not filtered_state.get("meets_consensus"):
            return False

        stable_label = str(filtered_state.get("stable_label", ""))
        if not stable_label:
            return False

        timestamp = float(filtered_state.get("timestamp", 0.0))
        if stable_label == self._last_emitted_label:
            elapsed_ms = (timestamp - self._last_emit_time) * 1000.0
            if elapsed_ms < self.debounce_ms:
                return False

        return True

    def mark_emitted(self, filtered_state: dict[str, Any]) -> None:
        """Record that an action was emitted for debounce tracking."""
        self._last_emitted_label = str(filtered_state.get("stable_label", ""))
        self._last_emit_time = float(filtered_state.get("timestamp", 0.0))
