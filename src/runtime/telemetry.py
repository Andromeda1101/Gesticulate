"""Runtime latency, FPS, and benchmark summarization."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any


def record_stage_timing(
    samples: list[dict[str, Any]],
    stage_name: str,
    start_time: float,
    end_time: float,
) -> None:
    """Append one stage timing sample in milliseconds."""
    samples.append(
        {
            "stage": stage_name,
            "duration_ms": (end_time - start_time) * 1000.0,
            "timestamp": end_time,
        }
    )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def compute_runtime_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate runtime quality metrics from per-frame samples.

    Each sample should include optional keys:
    ``capture_to_prediction_ms``, ``prediction_to_dispatch_ms``,
    ``end_to_end_ms``, ``fps``.
    """
    if not samples:
        return {
            "frame_count": 0,
            "avg_fps": 0.0,
            "fps_jitter_ms": 0.0,
            "latency_ms": {},
            "event_counts": {},
        }

    def _collect(key: str) -> list[float]:
        return [float(s[key]) for s in samples if key in s and s[key] is not None]

    capture_to_pred = _collect("capture_to_prediction_ms")
    pred_to_dispatch = _collect("prediction_to_dispatch_ms")
    end_to_end = _collect("end_to_end_ms")
    fps_values = _collect("fps")
    frame_intervals = _collect("frame_interval_ms")

    emitted = sum(1 for s in samples if s.get("action_emitted"))
    suppressed = sum(1 for s in samples if s.get("action_suppressed"))
    detections = sum(1 for s in samples if s.get("detection_ok"))

    fps_jitter = 0.0
    if len(frame_intervals) >= 2:
        fps_jitter = float(statistics.pstdev(frame_intervals))

    summary: dict[str, Any] = {
        "frame_count": len(samples),
        "avg_fps": float(statistics.mean(fps_values)) if fps_values else 0.0,
        "fps_jitter_ms": fps_jitter,
        "latency_ms": {
            "capture_to_prediction_avg": float(statistics.mean(capture_to_pred)) if capture_to_pred else 0.0,
            "capture_to_prediction_p95": _percentile(capture_to_pred, 95.0),
            "prediction_to_dispatch_avg": float(statistics.mean(pred_to_dispatch)) if pred_to_dispatch else 0.0,
            "prediction_to_dispatch_p95": _percentile(pred_to_dispatch, 95.0),
            "end_to_end_avg": float(statistics.mean(end_to_end)) if end_to_end else 0.0,
            "end_to_end_p95": _percentile(end_to_end, 95.0),
        },
        "event_counts": {
            "actions_emitted": emitted,
            "actions_suppressed": suppressed,
            "detections_ok": detections,
        },
    }
    return summary


@dataclass
class RuntimeTelemetry:
    """Collect per-frame runtime metrics during a live session."""

    log_fps_interval_sec: float = 5.0
    samples: list[dict[str, Any]] = field(default_factory=list)
    stage_timings: list[dict[str, Any]] = field(default_factory=list)
    _session_start: float = field(default_factory=time.perf_counter, init=False)
    _last_frame_time: float | None = field(default=None, init=False)
    _last_fps_log_time: float = field(default=time.perf_counter, init=False)
    _frames_since_log: int = field(default=0, init=False)

    def begin_frame(self) -> float:
        return time.perf_counter()

    def end_frame(
        self,
        *,
        frame_start: float,
        preprocess_end: float | None = None,
        predict_end: float | None = None,
        dispatch_end: float | None = None,
        prediction: dict[str, Any] | None = None,
        filtered_state: dict[str, Any] | None = None,
        action_result: dict[str, Any] | None = None,
        detection_ok: bool = False,
    ) -> dict[str, Any]:
        """Record one frame's timing and return the sample dict."""
        now = dispatch_end or predict_end or preprocess_end or time.perf_counter()

        sample: dict[str, Any] = {
            "timestamp": now,
            "detection_ok": detection_ok,
        }

        if preprocess_end is not None:
            sample["capture_to_prediction_ms"] = (preprocess_end - frame_start) * 1000.0
            record_stage_timing(self.stage_timings, "preprocess", frame_start, preprocess_end)
        if predict_end is not None and preprocess_end is not None:
            sample["prediction_ms"] = (predict_end - preprocess_end) * 1000.0
            record_stage_timing(self.stage_timings, "predict", preprocess_end, predict_end)
        if dispatch_end is not None and predict_end is not None:
            sample["prediction_to_dispatch_ms"] = (dispatch_end - predict_end) * 1000.0
            record_stage_timing(self.stage_timings, "dispatch", predict_end, dispatch_end)
        if dispatch_end is not None:
            sample["end_to_end_ms"] = (dispatch_end - frame_start) * 1000.0
        elif predict_end is not None:
            sample["end_to_end_ms"] = (predict_end - frame_start) * 1000.0

        if self._last_frame_time is not None:
            interval_ms = (now - self._last_frame_time) * 1000.0
            sample["frame_interval_ms"] = interval_ms
            if interval_ms > 0:
                sample["fps"] = 1000.0 / interval_ms
        self._last_frame_time = now

        if prediction:
            sample["predicted_label"] = prediction.get("label")
            sample["confidence"] = prediction.get("confidence")
        if filtered_state:
            sample["stable_label"] = filtered_state.get("stable_label")
            sample["consensus_ratio"] = filtered_state.get("consensus_ratio")
        if action_result:
            sample["action_emitted"] = bool(action_result.get("emitted"))
            sample["action_suppressed"] = not action_result.get("emitted")
            sample["action_reason"] = action_result.get("reason")
            sample["mapped_key"] = action_result.get("mapped_key")

        self.samples.append(sample)
        self._frames_since_log += 1
        return sample

    def maybe_log_fps(self, logger: Any) -> None:
        """Log rolling FPS if interval elapsed."""
        now = time.perf_counter()
        elapsed = now - self._last_fps_log_time
        if elapsed >= self.log_fps_interval_sec and self._frames_since_log > 0:
            fps = self._frames_since_log / elapsed
            logger.info("Runtime FPS: %.1f over last %.1fs", fps, elapsed)
            self._last_fps_log_time = now
            self._frames_since_log = 0

    def summary(self) -> dict[str, Any]:
        """Return aggregated runtime summary."""
        base = compute_runtime_summary(self.samples)
        base["session_duration_sec"] = time.perf_counter() - self._session_start
        base["stage_timings_count"] = len(self.stage_timings)
        return base
