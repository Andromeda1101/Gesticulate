"""Shared real-time inference loop for demo and benchmark entrypoints."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from src.common.config_loader import load_config, merge_overrides
from src.common.logger import get_logger
from src.common.path_manager import build_artifact_path, resolve_project_root
from src.common.run_registry import create_run_record, save_run_record
from src.runtime import camera_stream
from src.runtime.gesture_filter import GestureFilter
from src.runtime.key_mapper import dispatch_key_action, load_keymap
from src.runtime.model_runner import load_runtime_model
from src.runtime.preprocess import extract_runtime_features, prepare_frame
from src.runtime.session_logger import SessionLogger
from src.runtime.telemetry import RuntimeTelemetry

logger = get_logger("src.runtime.pipeline")


@dataclass
class RuntimeSessionConfig:
    model_path: Path
    runtime_config_path: Path
    feature_config_path: Path
    camera_index: int = 0
    dry_run: bool = True
    enable_key_dispatch: bool = False
    show_overlay: bool = False
    duration_seconds: float | None = None
    session_log_path: Path | None = None
    summary_output_path: Path | None = None


def _resolve_path(root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def build_session_config(
    *,
    model: str,
    runtime_config: str,
    feature_config: str = "configs/features/default.yaml",
    camera_index: int | None = None,
    dry_run: bool = True,
    enable_key_dispatch: bool = False,
    show_overlay: bool = False,
    duration_seconds: float | None = None,
    output: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> tuple[RuntimeSessionConfig, dict[str, Any], dict[str, Any]]:
    """Load configs and assemble a runtime session configuration."""
    root = resolve_project_root()
    runtime_cfg = load_config(str(_resolve_path(root, runtime_config)))
    if cli_overrides:
        runtime_cfg = merge_overrides(runtime_cfg, cli_overrides)

    feature_cfg_path = runtime_cfg.get("features", {}).get("config", feature_config)
    feature_cfg = load_config(str(_resolve_path(root, feature_cfg_path)))

    model_path = Path(model) if model else None
    if model_path is None or str(model_path) == "" or str(model_path).lower() == "none":
        champion = runtime_cfg.get("inference", {}).get("champion_model_path")
        if not champion:
            raise ValueError("Model path required via --model or inference.champion_model_path")
        model_path = _resolve_path(root, champion)
    else:
        model_path = _resolve_path(root, model_path)

    cam_idx = camera_index
    if cam_idx is None:
        cam_idx = int(runtime_cfg.get("camera", {}).get("index", 0))

    dispatch_cfg = runtime_cfg.get("dispatch", {})
    effective_dry_run = dry_run if enable_key_dispatch is False else False
    effective_enable = enable_key_dispatch or bool(dispatch_cfg.get("enable_key_dispatch"))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_log = build_artifact_path(
        "runtime",
        f"runtime_session_{timestamp}",
        "jsonl",
        project_root=root,
    )
    summary_path = Path(output) if output else build_artifact_path(
        "runtime",
        f"runtime_eval_{timestamp}",
        "json",
        project_root=root,
    )
    if not summary_path.is_absolute():
        summary_path = root / summary_path

    session = RuntimeSessionConfig(
        model_path=model_path,
        runtime_config_path=_resolve_path(root, runtime_config),
        feature_config_path=_resolve_path(root, feature_cfg_path),
        camera_index=cam_idx,
        dry_run=effective_dry_run,
        enable_key_dispatch=effective_enable,
        show_overlay=show_overlay,
        duration_seconds=duration_seconds,
        session_log_path=session_log,
        summary_output_path=summary_path,
    )
    return session, runtime_cfg, feature_cfg


def run_runtime_session(session: RuntimeSessionConfig, runtime_cfg: dict[str, Any], feature_cfg: dict[str, Any]) -> dict[str, Any]:
    """Execute the webcam inference loop and return session summary."""
    inference_cfg = runtime_cfg.get("inference", {})
    smoothing_cfg = runtime_cfg.get("smoothing", {})
    debounce_cfg = runtime_cfg.get("debounce", {})
    monitoring_cfg = runtime_cfg.get("monitoring", {})
    camera_cfg = runtime_cfg.get("camera", {})

    feature_family = str(inference_cfg.get("feature_family", "hybrid"))
    confidence_threshold = float(inference_cfg.get("confidence_threshold", 0.6))

    runtime_model = load_runtime_model(session.model_path)
    gesture_filter = GestureFilter(
        window_size=int(smoothing_cfg.get("window_size", 5)),
        min_consensus_ratio=float(smoothing_cfg.get("min_consensus_ratio", 0.6)),
        debounce_ms=float(debounce_cfg.get("ms_between_actions", 300)),
        confidence_threshold=confidence_threshold,
    )
    keymap = load_keymap(runtime_config=runtime_cfg)
    telemetry = RuntimeTelemetry(
        log_fps_interval_sec=float(monitoring_cfg.get("log_fps_interval_sec", 5.0)),
    )
    session_logger = SessionLogger(session.session_log_path)

    camera_backend = camera_cfg.get("backend", "auto")
    camera_fallback_backends = camera_cfg.get("fallback_backends")
    max_consecutive_read_failures = int(camera_cfg.get("max_consecutive_read_failures", 5))
    max_reopen_attempts = int(camera_cfg.get("max_reopen_attempts", 3))

    camera_stream.open_camera(
        session.camera_index,
        frame_width=camera_cfg.get("width"),
        frame_height=camera_cfg.get("height"),
        backend=camera_backend,
        fallback_backends=camera_fallback_backends,
    )

    window_name = "Gesticulate Runtime"
    if session.show_overlay:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    log_gesture_labels = bool(monitoring_cfg.get("log_gesture_labels", True))
    log_gesture_every_frame = str(monitoring_cfg.get("log_gesture_mode", "on_change")).lower() == "every_frame"

    start_time = time.perf_counter()
    frame_index = 0
    consecutive_read_failures = 0
    reopen_attempts = 0
    last_logged_stable_label: str | None = None

    try:
        while True:
            if session.duration_seconds is not None:
                if time.perf_counter() - start_time >= session.duration_seconds:
                    logger.info("Duration limit reached (%.1fs)", session.duration_seconds)
                    break

            frame_start = telemetry.begin_frame()
            ok, frame = camera_stream.read_frame()
            if not ok or frame is None:
                consecutive_read_failures += 1
                logger.warning(
                    "Failed to read camera frame (%s/%s consecutive failures, backend=%s)",
                    consecutive_read_failures,
                    max_consecutive_read_failures,
                    camera_stream.get_active_backend(),
                )
                if consecutive_read_failures < max_consecutive_read_failures:
                    time.sleep(0.05)
                    continue

                if reopen_attempts < max_reopen_attempts:
                    reopen_attempts += 1
                    logger.warning(
                        "Attempting camera reopen %s/%s after repeated read failures",
                        reopen_attempts,
                        max_reopen_attempts,
                    )
                    try:
                        camera_stream.reopen_camera()
                    except RuntimeError as exc:
                        logger.error("Camera reopen failed: %s", exc)
                        break
                    consecutive_read_failures = 0
                    continue

                logger.error(
                    "Camera read failures exceeded recovery limits; stopping loop "
                    "(reopen_attempts=%s, backend=%s)",
                    reopen_attempts,
                    camera_stream.get_active_backend(),
                )
                break

            consecutive_read_failures = 0

            frame = prepare_frame(frame, runtime_cfg)
            feature_record = extract_runtime_features(
                frame,
                feature_cfg,
                feature_family=feature_family,
            )
            preprocess_end = time.perf_counter()

            prediction = runtime_model.predict_from_record(feature_record)
            predict_end = time.perf_counter()

            filtered_state = gesture_filter.update_prediction(prediction, predict_end)

            if log_gesture_labels:
                stable_label = str(filtered_state.get("stable_label", ""))
                raw_label = str(prediction.get("label", ""))
                raw_conf = float(prediction.get("confidence", 0.0))
                avg_conf = float(filtered_state.get("avg_confidence", 0.0))
                consensus = float(filtered_state.get("consensus_ratio", 0.0))
                if log_gesture_every_frame or stable_label != last_logged_stable_label:
                    logger.info(
                        "Gesture stable=%s raw=%s (conf=%.2f, avg=%.2f, consensus=%.2f)",
                        stable_label,
                        raw_label,
                        raw_conf,
                        avg_conf,
                        consensus,
                    )
                    last_logged_stable_label = stable_label

            action_result: dict[str, Any] = {
                "emitted": False,
                "reason": "not_eligible",
                "gesture_label": filtered_state.get("stable_label"),
                "mapped_key": None,
                "dry_run": session.dry_run,
            }

            if gesture_filter.should_emit_action(filtered_state):
                action_result = dispatch_key_action(
                    str(filtered_state["stable_label"]),
                    keymap,
                    dry_run=session.dry_run,
                    enable_dispatch=session.enable_key_dispatch,
                )
                if action_result.get("emitted") or action_result.get("would_emit"):
                    gesture_filter.mark_emitted(filtered_state)
                    if log_gesture_labels:
                        logger.info(
                            "Key action %s: %s -> %s",
                            action_result.get("reason"),
                            action_result.get("gesture_label"),
                            action_result.get("mapped_key"),
                        )

            dispatch_end = time.perf_counter()

            sample = telemetry.end_frame(
                frame_start=frame_start,
                preprocess_end=preprocess_end,
                predict_end=predict_end,
                dispatch_end=dispatch_end,
                prediction=prediction,
                filtered_state=filtered_state,
                action_result=action_result,
                detection_ok=bool(feature_record.get("extraction_ok")),
            )

            session_logger.log_event(
                {
                    "frame_index": frame_index,
                    "predicted_gesture": prediction.get("label"),
                    "confidence": prediction.get("confidence"),
                    "stable_label": filtered_state.get("stable_label"),
                    "action_emitted": action_result.get("emitted"),
                    "action_reason": action_result.get("reason"),
                    "mapped_key": action_result.get("mapped_key"),
                    "latency_ms": sample.get("end_to_end_ms"),
                    "quality_flags": feature_record.get("quality_flags"),
                }
            )

            if session.show_overlay:
                overlay = frame.copy()
                label = str(filtered_state.get("stable_label", ""))
                conf = float(filtered_state.get("avg_confidence", 0.0))
                cv2.putText(
                    overlay,
                    f"{label} ({conf:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                mode = "DRY-RUN" if session.dry_run else "LIVE"
                cv2.putText(
                    overlay,
                    mode,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 255),
                    2,
                )
                cv2.imshow(window_name, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit key pressed; stopping runtime")
                    break

            telemetry.maybe_log_fps(logger)
            frame_index += 1

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received; shutting down runtime")
    finally:
        camera_stream.close_camera()
        if session.show_overlay:
            cv2.destroyAllWindows()

    summary = telemetry.summary()
    summary.update(
        {
            "experiment_id": "EXP-04",
            "model_path": str(session.model_path),
            "runtime_config": str(session.runtime_config_path),
            "feature_config": str(session.feature_config_path),
            "camera_index": session.camera_index,
            "camera_backend": camera_stream.get_active_backend(),
            "camera_reopen_attempts": reopen_attempts,
            "dry_run": session.dry_run,
            "enable_key_dispatch": session.enable_key_dispatch,
            "feature_family": feature_family,
            "frames_processed": frame_index,
        }
    )

    session_logger.write_summary(summary, session.summary_output_path)
    logger.info("Session summary written to %s", session.summary_output_path)

    _write_exp04_metrics(summary, session, runtime_cfg)
    return summary


def _write_exp04_metrics(
    summary: dict[str, Any],
    session: RuntimeSessionConfig,
    runtime_cfg: dict[str, Any],
) -> None:
    """Persist EXP-04 metrics record under artifacts/metrics/exp04_realtime_deployment/."""
    root = resolve_project_root()
    exp04_path = root / "configs/experiments/exp04_realtime_deployment.yaml"
    exp_config = load_config(str(exp04_path)) if exp04_path.is_file() else {}

    record = create_run_record(
        "EXP-04",
        exp_config or runtime_cfg,
        status="completed",
        artifacts={
            "session_log": str(session.session_log_path),
            "runtime_summary": str(session.summary_output_path),
            "model_path": str(session.model_path),
        },
    )
    record["metrics"] = {
        "primary": summary.get("latency_ms", {}).get("end_to_end_avg"),
        "fps": summary.get("avg_fps"),
        "end_to_end_p95_ms": summary.get("latency_ms", {}).get("end_to_end_p95"),
        "frames_processed": summary.get("frames_processed"),
        "event_counts": summary.get("event_counts"),
    }
    record["runtime_summary"] = summary
    metrics_path = save_run_record(record)
    logger.info("EXP-04 metrics record written to %s", metrics_path)
