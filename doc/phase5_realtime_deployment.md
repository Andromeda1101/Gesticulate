# Phase 5: Real-Time Runtime Pipeline and Deployment Packaging

## Phase Objectives
- Implement `EXP-04`, the end-to-end real-time gesture-to-keyboard control pipeline.
- Load the champion model artifact from earlier phases and run online inference on webcam frames.
- Map recognized gestures to keyboard events with smoothing and debounce logic.
- Measure runtime latency and FPS stability for deployment evaluation.

## Prerequisites
- Final champion model artifact and metadata from `doc/phase3_model_benchmarking.md`
- Robustness findings and deployment recommendation from `doc/phase4_robustness_evaluation.md`
- Feature extraction contracts from `doc/phase2_feature_pipeline.md`
- Runtime configuration contract from `doc/project_overview.md`

## Code Module Plan

### 1. `src/runtime/camera_stream.py`
- **Function**: Manage webcam initialization, frame retrieval, and cleanup.
- **Suggested interface**:
  - `open_camera(camera_index: int, frame_width: int | None = None, frame_height: int | None = None)`
  - `read_frame() -> tuple[bool, np.ndarray]`
  - `close_camera() -> None`
- **Core logic**:
  1. Open the webcam with OpenCV.
  2. Read frames in a loop.
  3. Surface camera failures cleanly.
- **Dependent libraries**: `opencv-python`

### 2. `src/runtime/preprocess.py`
- **Function**: Convert live frames into the format expected by the selected feature family.
- **Suggested interface**:
  - `prepare_frame(frame: np.ndarray, runtime_config: dict) -> np.ndarray`
  - `extract_runtime_features(frame: np.ndarray, feature_config: dict) -> dict`
- **Core logic**:
  1. Resize or crop frame if required.
  2. Run hand detection and feature extraction with the same logic used offline.
  3. Return a feature vector plus quality flags.
- **Dependent libraries**: `opencv-python`, `mediapipe`, `numpy`

### 3. `src/runtime/model_runner.py`
- **Function**: Load the serialized champion model and run predictions on live feature vectors.
- **Suggested interface**:
  - `load_runtime_model(model_path: str, metadata_path: str) -> "RuntimeModel"`
  - `predict_gesture(feature_vector: np.ndarray) -> dict`
- **Output guidance**:
  - predicted label
  - confidence or probability
  - raw score vector if available
- **Dependent libraries**: `joblib`, `numpy`

### 4. `src/runtime/gesture_filter.py`
- **Function**: Stabilize frame-level predictions before emitting keyboard events.
- **Suggested interface**:
  - `update_prediction(prediction: dict, timestamp: float) -> dict`
  - `should_emit_action(filtered_state: dict) -> bool`
- **Core logic**:
  1. Track a short rolling window of predictions.
  2. Apply majority vote or confidence smoothing.
  3. Enforce cooldown or debounce intervals.
- **Dependent libraries**: `collections`, `time`

### 5. `src/runtime/key_mapper.py`
- **Function**: Map filtered gesture labels to OS-level keyboard actions.
- **Suggested interface**:
  - `load_keymap(config_path: str) -> dict`
  - `dispatch_key_action(gesture_label: str, keymap: dict) -> dict`
- **Baseline mapping**:
  - `Palm -> space`
  - `Fist -> enter`
  - `Thumb_Up -> up`
  - `Peace -> down`
- **Dependent libraries**: `pynput` or `pyautogui`

### 6. `src/runtime/telemetry.py`
- **Function**: Measure runtime quality metrics for deployment evaluation.
- **Suggested interface**:
  - `record_stage_timing(stage_name: str, start_time: float, end_time: float) -> None`
  - `compute_runtime_summary(samples: list[dict]) -> dict`
- **Metrics to track**:
  - capture-to-prediction latency
  - prediction-to-key-dispatch latency
  - total end-to-end latency
  - average FPS
  - FPS jitter or stability
- **Dependent libraries**: `time`, `statistics`, `json`

### 7. `src/runtime/session_logger.py`
- **Function**: Save runtime events for later debugging and evaluation.
- **Suggested interface**:
  - `log_runtime_event(event: dict, output_path: str) -> None`
- **Stored event fields**:
  - timestamp
  - predicted gesture
  - confidence
  - action emitted or suppressed
  - latency metrics
  - frame quality flags
- **Dependent libraries**: `json`, `csv`

### 8. `scripts/run_realtime_demo.py`
- **Function**: Main interactive runtime entrypoint.
- **Suggested CLI arguments**:
  - `--model artifacts/models/EXP-01_svm_hybrid.joblib`
  - `--runtime-config configs/runtime/default.yaml`
  - `--camera-index 0`
  - `--show-overlay`
- **Core logic**:
  1. Load runtime config and model artifact.
  2. Start camera stream.
  3. For each frame: detect hand, extract features, predict gesture, smooth decision, dispatch key action if allowed.
  4. Track telemetry.
  5. Save a session summary on exit.

### 9. `scripts/benchmark_runtime.py`
- **Function**: Run a controlled runtime benchmark without enabling real keyboard events by default.
- **Suggested CLI arguments**:
  - `--dry-run`
  - `--duration-seconds 60`
  - `--output artifacts/runtime/runtime_eval_<timestamp>.json`
- **Core logic**:
  1. Run the same pipeline as the demo.
  2. Replace key dispatch with a logging stub in dry-run mode.
  3. Aggregate latency and FPS statistics.
  4. Export a benchmark report.

## Data Flow & Interaction Design
- The runtime should reuse the same feature-family contract as the chosen model artifact.
- Runtime configuration must specify:
  - model path
  - feature family and version
  - key mapping
  - debounce window
  - confidence threshold
  - camera settings
- Live predictions should pass through a smoothing stage before any OS event is emitted.

```text
CameraFrame -> RuntimePreprocess -> FeatureVector -> ModelRunner -> GestureFilter -> KeyMapper
      |                                                                       |
      +------------------------------> Telemetry <-----------------------------+
```

### Safety and Control Design
- Include a dry-run mode that never sends keyboard events.
- Add a hotkey or console interrupt path to stop the runtime safely.
- Require an explicit config flag before enabling real OS-level key dispatch.

## Verification & Testing Approach
- Validate that the runtime feature extractor matches the champion model's expected feature schema.
- Confirm that dry-run mode logs events without sending actual keyboard inputs.
- Measure average FPS and latency over short and longer runs.
- Test debounce and smoothing using repeated gesture holds and rapid gesture changes.
- Review session logs for false triggers and dropped detections.
- Confirm that runtime shutdown releases camera resources and writes the session summary.

## Code Execution Method

### Environment Setup
1. Use the completed project environment from earlier phases.
2. Ensure a webcam is connected and accessible.
3. Confirm the runtime config references the exported champion model.

### Dependency Installation
- Required packages:
  - `opencv-python`
  - `mediapipe`
  - `numpy`
  - `joblib`
  - `pynput`
  - `pyyaml`
- Example command:
  - `pip install opencv-python mediapipe numpy joblib pynput pyyaml`

### Execution Steps and Example Commands
1. Launch the runtime in safe dry-run mode:
   - `python scripts/run_realtime_demo.py --model artifacts/models/EXP-01_svm_hybrid.joblib --runtime-config configs/runtime/default.yaml --camera-index 0 --dry-run --show-overlay`
2. Run a timed runtime benchmark:
   - `python scripts/benchmark_runtime.py --model artifacts/models/EXP-01_svm_hybrid.joblib --runtime-config configs/runtime/default.yaml --duration-seconds 60 --dry-run --output artifacts/runtime/runtime_eval_001.json`
3. Run the live keyboard-control mode after validation:
   - `python scripts/run_realtime_demo.py --model artifacts/models/EXP-01_svm_hybrid.joblib --runtime-config configs/runtime/default.yaml --camera-index 0 --enable-key-dispatch`

### Expected Output or Result Example
- Live overlay window showing predicted gesture and confidence
- `artifacts/runtime/runtime_eval_001.json`
- `artifacts/runtime/runtime_session_<timestamp>.jsonl`
- Summary log with average FPS, p95 latency, and event counts

## Exit Criteria
- The project can run webcam-based gesture inference using the exported champion model.
- Dry-run benchmarking reports FPS and latency statistics aligned with the proposal's deployment goals.
- The runtime supports safe progression from offline evaluation to real keyboard control.
