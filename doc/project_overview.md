# Project Overview: ML-Based Visual Gesture Recognition System

## Purpose
This document defines the shared architecture, repository conventions, experiment IDs, artifact contracts, and implementation boundaries for the gesture-recognition research project described in `doc/ML_proposal.md`. All phase-level implementation agents should treat this overview as the source of truth for project structure and cross-phase compatibility.

## Research Objectives
- Build a webcam-based gesture recognition pipeline for keyboard control.
- Compare multiple classical and neural classifiers on a consistent feature stack.
- Evaluate generalization from an in-the-wild primary dataset to a clean static out-of-distribution dataset.
- Deploy the best-performing model in a real-time inference loop with gesture-to-key mapping and latency instrumentation.

## Phase Sequence
1. Phase 0: Project foundation and experiment contracts
2. Phase 1: Dataset ingestion and split preparation
3. Phase 2: Feature extraction and feature store generation
4. Phase 3: Model benchmarking and feature ablation
5. Phase 4: Robustness and cross-dataset evaluation
6. Phase 5: Real-time runtime pipeline and deployment packaging

## Intended Repository Layout
The following structure should remain stable unless a later architectural decision is explicitly approved:

```text
Gesticulate/
  README.md
  requirements.txt
  configs/
    datasets/
    features/
    models/
    runtime/
    experiments/
  data/
    raw/
      leapgestrecog/
      hagrid/
    interim/
    processed/
    splits/
  artifacts/
    features/
    models/
    metrics/
    runtime/
  reports/
    figures/
    tables/
    summaries/
  scripts/
  src/
    common/
    data/
    features/
    models/
    evaluation/
    runtime/
  tests/
    smoke/
    integration/
  doc/
    project_overview.md
    phase0_project_foundation.md
    phase1_dataset_ingestion.md
    phase2_feature_pipeline.md
    phase3_model_benchmarking.md
    phase4_robustness_evaluation.md
    phase5_realtime_deployment.md
```

## Documentation and Runbooks
- **将必要指令流程写入 README**：所有可执行的端到端流程（环境准备、数据索引、划分、特征提取、实验与评估）必须在仓库根目录的 [`README.md`](../README.md) 中维护一份可复制的命令清单，作为操作入口。
- Phase 文档（`doc/phase*.md`）保留模块设计、接口约定与阶段验收标准；与运行相关的示例命令以 README 为准，phase 文档可引用 README 中的对应章节，避免两处命令长期不一致。
- 新增或变更 CLI 脚本时，同步更新 README 中的 **End-to-end workflow** 小节，并标明该步骤所属阶段、输入/输出路径及对应 `EXP-*` 实验（如适用）。

## Core Design Principles
- Keep data processing, feature extraction, model training, evaluation, and runtime inference as separate layers.
- Prefer configuration-driven experiment execution over hard-coded dataset paths or model parameters.
- Treat every experiment output as a reproducible artifact with machine-readable metadata.
- Make offline feature generation compatible with both classical models and future deep-sequence extensions.
- Keep the real-time runtime decoupled from training code; it should consume exported artifacts only.
- Use a process pool to accelerate work wherever batch processing may become slow or congested. Use tqdm process bar for noticing if necessary. Offline stages that iterate over tens of thousands of samples should parallelize across CPU cores instead of running strictly serial loops in the parent process. Each worker must own its own heavy state (for example, a MediaPipe Hand Landmarker instance) and must be initialized under a `spawn` multiprocessing context so native libraries are not inherited in a broken state after `fork`. Preserve manifest row order when collecting results. Expose parallelism through configuration and CLI flags; use `--workers 1` for debugging. Do not apply process pools to the Phase 5 real-time frame loop, where latency and single-threaded camera capture dominate. 

## Canonical Data Contracts

### Raw Sample Record
Every raw sample should be mappable to a standardized metadata record:

```text
sample_id: str
dataset_name: str
subject_id: str | null
gesture_label: str
image_path: str
split: str | null
capture_context: dict
```

### Feature Record
Every extracted feature vector should follow a consistent schema:

```text
sample_id: str
dataset_name: str
gesture_label: str
feature_family: str
feature_version: str
vector_path: str | null
vector_inline: list[float] | null
quality_flags: dict
```

### Model Artifact Metadata
Each trained model should produce a sidecar metadata document:

```text
model_id: str
experiment_id: str
algorithm_name: str
feature_family: str
feature_version: str
train_split_id: str
validation_strategy: str
hyperparameters: dict
metrics_summary: dict
artifact_path: str
created_at: str
```

## Experiment IDs
- `EXP-01`: algorithm comparison on a common feature and split protocol
- `EXP-02`: feature ablation comparing keypoints-only, HOG-only, and hybrid features
- `EXP-03`: robustness study using train-on-HaGRID subset and test-on-LeapGestRecog (OOD)
- `EXP-04`: real-time deployment evaluation using the champion exported model

## Shared Module Responsibilities

### `src/common/`
- Configuration loading
- Path resolution
- Logging
- Reproducibility utilities
- Serialization helpers
- Run metadata management

### `src/data/`
- Dataset adapters
- Sample indexing
- Label normalization
- Split generation
- Dataset statistics summarization

### `src/features/`
- Hand landmark detection wrapper
- Geometric feature computation
- HOG extraction
- Feature concatenation
- Feature persistence and manifest generation
- Parallel batch extraction for offline manifests via a process pool

### `src/models/`
- Training interfaces
- Algorithm registry
- Hyperparameter search planners
- Model export packaging
- Inference adapter abstraction

### `src/evaluation/`
- Metrics computation
- Confusion matrix generation
- Cross-validation aggregation
- Runtime benchmark summarization
- Report assembly utilities

### `src/runtime/`
- Camera frame loop
- Runtime preprocessing
- Online feature extraction
- Model inference gateway
- Gesture smoothing
- Keyboard event dispatcher
- FPS and latency monitors

## Cross-Phase Dependency Rules
- Phase 1 may define dataset manifests but should not assume finalized feature dimensions.
- Phase 2 must preserve stable sample IDs from Phase 1 manifests.
- Phase 3 must read feature manifests from Phase 2 rather than rescanning raw data directly.
- Phase 4 must reuse trained-model interfaces from Phase 3 and must not create a separate evaluation stack.
- Phase 5 must consume exported champion artifacts and runtime configuration without retraining models.

## Configuration Strategy
Every configurable behavior should eventually live in YAML or JSON files under `configs/`. Suggested config scopes:
- `configs/datasets/`: root paths, label maps, sampling rules, split seeds
- `configs/features/`: MediaPipe options, HOG parameters, feature family toggles
- `configs/models/`: algorithm presets, search spaces, regularization defaults
- `configs/experiments/`: experiment IDs, datasets, feature families, metrics
- `configs/runtime/`: camera index, gesture debounce, smoothing window, key bindings

## Suggested Artifact Naming Convention
- Feature matrices: `artifacts/features/{dataset}_{feature_family}_{version}.parquet`
- Feature manifests: `artifacts/features/{dataset}_{feature_family}_{version}_manifest.json`
- Trained models: `artifacts/models/{experiment_id}_{algorithm}_{feature_family}.joblib`
- Metrics summaries: `artifacts/metrics/{experiment_id}_{run_id}.json`
- Runtime benchmarks: `artifacts/runtime/runtime_eval_{timestamp}.json`

## Gesture-to-Key Mapping Baseline
The initial deployment target should use the proposal's default mapping:
- `Palm -> space`
- `Fist -> enter`
- `Thumb_Up -> up`
- `Peace -> down`

This mapping should remain configuration-driven so later user studies can rebind actions without changing inference logic.

## Non-Goals During Initial Implementation
- Do not optimize for multi-hand interaction.
- Do not build a GUI dashboard in the first pass.
- Do not merge training and runtime code into one monolithic script.
- Do not skip artifact versioning, even for early experiments.

## Recommended Handoff Order
An implementation agent should execute the phase documents strictly in order. If a later phase uncovers an issue, the fix should be applied to the earliest responsible layer rather than patched only in downstream scripts.

## Success Criteria Across The Whole Project
- Offline pipeline produces reproducible datasets, features, models, and metrics artifacts.
- Benchmarking covers the required experiments with comparable outputs.
- The selected champion model reaches the proposal's target accuracy and latency ranges, or the reports clearly explain any gap.
- The runtime pipeline can map webcam-observed gestures to keyboard events with measurable FPS and end-to-end latency.
