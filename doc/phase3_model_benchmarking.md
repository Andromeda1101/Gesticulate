# Phase 3: Model Benchmarking and Feature Ablation

## Phase Objectives
- Implement the model training and evaluation framework for `EXP-01` and `EXP-02`.
- Benchmark the required algorithms under a unified interface and consistent data splits.
- Compare feature families including keypoints-only, HOG-only, and hybrid vectors.
- Export comparable metrics, training-time summaries, and serialized model artifacts.

## Prerequisites
- Shared config and artifact structure from `doc/project_overview.md`
- Feature artifacts and manifests from `doc/phase2_feature_pipeline.md`
- Train, validation, and fold definitions from `doc/phase1_dataset_ingestion.md`

## Code Module Plan

### 1. `src/models/model_registry.py`
- **Function**: Register all supported algorithms under a shared training API.
- **Algorithms to include in scope**:
  - KNN
  - SVM
  - Decision Tree
  - Random Forest
  - Naive Bayes
  - Logistic Regression
  - MLP
  - Placeholder wrappers for CNN and LSTM sequence variants if sequence data is added later
- **Suggested interface**:
  - `get_model_builder(algorithm_name: str)`
  - `list_supported_algorithms() -> list[str]`
- **Dependent libraries**: `scikit-learn`, optional `torch` or `tensorflow`

### 2. `src/models/trainers/classical_trainer.py`
- **Function**: Train classical ML models on a selected feature matrix and split definition.
- **Suggested interface**:
  - `train_model(features, labels, train_ids, val_ids, config: dict) -> dict`
- **Input/Output**:
  - Input: feature table, label series, train/validation IDs, model config
  - Output: trained estimator, validation predictions, fit-time metadata
- **Core logic**:
  1. Subset features by split IDs.
  2. Apply scaling if required by algorithm.
  3. Fit estimator.
  4. Generate validation predictions and training statistics.
- **Dependent libraries**: `scikit-learn`, `numpy`, `joblib`

### 3. `src/models/trainers/deep_baseline_trainer.py`
- **Function**: Provide a design placeholder for MLP and future CNN or LSTM baselines.
- **Suggested interface**:
  - `train_deep_baseline(train_dataset, val_dataset, config: dict) -> dict`
- **Implementation guidance**:
  - Keep interface compatible with the classical trainer output.
  - Focus on exported metrics and artifact metadata rather than framework-specific details.
- **Dependent libraries**: optional `torch` or `tensorflow`

### 4. `src/models/hyperparameter_plan.py`
- **Function**: Define search spaces and tuning plans without hard-coding them in scripts.
- **Suggested interface**:
  - `get_search_space(algorithm_name: str) -> dict`
  - `build_search_plan(experiment_config: dict) -> list[dict]`
- **Core logic**:
  1. Load model-specific tuning ranges.
  2. Emit candidate parameter sets.
  3. Associate each set with run metadata.
- **Dependent libraries**: none beyond Python and config loaders

### 5. `src/evaluation/metrics.py`
- **Function**: Compute classification and efficiency metrics for every run.
- **Metrics in scope**:
  - accuracy
  - macro precision
  - macro recall
  - macro F1
  - micro precision, recall, F1 if desired
  - confusion matrix
  - training wall-time
  - per-sample inference time on held-out data
- **Suggested interface**:
  - `compute_classification_metrics(y_true, y_pred) -> dict`
  - `compute_efficiency_metrics(timing_info: dict) -> dict`
- **Dependent libraries**: `scikit-learn`, `numpy`, `time`

### 6. `src/evaluation/report_builder.py`
- **Function**: Aggregate per-run outputs into experiment summaries and comparison tables.
- **Suggested interface**:
  - `build_experiment_summary(run_records: list[dict]) -> dict`
  - `export_leaderboard(summary: dict, output_path: str) -> None`
- **Core logic**:
  1. Read individual run outputs.
  2. Rank algorithms by primary metric.
  3. Export summary tables and figures.
- **Dependent libraries**: `pandas`, `matplotlib`, `seaborn`

### 7. `src/models/exporter.py`
- **Function**: Save trained models and inference metadata in a runtime-consumable format.
- **Suggested interface**:
  - `export_model(estimator, metadata: dict, output_dir: str) -> dict`
- **Output contents**:
  - serialized model file
  - label encoder or label map
  - feature schema
  - preprocessing requirements
  - training summary metadata
- **Dependent libraries**: `joblib`, `json`

### 8. `scripts/run_experiment.py`
- **Function**: Generic experiment runner for `EXP-01` and `EXP-02`.
- **Suggested CLI arguments**:
  - `--experiment-id EXP-01`
  - `--feature-family geometric`
  - `--algorithm svm`
  - `--config configs/experiments/exp01_model_comparison.yaml`
- **Core logic**:
  1. Load experiment config.
  2. Resolve feature store and split definition.
  3. Select model builder and tuning plan.
  4. Train and evaluate.
  5. Save model and metrics artifacts.

### 9. `scripts/run_ablation_suite.py`
- **Function**: Launch the feature-family comparison study for `EXP-02`.
- **Feature families to compare**:
  - keypoints or geometric only
  - HOG only
  - hybrid
- **Core logic**:
  1. Iterate across feature families.
  2. Reuse the same algorithms or a selected subset.
  3. Aggregate results into one ablation report.

## Data Flow & Interaction Design
- Feature artifacts from Phase 2 are the only accepted training input.
- Split files from Phase 1 determine train, validation, and cross-validation membership.
- Model training outputs should be stored under `artifacts/models/`.
- Metrics and leaderboards should be stored under `artifacts/metrics/` and `reports/`.

```text
FeatureStore + SplitFiles + ExperimentConfig
                |
                v
          ModelTrainer -> MetricsEngine -> RunRecord
                |
                +-> ExportedModel
                +-> ConfusionMatrix
                +-> LeaderboardSummary
```

### Standard Run Output Bundle
- serialized model artifact
- metrics JSON
- config snapshot
- timing summary
- confusion matrix image or CSV
- run metadata record

## Verification & Testing Approach
- Verify that every algorithm can be instantiated via the registry.
- Confirm that all runs consume identical split definitions for fair comparison.
- Inspect whether scaling and preprocessing are applied consistently where required.
- Validate that exported model metadata contains the matching feature family and vector dimension.
- Review confusion matrices for obvious label mapping issues.
- Confirm that `EXP-02` ablation compares feature families without changing unrelated settings.

## Code Execution Method

### Environment Setup
1. Use the virtual environment from earlier phases.
2. Ensure Phase 1 split files and Phase 2 feature stores are present.

### Dependency Installation
- Required packages for this phase:
  - `scikit-learn`
  - `joblib`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
- Optional packages for deep baselines:
  - `torch` or `tensorflow`
- Example command:
  - `pip install scikit-learn joblib numpy pandas matplotlib seaborn`

### Execution Steps and Example Commands
1. Run a single benchmark configuration:
   - `python scripts/run_experiment.py --experiment-id EXP-01 --feature-family geometric --algorithm svm --config configs/experiments/exp01_model_comparison.yaml`
2. Run a full algorithm sweep:
   - `python scripts/run_benchmark_suite.py --experiment-id EXP-01 --feature-family geometric --algorithms knn svm random_forest logistic_regression mlp`
3. Run feature ablation:
   - `python scripts/run_ablation_suite.py --experiment-id EXP-02 --algorithms svm random_forest mlp`
4. Export leaderboard summary:
   - `python scripts/export_phase3_report.py --input-dir artifacts/metrics --output reports/summaries/phase3_benchmark_summary.md`

### Expected Output or Result Example
- `artifacts/models/EXP-01_svm_geometric.joblib`
- `artifacts/models/EXP-01_random_forest_hybrid.joblib`
- `artifacts/metrics/EXP-01_<run_id>.json`
- `artifacts/metrics/EXP-02_<run_id>.json`
- `reports/tables/exp01_leaderboard.csv`
- `reports/figures/exp02_feature_ablation.png`

## Exit Criteria
- `EXP-01` and `EXP-02` can be run through a shared experiment interface.
- Multiple algorithms can be compared on the same splits and feature sets.
- At least one champion candidate is clearly identified for robustness testing in Phase 4.
