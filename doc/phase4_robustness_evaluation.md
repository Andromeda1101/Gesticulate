# Phase 4: Robustness and Cross-Dataset Evaluation

## Phase Objectives
- Implement `EXP-03`, the out-of-distribution robustness study.
- Evaluate how well models trained on LeapGestRecog generalize to the HaGRID subset without retraining.
- Quantify generalization drop and identify failure modes across gesture classes and conditions.
- Produce a report that explains whether the champion model is suitable for real-world deployment or requires mitigation.

## Prerequisites
- Champion candidates and exportable model artifacts from `doc/phase3_model_benchmarking.md`
- HaGRID subset manifest from `doc/phase1_dataset_ingestion.md`
- Feature extraction pipelines from `doc/phase2_feature_pipeline.md`
- Shared reporting conventions from `doc/project_overview.md`

## Code Module Plan

### 1. `src/evaluation/ood_loader.py`
- **Function**: Load OOD evaluation inputs using the same feature schema as the training environment.
- **Suggested interface**:
  - `load_ood_feature_set(dataset_name: str, feature_family: str, version: str)`
  - `validate_schema_compatibility(train_manifest: dict, test_manifest: dict) -> dict`
- **Core logic**:
  1. Resolve feature artifact paths.
  2. Confirm matching feature dimensions and label vocabulary.
  3. Return evaluation-ready tables.
- **Dependent libraries**: `pandas`, `json`

### 2. `src/evaluation/robustness_metrics.py`
- **Function**: Compute robustness-specific metrics on top of standard classification metrics.
- **Metrics to include**:
  - in-domain accuracy
  - OOD accuracy
  - absolute accuracy drop
  - relative performance retention
  - per-class drop
  - misclassification concentration by gesture class
- **Suggested interface**:
  - `compute_ood_drop(id_metrics: dict, ood_metrics: dict) -> dict`
  - `compute_per_class_shift(y_true, y_pred, domain_labels) -> dict`
- **Dependent libraries**: `numpy`, `pandas`

### 3. `src/evaluation/error_analysis.py`
- **Function**: Summarize common OOD error patterns and likely causes.
- **Suggested interface**:
  - `group_errors_by_context(predictions: pd.DataFrame) -> dict`
  - `sample_failure_cases(predictions: pd.DataFrame, n_per_class: int) -> pd.DataFrame`
- **Planned analysis dimensions**:
  - cluttered background
  - lighting variation
  - subject variation
  - gesture confusion pairs
- **Dependent libraries**: `pandas`

### 4. `src/evaluation/domain_report.py`
- **Function**: Turn robustness outputs into a deployment-oriented summary.
- **Suggested interface**:
  - `build_domain_shift_report(run_inputs: dict) -> dict`
  - `export_domain_shift_report(report: dict, output_path: str) -> None`
- **Core logic**:
  1. Load in-domain benchmark metrics.
  2. Load OOD evaluation metrics.
  3. Compare both domains.
  4. Export a concise summary for decision-making.
- **Dependent libraries**: `pandas`, `matplotlib`, `seaborn`

### 5. `scripts/phase4/run_robustness_eval.py`
- **Function**: Main CLI entrypoint for `EXP-03`.
- **Suggested CLI arguments**:
  - `--model-artifact artifacts/models/EXP-01_svm_hybrid.joblib`
  - `--in-domain-features artifacts/features/leapgestrecog_hybrid_v1.parquet`
  - `--ood-features artifacts/features/hagrid_subset_hybrid_v1.parquet`
  - `--config configs/experiments/exp03_robustness.yaml`
- **Core logic**:
  1. Load champion model artifact and metadata.
  2. Load held-out in-domain test features and HaGRID OOD features.
  3. Run prediction on both domains.
  4. Compute domain-shift metrics.
  5. Export metrics and report bundle.

### 6. `scripts/phase4/export_failure_gallery.py`
- **Function**: Assemble a review set of OOD misclassifications for qualitative analysis.
- **Input/Output**:
  - Input: prediction table with image paths, truth labels, predicted labels, and context metadata
  - Output: CSV or Markdown index of representative failure cases
- **Core logic**:
  1. Rank errors by confidence or confusion importance.
  2. Sample a balanced set across classes.
  3. Export a human-review artifact to `reports/`.

## Data Flow & Interaction Design
- Training-domain and OOD-domain features must use the same feature family and version.
- The trained model from Phase 3 should be treated as immutable during Phase 4.
- Robustness evaluation should produce both machine-readable metrics and human-readable diagnosis artifacts.

```text
ChampionModel + InDomainTestFeatures + OODFeatures
                      |
                      v
                PredictionRunner
                      |
          +-----------+------------+
          |                        |
          v                        v
   RobustnessMetrics         ErrorAnalysis
          |                        |
          +-----------+------------+
                      v
               DomainShiftReport
```

### Required Output Bundle
- in-domain prediction table
- OOD prediction table
- OOD metrics JSON
- per-class drop summary
- qualitative error review artifact
- deployment recommendation note

## Verification & Testing Approach
- Confirm that the OOD feature schema matches the feature schema used during training.
- Validate that label names align exactly between the two datasets.
- Check that the model artifact loaded in Phase 4 is the same one selected in Phase 3.
- Review whether OOD accuracy drop is computed against the correct in-domain reference score.
- Inspect representative failure cases to separate detector failures from classifier failures.
- Verify that the report explicitly documents assumptions and known limits of zero-shot cross-dataset transfer.

## Code Execution Method

### Environment Setup
1. Use the environment from earlier phases.
2. Ensure the selected model artifact and both in-domain and OOD feature stores are available.

### Dependency Installation
- Required packages:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
- Example command:
  - `pip install numpy pandas scikit-learn matplotlib seaborn`

### Execution Steps and Example Commands
1. Evaluate the champion model on in-domain and OOD features:
   - `python scripts/phase4/run_robustness_eval.py --model-artifact artifacts/models/EXP-01_svm_hybrid.joblib --in-domain-features artifacts/features/leapgestrecog_hybrid_v1.parquet --ood-features artifacts/features/hagrid_subset_hybrid_v1.parquet --config configs/experiments/exp03_robustness.yaml`
2. Export a failure-case index:
   - `python scripts/phase4/export_failure_gallery.py --predictions artifacts/metrics/EXP-03_predictions.csv --output reports/summaries/exp03_failure_gallery.md`
3. Export the domain-shift summary:
   - `python scripts/phase4/export_phase4_report.py --metrics artifacts/metrics/EXP-03_<run_id>.json --output reports/summaries/phase4_robustness_summary.md`

### Expected Output or Result Example
- `artifacts/metrics/EXP-03_<run_id>.json`
- `artifacts/metrics/EXP-03_predictions.csv`
- `reports/figures/exp03_per_class_drop.png`
- `reports/summaries/phase4_robustness_summary.md`
- `reports/summaries/exp03_failure_gallery.md`

## Exit Criteria
- The project has a quantified estimate of cross-dataset generalization performance.
- Failure modes are documented clearly enough to guide deployment decisions.
- One final champion model configuration is selected for Phase 5 runtime deployment.
