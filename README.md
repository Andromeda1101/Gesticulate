# Gesticulate

ML-based visual gesture recognition for keyboard control. This repository follows a phased implementation plan documented under `doc/`.

**Primary dataset:** HaGRID subset (in-domain training and evaluation). **OOD dataset:** LeapGestRecog (cross-domain evaluation in EXP-03).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional: scaffold missing config placeholders:

```bash
python scripts/bootstrap_project.py --project-root . --with-placeholders
```

Place raw data under:

- `data/raw/hagrid/HaGRIDv2_dataset_512/` — HaGRID subset (primary)
- `data/raw/leapgestrecog/` — LeapGestRecog (OOD)

## End-to-end workflow

Run from the repository root with the virtual environment activated. Commands below use the current dataset roles (HaGRID = in-domain, LeapGestRecog = OOD).

### 1. Dataset manifests and splits

Build manifests:

```bash
python scripts/build_dataset_manifests.py \
  --dataset hagrid_subset \
  --config configs/datasets/hagrid_subset.yaml \
  --output data/interim/hagrid_subset_manifest.parquet

python scripts/build_dataset_manifests.py \
  --dataset leapgestrecog \
  --config configs/datasets/leapgestrecog.yaml \
  --output data/interim/leapgestrecog_manifest.parquet
```

Generate stratified train/val/test splits and CV folds on the primary dataset:

```bash
python scripts/generate_splits.py \
  --manifest data/interim/hagrid_subset_manifest.parquet \
  --config configs/datasets/hagrid_subset.yaml \
  --seed 42 \
  --folds 5
```

Export a dataset summary (optional):

```bash
python scripts/export_dataset_report.py \
  --manifest data/interim/hagrid_subset_manifest.parquet \
  --output reports/summaries/hagrid_subset_summary.json
```

**Outputs:** `data/interim/hagrid_subset_manifest.parquet`, `data/interim/leapgestrecog_manifest.parquet`, `data/splits/hagrid_subset_train_val_test.json`, `data/splits/hagrid_subset_cv_folds.json`.

### 2. Feature extraction

In-domain features (HaGRID) for EXP-01 and EXP-02:

```bash
python scripts/extract_features.py \
  --manifest data/interim/hagrid_subset_manifest.parquet \
  --feature-family geometric \
  --config configs/features/default.yaml \
  --output artifacts/features/hagrid_subset_geometric_v1.parquet

python scripts/extract_features.py \
  --manifest data/interim/hagrid_subset_manifest.parquet \
  --feature-family hog \
  --config configs/features/default.yaml \
  --output artifacts/features/hagrid_subset_hog_v1.parquet

python scripts/build_hybrid_features.py \
  --keypoint-features artifacts/features/hagrid_subset_geometric_v1.parquet \
  --hog-features artifacts/features/hagrid_subset_hog_v1.parquet \
  --output artifacts/features/hagrid_subset_hybrid_v1.parquet

python scripts/export_feature_report.py \
  --feature-manifest artifacts/features/hagrid_subset_geometric_v1_manifest.json \
  --output reports/summaries/feature_report_hagrid_geometric_v1.json
```

OOD features (LeapGestRecog) for EXP-03 — repeat with `leapgestrecog` manifest and `leapgestrecog_*` artifact prefixes:

```bash
python scripts/extract_features.py \
  --manifest data/interim/leapgestrecog_manifest.parquet \
  --feature-family geometric \
  --config configs/features/default.yaml \
  --output artifacts/features/leapgestrecog_geometric_v1.parquet

python scripts/extract_features.py \
  --manifest data/interim/leapgestrecog_manifest.parquet \
  --feature-family hog \
  --config configs/features/default.yaml \
  --output artifacts/features/leapgestrecog_hog_v1.parquet

python scripts/build_hybrid_features.py \
  --keypoint-features artifacts/features/leapgestrecog_geometric_v1.parquet \
  --hog-features artifacts/features/leapgestrecog_hog_v1.parquet \
  --output artifacts/features/leapgestrecog_hybrid_v1.parquet
```

Parallel extraction (optional; use `--workers 1` for debugging):

```bash
python scripts/extract_features.py \
  --manifest data/interim/hagrid_subset_manifest.parquet \
  --feature-family geometric \
  --config configs/features/default.yaml \
  --output artifacts/features/hagrid_subset_geometric_v1.parquet \
  --workers 8
```

### 3. Model benchmarking

Custom classical models (KNN, SVM, Decision Tree, Random Forest, Naive Bayes, Logistic Regression) and PyTorch baselines (MLP, CNN, LSTM) share one experiment runner. Model banchmarking reads feature parquet files and split JSON only.

```bash
# EXP-01: single algorithm run
python scripts/run_experiment.py \
  --experiment-id EXP-01 \
  --feature-family hybrid \
  --algorithm svm \
  --config configs/experiments/exp01_model_comparison.yaml

# EXP-01: full classical + deep sweep on hybrid features
python scripts/run_benchmark_suite.py \
  --experiment-id EXP-01 \
  --feature-family hybrid \
  --algorithms knn svm decision_tree random_forest naive_bayes logistic_regression mlp

# EXP-02: feature ablation (keypoints / HOG / hybrid)
python scripts/run_ablation_suite.py \
  --experiment-id EXP-02 \
  --config configs/experiments/exp02_feature_ablation.yaml

# Aggregate completed runs into a summary report
python scripts/export_phase3_report.py \
  --input-dir artifacts/metrics \
  --output reports/summaries/phase3_benchmark_summary.md
```

Expected outputs:

- `artifacts/models/{experiment_id}_{algorithm}_{feature_family}.joblib` (or `.pt` for deep models)
- `artifacts/models/{experiment_id}_{algorithm}_{feature_family}.meta.json`
- `artifacts/metrics/{experiment_id}_{run_id}.json`
- `reports/tables/*_leaderboard.csv` and `reports/figures/*_confusion.png`

### 4. Robustness evaluation (Phase 4, planned)

Train on HaGRID, evaluate on LeapGestRecog without retraining (`configs/experiments/exp03_robustness.yaml`):

```bash
python scripts/run_robustness_eval.py \
  --model-artifact artifacts/models/EXP-01_svm_hybrid.joblib \
  --in-domain-features artifacts/features/hagrid_subset_hybrid_v1.parquet \
  --ood-features artifacts/features/leapgestrecog_hybrid_v1.parquet \
  --config configs/experiments/exp03_robustness.yaml
```

### 5. Real-time deployment (Phase 5, planned)

See `doc/phase5_realtime_deployment.md` for `run_realtime_demo.py` and `benchmark_runtime.py`.

## Smoke tests

```bash
pytest tests/smoke/test_phase0_foundation.py -q
pytest tests/smoke/test_phase1_dataset_ingestion.py -q
pytest tests/smoke/ -q
```

Phase 2 tests require `opencv-python` and related packages from `requirements.txt`.

```bash
pytest tests/smoke/test_phase3_model_benchmarking.py -q
```

Phase 3 deep baselines require `torch` (included in `requirements.txt`).

## Repository layout

See [doc/project_overview.md](doc/project_overview.md) for the canonical directory structure, experiment IDs (`EXP-01`–`EXP-04`), and artifact naming rules.

## Experiment IDs

| ID | Description |
|----|-------------|
| EXP-01 | Model comparison on common features and splits (HaGRID subset) |
| EXP-02 | Feature ablation: keypoints / HOG / hybrid (HaGRID subset) |
| EXP-03 | Robustness: train HaGRID subset, test LeapGestRecog (OOD) |
| EXP-04 | Real-time deployment evaluation |
