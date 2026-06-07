# Gesticulate

ML-based visual gesture recognition for keyboard control. This repository follows a phased implementation plan documented under `doc/`.

Primary dataset: HaGRID subset (in-domain training and evaluation). 
OOD dataset: LeapGestRecog (cross-domain evaluation in EXP-03).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download datasets [HaDRID](https://github.com/hukenovs/hagrid) and [LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog/data) from official websets.

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

Outputs: `data/interim/hagrid_subset_manifest.parquet`, `data/interim/leapgestrecog_manifest.parquet`, `data/splits/hagrid_subset_train_val_test.json`, `data/splits/hagrid_subset_cv_folds.json`.

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

LeapGestRecog labels are mapped to the shared 10-class vocabulary (`Palm`, `Fist`, …) at manifest time. If an older manifest used subject ids (`00`–`09`) as `gesture_label`, rebuild the manifest and sync existing feature files (faster than re-extracting):

```bash
python scripts/build_dataset_manifests.py \
  --dataset leapgestrecog \
  --config configs/datasets/leapgestrecog.yaml \
  --output data/interim/leapgestrecog_manifest.parquet

python scripts/sync_feature_labels_from_manifest.py \
  --manifest data/interim/leapgestrecog_manifest.parquet \
  --matrix artifacts/features/leapgestrecog_geometric_v1.parquet \
  --matrix artifacts/features/leapgestrecog_hog_v1.parquet \
  --matrix artifacts/features/leapgestrecog_hybrid_v1.parquet \
  --refresh-manifest
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
python scripts/export_benchmark_report.py \
  --input-dir artifacts/metrics \
  --output reports/summaries/benchmark_summary.md
```

Expected outputs:

- `artifacts/models/{experiment_id}_{algorithm}_{feature_family}.joblib` (or `.pt` for deep models)
- `artifacts/models/{experiment_id}_{algorithm}_{feature_family}.meta.json`
- `artifacts/metrics/exp0N_<experiment_slug>/{experiment_id}_{run_id}.json`
- `reports/tables/*_leaderboard.csv` and `reports/figures/*_confusion.png`

### 4. Robustness evaluation (EXP-03)

Train on HaGRID , then evaluate the champion model on in-domain test split and LeapGestRecog OOD features without retraining (`configs/experiments/exp03_robustness.yaml`):

```bash
# EXP-03 suite: all feature families × models
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/run_robustness_suite.py \
  --config configs/experiments/exp03_robustness.yaml \
  --batch-size 128 \
  --skip-missing

# Single champion run (WSL-safe defaults: streamed parquet batches, no predict_proba)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/run_robustness_eval.py \
  --model-artifact artifacts/models/EXP-01_svm_hybrid.joblib \
  --in-domain-features artifacts/features/hagrid_subset_hybrid_v1.parquet \
  --ood-features artifacts/features/leapgestrecog_hybrid_v1.parquet \
  --config configs/experiments/exp03_robustness.yaml \
  --batch-size 128

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/run_robustness_eval.py   \
  --model-artifact artifacts/models/EXP-02_cnn_keypoints_only.pt   \
  --in-domain-features artifacts/features/hagrid_subset_geometric_v1.parquet   \
  --ood-features artifacts/features/leapgestrecog_geometric_v1.parquet   \
  --config configs/experiments/exp03_robustness.yaml   \
  --batch-size 128

# If memory is still tight, try --batch-size 64.
# Add --include-proba for masked shared-class argmax OOD metrics (uses more memory on large SVMs).

python scripts/export_failure_gallery.py \
  --predictions artifacts/metrics/exp03_robustness/EXP-03_<run_id>_predictions.csv \
  --output reports/summaries/exp03_failure_gallery.md

python scripts/export_ood_report.py \
  --metrics artifacts/metrics/exp03_robustness/EXP-03_<run_id>.json \
  --output reports/summaries/robustness_summary.md
```

Replace `<run_id>` with the UUID printed by `run_robustness_eval.py`. Pick `--model-artifact` from the Phase 3 leaderboard (e.g. top `EXP-01_*_hybrid` under `artifacts/models/`).

The run JSON and `robustness_summary.md` also report restricted OOD protocols:

- Shared-class subset: OOD accuracy only on the 7 classes present in both HaGRID and LeapGestRecog (excludes `L`, `Down`, `Palm_Moved`).
- Masked unknown: predictions outside the 10-class OOD vocabulary map to `unknown`.
- Masked shared argmax (requires `--include-proba`): each prediction is the argmax over `predict_proba` restricted to the 7 shared classes.

Expected outputs:

- `artifacts/metrics/exp03_robustness/EXP-03_<run_id>.json` — run record with in-domain/OOD metrics and robustness drop
- `reports/tables/EXP-03_<run_id>_in_domain_predictions.csv` and `*_ood_predictions.csv`
- `artifacts/metrics/exp03_robustness/EXP-03_<run_id>_predictions.csv` — combined predictions
- `reports/tables/EXP-03_<run_id>_per_class_drop.csv` and `reports/figures/exp03_per_class_drop.png`
- `reports/tables/EXP-03_<run_id>_ood_per_class_accuracy.csv` and `reports/figures/EXP-03_<run_id>_ood_per_class_accuracy.png` — OOD 10-class per-class accuracy
- `reports/tables/EXP-03_<run_id>_ood_confusion.csv` and `reports/figures/EXP-03_<run_id>_ood_confusion.png` — OOD confusion matrix (canonical vocab + `_other_` column for train-only predictions)
- `reports/summaries/robustness_summary.md` — deployment-oriented summary
- `reports/summaries/exp03_failure_gallery.md` — qualitative OOD error index
- `reports/tables/exp03_robustness_suite_leaderboard.csv` — batch suite comparison (all feature × model runs)

### 5. Real-time deployment (EXP-04)

Run the webcam runtime in safe dry-run mode (no keyboard events):

```bash
python scripts/run_realtime_demo.py \
  --model artifacts/models/EXP-01_svm_hybrid.joblib \
  --runtime-config configs/runtime/default.yaml \
  --camera-index 0 \
  --dry-run \
  --show-overlay
```

Timed runtime benchmark (latency/FPS report, dry-run by default):

```bash
python scripts/benchmark_runtime.py \
  --model artifacts/models/EXP-01_svm_hybrid.joblib \
  --runtime-config configs/runtime/default.yaml \
  --duration-seconds 60 \
  --dry-run \
  --output artifacts/runtime/runtime_eval_001.json
```

Live keyboard control after validating dry-run behavior (explicit opt-in):

```bash
python scripts/run_realtime_demo.py --model artifacts/models/EXP-01_cnn_hybrid.pt --runtime-config configs/runtime/default.yaml --camera-index 0 --enable-key-dispatch
```

Replace the model path with your Phase 3 champion artifact (e.g. top `EXP-01_*_hybrid` under `artifacts/models/`). Press `q` in the overlay window or `Ctrl+C` to stop safely.

Expected outputs:

- Live overlay with predicted gesture and confidence (when `--show-overlay` is set)
- `artifacts/runtime/runtime_eval_<timestamp>.json` — latency/FPS summary
- `artifacts/runtime/runtime_session_<timestamp>.jsonl` — per-frame event log
- `artifacts/metrics/exp04_realtime_deployment/EXP-04_<run_id>.json` — EXP-04 metrics record

## Experiment IDs

| ID | Description |
|----|-------------|
| EXP-01 | Model comparison on common features and splits (HaGRID subset) |
| EXP-02 | Feature ablation: keypoints / HOG / hybrid (HaGRID subset) |
| EXP-03 | Robustness: train HaGRID subset, test LeapGestRecog (OOD) |
| EXP-04 | Real-time deployment evaluation |
