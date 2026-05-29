#!/bin/bash
# End-to-end dataset manifest, splits, and feature pipeline.
# Run inside an existing tmux pane (e.g. `tmux attach`) — do not use
# `tmux new-session './dataset_pipline.sh'` or the session closes when this exits.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
if ! "$PYTHON" -c "import cv2" 2>/dev/null; then
  echo "ERROR: OpenCV (cv2) not found for $PYTHON. Activate the project env first, e.g.:" >&2
  echo "  conda activate gesticulate   # or: source .venv/bin/activate" >&2
  exit 1
fi

# Limit parallel MediaPipe workers on memory-constrained hosts (override: HAGRID_WORKERS=4 ./...)
HAGRID_WORKERS="${HAGRID_WORKERS:-2}"
# Manifest rows per extract-and-flush cycle (uses configs/features/default.yaml if unset)
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-1000}"

_run() {
  echo "==> $*"
  "$@"
}

# build dataset manifests
_run "$PYTHON" scripts/build_dataset_manifests.py \
  --dataset hagrid_subset \
  --config configs/datasets/hagrid_subset.yaml \
  --output data/interim/hagrid_subset_manifest.parquet

_run "$PYTHON" scripts/build_dataset_manifests.py \
  --dataset leapgestrecog \
  --config configs/datasets/leapgestrecog.yaml \
  --output data/interim/leapgestrecog_manifest.parquet

# generate splits
_run "$PYTHON" scripts/generate_splits.py \
  --manifest data/interim/hagrid_subset_manifest.parquet \
  --config configs/datasets/hagrid_subset.yaml \
  --seed 42 \
  --folds 5

# extract features (HaGRID subset)
_run "$PYTHON" scripts/extract_features.py \
  --manifest data/interim/hagrid_subset_manifest.parquet \
  --feature-family geometric \
  --config configs/features/default.yaml \
  --output artifacts/features/hagrid_subset_geometric_v1.parquet \
  --workers "$HAGRID_WORKERS" \
  --batch-size "$EXTRACT_BATCH_SIZE"

_run "$PYTHON" scripts/extract_features.py \
  --manifest data/interim/hagrid_subset_manifest.parquet \
  --feature-family hog \
  --config configs/features/default.yaml \
  --output artifacts/features/hagrid_subset_hog_v1.parquet \
  --workers "$HAGRID_WORKERS" \
  --batch-size "$EXTRACT_BATCH_SIZE"

# build hybrid features
_run "$PYTHON" scripts/build_hybrid_features.py \
  --keypoint-features artifacts/features/hagrid_subset_geometric_v1.parquet \
  --hog-features artifacts/features/hagrid_subset_hog_v1.parquet \
  --output artifacts/features/hagrid_subset_hybrid_v1.parquet

# extract features for leapgestrecog
_run "$PYTHON" scripts/extract_features.py \
  --manifest data/interim/leapgestrecog_manifest.parquet \
  --feature-family geometric \
  --config configs/features/default.yaml \
  --output artifacts/features/leapgestrecog_geometric_v1.parquet \
  --workers "$HAGRID_WORKERS" \
  --batch-size "$EXTRACT_BATCH_SIZE"

_run "$PYTHON" scripts/extract_features.py \
  --manifest data/interim/leapgestrecog_manifest.parquet \
  --feature-family hog \
  --config configs/features/default.yaml \
  --output artifacts/features/leapgestrecog_hog_v1.parquet \
  --workers "$HAGRID_WORKERS" \
  --batch-size "$EXTRACT_BATCH_SIZE"

# build hybrid features
_run "$PYTHON" scripts/build_hybrid_features.py \
  --keypoint-features artifacts/features/leapgestrecog_geometric_v1.parquet \
  --hog-features artifacts/features/leapgestrecog_hog_v1.parquet \
  --output artifacts/features/leapgestrecog_hybrid_v1.parquet

echo "Pipeline finished successfully."
