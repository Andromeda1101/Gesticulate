# Phase 2: Feature Extraction and Feature Store Generation

## Phase Objectives
- Build a reusable feature extraction pipeline for the two feature families required by the proposal:
  - hand keypoints and geometric descriptors
  - HOG-based appearance features
- Standardize feature vector storage so classical models and future deep models can consume the same sample IDs.
- Produce feature manifests for keypoints-only, HOG-only, and hybrid feature sets.

## Prerequisites
- Dataset manifests and split files from `doc/phase1_dataset_ingestion.md`
- Shared artifact and configuration contracts from `doc/project_overview.md`
- Access to raw images referenced by the generated manifests

## Code Module Plan

### 1. `src/features/hand_detector.py`
- **Function**: Wrap MediaPipe Hands or an equivalent detector behind a clean interface.
- **Suggested interface**:
  - `detect_hand_landmarks(image: np.ndarray, config: dict) -> dict | None`
- **Input/Output**:
  - Input: image array and detector settings
  - Output: wrist and 21 landmark coordinates, handedness metadata, confidence scores, or `None` if detection fails
- **Core logic**:
  1. Convert input image to detector-compatible format.
  2. Run hand landmark detection.
  3. Return normalized landmark object or failure state.
- **Dependent libraries**: `mediapipe`, `opencv-python`, `numpy`

### 2. `src/features/geometric_features.py`
- **Function**: Convert detected landmarks into translation- and scale-invariant geometric descriptors.
- **Suggested interface**:
  - `normalize_landmarks(landmarks: np.ndarray) -> np.ndarray`
  - `compute_pairwise_distances(landmarks: np.ndarray) -> np.ndarray`
  - `compute_joint_angles(landmarks: np.ndarray) -> np.ndarray`
  - `build_geometric_vector(landmarks: np.ndarray) -> np.ndarray`
- **Core logic**:
  1. Use wrist-relative centering.
  2. Scale by hand bounding-box diagonal or equivalent hand size proxy.
  3. Compute keypoint coordinates, pairwise distances, and selected joint angles.
  4. Concatenate outputs into a fixed-length vector.
- **Dependent libraries**: `numpy`

### 3. `src/features/hog_features.py`
- **Function**: Extract HOG descriptors from a cropped hand region.
- **Suggested interface**:
  - `crop_hand_region(image: np.ndarray, landmarks: np.ndarray | None) -> np.ndarray`
  - `extract_hog_descriptor(image_crop: np.ndarray, config: dict) -> np.ndarray`
- **Input/Output**:
  - Input: source image, optional landmarks or bounding box, HOG config
  - Output: fixed-length HOG feature vector
- **Core logic**:
  1. Determine crop region using hand landmarks or fallback heuristics.
  2. Resize crop to configured dimensions.
  3. Compute HOG descriptor.
  4. Return vector and optional quality metadata.
- **Dependent libraries**: `opencv-python`, `scikit-image`, `numpy`

### 4. `src/features/feature_combiner.py`
- **Function**: Merge multiple feature families into a single hybrid feature vector.
- **Suggested interface**:
  - `concatenate_features(feature_blocks: dict[str, np.ndarray]) -> np.ndarray`
  - `build_feature_record(sample_id: str, feature_family: str, vector: np.ndarray, metadata: dict) -> dict`
- **Core logic**:
  1. Enforce stable family ordering.
  2. Concatenate vectors.
  3. Store feature dimensions and quality flags.
- **Dependent libraries**: `numpy`

### 5. `src/features/feature_store.py`
- **Function**: Persist extracted features and manifests in a reproducible layout.
- **Suggested interface**:
  - `save_feature_matrix(records: list[dict], output_path: str) -> None`
  - `load_feature_matrix(path: str) -> "FeatureTable"`
  - `save_feature_manifest(manifest: dict, output_path: str) -> None`
- **Storage guidance**:
  - feature table in Parquet, NumPy archive, or HDF5
  - metadata manifest in JSON
- **Dependent libraries**: `pandas`, `numpy`, optional `pyarrow`

### 6. `src/features/quality_checks.py`
- **Function**: Track extraction failures and coverage quality.
- **Suggested interface**:
  - `evaluate_feature_coverage(records: list[dict]) -> dict`
  - `flag_low_confidence_samples(records: list[dict], threshold: float) -> list[str]`
- **Core logic**:
  1. Count detector failures.
  2. Track missing values.
  3. Flag samples needing exclusion or review.
- **Dependent libraries**: `pandas`

### 7. `scripts/phase2/extract_features.py`
- **Function**: Main CLI entrypoint for batch feature extraction.
- **Suggested CLI arguments**:
  - `--manifest data/interim/leapgestrecog_manifest.parquet`
  - `--feature-family keypoints`
  - `--config configs/features/default.yaml`
  - `--output artifacts/features/leapgestrecog_keypoints_v1.parquet`
- **Core logic**:
  1. Load sample manifest.
  2. Iterate through images.
  3. Extract requested feature family.
  4. Persist feature matrix and manifest.
  5. Save extraction summary.

### 8. `scripts/phase2/build_hybrid_features.py`
- **Function**: Join previously generated feature families into a hybrid store.
- **Suggested CLI arguments**:
  - `--keypoint-features ...`
  - `--hog-features ...`
  - `--output artifacts/features/leapgestrecog_hybrid_v1.parquet`
- **Core logic**:
  1. Align by `sample_id`.
  2. Validate matching label assignments.
  3. Concatenate feature blocks.
  4. Write hybrid features plus a schema manifest.

## Data Flow & Interaction Design
- Input manifests from Phase 1 define the authoritative sample order.
- Feature extraction should never generate new sample IDs.
- Each feature family should produce:
  - a feature matrix
  - a schema manifest with vector length, version, and extraction settings
  - a quality report with failure counts

```text
SampleManifest -> HandDetector -> GeometricVector ----\
                                                      +-> FeatureStore -> FeatureManifest
SampleManifest -> CropAndHOG -------------------------/
```

### Suggested Feature Families
- `keypoints_raw`
- `geometric`
- `hog`
- `hybrid_keypoints_hog`

### Suggested Versioning Scheme
- `v1`: initial baseline parameters
- `v2`: tuned parameters after ablation or quality fixes

## Verification & Testing Approach
- Confirm fixed feature dimensionality for each family.
- Check that the number of feature rows matches the number of valid input samples.
- Measure the percentage of detection failures and inspect representative failures.
- Validate that hybrid vectors use the same sample order as their source feature families.
- Compare a few manual calculations of distances or angles against the implementation design.
- Confirm that all feature artifacts include config metadata for reproducibility.

## Code Execution Method

### Environment Setup
1. Use the environment from Phase 0.
2. Ensure raw images and Phase 1 manifests exist.

### Dependency Installation
- Required packages for this phase:
  - `opencv-python`
  - `mediapipe`
  - `numpy`
  - `pandas`
  - `scikit-image`
  - optional `pyarrow`
- Example command:
  - `pip install opencv-python mediapipe numpy pandas scikit-image pyarrow`

### Execution Steps and Example Commands
1. Extract keypoint and geometric features:
   - `python scripts/phase2/extract_features.py --manifest data/interim/leapgestrecog_manifest.parquet --feature-family geometric --config configs/features/default.yaml --output artifacts/features/leapgestrecog_geometric_v1.parquet`
2. Extract HOG features:
   - `python scripts/phase2/extract_features.py --manifest data/interim/leapgestrecog_manifest.parquet --feature-family hog --config configs/features/default.yaml --output artifacts/features/leapgestrecog_hog_v1.parquet`
3. Build hybrid features:
   - `python scripts/phase2/build_hybrid_features.py --keypoint-features artifacts/features/leapgestrecog_geometric_v1.parquet --hog-features artifacts/features/leapgestrecog_hog_v1.parquet --output artifacts/features/leapgestrecog_hybrid_v1.parquet`
4. Export quality summary:
   - `python scripts/phase2/export_feature_report.py --feature-manifest artifacts/features/leapgestrecog_geometric_v1_manifest.json --output reports/summaries/feature_report_geometric_v1.json`

### Expected Output or Result Example
- `artifacts/features/leapgestrecog_geometric_v1.parquet`
- `artifacts/features/leapgestrecog_hog_v1.parquet`
- `artifacts/features/leapgestrecog_hybrid_v1.parquet`
- Matching manifest JSON files with feature dimensions and config hashes
- Quality report summarizing detection success rate and excluded samples

## Exit Criteria
- All required feature families can be produced from the Phase 1 manifests.
- Feature stores are versioned and aligned by `sample_id`.
- The outputs are ready for direct consumption by model training scripts in Phase 3.
