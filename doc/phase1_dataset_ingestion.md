# Phase 1: Dataset Ingestion and Split Preparation

## Phase Objectives
- Ingest the LeapGestRecog dataset and the planned HaGRID subset into a common internal format.
- Normalize gesture labels, directory metadata, and sample identifiers.
- Generate reproducible train, validation, and test splits for the main dataset.
- Produce dataset manifests and summary statistics that all later phases can consume.

## Prerequisites
- Directory and config scaffolding from `doc/phase0_project_foundation.md`
- Shared contracts from `doc/project_overview.md`
- Access to raw datasets placed under:
  - `data/raw/leapgestrecog/`
  - `data/raw/hagrid/`

## Code Module Plan

### 1. `src/data/dataset_registry.py`
- **Function**: Register dataset-specific adapters under a shared API.
- **Suggested interface**:
  - `get_dataset_adapter(dataset_name: str)`
  - `list_supported_datasets() -> list[str]`
- **Input/Output**:
  - Input: dataset name such as `leapgestrecog` or `hagrid_subset`
  - Output: adapter class or callable with parsing behavior
- **Core logic**:
  1. Map dataset names to adapter classes.
  2. Validate dataset availability from config.
  3. Return adapter for downstream indexing.
- **Dependent libraries**: none beyond core Python

### 2. `src/data/adapters/leapgestrecog_adapter.py`
- **Function**: Parse LeapGestRecog folder structure into standardized sample records.
- **Suggested interface**:
  - `index_samples(root_dir: str, label_map: dict) -> list[dict]`
- **Input/Output**:
  - Input: dataset root path, label map config
  - Output: list of sample metadata dictionaries
- **Core logic**:
  1. Traverse subject and gesture folders.
  2. Extract image paths and inferred labels.
  3. Generate stable `sample_id` values.
  4. Attach dataset-specific context such as subject ID.
- **Dependent libraries**: `pathlib`, `hashlib`

### 3. `src/data/adapters/hagrid_adapter.py`
- **Function**: Parse the selected HaGRID subset into the same sample schema used by the primary dataset.
- **Suggested interface**:
  - `index_samples(root_dir: str, subset_spec: dict) -> list[dict]`
- **Input/Output**:
  - Input: raw subset root, subset selection rules
  - Output: standardized sample metadata dictionaries
- **Core logic**:
  1. Read annotation files or folder metadata.
  2. Filter to target gesture classes overlapping with the primary dataset.
  3. Convert labels to the shared vocabulary.
  4. Store background and capture-context metadata useful for robustness analysis.
- **Dependent libraries**: `json`, `pathlib`, `pandas`

### 4. `src/data/label_mapper.py`
- **Function**: Convert dataset-native labels to a canonical project label set.
- **Suggested interface**:
  - `normalize_label(raw_label: str, dataset_name: str) -> str`
  - `validate_label_coverage(samples: list[dict], canonical_labels: list[str]) -> dict`
- **Core logic**:
  1. Apply dataset-specific mapping rules.
  2. Flag unknown labels.
  3. Return label coverage statistics.
- **Dependent libraries**: none beyond core Python

### 5. `src/data/split_generator.py`
- **Function**: Generate reproducible splits for training, validation, testing, and cross-validation.
- **Suggested interface**:
  - `create_primary_splits(samples: list[dict], seed: int) -> dict`
  - `create_stratified_folds(samples: list[dict], n_folds: int, seed: int) -> list[dict]`
- **Input/Output**:
  - Input: indexed sample list and split policy
  - Output: split manifest files identifying sample membership
- **Core logic**:
  1. Group by canonical label.
  2. Perform stratified 70/15/15 split.
  3. Generate 5-fold cross-validation memberships for training records.
  4. Save both flat split files and fold definitions.
- **Dependent libraries**: `scikit-learn`, `pandas`

### 6. `src/data/dataset_summary.py`
- **Function**: Produce dataset statistics and quality checks.
- **Suggested interface**:
  - `summarize_dataset(samples: list[dict]) -> dict`
  - `export_summary(summary: dict, output_path: str) -> None`
- **Output examples**:
  - sample counts per class
  - subject distribution
  - missing-file warnings
  - overlap between training label space and HaGRID subset
- **Dependent libraries**: `pandas`, `json`

### 7. `scripts/phase1/build_dataset_manifests.py`
- **Function**: CLI entrypoint for indexing raw data and producing manifests.
- **Suggested CLI arguments**:
  - `--dataset leapgestrecog`
  - `--config configs/datasets/leapgestrecog.yaml`
  - `--output data/interim/leapgestrecog_manifest.parquet`
- **Core logic**:
  1. Load dataset config.
  2. Select adapter.
  3. Index raw files.
  4. Normalize labels.
  5. Save manifest and summary report.

### 8. `scripts/phase1/generate_splits.py`
- **Function**: Create split and fold artifacts from the primary dataset manifest.
- **Suggested CLI arguments**:
  - `--manifest data/interim/leapgestrecog_manifest.parquet`
  - `--seed 42`
  - `--folds 5`
- **Core logic**:
  1. Load manifest.
  2. Validate label balance.
  3. Create train, val, test split assignments.
  4. Save split files to `data/splits/`.

## Data Flow & Interaction Design
- Raw image directories remain untouched under `data/raw/`.
- Indexed sample metadata is written to `data/interim/` as Parquet or CSV plus a JSON summary.
- Split membership should be stored separately from the base manifest to avoid duplicating metadata.
- Stable `sample_id` values must be reused by feature extraction and evaluation scripts.

```text
RawImages -> DatasetAdapter -> CanonicalManifest -> SplitGenerator -> SplitFiles
                                  |
                                  -> DatasetSummary -> reports/summaries/
```

### Suggested Storage Formats
- Manifest table: Parquet for efficient downstream loading
- Human-readable summary: JSON or Markdown
- Split definitions: JSON lines, CSV, or Parquet keyed by `sample_id`

## Verification & Testing Approach
- Check that every raw sample referenced in the manifest exists on disk.
- Validate that each sample has exactly one canonical label.
- Confirm class balance across train, validation, and test splits.
- Review a small random sample of records from both datasets to ensure label normalization is correct.
- Ensure there is no `sample_id` duplication across manifest rows.
- For HaGRID subset preparation, verify that only intended gesture classes are retained.

## Code Execution Method

### Environment Setup
1. Use the Python environment established in Phase 0.
2. Confirm dataset root paths in `configs/datasets/`.

### Dependency Installation
- Required packages for this phase:
  - `pandas`
  - `pyyaml`
  - `scikit-learn`
  - optionally `pyarrow` for Parquet support
- Example command:
  - `pip install pandas pyyaml scikit-learn pyarrow`

### Execution Steps and Example Commands
1. Build the LeapGestRecog manifest:
   - `python scripts/phase1/build_dataset_manifests.py --dataset leapgestrecog --config configs/datasets/leapgestrecog.yaml --output data/interim/leapgestrecog_manifest.parquet`
2. Build the HaGRID subset manifest:
   - `python scripts/phase1/build_dataset_manifests.py --dataset hagrid_subset --config configs/datasets/hagrid_subset.yaml --output data/interim/hagrid_subset_manifest.parquet`
3. Generate primary splits and folds:
   - `python scripts/phase1/generate_splits.py --manifest data/interim/leapgestrecog_manifest.parquet --seed 42 --folds 5`
4. Export a dataset summary:
   - `python scripts/phase1/export_dataset_report.py --manifest data/interim/leapgestrecog_manifest.parquet --output reports/summaries/leapgestrecog_summary.json`

### Expected Output or Result Example
- `data/interim/leapgestrecog_manifest.parquet`
- `data/interim/hagrid_subset_manifest.parquet`
- `data/splits/leapgestrecog_train_val_test.json`
- `data/splits/leapgestrecog_cv_folds.json`
- `reports/summaries/leapgestrecog_summary.json`
- Log output confirming class counts and split distributions

## Exit Criteria
- Both datasets can be indexed into a shared schema.
- Primary dataset splits are reproducible and stratified.
- Later phases can consume canonical manifests without reading raw folders directly.
