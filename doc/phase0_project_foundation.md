# Phase 0: Project Foundation and Experiment Contracts

## Phase Objectives
- Establish the base repository structure needed by all later phases.
- Define the shared configuration, logging, artifact, and metadata conventions.
- Create the minimum scaffolding for reproducible experiments without implementing the full experimental logic yet.
- Fix experiment identifiers, file naming rules, and storage locations so downstream phases do not diverge.

## Prerequisites
- Source proposal in `doc/ML_proposal.md`
- Shared architecture reference in `doc/project_overview.md`
- No prior implementation output is required because this is the foundation phase.

## Code Module Plan

### 1. `src/common/config_loader.py`
- **Function**: Load and validate YAML or JSON configuration files for datasets, features, models, experiments, and runtime settings.
- **Suggested interface**:
  - `load_config(config_path: str) -> dict`
  - `merge_overrides(base_config: dict, cli_args: dict) -> dict`
- **Input/Output**:
  - Input: path to config file, optional command-line overrides
  - Output: normalized configuration dictionary
- **Core logic**:
  1. Read config file.
  2. Validate required keys by config type.
  3. Apply CLI overrides.
  4. Return normalized config object.
- **Suggested file path**: `src/common/config_loader.py`
- **Dependent libraries**: `pyyaml`, `json`, `pathlib`

### 2. `src/common/path_manager.py`
- **Function**: Centralize project-root discovery and artifact path generation.
- **Suggested interface**:
  - `resolve_project_root() -> Path`
  - `build_artifact_path(category: str, name: str, extension: str) -> Path`
- **Input/Output**:
  - Input: artifact category and logical artifact name
  - Output: canonical filesystem path
- **Core logic**:
  1. Detect project root.
  2. Create output directories when missing.
  3. Return standardized paths for artifacts, reports, and caches.
- **Dependent libraries**: `pathlib`, `os`

### 3. `src/common/logger.py`
- **Function**: Define a shared logging format for scripts and future runtime modules.
- **Suggested interface**:
  - `get_logger(module_name: str, run_id: str | None = None)`
- **Input/Output**:
  - Input: module name, optional run identifier
  - Output: configured logger instance
- **Core logic**:
  1. Configure console and file handlers.
  2. Attach timestamp, phase, run ID, and log level.
  3. Reuse the same format across scripts.
- **Dependent libraries**: `logging`

### 4. `src/common/run_registry.py`
- **Function**: Record experiment metadata and artifact pointers in machine-readable form.
- **Suggested interface**:
  - `create_run_record(experiment_id: str, config: dict) -> dict`
  - `save_run_record(record: dict, output_path: str) -> None`
- **Input/Output**:
  - Input: experiment ID, resolved configuration, timestamps
  - Output: JSON metadata file under `artifacts/metrics/` or `reports/summaries/`
- **Core logic**:
  1. Generate a unique run ID.
  2. Snapshot config values.
  3. Store intended input datasets and output paths.
  4. Save a run manifest before and after execution.
- **Dependent libraries**: `uuid`, `json`, `datetime`

### 5. `configs/` starter files
- **Function**: Provide stable config surfaces for later phases.
- **Files to create**:
  - `configs/datasets/leapgestrecog.yaml`
  - `configs/datasets/hagrid_subset.yaml`
  - `configs/features/default.yaml`
  - `configs/models/baselines.yaml`
  - `configs/experiments/exp01_model_comparison.yaml`
  - `configs/experiments/exp02_feature_ablation.yaml`
  - `configs/experiments/exp03_robustness.yaml`
  - `configs/runtime/default.yaml`
- **Content guidance**:
  - dataset roots
  - label vocabulary
  - split strategy
  - feature toggles
  - algorithm registry
  - runtime key mapping and debounce settings
- **Dependent libraries**: none at creation time

### 6. `scripts/phase0/bootstrap_project.py`
- **Function**: Create the expected directory tree and placeholder config files.
- **Suggested interface**:
  - CLI entrypoint with arguments such as `--project-root`, `--force`, `--with-placeholders`
- **Input/Output**:
  - Input: target root directory and bootstrap options
  - Output: created folders, copied template configs, bootstrap log
- **Core logic**:
  1. Create directories from the canonical layout.
  2. Create placeholder config files if missing.
  3. Write a bootstrap summary to `reports/summaries/`.
- **Dependent libraries**: `argparse`, `pathlib`

### 7. `requirements.txt`
- **Function**: Central dependency declaration for all phases.
- **Planned package groups**:
  - core utilities: `pyyaml`, `numpy`, `pandas`
  - CV and features: `opencv-python`, `mediapipe`, `scikit-image`
  - ML: `scikit-learn`, `joblib`
  - optional deep learning: `torch` or `tensorflow`
  - visualization/reporting: `matplotlib`, `seaborn`
  - runtime control: `pynput` or `pyautogui`
- **Note**: keep optional deep-learning dependencies marked clearly if not required for early phases.

## Data Flow & Interaction Design
- This phase defines control-plane data rather than experiment data.
- Config files should drive script behavior in later phases.
- The output of this phase is a stable folder tree plus config and manifest conventions.
- All future phases should write artifact metadata with a run ID and config fingerprint.

```text
Proposal -> ProjectOverview -> ConfigTemplates -> BootstrapScript
                                         |
                                         -> DirectoryTree
                                         -> RunMetadataContract
                                         -> ArtifactNamingRules
```

## Verification & Testing Approach
- Confirm that the bootstrap script would create every required folder in `doc/project_overview.md`.
- Review config templates to ensure every later phase has a home for its settings.
- Check that experiment IDs `EXP-01` to `EXP-04` are defined consistently.
- Perform a dry-run design review: verify that a later phase could resolve paths, load configs, and emit artifacts without inventing new conventions.
- Add lightweight smoke checks later to confirm config loading and path creation behavior.

## Code Execution Method

### Environment Setup
1. Install Python 3.10 or 3.11.
2. Create a virtual environment.
3. Activate the environment before installing packages.

### Dependency Installation
- Planned `requirements.txt` contents should include:
  - `numpy`
  - `pandas`
  - `pyyaml`
  - `opencv-python`
  - `mediapipe`
  - `scikit-image`
  - `scikit-learn`
  - `joblib`
  - `matplotlib`
  - `seaborn`
  - `pynput`
- Example install command:
  - `pip install -r requirements.txt`

### Execution Steps and Example Commands
1. Bootstrap the directory structure:
   - `python scripts/phase0/bootstrap_project.py --project-root . --with-placeholders`
2. Validate a config file shape:
   - `python -m scripts.phase0.validate_config --config configs/experiments/exp01_model_comparison.yaml`
3. Generate a run manifest template:
   - `python -m scripts.phase0.init_run --experiment-id EXP-01`

### Expected Output or Result Example
- Newly created folders under `configs/`, `data/`, `artifacts/`, `reports/`, `scripts/`, and `src/`
- Placeholder config files ready for later editing
- Bootstrap log file under `reports/summaries/`
- Run metadata template under `artifacts/metrics/`

## Exit Criteria
- The repository layout matches `doc/project_overview.md`.
- Shared configs exist for datasets, features, experiments, and runtime.
- The project has a reproducibility contract for later scripts to follow.
