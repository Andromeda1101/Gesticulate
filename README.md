# Gesticulate

ML-based visual gesture recognition for keyboard control. This repository follows a phased implementation plan documented under `doc/`.

## Phase 0: Project Foundation

Phase 0 establishes shared configuration, logging, artifact paths, and run metadata conventions.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Bootstrap layout

```bash
python scripts/bootstrap_project.py --project-root . --with-placeholders
```

### Smoke tests

```bash
pytest tests/smoke/test_phase0_foundation.py -q
```

## Repository layout

See [doc/project_overview.md](doc/project_overview.md) for the canonical directory structure, experiment IDs (`EXP-01`–`EXP-04`), and artifact naming rules.

## Experiment IDs

| ID | Description |
|----|-------------|
| EXP-01 | Model comparison on common features and splits |
| EXP-02 | Feature ablation (keypoints / HOG / hybrid) |
| EXP-03 | Robustness: train LeapGestRecog, test HaGRID subset |
| EXP-04 | Real-time deployment evaluation |
