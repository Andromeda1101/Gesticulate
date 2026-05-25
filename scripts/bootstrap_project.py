#!/usr/bin/env python3
"""Bootstrap canonical directory layout and optional placeholder configs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.logger import get_logger
from src.common.path_manager import (
    CANONICAL_DIRECTORIES,
    ensure_directories,
    resolve_project_root,
)

# Minimal placeholder templates when --with-placeholders creates missing files
_PLACEHOLDER_CONFIGS: dict[str, str] = {
    "configs/datasets/leapgestrecog.yaml": (
        "dataset_name: leapgestrecog\n"
        "root_path: data/raw/leapgestrecog\n"
        "label_vocabulary: []\n"
        "split_strategy:\n  method: stratified\n  seed: 42\n"
    ),
    "configs/datasets/hagrid_subset.yaml": (
        "dataset_name: hagrid_subset\n"
        "root_path: data/raw/hagrid\n"
        "label_vocabulary: []\n"
        "split_strategy:\n  method: fixed_eval\n  seed: 42\n"
    ),
    "configs/features/default.yaml": "feature_version: v1\n",
    "configs/models/baselines.yaml": "algorithm_registry: {}\n",
    "configs/runtime/default.yaml": (
        "camera:\n  index: 0\n"
        "gesture_mapping: {}\n"
    ),
    "configs/experiments/exp01_model_comparison.yaml": (
        "experiment_id: EXP-01\n"
        "name: model_comparison\n"
    ),
    "configs/experiments/exp02_feature_ablation.yaml": (
        "experiment_id: EXP-02\n"
        "name: feature_ablation\n"
    ),
    "configs/experiments/exp03_robustness.yaml": (
        "experiment_id: EXP-03\n"
        "name: robustness_cross_dataset\n"
    ),
    "configs/experiments/exp04_realtime_deployment.yaml": (
        "experiment_id: EXP-04\n"
        "name: realtime_deployment\n"
    ),
}


def _write_placeholders(
    project_root: Path,
    *,
    force: bool,
) -> tuple[list[str], list[str]]:
    created: list[str] = []
    skipped: list[str] = []

    for rel_path, content in _PLACEHOLDER_CONFIGS.items():
        target = project_root / rel_path
        if target.exists() and not force:
            skipped.append(rel_path)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        created.append(rel_path)

    return created, skipped


def _write_summary(
    project_root: Path,
    *,
    dirs_created: list[Path],
    config_created: list[str],
    config_skipped: list[str],
) -> Path:
    summary_dir = project_root / "reports" / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = summary_dir / f"bootstrap_summary_{timestamp}.json"

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "directories_created": [str(p.relative_to(project_root)) for p in dirs_created],
        "placeholders_created": config_created,
        "placeholders_skipped": config_skipped,
    }

    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap Gesticulate project layout and configs.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Target project root (default: repository root)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite placeholder configs when using --with-placeholders",
    )
    parser.add_argument(
        "--with-placeholders",
        action="store_true",
        help="Create minimal placeholder config files if missing",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    logger = get_logger("bootstrap_project")

    logger.info("Bootstrapping project at %s", project_root)

    dirs_created = ensure_directories(project_root, CANONICAL_DIRECTORIES)
    logger.info("Ensured %d new directories", len(dirs_created))

    config_created: list[str] = []
    config_skipped: list[str] = []
    if args.with_placeholders:
        config_created, config_skipped = _write_placeholders(
            project_root,
            force=args.force,
        )
        logger.info(
            "Placeholders: created=%d skipped=%d",
            len(config_created),
            len(config_skipped),
        )

    summary_path = _write_summary(
        project_root,
        dirs_created=dirs_created,
        config_created=config_created,
        config_skipped=config_skipped,
    )
    logger.info("Bootstrap summary written to %s", summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
