"""Centralize project-root discovery and artifact path generation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

# Canonical directories from doc/project_overview.md
CANONICAL_DIRECTORIES: tuple[str, ...] = (
    "configs/datasets",
    "configs/features",
    "configs/models",
    "configs/runtime",
    "configs/experiments",
    "data/raw/leapgestrecog",
    "data/raw/hagrid",
    "data/interim",
    "data/processed",
    "data/splits",
    "data/models",
    "artifacts/features",
    "artifacts/models",
    "artifacts/metrics",
    "artifacts/runtime",
    "reports/figures",
    "reports/tables",
    "reports/summaries",
    "scripts/",
    "src/common",
    "src/data",
    "src/features",
    "src/models",
    "src/evaluation",
    "src/runtime",
    "tests/smoke",
    "tests/integration",
)

# Marker files/dirs used to detect repository root
_ROOT_MARKERS: tuple[str, ...] = ("README.md", "requirements.txt", "configs")

_ARTIFACT_CATEGORIES = frozenset({"features", "models", "metrics", "runtime"})


def resolve_project_root(start: Path | str | None = None) -> Path:
    """Walk upward from *start* (or cwd) until a project root marker is found."""
    current = Path(start or os.getcwd()).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate

    raise FileNotFoundError(
        "Could not resolve project root. Expected one of: "
        + ", ".join(_ROOT_MARKERS)
    )


def ensure_directories(
    project_root: Path | str,
    directories: Iterable[str] = CANONICAL_DIRECTORIES,
) -> list[Path]:
    """Create canonical directories under *project_root*; return paths created."""
    root = Path(project_root).resolve()
    created: list[Path] = []
    for rel in directories:
        path = root / rel
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
    return created


def build_artifact_path(
    category: str,
    name: str,
    extension: str,
    *,
    project_root: Path | str | None = None,
    create_parents: bool = True,
) -> Path:
    """
    Build a canonical artifact path under ``artifacts/<category>/``.

    Naming follows doc/project_overview.md, e.g.
    ``artifacts/metrics/{experiment_id}_{run_id}.json``.
    """
    if category not in _ARTIFACT_CATEGORIES:
        raise ValueError(
            f"Invalid artifact category '{category}'. "
            f"Expected one of: {sorted(_ARTIFACT_CATEGORIES)}"
        )

    ext = extension.lstrip(".")
    root = Path(project_root) if project_root else resolve_project_root()
    path = root / "artifacts" / category / f"{name}.{ext}"

    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)

    return path
