"""Shared utilities: config, paths, logging, run metadata."""

from src.common.config_loader import load_config, merge_overrides
from src.common.logger import get_logger
from src.common.path_manager import build_artifact_path, resolve_project_root
from src.common.run_registry import create_run_record, save_run_record

__all__ = [
    "load_config",
    "merge_overrides",
    "get_logger",
    "resolve_project_root",
    "build_artifact_path",
    "create_run_record",
    "save_run_record",
]
