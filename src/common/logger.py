"""Shared logging format for scripts and runtime modules."""

from __future__ import annotations

import logging
from pathlib import Path

from src.common.path_manager import resolve_project_root

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s"
    "%(run_id_suffix)s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured_loggers: set[str] = set()


class _RunIdFilter(logging.Filter):
    def __init__(self, run_id: str | None) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id_suffix = f" | run={self.run_id}" if self.run_id else ""
        return True


def get_logger(
    module_name: str,
    run_id: str | None = None,
    *,
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    """Return a configured logger with console and optional file handlers."""
    logger = logging.getLogger(module_name)

    if module_name in _configured_loggers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_RunIdFilter(run_id))
    logger.addHandler(console_handler)

    if log_to_file:
        try:
            root = resolve_project_root()
            log_dir = root / "reports" / "summaries"
            log_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"_{run_id}" if run_id else ""
            log_path = log_dir / f"{module_name.replace('.', '_')}{suffix}.log"
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler.addFilter(_RunIdFilter(run_id))
            logger.addHandler(file_handler)
        except FileNotFoundError:
            pass

    _configured_loggers.add(module_name)
    return logger
