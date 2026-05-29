"""Parallel batch feature extraction using a process pool."""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from src.features.extraction import extract_sample_features
from src.features.quality_checks import apply_quality_flags


def default_worker_count() -> int:
    """Default pool size: leave one CPU core free when possible."""
    count = os.cpu_count() or 1
    return max(1, count - 1)


def resolve_num_workers(requested: int | None, n_samples: int) -> int:
    """Resolve worker count; 1 means serial extraction in the parent process."""
    if requested is None:
        requested = default_worker_count()
    if requested <= 1 or n_samples <= 1:
        return 1
    return min(int(requested), n_samples)


def _pool_initializer() -> None:
    """Ensure each child process owns a fresh MediaPipe detector instance."""
    from src.features.hand_detector import reset_detector

    reset_detector()


def _extract_one(
    task: tuple[int, dict[str, Any], str, dict[str, Any], bool, float, int | None],
) -> tuple[int, dict[str, Any]]:
    idx, sample, feature_family, config, apply_quality, min_confidence, min_visible = task
    record = extract_sample_features(sample, feature_family, config)
    if apply_quality:
        apply_quality_flags(
            record,
            min_confidence=min_confidence,
            min_visible_landmarks=min_visible,
        )
    return idx, record


def _extract_serial(
    samples: list[dict[str, Any]],
    feature_family: str,
    config: dict[str, Any],
    *,
    apply_quality: bool,
    min_confidence: float,
    min_visible_landmarks: int | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample in samples:
        record = extract_sample_features(sample, feature_family, config)
        if apply_quality:
            apply_quality_flags(
                record,
                min_confidence=min_confidence,
                min_visible_landmarks=min_visible_landmarks,
            )
        records.append(record)
    return records


def extract_samples_batch(
    samples: list[dict[str, Any]],
    feature_family: str,
    config: dict[str, Any],
    *,
    num_workers: int | None = None,
    chunksize: int | None = None,
    apply_quality: bool = False,
    min_confidence: float = 0.5,
    min_visible_landmarks: int | None = None,
) -> list[dict[str, Any]]:
    """
    Extract features for all manifest samples, optionally using a process pool.

    Results preserve manifest row order. Uses the ``spawn`` multiprocessing
    context so MediaPipe / OpenGL state is not inherited from the parent.
    """
    if not samples:
        return []

    workers = resolve_num_workers(num_workers, len(samples))
    if workers == 1:
        return _extract_serial(
            samples,
            feature_family,
            config,
            apply_quality=apply_quality,
            min_confidence=min_confidence,
            min_visible_landmarks=min_visible_landmarks,
        )

    if chunksize is None:
        chunksize = max(1, len(samples) // (workers * 4))

    tasks = [
        (
            index,
            sample,
            feature_family,
            config,
            apply_quality,
            min_confidence,
            min_visible_landmarks,
        )
        for index, sample in enumerate(samples)
    ]

    ordered: list[dict[str, Any] | None] = [None] * len(samples)
    mp_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_pool_initializer,
        mp_context=mp_ctx,
    ) as executor:
        for idx, record in executor.map(_extract_one, tasks, chunksize=chunksize):
            ordered[idx] = record

    if any(record is None for record in ordered):
        raise RuntimeError("Parallel extraction returned incomplete results")
    return ordered  # type: list[dict[str, Any]]
