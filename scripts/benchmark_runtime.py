#!/usr/bin/env python3
"""Controlled runtime benchmark without keyboard dispatch by default (EXP-04)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.path_manager import build_artifact_path, resolve_project_root
from src.runtime.pipeline import build_session_config, run_runtime_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark runtime latency and FPS over a timed session (EXP-04)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Champion model artifact (.joblib or .pt)",
    )
    parser.add_argument(
        "--runtime-config",
        default="configs/runtime/default.yaml",
        help="Runtime YAML config path",
    )
    parser.add_argument(
        "--feature-config",
        default="configs/features/default.yaml",
        help="Feature extraction YAML config path",
    )
    parser.add_argument("--camera-index", type=int, default=None, help="Webcam device index")
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress OS keyboard events (default: true)",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=60.0,
        help="Benchmark duration in seconds",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Benchmark report JSON path (default: artifacts/runtime/runtime_eval_<timestamp>.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = resolve_project_root()

    output = args.output
    if output is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = str(build_artifact_path("runtime", f"runtime_eval_{timestamp}", "json", project_root=root))
    elif not Path(output).is_absolute():
        output = str(root / output)

    session, runtime_cfg, feature_cfg = build_session_config(
        model=args.model or "",
        runtime_config=args.runtime_config,
        feature_config=args.feature_config,
        camera_index=args.camera_index,
        dry_run=args.dry_run,
        enable_key_dispatch=False,
        show_overlay=False,
        duration_seconds=args.duration_seconds,
        output=output,
    )

    summary = run_runtime_session(session, runtime_cfg, feature_cfg)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
