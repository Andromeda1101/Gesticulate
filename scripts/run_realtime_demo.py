#!/usr/bin/env python3
"""Interactive real-time gesture-to-keyboard demo (EXP-04)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.path_manager import resolve_project_root
from src.runtime.pipeline import build_session_config, run_runtime_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run webcam gesture recognition with optional keyboard dispatch (EXP-04)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Champion model artifact (.joblib or .pt); falls back to runtime config",
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
        help="Log actions without sending OS keyboard events (default: true)",
    )
    parser.add_argument(
        "--enable-key-dispatch",
        action="store_true",
        help="Enable real OS keyboard events (disables dry-run)",
    )
    parser.add_argument(
        "--show-overlay",
        action="store_true",
        help="Display live prediction overlay window (press q to quit)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional session summary JSON output path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dry_run = False if args.enable_key_dispatch else args.dry_run

    session, runtime_cfg, feature_cfg = build_session_config(
        model=args.model or "",
        runtime_config=args.runtime_config,
        feature_config=args.feature_config,
        camera_index=args.camera_index,
        dry_run=dry_run,
        enable_key_dispatch=args.enable_key_dispatch,
        show_overlay=args.show_overlay,
        output=args.output,
    )

    summary = run_runtime_session(session, runtime_cfg, feature_cfg)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
