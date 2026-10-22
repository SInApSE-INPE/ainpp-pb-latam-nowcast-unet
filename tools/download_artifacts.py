"""Compatibility wrapper for the installed artifact download command."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
main = importlib.import_module("unet_nowcast.download_artifacts").main

if __name__ == "__main__":
    raise SystemExit(main())
