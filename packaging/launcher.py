"""Entry point of the standalone Windows package (ForecastStudio.exe).

Double-click: starts the web interface and opens the browser.
With arguments: `ForecastStudio.exe run --dataset ...` works like `python -m tsfstudio run ...`.

Data, runs and uploads live next to the executable (data\\tsf, runs, uploads) unless TSFSTUDIO_* variables say otherwise.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path

COMMANDS = {"data-check", "index", "hash", "template", "convert", "engines", "estimate", "run", "doctor", "golden", "rank", "app"}


def _base_dir() -> Path:
    return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path.cwd()


def main() -> int:
    base = _base_dir()
    for var, sub in (("TSFSTUDIO_DATA_DIR", "data/tsf"), ("TSFSTUDIO_RUNS_DIR", "runs"), ("TSFSTUDIO_UPLOADS_DIR", "uploads")):
        if not os.environ.get(var):
            p = base / sub
            p.mkdir(parents=True, exist_ok=True)
            os.environ[var] = str(p)
    # numba (statsforecast) caches compiled functions; inside the package it must be a writable folder.
    os.environ.setdefault("NUMBA_CACHE_DIR", str(base / "cache" / "numba"))
    os.environ.setdefault("PYTHONUTF8", "1")

    from tsfstudio.cli import main as cli_main

    args = sys.argv[1:]
    if not args or args[0] not in COMMANDS:
        args = ["app", *args]
    return cli_main(args)


if __name__ == "__main__":
    multiprocessing.freeze_support()          # worker processes (spawn) re-enter here in the frozen app
    sys.exit(main())
