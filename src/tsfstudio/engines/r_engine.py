"""R motoru: orijinal R modellerini `Rscript` ile çalıştırır.

İş protokolü (Python <-> R), temp bir klasörde:
  yazılır  : job.json, series.bin (little-endian float64, seriler art arda)
  okunur   : forecasts.bin (float64, SÜTUN-ana), summary.json, progress.json, error.txt
Hiçbir `rm`/`os.system` kullanılmaz; klasör Python tarafından temizlenir (Windows/Linux uyumlu).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from .base import Engine, EngineCancelled, EngineError, EngineJob, EngineResult, ProgressFn

_POLL_SEC = 0.4
_LOG_TAIL_CHARS = 2000


def default_worker_path() -> Path:
    env = os.environ.get("TSFSTUDIO_R_WORKER")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "r" / "worker.R"


class REngine(Engine):
    name = "r"
    LOCAL_METHODS = ("ses", "ets", "theta", "tbats", "arima", "dhr_arima", "snaive")
    GLOBAL_METHODS = ("pooled_regression", "catboost")

    def __init__(self, rscript: str | None = None, worker: str | Path | None = None,
                 workdir: str | Path | None = None, keep_job_dir: bool = False):
        self.rscript = rscript or os.environ.get("TSFSTUDIO_RSCRIPT") or shutil.which("Rscript")
        self.worker = Path(worker) if worker else default_worker_path()
        self.workdir = str(workdir) if workdir else None
        self.keep_job_dir = keep_job_dir

    # ------------------------------------------------------------------ availability
    def available(self) -> tuple[bool, str]:
        if not self.rscript:
            return False, "Rscript not found (R is installed in the Docker image; locally, install R or set TSFSTUDIO_RSCRIPT)."
        if not self.worker.is_file():
            return False, f"R worker not found: {self.worker}"
        try:
            out = subprocess.run([self.rscript, "--version"], capture_output=True, text=True, timeout=30)
            return True, (out.stderr or out.stdout).strip().splitlines()[0]
        except Exception as exc:  # noqa: BLE001
            return False, f"Rscript could not be run: {exc}"

    # ------------------------------------------------------------------ run
    def run(self, job: EngineJob, progress: ProgressFn | None = None,
            cancel: threading.Event | None = None) -> EngineResult:
        ok, msg = self.available()
        if not ok:
            raise EngineError(msg)
        job.validate()
        if not self.supports(job.task, job.method):
            raise EngineError(f"The R engine does not support the method '{job.method}' for the task '{job.task}'.")

        job_dir = Path(tempfile.mkdtemp(prefix="tsfjob_", dir=self.workdir))
        try:
            return self._run_in(job_dir, job, progress, cancel)
        finally:
            if not self.keep_job_dir:
                shutil.rmtree(job_dir, ignore_errors=True)

    def _write_inputs(self, job_dir: Path, job: EngineJob) -> None:
        lengths = [int(len(s)) for s in job.train]
        with open(job_dir / "series.bin", "wb") as fh:
            for s in job.train:
                np.ascontiguousarray(s, dtype="<f8").tofile(fh)
        spec = {
            "task": job.task, "method": job.method, "horizon": int(job.horizon),
            "horizons": None if job.horizons is None else [int(h) for h in job.horizons],
            "seasonality": [float(x) for x in job.seasonality], "lengths": lengths,
            "integer_conversion": bool(job.integer_conversion), "seed": int(job.seed),
            "workers": int(max(1, job.workers)), "lag": None if job.lag is None else int(job.lag),
            "params": job.params or {},
        }
        (job_dir / "job.json").write_text(json.dumps(spec), encoding="utf-8")

    def _run_in(self, job_dir: Path, job: EngineJob, progress, cancel) -> EngineResult:
        self._write_inputs(job_dir, job)
        log_path = job_dir / "worker.log"
        t0 = time.time()
        with open(log_path, "wb") as log:
            proc = subprocess.Popen([self.rscript, "--vanilla", str(self.worker), str(job_dir)],
                                    stdout=log, stderr=subprocess.STDOUT)
            last = None
            while True:
                try:
                    rc = proc.wait(timeout=_POLL_SEC)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if cancel is not None and cancel.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise EngineCancelled("The job was cancelled by the user.")
                last = self._report_progress(job_dir, progress, last)
        if progress:
            progress("finished", len(job.train), len(job.train))
        if rc != 0:
            raise EngineError(self._failure_text(job_dir, log_path, rc))
        return self._read_outputs(job_dir, job, time.time() - t0)

    @staticmethod
    def _report_progress(job_dir: Path, progress, last):
        if progress is None:
            return last
        try:
            p = json.loads((job_dir / "progress.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return last
        key = (p.get("stage"), p.get("done"))
        if key != last:
            progress(str(p.get("stage")), int(p.get("done", 0)), int(p.get("total", 0)))
        return key

    @staticmethod
    def _failure_text(job_dir: Path, log_path: Path, rc: int) -> str:
        err = job_dir / "error.txt"
        if err.is_file():
            return f"The R worker reported an error (code {rc}): {err.read_text(encoding='utf-8', errors='replace').strip()}"
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-_LOG_TAIL_CHARS:] if log_path.is_file() else ""
        return f"The R worker terminated unexpectedly (code {rc}). End of the log:\n{tail}"

    @staticmethod
    def _read_outputs(job_dir: Path, job: EngineJob, wall: float) -> EngineResult:
        n, h = len(job.train), int(job.horizon)
        fpath = job_dir / "forecasts.bin"
        if not fpath.is_file():
            raise EngineError("The R worker produced no output (forecasts.bin is missing).")
        arr = np.fromfile(fpath, dtype="<f8")
        if arr.size != n * h:
            raise EngineError(f"The forecast file does not have the expected size: {arr.size} != {n}x{h}")
        forecasts = arr.reshape((n, h), order="F")       # R: as.vector(matrix) sütun-ana
        summary = json.loads((job_dir / "summary.json").read_text(encoding="utf-8")) if (job_dir / "summary.json").is_file() else {}
        notes = []
        fb = summary.get("fallbacks") or []
        if fb:
            notes.append(f"{len(fb)} series received a fallback (SNaive/naive) forecast instead of the model.")
        return EngineResult(
            forecasts=forecasts,
            elapsed_sec=float(summary.get("elapsed_sec", wall)),
            fallbacks=[{"index": int(f["index"]), "message": str(f["message"])} for f in fb],
            na_replaced=int(summary.get("na_replaced") or 0),
            versions=summary.get("versions") or {},
            notes=notes,
        )
