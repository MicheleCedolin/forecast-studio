"""Python referans (baseline) motoru: naive ve mevsimsel naive.

Her ortamda çalışır (R/torch gerekmez); koşucu, manifest ve metrik hattının testi ile "model yok" referansı içindir.
Dikkat: Burada SNaive, ilk mevsimsel periyodu (min(mevsimsellik), tamsayıya kesilmiş) kullanır. R motorundaki
`snaive(msts)` ise msts nesnesinin frekansını (floor(max(periyot))) kullanır; çok-mevsimli veride sonuçlar
farklıdır. Karşılaştırma için R motoru referanstır.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from ..seasonality import primary_period
from .base import Engine, EngineJob, EngineResult, ProgressFn


class PythonBaselineEngine(Engine):
    name = "python-baseline"
    LOCAL_METHODS = ("naive", "snaive")
    GLOBAL_METHODS = ()

    def available(self) -> tuple[bool, str]:
        return True, f"numpy {np.__version__}"

    def run(self, job: EngineJob, progress: ProgressFn | None = None,
            cancel: threading.Event | None = None) -> EngineResult:
        job.validate()
        if not self.supports(job.task, job.method):
            raise ValueError(f"Baseline motoru '{job.task}/{job.method}' desteklemiyor.")
        t0 = time.time()
        n, H = len(job.train), job.horizon
        m = primary_period(job.seasonality) if job.method == "snaive" else 1
        out = np.full((n, H), np.nan)
        for i, y in enumerate(job.train):
            h = job.horizons[i] if job.horizons is not None else H
            mm = m if len(y) >= m else 1
            idx = len(y) - mm + (np.arange(h) % mm)
            out[i, :h] = y[idx]
        if job.integer_conversion:
            out = np.round(out)
        if progress:
            progress("forecasting", n, n)
        return EngineResult(forecasts=out, elapsed_sec=time.time() - t0, versions={"numpy": np.__version__})
