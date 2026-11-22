"""C++ engine (`--engine cpp`): the same methods as the Python engine, computed by the native library tsfcore.

tsfcore (cpp/) is a C++17 library with a plain C interface, built as a DLL (cpp/build.ps1). All series of a job go to the
library in ONE call and are spread over CPU threads inside C++, so there is no per-series Python or process overhead.
Each method follows its Python counterpart (py_models.py) step by step; tests/test_cpp_engine.py checks the agreement.

Library lookup: $TSFSTUDIO_CPP_LIB, else src/tsfstudio/native/tsfcore.dll (.so on Linux). If it is missing the engine
reports itself unavailable; the Python engine keeps working.
"""
from __future__ import annotations

import ctypes
import os
import platform
import threading
import time
from pathlib import Path

import numpy as np

from .base import Engine, EngineError, EngineJob, EngineResult, ProgressFn

_LIB = None
_LIB_ERR: str | None = None
_LOCK = threading.Lock()


def _lib_path() -> Path:
    env = os.environ.get("TSFSTUDIO_CPP_LIB")
    if env:
        return Path(env)
    name = "tsfcore.dll" if platform.system() == "Windows" else "tsfcore.so"
    return Path(__file__).resolve().parents[1] / "native" / name


def load_lib():
    """The loaded library, or None (reason in `_LIB_ERR`)."""
    global _LIB, _LIB_ERR
    with _LOCK:
        if _LIB is not None or _LIB_ERR is not None:
            return _LIB
        p = _lib_path()
        if not p.is_file():
            _LIB_ERR = f"native library not found: {p} (build it with cpp/build.ps1)"
            return None
        try:
            lib = ctypes.CDLL(str(p))
            lib.tsf_version.restype = ctypes.c_char_p
            lib.tsf_methods.restype = ctypes.c_char_p
            dp, i64p, i32p = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32)
            lib.tsf_forecast_batch.restype = ctypes.c_int32
            lib.tsf_forecast_batch.argtypes = [ctypes.c_char_p, dp, i64p, ctypes.c_int32, ctypes.c_int32, i32p, dp,
                                               ctypes.c_int32, ctypes.c_int32, dp, i32p, ctypes.c_char_p, ctypes.c_int32]
            lib.tsf_holt_sse.restype = ctypes.c_double
            lib.tsf_holt_sse.argtypes = [dp, ctypes.c_int32, ctypes.c_int32]
            _LIB = lib
        except OSError as exc:
            _LIB_ERR = f"could not load {p}: {exc}"
        return _LIB


def _ptr(a: np.ndarray, ctype):
    return a.ctypes.data_as(ctypes.POINTER(ctype))


def forecast_batch(method: str, train: list[np.ndarray], h: int, seasonality, horizons=None, threads: int = 0):
    """(forecasts n x h, fallback flags n, first fallback message). Raises EngineError if the library is missing."""
    lib = load_lib()
    if lib is None:
        raise EngineError(_LIB_ERR or "native library not available")
    n = len(train)
    lens = np.array([len(s) for s in train], dtype=np.int64)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    values = np.ascontiguousarray(np.concatenate([np.asarray(s, dtype=np.float64) for s in train]) if n else np.zeros(0))
    periods = np.ascontiguousarray(np.asarray(seasonality, dtype=np.float64).reshape(-1))
    hz = None if horizons is None else np.ascontiguousarray(np.asarray(horizons, dtype=np.int32))
    out = np.empty((n, h), dtype=np.float64)
    flags = np.zeros(n, dtype=np.int32)
    msg = ctypes.create_string_buffer(512)
    rc = lib.tsf_forecast_batch(method.encode(), _ptr(values, ctypes.c_double), _ptr(offsets, ctypes.c_int64), n, int(h),
                                None if hz is None else _ptr(hz, ctypes.c_int32), _ptr(periods, ctypes.c_double),
                                len(periods), int(threads), _ptr(out, ctypes.c_double), _ptr(flags, ctypes.c_int32), msg, 512)
    if rc == -1:
        raise EngineError(f"The C++ engine does not know the method '{method}'.")
    if rc != 0:
        raise EngineError(f"The C++ engine rejected the job (code {rc}).")
    return out, flags, msg.value.decode("utf-8", "replace")


class CppEngine(Engine):
    name = "cpp"
    LOCAL_METHODS = ("naive", "snaive", "drift", "ses", "theta", "naive2", "comb_shd", "ets", "holt_winters", "arima", "dhr_arima",
                     "croston_classic", "croston_sba", "croston_optimized", "tsb", "adida", "imapa", "dotm", "auto_ces", "scum", "auto_theta", "mstl")
    GLOBAL_METHODS = ()

    def methods(self) -> dict[str, tuple[str, ...]]:
        return {"local": self.LOCAL_METHODS if load_lib() is not None else (), "global": ()}

    def versions(self) -> dict:
        lib = load_lib()
        return {"python": platform.python_version(), "numpy": np.__version__,
                "tsfcore": lib.tsf_version().decode() if lib is not None else None}

    def available(self) -> tuple[bool, str]:
        lib = load_lib()
        if lib is None:
            return False, _LIB_ERR or "native library not available"
        return True, f"{lib.tsf_version().decode()} ({_lib_path().name})"

    def run(self, job: EngineJob, progress: ProgressFn | None = None, cancel: threading.Event | None = None) -> EngineResult:
        job.validate()
        if job.task != "local" or job.method not in self.LOCAL_METHODS:
            raise EngineError(f"The C++ engine does not support '{job.task}/{job.method}'. Available: {self.LOCAL_METHODS}")
        if job.method == "dhr_arima":
            k = (job.params or {}).get("fourier_k", 1)
            if int(k) != 1 or (job.params or {}).get("fourier_k_per_period"):
                raise EngineError("The C++ dhr_arima uses K = 1 Fourier terms (the study's setting); "
                                  f"fourier_k = {k} / fourier_k_per_period need the R or Python engine.")
        if job.method == "holt_winters" and str((job.params or {}).get("error_type", "A")).upper() != "A":
            raise EngineError("The C++ holt_winters has additive errors (error_type = 'A'); 'M' needs the Python engine.")
        if job.method in ("dotm", "auto_theta") and str((job.params or {}).get("decomposition_type", "multiplicative")) != "multiplicative":
            raise EngineError("The C++ dotm / auto_theta use decomposition_type = 'multiplicative' (additive automatically for data <= 0); "
                              "other settings need the Python engine.")
        if job.method == "auto_ces" and (str((job.params or {}).get("model", "Z")) != "Z"
                                         or int((job.params or {}).get("max_season", 24)) != 24):
            raise EngineError("The C++ auto_ces uses model = 'Z' and max_season = 24 (the defaults); other settings need the Python engine.")
        if job.method == "scum" and int((job.params or {}).get("arima_max_season", 200)) != 200:
            raise EngineError("The C++ scum uses arima_max_season = 200 (the default); other values need the Python engine.")
        if job.method == "tsb":
            ad, ap = (float((job.params or {}).get(k, 0.1)) for k in ("alpha_d", "alpha_p"))
            if ad != 0.1 or ap != 0.1:
                raise EngineError("The C++ tsb uses alpha_d = alpha_p = 0.1 (the defaults); other values need the Python engine.")
        t0 = time.time()
        n = len(job.train)
        if progress:
            progress("forecasting", 0, n)
        threads = int(job.workers) if job.workers and int(job.workers) > 1 else 0      # 0 = all CPU cores
        f, flags, msg = forecast_batch(job.method, list(job.train), job.horizon, job.seasonality, job.horizons, threads)
        if job.integer_conversion:
            f = np.round(f)
        fallbacks = [{"index": int(i), "message": msg or f"{job.method}: fallback"} for i in np.flatnonzero(flags)]
        notes = []
        if fallbacks:
            notes.append(f"{len(fallbacks)} series received a fallback (SNaive/naive) forecast instead of the model. "
                         f"First reason: {msg[:300]}")
        if progress:
            progress("finished", n, n)
        return EngineResult(forecasts=f, elapsed_sec=time.time() - t0, fallbacks=fallbacks, na_replaced=0,
                            versions=self.versions(), notes=notes)
