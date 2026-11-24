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
            lib.tsf_pooled_regression.restype = ctypes.c_int32
            lib.tsf_pooled_regression.argtypes = [dp, i64p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                                                  dp, i32p, i32p, ctypes.c_char_p, ctypes.c_int32]
            lib.tsf_catboost.restype = ctypes.c_int32
            lib.tsf_catboost.argtypes = [dp, i64p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                                         ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32, dp,
                                         ctypes.c_char_p, ctypes.c_int32]
            lib.tsf_lightgbm.restype = ctypes.c_int32
            lib.tsf_lightgbm.argtypes = [dp, i64p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                                         ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32, dp, ctypes.c_char_p, ctypes.c_int32]
            lib.tsf_lightgbm_feat.restype = ctypes.c_int32
            lib.tsf_lightgbm_feat.argtypes = [dp, i64p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                                              ctypes.c_int32, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32, dp,
                                              ctypes.c_char_p, ctypes.c_int32]
            lib.tsf_lightgbm_direct.restype = ctypes.c_int32
            lib.tsf_lightgbm_direct.argtypes = lib.tsf_lightgbm_feat.argtypes
            lib.tsf_holt_sse.restype = ctypes.c_double
            lib.tsf_holt_sse.argtypes = [dp, ctypes.c_int32, ctypes.c_int32]
            _LIB = lib
        except OSError as exc:
            _LIB_ERR = f"could not load {p}: {exc}"
        return _LIB


def _ptr(a: np.ndarray, ctype):
    return a.ctypes.data_as(ctypes.POINTER(ctype))


def _int_in(lo: int, hi: int | None = None):
    def check(key, v):
        if isinstance(v, bool) or float(v) != int(float(v)):
            raise EngineError(f"{key} must be a whole number; given: {v!r}")
        v = int(float(v))
        if v < lo or (hi is not None and v > hi):
            raise EngineError(f"{key} must be in [{lo}, {hi if hi is not None else '∞'}]; given: {v}")
        return str(v)
    return check


def _alpha(key, v):
    v = float(v)
    if not 0.0 < v <= 1.0:
        raise EngineError(f"{key} must be in (0, 1]; given: {v}")
    return repr(v)


def _choice(*allowed, upper=False):
    def check(key, v):
        s = str(v).upper() if upper else str(v)
        if s not in allowed:
            raise EngineError(f"{key} must be one of {', '.join(allowed)}; given: {v!r}")
        return s
    return check


# Settings each C++ method accepts (same keys, meanings and defaults as the Python engine's; cpp/src/capi.cpp PMETHODS).
CPP_METHOD_PARAMS = {
    "holt_winters": {"error_type": _choice("A", "M", upper=True), "max_season": _int_in(2, 100000)},
    "holt_ets": {"error_type": _choice("A", "M", upper=True)},
    "dhr_arima": {"fourier_k": _int_in(1, 1000)},
    "tsb": {"alpha_d": _alpha, "alpha_p": _alpha},
    "dotm": {"decomposition_type": _choice("multiplicative", "additive")},
    "auto_theta": {"decomposition_type": _choice("multiplicative", "additive")},
    "auto_ces": {"model": _choice("Z", "N", "S", "P", "F", upper=True), "max_season": _int_in(2, 100000)},
    "auto_mfles": {"n_windows": _int_in(1, 1000), "metric": _choice("smape", "mape", "mse", "mae")},
    "window_average": {"window_size": _int_in(1)},
    "seasonal_window_average": {"window_size": _int_in(1)},
    "bagged_ets": {"n_boot": _int_in(1, 100000), "seed": _int_in(0, 2**31 - 1)},
    "scum": {"arima_max_season": _int_in(2, 100000)},
}


def method_with_params(method: str, params: dict | None) -> str:
    """The method name for tsf_forecast_batch: "name?key=value&..." when the job sets any of the method's settings
    (validated here), else the plain name (the method's defaults)."""
    checks = CPP_METHOD_PARAMS.get(method)
    if not checks or not params:
        return method
    parts = []
    for key, check in checks.items():
        v = params.get(key)
        if v is None or v == "":
            continue
        try:
            parts.append(f"{key}={check(key, v)}")
        except (TypeError, ValueError) as exc:
            raise EngineError(f"{method}: invalid {key} = {v!r} ({exc})") from exc
    return f"{method}?{'&'.join(parts)}" if parts else method


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
        raise EngineError(f"The C++ engine does not know the method '{method}'"
                          + (" or does not take settings for it." if "?" in method else "."))
    if rc != 0:
        raise EngineError(f"The C++ engine rejected the job (code {rc}).")
    return out, flags, msg.value.decode("utf-8", "replace")


def pooled_regression(train: list[np.ndarray], lag: int, h: int, max_obs: int | None = None):
    """(forecasts n x h, aliased 1-based lag numbers). Global pooled regression as R glm / py_global."""
    lib = load_lib()
    if lib is None:
        raise EngineError(_LIB_ERR or "native library not available")
    n = len(train)
    lens = np.array([len(s) for s in train], dtype=np.int64)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    values = np.ascontiguousarray(np.concatenate([np.asarray(s, dtype=np.float64) for s in train]) if n else np.zeros(0))
    out = np.empty((n, h), dtype=np.float64)
    aliased = np.zeros(max(int(lag), 1), dtype=np.int32)
    na = ctypes.c_int32(0)
    msg = ctypes.create_string_buffer(512)
    rc = lib.tsf_pooled_regression(_ptr(values, ctypes.c_double), _ptr(offsets, ctypes.c_int64), n, int(lag), int(h),
                                   int(max_obs or 0), _ptr(out, ctypes.c_double), _ptr(aliased, ctypes.c_int32),
                                   ctypes.byref(na), msg, 512)
    if rc != 0:
        raise EngineError(msg.value.decode(errors="replace"))
    return out, [int(a) for a in aliased[: na.value]]


def catboost_paths() -> tuple[Path, Path] | None:
    """(catboost executable, model library) or None. Looked up in TSFSTUDIO_CATBOOST_DIR, the package's native/catboost
    folder (release build) and the repository's third_party/catboost folder (downloaded from the CatBoost releases)."""
    exe = "catboost.exe" if platform.system() == "Windows" else "catboost"
    lib = "catboostmodel.dll" if platform.system() == "Windows" else "libcatboostmodel.so"
    here = Path(__file__).resolve()
    dirs = [Path(os.environ["TSFSTUDIO_CATBOOST_DIR"])] if os.environ.get("TSFSTUDIO_CATBOOST_DIR") else []
    dirs += [here.parents[1] / "native" / "catboost", here.parents[3] / "third_party" / "catboost"]
    for d in dirs:
        if (d / exe).is_file() and (d / lib).is_file():
            return d / exe, d / lib
    return None


def catboost(train: list[np.ndarray], lag: int, h: int, max_obs: int | None = None, threads: int = -1) -> np.ndarray:
    """Global CatBoost (CatBoost's defaults, RMSE, random seed 0) trained by the CatBoost command-line program and
    predicted through its model library, both driven from C++."""
    import tempfile
    lib = load_lib()
    if lib is None:
        raise EngineError(_LIB_ERR or "native library not available")
    paths = catboost_paths()
    if paths is None:
        raise EngineError("CatBoost binaries not found (catboost.exe and catboostmodel.dll from the CatBoost 1.2.10 release; "
                          "see cpp/README.md)")
    n = len(train)
    lens = np.array([len(s) for s in train], dtype=np.int64)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    values = np.ascontiguousarray(np.concatenate([np.asarray(s, dtype=np.float64) for s in train]) if n else np.zeros(0))
    out = np.empty((n, h), dtype=np.float64)
    msg = ctypes.create_string_buffer(1024)
    if threads <= 0:   # all cores; the catboost program rejects --thread-count -1 (the Python package accepts it)
        threads = os.cpu_count() or 1
    with tempfile.TemporaryDirectory(prefix="tsf_catboost_") as wd:
        rc = lib.tsf_catboost(_ptr(values, ctypes.c_double), _ptr(offsets, ctypes.c_int64), n, int(lag), int(h),
                              int(max_obs or 0), str(paths[0]).encode(), str(paths[1]).encode(), wd.encode(),
                              int(threads), _ptr(out, ctypes.c_double), msg, 1024)
    if rc != 0:
        raise EngineError(msg.value.decode(errors="replace"))
    return out


def lightgbm_lib_path() -> Path | None:
    """LightGBM's C library, or None. Looked up in TSFSTUDIO_LIGHTGBM_LIB, the package's native/lightgbm folder (release
    build) and the installed lightgbm Python package (found without importing it)."""
    import importlib.util
    name = "lib_lightgbm.dll" if platform.system() == "Windows" else "lib_lightgbm.so"
    cands = [Path(os.environ["TSFSTUDIO_LIGHTGBM_LIB"])] if os.environ.get("TSFSTUDIO_LIGHTGBM_LIB") else []
    cands.append(Path(__file__).resolve().parents[1] / "native" / "lightgbm" / name)
    try:
        spec = importlib.util.find_spec("lightgbm")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.submodule_search_locations:
        for d in spec.submodule_search_locations:
            cands += [Path(d) / "bin" / name, Path(d) / "lib" / name]
    return next((p for p in cands if p.is_file()), None)


def _lgb_param_str(p: dict) -> str:
    """LightGBM parameter string as the Python package writes it ("key=value", space separated)."""
    return " ".join(f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in p.items())


def lightgbm(train: list[np.ndarray], lag: int, h: int, max_obs: int | None = None, params: dict | None = None,
             workers: int = 1) -> np.ndarray:
    """Global LightGBM trained and predicted through LightGBM's C library from C++ (py_global.lightgbm_forecast's
    parameters and data pipeline)."""
    from .py_global import lightgbm_params
    lib = load_lib()
    if lib is None:
        raise EngineError(_LIB_ERR or "native library not available")
    path = lightgbm_lib_path()
    if path is None:
        raise EngineError("LightGBM's C library (lib_lightgbm) not found; install the lightgbm package or set TSFSTUDIO_LIGHTGBM_LIB")
    try:
        p = lightgbm_params(params, workers)
    except ValueError as exc:
        raise EngineError(str(exc)) from exc
    rounds = p.pop("num_iterations")
    n = len(train)
    lens = np.array([len(s) for s in train], dtype=np.int64)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    values = np.ascontiguousarray(np.concatenate([np.asarray(s, dtype=np.float64) for s in train]) if n else np.zeros(0))
    out = np.empty((n, h), dtype=np.float64)
    msg = ctypes.create_string_buffer(1024)
    rc = lib.tsf_lightgbm(_ptr(values, ctypes.c_double), _ptr(offsets, ctypes.c_int64), n, int(lag), int(h), int(max_obs or 0),
                          str(path).encode(), _lgb_param_str(p).encode(), int(rounds), _ptr(out, ctypes.c_double), msg, 1024)
    if rc != 0:
        raise EngineError(msg.value.decode(errors="replace"))
    return out


def lightgbm_feat(train: list[np.ndarray], lag: int, h: int, m: int, max_obs: int | None = None, params: dict | None = None,
                  workers: int = 1, direct: bool = False) -> np.ndarray:
    """Global LightGBM with M5-style features through LightGBM's C library: recursive (py_global.lightgbm_feat_forecast)
    or, with direct=True, one model per forecast step (py_global.lightgbm_direct_forecast)."""
    from .py_global import lightgbm_params
    lib = load_lib()
    if lib is None:
        raise EngineError(_LIB_ERR or "native library not available")
    path = lightgbm_lib_path()
    if path is None:
        raise EngineError("LightGBM's C library (lib_lightgbm) not found; install the lightgbm package or set TSFSTUDIO_LIGHTGBM_LIB")
    try:
        p = lightgbm_params(params, workers)
    except ValueError as exc:
        raise EngineError(str(exc)) from exc
    rounds = p.pop("num_iterations")
    n = len(train)
    lens = np.array([len(s) for s in train], dtype=np.int64)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    values = np.ascontiguousarray(np.concatenate([np.asarray(s, dtype=np.float64) for s in train]) if n else np.zeros(0))
    out = np.empty((n, h), dtype=np.float64)
    msg = ctypes.create_string_buffer(1024)
    fn = lib.tsf_lightgbm_direct if direct else lib.tsf_lightgbm_feat
    rc = fn(_ptr(values, ctypes.c_double), _ptr(offsets, ctypes.c_int64), n, int(lag), int(h), int(m),
            int(max_obs or 0), str(path).encode(), _lgb_param_str(p).encode(), int(rounds),
            _ptr(out, ctypes.c_double), msg, 1024)
    if rc != 0:
        raise EngineError(msg.value.decode(errors="replace"))
    return out


def _season(seasonality) -> int:
    vals = [float(v) for v in np.asarray(seasonality, dtype=np.float64).reshape(-1)] or [1.0]
    return max(1, int(np.floor(max(vals))))


class CppEngine(Engine):
    name = "cpp"
    LOCAL_METHODS = ("naive", "snaive", "drift", "ses", "theta", "naive2", "comb_shd", "ets", "holt_winters", "arima", "dhr_arima",
                     "croston_classic", "croston_sba", "croston_optimized", "tsb", "adida", "imapa", "dotm", "auto_ces", "scum", "comb_eat", "comb_eat_median", "auto_theta", "mstl", "auto_mfles", "tbats",
                     "holt_ets", "historic_average", "window_average", "seasonal_window_average", "seasonal_es", "zero_model",
                     "stl_ets", "stl_arima", "bagged_ets", "thief", "mapa")
    GLOBAL_METHODS = ("pooled_regression", "catboost", "lightgbm", "lightgbm_feat", "lightgbm_direct")

    def methods(self) -> dict[str, tuple[str, ...]]:
        ok = load_lib() is not None
        glob = tuple(m for m in self.GLOBAL_METHODS if (m != "catboost" or catboost_paths() is not None)
                     and (not m.startswith("lightgbm") or lightgbm_lib_path() is not None))
        return {"local": self.LOCAL_METHODS if ok else (), "global": glob if ok else ()}

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
        if job.task == "global" and job.method in self.GLOBAL_METHODS:
            t0 = time.time()
            if progress:
                progress("fitting", 0, 1)
            max_obs = (job.params or {}).get("max_obs")
            max_obs = int(max_obs) if max_obs else None
            aliased = []
            if job.method == "catboost":
                if (job.params or {}).get("catboost"):
                    raise EngineError("The C++ catboost uses CatBoost's defaults (RMSE, random seed 0); custom CatBoost "
                                      "parameters need the Python engine.")
                threads = int(job.workers) if job.workers and int(job.workers) > 1 else -1
                f = catboost(list(job.train), int(job.lag), job.horizon, max_obs, threads)
            elif job.method == "lightgbm":
                f = lightgbm(list(job.train), int(job.lag), job.horizon, max_obs, (job.params or {}).get("lightgbm"),
                             job.workers or 1)
            elif job.method in ("lightgbm_feat", "lightgbm_direct"):
                f = lightgbm_feat(list(job.train), int(job.lag), job.horizon, _season(job.seasonality), max_obs,
                                  (job.params or {}).get("lightgbm"), job.workers or 1, direct=job.method == "lightgbm_direct")
            else:
                f, aliased = pooled_regression(list(job.train), int(job.lag), job.horizon, max_obs)
            if job.integer_conversion:
                f = np.round(f)
            notes = []
            if aliased:
                notes.append(f"{len(aliased)} linearly dependent lag columns (Lag {aliased[:8]}{'…' if len(aliased) > 8 else ''}) "
                             "were treated as NA coefficients and set to 0, as R glm does.")
            if progress:
                progress("finished", 1, 1)
            return EngineResult(forecasts=f, elapsed_sec=time.time() - t0, fallbacks=[], na_replaced=0,
                                versions=self.versions(), notes=notes)
        if job.task != "local" or job.method not in self.LOCAL_METHODS:
            raise EngineError(f"The C++ engine does not support '{job.task}/{job.method}'. Available: {self.LOCAL_METHODS}")
        if job.method == "dhr_arima" and (job.params or {}).get("fourier_k_per_period"):
            raise EngineError("fourier_k_per_period (K for each of several periods) needs the R engine.")
        spec = method_with_params(job.method, job.params)
        t0 = time.time()
        n = len(job.train)
        if progress:
            progress("forecasting", 0, n)
        threads = int(job.workers) if job.workers and int(job.workers) > 1 else 0      # 0 = all CPU cores
        f, flags, msg = forecast_batch(spec, list(job.train), job.horizon, job.seasonality, job.horizons, threads)
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
