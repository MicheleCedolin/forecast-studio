"""Python motoru: R modellerinin Python karşılıkları. R/Docker gerekmez.

Yerleşik (yalnızca numpy/scipy; R algoritmasıyla birebir sınanmıştır, docs/VALIDATION.md):
    naive, snaive, ses, theta        yerel
    drift, naive2, comb_shd          yerel M3/M4 kıyasları (2026-11-22; R ile birebir sınanmadı, bkz. py_models)
    pooled_regression                global (akış hâlinde QR; R `glm` ile aynı sonuç)
İsteğe bağlı kütüphaneler:
    statsforecast  -> ets, arima, dhr_arima          (kurulu değilse yöntem listesinde görünmez)
                      + C3: croston_classic/sba/optimized, tsb, adida, imapa (aralıklı talep);
                            auto_theta, auto_ces, holt_winters, mstl, auto_mfles (ek klasik)
    (tbats: kullanıcı kararıyla devre dışı, bkz. aşağıdaki not)
    catboost       -> catboost                       (global)

R ile aynı hata semantiği: bir serinin modeli hata verirse o seri `snaive` (R `msts` frekansıyla) tahmini alır ve
bu SAYILIR (`fallbacks`). Fark: model NaN/Inf üretirse R bunu 0 yapar (`na_replaced`); burada da yedek tahmine düşülür
(0 uydurmak yerine). Bu bilinçli bir farktır.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import platform
import threading
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

from . import py_global as G
from . import py_models as M
from . import py_sf as S
from .base import Engine, EngineCancelled, EngineError, EngineJob, EngineResult, ProgressFn

NATIVE_LOCAL = ("naive", "snaive", "ses", "theta",
                "drift", "naive2", "comb_shd")                 # 2026-11-22: M3/M4 kıyasları (py_models)
# TBATS (2026-11-03, kullanıcı kararı): Python motorundan ÇIKARILDI. statsforecast `AutoTBATS`, R `tbats()` ile aynı modeli
# bulmuyor (NN5 haftalık, 8 seri: ortanca tahmin farkı %3,1) ve amaç R'ı Python'a çevirmek değil. Orijinal R TBATS, R motorunda
# kalır. Adaptör `py_sf.forecast_tbats` (düzeltilmiş imza) yeniden açmak için durur: SF_LOCAL'e "tbats": "AutoTBATS" ekleyip
# `forecast_one`'a dalını geri koyun.
SF_LOCAL = {"ets": "AutoETS", "arima": "AutoARIMA", "dhr_arima": "AutoARIMA",
            **{k: v["cls"] for k, v in S.SPECS.items()},      # C3: aralıklı talep + ek klasik modeller
            "scum": "DynamicOptimizedTheta"}                   # 2026-11-22: ETS/CES/ARIMA/DOTM medyanı


# ------------------------------------------------------------------------------------------------
# Seri başına tahmin (modül düzeyinde: çok süreçli çalışmada seçilebilir olmalı)
# ------------------------------------------------------------------------------------------------
def forecast_one(method: str, y: np.ndarray, h: int, seasonality: tuple[float, ...], params: dict) -> np.ndarray:
    m = M.r_frequency(seasonality)
    if method == "naive":
        return M.naive(y, h)
    if method == "snaive":
        return M.snaive(y, h, m)
    if method == "ses":
        return M.ses_forecast(y, h)
    if method == "theta":
        return M.theta_forecast(y, h, m)
    if method == "drift":
        return M.drift(y, h)
    if method == "naive2":
        return M.naive2(y, h, m)
    if method == "comb_shd":
        return M.comb_shd(y, h, m)
    if method == "scum":
        return S.forecast_scum(y, h, m, seasonality, params)
    if method == "ets":
        return S.forecast_ets(y, h, m)
    if method == "arima":
        return S.forecast_arima(y, h, m, int(params.get("arima_max_season", S.ARIMA_MAX_SEASON_DEFAULT)))
    if method == "dhr_arima":
        return S.forecast_dhr_arima(y, h, seasonality, params.get("fourier_k", 1))
    if method in S.SPECS:
        return S.forecast_spec(method, y, h, m, seasonality, params)
    raise ValueError(f"Unknown local method: {method}")


def _safe_forecast(method: str, y: np.ndarray, h: int, seasonality, params) -> tuple[np.ndarray, str | None, int, tuple]:
    """(tahmin, yedek nedeni | None, uyarı bayrakları, adaptör notları). Bayraklar: 1 = yakınsama uyarısı, 2 = başka uyarı.

    Kütüphane uyarıları (ör. statsforecast "possible convergence problem") seri başına yüzlerce satır basıp ekranı
    doldurduğundan yakalanır, yutulmaz: sayılır ve motor notlarında raporlanır. Adaptör notları (`py_sf.AdapterNote`:
    mevsimselliğin atılması gibi sessiz uyarlamalar) ayrıca toplanır."""
    m = M.r_frequency(seasonality)
    flags = 0
    msg = None
    result = None
    notes: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            f = np.asarray(forecast_one(method, y, h, seasonality, params), dtype=np.float64).reshape(-1)
            if f.shape[0] < h or not np.all(np.isfinite(f[:h])):
                raise ValueError("the model produced a non-finite value or an incomplete horizon")
            result = f[:h]
        except Exception as exc:  # noqa: BLE001
            msg = f"{method}: {type(exc).__name__}: {exc}"
    for w in caught:
        if issubclass(w.category, S.AdapterNote):
            notes.append(str(w.message))
        elif "converg" in str(w.message).lower():
            flags |= 1
        else:
            flags |= 2
    if result is not None:
        return result, None, flags, tuple(notes)
    try:
        return M.snaive(y, h, m), msg, flags, tuple(notes)
    except Exception as exc2:  # noqa: BLE001
        return M.naive(y, h), f"{msg} | snaive: {exc2}", flags, tuple(notes)


def _run_chunk(items: list[tuple]) -> list[tuple[int, np.ndarray, str | None, int, tuple]]:
    out = []
    for i, method, y, h, seasonality, params in items:
        f, fb, fl, nt = _safe_forecast(method, y, h, seasonality, params)
        out.append((i, f, fb, fl, nt))
    return out


def _mp_context():
    """Windows'ta (ve TSFSTUDIO_MP_START ile istenirse) 'spawn'; diğerlerinde platform varsayılanı."""
    method = os.environ.get("TSFSTUDIO_MP_START") or ("spawn" if platform.system() == "Windows" else None)
    return mp.get_context(method) if method else mp.get_context()


# ------------------------------------------------------------------------------------------------
class PythonEngine(Engine):
    name = "python"
    LOCAL_METHODS = NATIVE_LOCAL + tuple(SF_LOCAL)          # üst küme; `methods()` kuruluma göre süzer
    GLOBAL_METHODS = ("pooled_regression", "catboost")
    # Bellek: pooled_regression akış hâlinde (tasarım matrisi yok); catboost float32 matris + Pool kopyası.
    MEMORY_FACTOR = {"pooled_regression": None, "catboost": 1.2}

    def methods(self) -> dict[str, tuple[str, ...]]:
        local = list(NATIVE_LOCAL)
        if S.sf_available():
            local += [k for k, cls in SF_LOCAL.items() if S.has_model(cls) and not S.missing_requirements(k)]
        glob = ["pooled_regression"] + (["catboost"] if G.catboost_available() else [])
        return {"local": tuple(local), "global": tuple(glob)}

    def versions(self) -> dict:
        import scipy
        v = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
             "statsforecast": S.sf_version(), "catboost": None}
        if G.catboost_available():
            import catboost
            v["catboost"] = catboost.__version__
        return v

    def available(self) -> tuple[bool, str]:
        v = self.versions()
        ms = self.methods()
        miss = [m for m in self.LOCAL_METHODS + self.GLOBAL_METHODS if m not in ms["local"] + ms["global"]]
        txt = f"numpy {v['numpy']}, scipy {v['scipy']}, statsforecast {v['statsforecast'] or 'MISSING'}, catboost {v['catboost'] or 'MISSING'}"
        if miss:
            txt += f" | methods unavailable because they are not installed: {', '.join(miss)}"
            pips = sorted({p for m in miss for p in S.missing_requirements(m)})
            if pips:
                txt += f" (missing package: pip install {' '.join(pips)})"
        return True, txt

    # ----------------------------------------------------------------------------------------
    def run(self, job: EngineJob, progress: ProgressFn | None = None,
            cancel: threading.Event | None = None) -> EngineResult:
        job.validate()
        if not self.supports(job.task, job.method):
            raise EngineError(f"The Python engine does not support '{job.task}/{job.method}', or the required library is not installed. "
                              f"Available: {self.methods()}")
        t0 = time.time()
        if job.task == "local":
            f, fallbacks, notes = self._run_local(job, progress, cancel)
        else:
            f, fallbacks, notes = self._run_global(job, progress)
        n, H = len(job.train), job.horizon
        if job.horizons is not None:
            for i, hi in enumerate(job.horizons):
                if hi < H:
                    f[i, hi:] = np.nan
        if job.integer_conversion:
            f = np.round(f)
        if S.DROPPED:
            notes.append("Keyword arguments not recognised by your statsforecast version were dropped: " + ", ".join(sorted(S.DROPPED)))
        if fallbacks:
            distinct: list[str] = []
            for d in fallbacks:
                if d["message"] not in distinct:
                    distinct.append(d["message"])
            shown = " || ".join(x[:300] for x in distinct[:3])
            notes.append(f"{len(fallbacks)} series received a fallback (SNaive/naive) forecast instead of the model. First distinct reason(s): {shown}")
        if progress:
            progress("finished", n, n)
        return EngineResult(forecasts=f, elapsed_sec=time.time() - t0, fallbacks=fallbacks, na_replaced=0,
                            versions=self.versions(), notes=notes)

    # ----------------------------------------------------------------------------------------
    def _run_local(self, job: EngineJob, progress, cancel):
        n, H = len(job.train), job.horizon
        hs = job.horizons if job.horizons is not None else [H] * n
        seas = tuple(float(s) for s in job.seasonality)
        items = [(i, job.method, np.ascontiguousarray(job.train[i], dtype=np.float64), int(hs[i]), seas, job.params or {})
                 for i in range(n)]
        out = np.full((n, H), np.nan)
        fallbacks: list[dict] = []
        n_conv = n_other = 0
        adapter_notes: dict[str, int] = {}
        done = 0
        if progress:
            progress("forecasting", 0, n)

        def absorb(res):
            nonlocal done, n_conv, n_other
            for i, f, fb, fl, nt in res:
                out[i, : len(f)] = f
                if fb:
                    fallbacks.append({"index": int(i), "message": fb})
                n_conv += bool(fl & 1)
                n_other += bool(fl & 2)
                for t in set(nt):
                    adapter_notes[t] = adapter_notes.get(t, 0) + 1
            done += len(res)
            if progress:
                progress("forecasting", done, n)

        workers = max(1, int(job.workers))
        chunk = max(workers * 10, 50)
        if workers > 1 and n > chunk:
            ex = ProcessPoolExecutor(max_workers=workers, mp_context=_mp_context())
            try:
                pending = {ex.submit(_run_chunk, items[s: s + chunk]) for s in range(0, n, chunk)}
                while pending:
                    ready, pending = wait(pending, timeout=0.4, return_when=FIRST_COMPLETED)
                    for fut in ready:
                        absorb(fut.result())
                    if cancel is not None and cancel.is_set():
                        self._kill(ex)
                        raise EngineCancelled("The job was cancelled by the user.")
            except BaseException:
                self._kill(ex)
                raise
            else:
                ex.shutdown()
        else:
            for s in range(0, n, 10):
                if cancel is not None and cancel.is_set():
                    raise EngineCancelled("The job was cancelled by the user.")
                absorb(_run_chunk(items[s: s + 10]))
        fallbacks.sort(key=lambda d: d["index"])
        notes: list[str] = []
        if n_conv:
            notes.append(f"The library issued an optimisation convergence warning for {n_conv} series (the forecasts were used; warning texts "
                         f"are not printed). This can also be seen in R; if there are many, review the model/settings.")
        if n_other:
            notes.append(f"Other library warnings were caught for {n_other} series (not printed).")
        for t, c in sorted(adapter_notes.items(), key=lambda kv: -kv[1])[:6]:
            notes.append(f"{c} series: {t}")
        return out, fallbacks, notes

    @staticmethod
    def _kill(ex: ProcessPoolExecutor) -> None:
        try:
            ex.shutdown(wait=False, cancel_futures=True)
            for p in list(getattr(ex, "_processes", {}).values()):
                p.terminate()
        except Exception:  # noqa: BLE001
            pass

    def _run_global(self, job: EngineJob, progress):
        lag, H = int(job.lag), job.horizon
        max_obs = (job.params or {}).get("max_obs")
        max_obs = int(max_obs) if max_obs else None
        if progress:
            progress("fitting", 0, 1)
        try:
            if job.method == "pooled_regression":
                f, info = G.pooled_regression_forecast(job.train, lag, H, max_obs)
            else:
                f, info = G.catboost_forecast(job.train, lag, H, max_obs, (job.params or {}).get("catboost"), job.workers)
        except ValueError as exc:
            raise EngineError(str(exc)) from exc
        if progress:
            progress("fitting", 1, 1)
        return f, [], list(info.notes)
