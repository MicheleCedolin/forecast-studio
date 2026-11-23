"""Koşucu: veri kümesi + yapılandırma -> bölme -> motor -> metrikler -> çıktı dosyaları + manifest.

Akış (tamamı otomatik; kullanıcı yalnızca veri kümesi ve model seçer):
  1. `prepare`   : eksik değer politikası, eğitim/test bölmesi (son `horizon` gözlem test), çok kısa serilerin dışlanması.
  2. `preflight` : bellek/süre tahmini (global modellerde bellek yetersizse koşu başlamadan durdurulur).
  3. motor       : R (orijinal) ya da başka bir motor.
  4. metrikler   : `metrics.compute_error_table` + özet.
  5. çıktılar    : forecasts.csv/xlsx, test_actuals.csv, metrics.xlsx, fallbacks.csv, manifest.json.

Yeniden üretilebilirlik: manifest; yapılandırma, veri kümesi SHA-256, tohum, motor/paket sürümleri, atlanan seriler,
yedek (fallback) sayıları ve uyarıları kaydeder.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import __version__
from .catalog import DatasetEntry
from .engines import Engine, EngineJob
from .engines.registry import get_engine
from .estimate import PreflightReport, preflight
from .metrics import DEFAULT_METRICS, compute_error_table, summarize, write_metrics_xlsx
from .seasonality import seasonality_for
from .tsf.reader import TSFDataset

MISSING_POLICIES = ("error", "zero", "locf", "interpolate")
_XLSX_MAX_CELLS = 2_000_000
ProgressFn = Callable[[str, int, int], None]


class RunError(RuntimeError):
    """Koşu başlamadan ya da sırasında kullanıcıya anlatılabilir hata."""


@dataclass
class RunConfig:
    engine: str = "r"
    task: str = "local"                      # "local" | "global"
    method: str = "ets"
    horizon: int | None = None               # None -> katalog/başlık
    lag: int | None = None                   # yalnızca global
    seasonality: tuple[float, ...] | None = None   # None -> frekanstan
    integer_conversion: bool | None = None   # None -> katalog girdisi (yoksa False)
    missing_policy: str = "error"            # eğitim kısmındaki eksikler için
    params: dict = field(default_factory=dict)
    seed: int = 1
    workers: int = 1
    max_series: int | None = None            # hızlı deneme için ilk N seri
    metrics: tuple[str, ...] = DEFAULT_METRICS
    mase_digits: int | None = None
    output_dir: str | Path | None = None
    force: bool = False                      # bellek engelini aşar (önerilmez)
    write_xlsx: bool = True
    origins: int = 1                         # rolling origin: number of forecast origins (1 = last H only)
    origin_step: int | None = None           # observations between origins (None -> horizon: windows do not overlap)
    intervals: tuple[int, ...] = ()          # prediction-interval levels in percent (e.g. (80, 95)); () = none

    def to_dict(self) -> dict:
        d = asdict(self)
        d["output_dir"] = None if self.output_dir is None else str(self.output_dir)
        d["seasonality"] = None if self.seasonality is None else list(self.seasonality)
        d["metrics"] = list(self.metrics)
        d["intervals"] = list(self.intervals)
        return d


@dataclass
class Prepared:
    names: list[str]
    train: list[np.ndarray]
    test: np.ndarray                         # (n, H)
    horizon: int
    seasonality: tuple[float, ...]
    integer_conversion: bool
    excluded: list[dict]
    missing_info: dict
    warnings: list[str]


@dataclass
class RunResult:
    run_dir: Path
    manifest: dict
    names: list[str]
    train: list[np.ndarray]
    test: np.ndarray
    forecasts: np.ndarray
    per_series: pd.DataFrame
    summary: pd.DataFrame
    fallbacks: list[dict]
    preflight: PreflightReport


# ------------------------------------------------------------------------------------------------
def _apply_missing_policy(y: np.ndarray, policy: str) -> np.ndarray:
    if policy == "zero":
        return np.nan_to_num(y, nan=0.0)
    s = pd.Series(y)
    if policy == "locf":
        return s.ffill().bfill().to_numpy(dtype=np.float64)
    if policy == "interpolate":
        return s.interpolate(method="linear", limit_direction="both").to_numpy(dtype=np.float64)
    raise ValueError(policy)


def prepare(dataset: TSFDataset, cfg: RunConfig, entry: DatasetEntry | None = None, cut: int = 0,
            min_extra: int = 0) -> Prepared:
    """Split every series into training part and the last H observations, after dropping its last cut observations
    (rolling origin). Series shorter than H + 2 + min_extra are excluded, so every origin keeps the same series."""
    if cfg.missing_policy not in MISSING_POLICIES:
        raise RunError(f"Invalid missing-value policy: {cfg.missing_policy!r} (valid: {', '.join(MISSING_POLICIES)})")
    warnings: list[str] = []

    H = cfg.horizon or (entry.effective_horizon(dataset.header) if entry else None) or dataset.horizon
    if not H or H < 1:
        raise RunError("The forecast horizon could not be determined; the data header has no @horizon. Enter the horizon manually.")
    H = int(H)

    seas = tuple(cfg.seasonality) if cfg.seasonality else seasonality_for(dataset.frequency)
    integer = cfg.integer_conversion if cfg.integer_conversion is not None else bool(entry.integer_conversion if entry else False)

    names = dataset.series_names
    values = dataset.values
    if cfg.max_series:
        names, values = names[: cfg.max_series], values[: cfg.max_series]

    train: list[np.ndarray] = []
    test_rows: list[np.ndarray] = []
    kept_names: list[str] = []
    excluded: list[dict] = []
    imputed = 0
    series_with_missing = 0
    test_missing = 0
    for i, (nm, y) in enumerate(zip(names, values)):
        y = np.asarray(y, dtype=np.float64)
        if len(y) < H + 2 + min_extra:
            need = f"horizon+2 = {H + 2}" if not min_extra else f"horizon+2+(origins-1)*step = {H + 2 + min_extra}"
            excluded.append({"index": i, "name": nm, "reason": f"series too short (length {len(y)} < {need})"})
            continue
        if cut:
            y = y[: len(y) - cut]
        tr, te = y[:-H], y[-H:]
        miss = int(np.isnan(tr).sum())
        if miss:
            series_with_missing += 1
            if cfg.missing_policy == "error":
                raise RunError(
                    f"Series '{nm}' has {miss} missing values in its training part. Choose a missing-value policy: "
                    f"'zero', 'locf' (carry the last value forward) or 'interpolate' (linear interpolation). "
                    f"We do not change these values silently; the choice affects the results.")
            if np.isnan(tr).all():
                excluded.append({"index": i, "name": nm, "reason": "training part entirely missing"})
                continue
            tr = _apply_missing_policy(tr, cfg.missing_policy)
            imputed += miss
        test_missing += int(np.isnan(te).sum())
        train.append(tr)
        test_rows.append(te)
        kept_names.append(nm)
    if not train:
        raise RunError("No series left to run (all were excluded).")
    from .engines.py_sf import INTERMITTENT
    if integer and cfg.method in INTERMITTENT:
        warnings.append(
            "Rounding to integers is ON (catalogue or choice): intermittent-demand methods forecast a flat demand RATE (often < 0.5), which "
            "becomes 0 when rounded; all forecasts can come out as 0. To see the rate forecast, turn rounding off (in the interface: Settings, 'Round to integers' = Off; on the command line: --integer-conversion off).")
    if excluded:
        warnings.append(f"{len(excluded)} series excluded (details in manifest.json -> split.excluded).")
    if test_missing:
        warnings.append(f"There are {test_missing} missing values in the test part; the measures exclude these pairs.")
    return Prepared(kept_names, train, np.vstack(test_rows), H, seas, integer, excluded,
                    {"policy": cfg.missing_policy, "series_with_missing": series_with_missing,
                     "imputed_values": imputed, "test_missing_values": test_missing}, warnings)


def prepare_origins(dataset: TSFDataset, cfg: RunConfig, entry: DatasetEntry | None = None) -> tuple[Prepared, int, int]:
    """The split of the latest origin with the exclusions of a rolling-origin run: (prepared, origins, step)."""
    n_orig = 1 if cfg.origins is None else int(cfg.origins)
    if n_orig < 1:
        raise RunError("The number of forecast origins must be at least 1.")
    prep = prepare(dataset, cfg, entry)
    step = int(cfg.origin_step or prep.horizon)
    if step < 1:
        raise RunError("The step between forecast origins must be at least 1.")
    levels = tuple(cfg.intervals or ())
    if any(int(lv) != lv or not 50 <= lv <= 99 for lv in levels):
        raise RunError("Prediction-interval levels must be whole percentages between 50 and 99.")
    extra = (n_orig - 1) * step + (prep.horizon if levels else 0)     # intervals: one calibration window before
    if extra:
        prep = prepare(dataset, cfg, entry, cut=0, min_extra=extra)
    return prep, n_orig, step


# ------------------------------------------------------------------------------------------------
def build_job(prep: Prepared, cfg: RunConfig, entry: DatasetEntry | None = None) -> EngineJob:
    """Hazırlanmış veriden motor işi üretir (lag: yapılandırma > katalog)."""
    lag = cfg.lag
    if cfg.task == "global" and lag is None and entry is not None:
        lag = entry.lag
    job = EngineJob(task=cfg.task, method=cfg.method, train=prep.train, horizon=prep.horizon, seasonality=prep.seasonality,
                    integer_conversion=prep.integer_conversion, lag=lag, params=cfg.params, seed=cfg.seed, workers=cfg.workers)
    try:
        job.validate()
    except Exception as exc:  # noqa: BLE001
        raise RunError(str(exc)) from exc
    return job


def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _run_id(dataset_label: str, cfg: RunConfig) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in f"{dataset_label}__{cfg.engine}-{cfg.method}")
    return f"{safe}__{stamp}"


def _write_outputs(run_dir: Path, prep: Prepared, forecasts: np.ndarray, fallbacks: list[dict],
                   per_series: pd.DataFrame, summary: pd.DataFrame, run_info: dict, write_xlsx: bool) -> dict:
    H = prep.horizon
    cols = [f"h{k}" for k in range(1, H + 1)]
    out: dict[str, str] = {}

    fc = pd.DataFrame(forecasts, columns=cols)
    fc.insert(0, "series_name", prep.names)
    fc.to_csv(run_dir / "forecasts.csv", index=False)
    out["forecasts_csv"] = "forecasts.csv"

    ac = pd.DataFrame(prep.test, columns=cols)
    ac.insert(0, "series_name", prep.names)
    ac.to_csv(run_dir / "test_actuals.csv", index=False)
    out["test_actuals_csv"] = "test_actuals.csv"

    if write_xlsx:
        if fc.shape[0] * fc.shape[1] <= _XLSX_MAX_CELLS:
            fc.to_excel(run_dir / "forecasts.xlsx", index=False, engine="xlsxwriter")
            out["forecasts_xlsx"] = "forecasts.xlsx"
        write_metrics_xlsx(run_dir / "metrics.xlsx", per_series, summary, run_info)
        out["metrics_xlsx"] = "metrics.xlsx"

    names = prep.names
    fb = pd.DataFrame([{"series_index": f["index"], "series_name": names[f["index"]] if 0 <= f["index"] < len(names) else "",
                        "message": f["message"]} for f in fallbacks], columns=["series_index", "series_name", "message"])
    fb.to_csv(run_dir / "fallbacks.csv", index=False)
    out["fallbacks_csv"] = "fallbacks.csv"

    # Excel'siz okunabilir metrik çıktıları (arayüz ve betikler için; xlsx kapalı olsa da yazılır)
    summary.to_csv(run_dir / "summary.csv", index=False)
    out["summary_csv"] = "summary.csv"
    per_series.reset_index().to_csv(run_dir / "errors_per_series.csv", index=False)
    out["errors_per_series_csv"] = "errors_per_series.csv"
    return out


def _write_origin_outputs(run_dir: Path, origin_rows: list) -> dict:
    """Rolling origin: forecasts, actuals and measures of every origin (origin 1 = oldest), plus a summary per origin."""
    fcs, acs, ers, sums = [], [], [], []
    for k, (cut, p, fc, ps) in enumerate(origin_rows, 1):
        cols = [f"h{j}" for j in range(1, p.horizon + 1)]
        head = {"origin": k, "cut": cut}
        f = pd.DataFrame(fc, columns=cols)
        f.insert(0, "series_name", p.names)
        a = pd.DataFrame(p.test, columns=cols)
        a.insert(0, "series_name", p.names)
        e = ps.reset_index()
        s = summarize(ps)
        for df in (f, a, e, s):
            for i, (key, val) in enumerate(head.items()):
                df.insert(i, key, val)
        fcs.append(f)
        acs.append(a)
        ers.append(e)
        sums.append(s)
    out = {}
    for key, name, frames in (("forecasts_by_origin_csv", "forecasts_by_origin.csv", fcs),
                              ("test_actuals_by_origin_csv", "test_actuals_by_origin.csv", acs),
                              ("errors_by_origin_csv", "errors_by_origin.csv", ers),
                              ("summary_by_origin_csv", "summary_by_origin.csv", sums)):
        pd.concat(frames, ignore_index=True).to_csv(run_dir / name, index=False)
        out[key] = name
    return out


def run(dataset: TSFDataset, cfg: RunConfig, *, entry: DatasetEntry | None = None, dataset_id: str | None = None,
        engine: Engine | None = None, progress: ProgressFn | None = None,
        cancel: threading.Event | None = None) -> RunResult:
    t_start = time.time()
    label = dataset_id or (entry.id if entry else None) or (Path(dataset.source).stem if dataset.source else "dataset")
    if progress:
        progress("prepare", 0, 1)
    prep, n_orig, step = prepare_origins(dataset, cfg, entry)
    levels = tuple(sorted(set(int(lv) for lv in (cfg.intervals or ()))))
    extra = (n_orig - 1) * step + (prep.horizon if levels else 0)
    latest_bounds = None
    eng = engine or get_engine(cfg.engine)

    ok, msg = eng.available()
    if not ok:
        raise RunError(f"Engine '{eng.name}' is not available: {msg}")
    if not eng.supports(cfg.task, cfg.method):
        raise RunError(f"Engine '{eng.name}' does not support the method '{cfg.task}/{cfg.method}'. Supported: {eng.methods()}")

    job = build_job(prep, cfg, entry)
    lag = job.lag

    pre = preflight(job, n_total_series=len(prep.train), engine=eng)
    if pre.blocked and not cfg.force:
        raise RunError("Memory looks insufficient; the run was not started. " + (pre.memory.message if pre.memory else ""))

    t_engine = time.time()
    # Rolling origin: one complete fit per origin, oldest first, each on the data before its origin only (a pooled fit
    # over origins would let global models see later test values). The last origin is the usual split.
    origin_rows: list[tuple[int, Prepared, np.ndarray, pd.DataFrame]] = []
    fallbacks_all: list[dict] = []
    notes_all: list[str] = []
    na_replaced = 0
    elapsed = 0.0
    res = None
    for o in range(n_orig - 1, -1, -1):
        cut = o * step
        p_o = prep if cut == 0 else prepare(dataset, cfg, entry, cut=cut, min_extra=extra)
        job_o = job if cut == 0 else build_job(p_o, cfg, entry)
        prog = progress
        if progress and n_orig > 1:
            k = n_orig - o
            prog = lambda st, d, tot, k=k: progress(f"{st} (origin {k}/{n_orig})", d, tot)   # noqa: E731
        r_o = eng.run(job_o, progress=prog, cancel=cancel)
        if r_o.forecasts.shape != p_o.test.shape:
            raise RunError(f"Engine output has an unexpected shape: {r_o.forecasts.shape} != {p_o.test.shape}")
        ps_o = compute_error_table(r_o.forecasts, p_o.test, p_o.train, p_o.seasonality, series_names=p_o.names,
                                   metrics=cfg.metrics, mase_digits=cfg.mase_digits)
        if levels:
            # conformal intervals: the same model fitted one horizon earlier gives the calibration errors
            from . import intervals as IV
            p_c = prepare(dataset, cfg, entry, cut=cut + prep.horizon, min_extra=extra)
            cprog = (lambda st, d, tot: progress(f"calibration: {st}", d, tot)) if progress else None
            r_c = eng.run(build_job(p_c, cfg, entry), progress=cprog, cancel=cancel)
            probs = IV.probs_for(levels)
            q = IV.conformal_quantiles(r_c.forecasts, p_c.test, IV.series_scales(p_c.train, p_c.seasonality), probs)
            scales = IV.series_scales(p_o.train, p_o.seasonality)
            bnds = IV.bounds(r_o.forecasts, scales, q, probs)
            ps_o = ps_o.join(IV.interval_metrics(r_o.forecasts, p_o.test, scales, levels, bnds, p_o.names))
            elapsed += r_c.elapsed_sec
            if cut == 0:
                latest_bounds = (probs, bnds)
        origin_rows.append((cut, p_o, r_o.forecasts, ps_o))
        fallbacks_all += [dict(f, origin_cut=cut) for f in r_o.fallbacks] if n_orig > 1 else list(r_o.fallbacks)
        notes_all += [m for m in r_o.notes if m not in notes_all]
        na_replaced += r_o.na_replaced
        elapsed += r_o.elapsed_sec
        res = r_o
    engine_sec = time.time() - t_engine
    res.fallbacks, res.notes, res.na_replaced, res.elapsed_sec = fallbacks_all, notes_all, na_replaced, elapsed

    if progress:
        progress("metrics", 0, 1)
    if n_orig == 1:
        per_series = origin_rows[0][3]
    else:
        # per series: the mean of its measures over the origins (finite values only)
        stacked = pd.concat([ps for *_, ps in origin_rows])
        per_series = stacked.groupby(level=0, sort=False).mean()
        per_series = per_series.reindex(prep.names)
    summary = summarize(per_series)

    warnings = list(prep.warnings) + list(pre.warnings) + list(res.notes)
    n = len(prep.train)
    frac_fb = len(res.fallbacks) / (n * n_orig) if n else 0.0
    if frac_fb > 0.5:
        warnings.append(f"{100 * frac_fb:.0f}% of the series received a fallback (SNaive) forecast instead of the model; the results may not represent this model.")
    if res.na_replaced:
        warnings.append(f"{res.na_replaced} forecast values were NA and were replaced with 0 (as in the original R behaviour).")

    run_id = _run_id(label, cfg)
    base = Path(cfg.output_dir) if cfg.output_dir else Path(os.environ.get("TSFSTUDIO_RUNS_DIR", "runs"))
    run_dir = base / run_id
    k = 1
    while True:                                   # aynı saniyedeki iki koşu birbirinin çıktısını ezmesin
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            k += 1
            run_id = f"{_run_id(label, cfg)}-{k}"
            run_dir = base / run_id

    src = Path(dataset.source) if dataset.source else None
    manifest = {
        "schema": 1,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {**cfg.to_dict(), "lag": lag, "effective_horizon": prep.horizon, "effective_seasonality": list(prep.seasonality),
                   "effective_integer_conversion": prep.integer_conversion},
        "dataset": {
            "id": label, "source": str(src) if src else None,
            "sha256": _sha256(src) if src and src.is_file() else None,
            "frequency": dataset.frequency, "n_series_total": dataset.n_series, "n_series_run": n,
            "header_horizon": dataset.horizon,
        },
        "split": {"horizon": prep.horizon, "n_excluded": len(prep.excluded), "excluded": prep.excluded,
                  "origins": n_orig, "origin_step": step if n_orig > 1 else None,
                  "interval_levels": list(levels), "interval_method": "split conformal, one calibration window" if levels else None,
                  "origin_cuts": [c for c, *_ in origin_rows]},
        "missing": prep.missing_info,
        "engine": {"name": eng.name, "versions": res.versions, "elapsed_sec": res.elapsed_sec},
        "fallbacks": {"count": len(res.fallbacks), "fraction": frac_fb, "na_replaced": res.na_replaced},
        "estimates": {
            "memory": None if pre.memory is None else asdict(pre.memory),
            "runtime": None if pre.runtime is None else asdict(pre.runtime),
        },
        "timing": {"engine_wall_sec": engine_sec, "total_sec": None},
        "software": {"tsfstudio": __version__, "python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "platform": platform.platform()},
        "warnings": warnings,
        "outputs": {},
    }
    if progress:
        progress("writing", 0, 1)
    run_info = {"run_id": run_id, "dataset": label, "engine": eng.name, "task": cfg.task, "method": cfg.method,
                "horizon": prep.horizon, "seasonality": list(prep.seasonality), "n_series": n, "seed": cfg.seed,
                "fallbacks": len(res.fallbacks), "engine_sec": f"{res.elapsed_sec:.2f}"}
    run_info["origins"] = n_orig
    manifest["outputs"] = _write_outputs(run_dir, prep, res.forecasts, res.fallbacks, per_series, summary, run_info, cfg.write_xlsx)
    if n_orig > 1:
        manifest["outputs"].update(_write_origin_outputs(run_dir, origin_rows))
    if latest_bounds is not None:
        probs, bnds = latest_bounds
        cols = [f"h{k}" for k in range(1, prep.horizon + 1)]
        frames = []
        for p in probs:
            fr = pd.DataFrame(bnds[p], columns=cols)
            fr.insert(0, "quantile", p)
            fr.insert(0, "series_name", prep.names)
            frames.append(fr)
        pd.concat(frames, ignore_index=True).to_csv(run_dir / "intervals.csv", index=False)
        manifest["outputs"]["intervals_csv"] = "intervals.csv"
    manifest["timing"]["total_sec"] = time.time() - t_start
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    manifest["outputs"]["manifest"] = "manifest.json"
    if progress:
        progress("done", 1, 1)
    return RunResult(run_dir, manifest, prep.names, prep.train, prep.test, res.forecasts, per_series, summary,
                     res.fallbacks, pre)
