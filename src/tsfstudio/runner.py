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

    def to_dict(self) -> dict:
        d = asdict(self)
        d["output_dir"] = None if self.output_dir is None else str(self.output_dir)
        d["seasonality"] = None if self.seasonality is None else list(self.seasonality)
        d["metrics"] = list(self.metrics)
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


def prepare(dataset: TSFDataset, cfg: RunConfig, entry: DatasetEntry | None = None) -> Prepared:
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
        if len(y) < H + 2:
            excluded.append({"index": i, "name": nm, "reason": f"series too short (length {len(y)} < horizon+2 = {H + 2})"})
            continue
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
            "becomes 0 when rounded; all forecasts can come out as 0. To see the rate forecast, run with --integer-conversion off.")
    if excluded:
        warnings.append(f"{len(excluded)} series excluded (details in manifest.json -> split.excluded).")
    if test_missing:
        warnings.append(f"There are {test_missing} missing values in the test part; the measures exclude these pairs.")
    return Prepared(kept_names, train, np.vstack(test_rows), H, seas, integer, excluded,
                    {"policy": cfg.missing_policy, "series_with_missing": series_with_missing,
                     "imputed_values": imputed, "test_missing_values": test_missing}, warnings)


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


def run(dataset: TSFDataset, cfg: RunConfig, *, entry: DatasetEntry | None = None, dataset_id: str | None = None,
        engine: Engine | None = None, progress: ProgressFn | None = None,
        cancel: threading.Event | None = None) -> RunResult:
    t_start = time.time()
    label = dataset_id or (entry.id if entry else None) or (Path(dataset.source).stem if dataset.source else "dataset")
    if progress:
        progress("prepare", 0, 1)
    prep = prepare(dataset, cfg, entry)
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
    res = eng.run(job, progress=progress, cancel=cancel)
    engine_sec = time.time() - t_engine
    if res.forecasts.shape != prep.test.shape:
        raise RunError(f"Engine output has an unexpected shape: {res.forecasts.shape} != {prep.test.shape}")

    if progress:
        progress("metrics", 0, 1)
    per_series = compute_error_table(res.forecasts, prep.test, prep.train, prep.seasonality, series_names=prep.names,
                                     metrics=cfg.metrics, mase_digits=cfg.mase_digits)
    summary = summarize(per_series)

    warnings = list(prep.warnings) + list(pre.warnings) + list(res.notes)
    n = len(prep.train)
    frac_fb = len(res.fallbacks) / n if n else 0.0
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
        "split": {"horizon": prep.horizon, "n_excluded": len(prep.excluded), "excluded": prep.excluded},
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
    manifest["outputs"] = _write_outputs(run_dir, prep, res.forecasts, res.fallbacks, per_series, summary, run_info, cfg.write_xlsx)
    manifest["timing"]["total_sec"] = time.time() - t_start
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    manifest["outputs"]["manifest"] = "manifest.json"
    if progress:
        progress("done", 1, 1)
    return RunResult(run_dir, manifest, prep.names, prep.train, prep.test, res.forecasts, per_series, summary,
                     res.fallbacks, pre)
