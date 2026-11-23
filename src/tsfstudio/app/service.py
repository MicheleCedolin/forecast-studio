"""Interface service layer: business logic independent of HTTP (testable).

Responsibilities: datasets (catalogue + uploads), model registry, configuration validation, pre-run check (memory/time), background
run (one job at a time; progress/cancel), past runs, series chart data, error distribution, Excel/TSF upload.

Security: the client never supplies a file PATH. A dataset is selected only by catalogue id or by a name in the uploads folder;
downloadable files are limited to the outputs listed in the run manifest.
"""
from __future__ import annotations

import json
import math
import os
import platform
import re
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import __version__
from .. import catalog as C
from ..engines.base import EngineCancelled, EngineError
from ..engines.registry import get_engine
from ..estimate import available_memory_bytes, preflight
from ..metrics import ALL_METRICS, DEFAULT_METRICS
from ..runner import MISSING_POLICIES, RunConfig, RunError, build_job, prepare, run
from ..seasonality import primary_period, seasonality_for
from ..tsf.reader import TSFDataset, read_tsf, read_tsf_header
from . import models_info as MI

PREVIEW_SERIES = 200            # veri önizlemesi/profili için okunacak ilk seri sayısı
SPARK_SERIES = 6                # küçük çizgi grafik sayısı
CACHE_MAX_BYTES = 40_000_000    # bundan büyük dosyalar bellekte önbelleğe alınmaz (8 GB bellek)
STATS_MAX_BYTES = 60_000_000    # dizin yoksa tam özet yalnızca bundan küçük dosyalarda hesaplanır
MAX_UPLOAD = 400_000_000
UPLOAD_PREFIX = "upload:"


class AppError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def clean(obj):
    """JSON için: NaN/Inf -> None, numpy -> python."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.\-]", "_", Path(name).stem)
    return s.strip("._") or "veri"


# ----------------------------------------------------------------------------------------------------
@dataclass
class Job:
    id: str
    payload: dict
    status: str = "running"          # running | done | error | cancelled
    stage: str = "starting"
    done: int = 0
    total: int = 0
    t0: float = field(default_factory=time.time)
    t1: float | None = None
    error: str | None = None
    run_id: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    cancelling: bool = False

    def view(self) -> dict:
        end = self.t1 if self.t1 else time.time()
        return {"id": self.id, "status": self.status, "stage": self.stage, "done": self.done, "total": self.total,
                "elapsed": end - self.t0, "error": self.error, "run_id": self.run_id, "cancelling": self.cancelling,
                "method": self.payload.get("method"), "engine": self.payload.get("engine"), "dataset": self.payload.get("dataset")}


class AppService:
    def __init__(self, data_dir: str | os.PathLike | None = None, runs_dir: str | os.PathLike | None = None,
                 uploads_dir: str | os.PathLike | None = None):
        self.data_dir = Path(data_dir) if data_dir else C.default_data_dir()
        self.runs_dir = Path(runs_dir) if runs_dir else Path(os.environ.get("TSFSTUDIO_RUNS_DIR", "runs"))
        self.uploads_dir = Path(uploads_dir) if uploads_dir else Path(os.environ.get("TSFSTUDIO_UPLOADS_DIR", "uploads"))
        self.catalog = C.load_catalog()
        self._lock = threading.RLock()
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._ds_cache: OrderedDict[tuple, TSFDataset] = OrderedDict()
        self._engine_cache: tuple[float, dict] | None = None
        self._run_cache: OrderedDict[str, dict] = OrderedDict()
        self._runs_index: dict[str, tuple[float, dict]] = {}
        self._stats_cache: dict[str, dict] = {}
        self._seq = 0

    # ============================================================================================ sistem / motorlar
    def engines(self, refresh: bool = False) -> dict:
        with self._lock:
            if self._engine_cache and not refresh and time.time() - self._engine_cache[0] < 90:
                return self._engine_cache[1]
        out = {}
        for name in ("python", "r", "cpp"):
            eng = get_engine(name)
            try:
                ok, msg = eng.available()
            except Exception as exc:  # noqa: BLE001
                ok, msg = False, f"{type(exc).__name__}: {exc}"
            out[name] = {"available": bool(ok), "message": msg, "methods": eng.methods() if ok else None}
        with self._lock:
            self._engine_cache = (time.time(), out)
        return out

    def system(self, refresh: bool = False) -> dict:
        mem = available_memory_bytes()
        eng = self.engines(refresh)
        cur = self.current_job()
        return {"version": __version__, "python": platform.python_version(), "platform": platform.platform(),
                "cpu": os.cpu_count(), "free_memory_gb": None if not mem else round(mem / 1024 ** 3, 2),
                "data_dir": str(self.data_dir.resolve()), "runs_dir": str(self.runs_dir.resolve()),
                "uploads_dir": str(self.uploads_dir.resolve()),
                "engines": {k: {"available": v["available"], "message": v["message"]} for k, v in eng.items()},
                "busy": cur is not None,
                # The running job itself, so a reloaded page can show its progress and Cancel again (UI review H2).
                "job": cur.view() if cur is not None else None}

    def models(self) -> dict:
        eng = self.engines()
        cat = MI.build_catalog({k: (v["methods"] if v["available"] else None) for k, v in eng.items()})
        return {"groups": [{"id": g, "title": t, "text": x} for g, t, x in MI.GROUPS], "models": cat,
                "metrics": MI.METRICS_INFO, "metric_defaults": list(DEFAULT_METRICS),
                "metric_all": list(ALL_METRICS),
                "metric_refs": [dict(MI.REFS[k], key=k) for k in MI.METRIC_REFS],
                "ref_table": {k: dict(v, key=k) for k, v in MI.REFS.items()},
                "missing_policies": [
                    {"value": "error", "label": "Stop with an error (default)",
                     "help": "If the training part has missing values the run does not start; nothing is changed silently."},
                    {"value": "locf", "label": "Carry last value forward (LOCF)", "help": "A missing value is filled with the previous observation."},
                    {"value": "interpolate", "label": "Linear interpolation", "help": "Missing values are filled linearly between neighbouring observations."},
                    {"value": "zero", "label": "Fill with zero", "help": "Missing values are treated as 0 (only meaningful when missing means no demand)."}]}

    # ============================================================================================ veri kümeleri
    def _upload_path(self, ds_id: str) -> Path:
        name = ds_id[len(UPLOAD_PREFIX):]
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name or ""):
            raise AppError("Invalid uploaded dataset name.", 404)
        p = self.uploads_dir / f"{name}.tsf"
        if not p.is_file():
            raise AppError(f"Uploaded dataset not found: {name}", 404)
        return p

    def resolve(self, ds_id: str) -> tuple[C.DatasetEntry | None, Path]:
        if not isinstance(ds_id, str) or not ds_id:
            raise AppError("No dataset selected.")
        if ds_id.startswith(UPLOAD_PREFIX):
            return None, self._upload_path(ds_id)
        try:
            entry = self.catalog.get(ds_id)
        except KeyError:
            raise AppError(f"Unknown dataset: {ds_id}", 404) from None
        st = C.availability(entry, self.data_dir)
        if st != "ok":
            what = "was not found" if st == "missing" else "has a size that does not match the catalogue (possibly corrupted or a different version)"
            raise AppError(f"Data file for '{ds_id}' {what}: {self.data_dir / entry.file}")
        return entry, self.data_dir / entry.file

    def _load(self, ds_id: str, max_series: int | None = None) -> tuple[TSFDataset, C.DatasetEntry | None]:
        entry, path = self.resolve(ds_id)
        key = (ds_id, max_series)
        with self._lock:
            if key in self._ds_cache:
                self._ds_cache.move_to_end(key)
                return self._ds_cache[key], entry
        ds = read_tsf(path, max_series=max_series)
        if path.stat().st_size <= CACHE_MAX_BYTES:
            with self._lock:
                self._ds_cache[key] = ds
                while len(self._ds_cache) > 3:
                    self._ds_cache.popitem(last=False)
        return ds, entry

    def _index(self) -> dict:
        try:
            return C.load_index(self.data_dir)
        except Exception:  # noqa: BLE001
            return {}

    def datasets(self) -> dict:
        idx = self._index()
        items = []
        for e in self.catalog.entries:
            st = C.availability(e, self.data_dir)
            s = idx.get(e.id) or {}
            items.append({
                "id": e.id, "kind": "catalog", "title": e.domain, "bundle": e.bundle, "file": e.file, "bytes": e.bytes, "status": st,
                "frequency": s.get("frequency"), "n_series": s.get("n_series"), "len_median": s.get("len_median"),
                "horizon": e.horizon_override if e.horizon_override is not None else s.get("horizon"),
                "lag": e.lag, "lag_origin": e.lag_origin, "integer_conversion": e.integer_conversion,
                "local_methods": list(e.local_methods), "global_methods": list(e.global_methods)})
        for p in sorted(self.uploads_dir.glob("*.tsf")) if self.uploads_dir.is_dir() else []:
            try:
                h = read_tsf_header(p)
            except Exception:  # noqa: BLE001
                continue
            items.append({"id": f"{UPLOAD_PREFIX}{p.stem}", "kind": "upload", "title": h.relation or p.stem, "bundle": "user", "file": p.name,
                          "bytes": p.stat().st_size, "status": "user", "frequency": h.frequency, "n_series": None, "len_median": None,
                          "horizon": h.horizon, "lag": None, "lag_origin": None, "integer_conversion": False,
                          "local_methods": [], "global_methods": []})
        return {"items": items, "missing": sum(1 for i in items if i["status"] == "missing")}

    @staticmethod
    def demand_profile(values: list[np.ndarray]) -> dict:
        """Syntetos–Boylan–Croston (2005) sınıflandırması: ADI >= 1,32 ve CV² >= 0,49 eşikleri."""
        counts = {"smooth": 0, "erratic": 0, "intermittent": 0, "lumpy": 0, "all_zero": 0}
        for v in values:
            x = np.asarray(v, dtype=float)
            x = x[~np.isnan(x)]
            nz = x[x != 0]
            if x.size == 0 or nz.size == 0:
                counts["all_zero"] += 1
                continue
            adi = x.size / nz.size
            cv2 = float((nz.std() / nz.mean()) ** 2) if nz.mean() != 0 else math.inf
            key = ("smooth" if cv2 < 0.49 else "erratic") if adi < 1.32 else ("intermittent" if cv2 < 0.49 else "lumpy")
            counts[key] += 1
        n = max(sum(counts.values()), 1)
        return {"n": sum(counts.values()), "counts": counts, "share": {k: v / n for k, v in counts.items()}}

    @staticmethod
    def _downsample(y: np.ndarray, max_points: int = 260) -> list:
        y = np.asarray(y, dtype=float)
        if y.size > max_points:
            idx = np.linspace(0, y.size - 1, max_points).round().astype(int)
            y = y[idx]
        return [None if not math.isfinite(v) else round(float(v), 4) for v in y]

    def dataset_detail(self, ds_id: str) -> dict:
        entry, path = self.resolve(ds_id)
        header = read_tsf_header(path)
        ds = read_tsf(path, max_series=PREVIEW_SERIES)
        freq = header.frequency
        seas = seasonality_for(freq)
        stats = (self._index().get(ds_id) if entry else None) or self._stats_cache.get(ds_id)
        stats_note = None
        if not stats:
            size = path.stat().st_size
            if size <= STATS_MAX_BYTES:
                full, _ = self._load(ds_id)
                stats = C.dataset_stats(full)
                self._stats_cache[ds_id] = stats
                if entry is not None:
                    try:
                        C.build_index(self.data_dir, [ds_id])
                    except Exception:  # noqa: BLE001
                        pass
            else:
                stats_note = ("File is large; the full summary was not computed. You can generate it once in a terminal with `python -m tsfstudio index --ids "
                              f"{ds_id}`.")
        lens = ds.lengths
        zero_share = float(np.mean([np.mean(v[~np.isnan(v)] == 0) if (~np.isnan(v)).any() else 0 for v in ds.values])) if ds.values else None
        horizon = (entry.effective_horizon(header) if entry else header.horizon)
        names = ds.series_names
        return clean({
            "id": ds_id, "title": (entry.domain if entry else header.relation), "file": path.name, "bytes": path.stat().st_size,
            "frequency": freq, "header_horizon": header.horizon, "horizon": horizon,
            "seasonality": list(seas), "primary_period": primary_period(seas),
            "lag": entry.lag if entry else None, "lag_origin": entry.lag_origin if entry else None,
            "lag_suggestion": None if entry and entry.lag else max(2, min(primary_period(seas), 60)),
            "integer_conversion": bool(entry.integer_conversion) if entry else False,
            "recommended_local": list(entry.recommended_local_methods(header)) if entry else [],
            "recommended_global": list(entry.global_methods) if entry else [],
            "study_methods": list(entry.local_methods) if entry else [],
            "stats": stats, "stats_note": stats_note,
            "preview": {"n_read": ds.n_series, "len_min": int(lens.min()) if lens.size else None, "len_max": int(lens.max()) if lens.size else None,
                        "zero_share": zero_share, "profile": self.demand_profile(ds.values),
                        "series": [{"name": names[i], "n": int(len(ds.values[i])), "y": self._downsample(ds.values[i])}
                                   for i in range(min(SPARK_SERIES, ds.n_series))]},
            "consistency": ds.consistency_warnings()})

    # ============================================================================================ yapılandırma
    def parse_config(self, payload: dict) -> tuple[RunConfig, C.DatasetEntry | None, str]:
        if not isinstance(payload, dict):
            raise AppError("Invalid request body.")
        ds_id = payload.get("dataset")
        self.resolve(ds_id)                                          # varlık denetimi
        entry = None if str(ds_id).startswith(UPLOAD_PREFIX) else self.catalog.get(ds_id)
        method = payload.get("method")
        engine = payload.get("engine") or "python"
        if method not in MI.MODELS:
            raise AppError(f"Unknown method: {method!r}")
        if engine not in ("python", "r", "cpp"):
            raise AppError(f"Unknown engine: {engine!r}")
        info = MI.MODELS[method]
        if engine not in info["engines"]:
            raise AppError(f"'{info['name']}' is not available in the {engine} engine.")
        task = info["task"]

        def opt_int(key, lo=1):
            v = payload.get(key)
            if v in (None, ""):
                return None
            try:
                f = float(v)
            except (TypeError, ValueError):
                raise AppError(f"'{key}' must be a number.") from None
            if f != int(f) or int(f) < lo:
                raise AppError(f"'{key}' must be a whole number of at least {lo}.")
            return int(f)

        ic = payload.get("integer_conversion", "auto")
        if ic not in ("auto", "on", "off", None):
            raise AppError("integer_conversion must be 'auto', 'on' or 'off'.")
        mp = payload.get("missing_policy") or "error"
        if mp not in MISSING_POLICIES:
            raise AppError(f"Invalid missing-value policy: {mp}")
        metrics = payload.get("metrics") or list(DEFAULT_METRICS)
        if not isinstance(metrics, list) or not metrics or any(m not in ALL_METRICS for m in metrics):
            raise AppError("Invalid metric selection.")
        workers = opt_int("workers") or 1
        workers = min(workers, max(1, (os.cpu_count() or 1)))
        cfg = RunConfig(engine=engine, task=task, method=method, horizon=opt_int("horizon"), lag=opt_int("lag") if task == "global" else None,
                        integer_conversion={"auto": None, None: None, "on": True, "off": False}[ic], missing_policy=mp,
                        params=self._params(info, engine, payload.get("params") or {}), seed=opt_int("seed", 0) if payload.get("seed") not in (None, "") else 1,
                        workers=workers, max_series=opt_int("max_series"), metrics=tuple(metrics), output_dir=self.runs_dir)
        return cfg, entry, ds_id

    @staticmethod
    def _params(info: dict, engine: str, given: dict) -> dict:
        """Şemaya göre doğrular; varsayılanı olan parametreler AÇIKÇA yazılır (manifest tam parametre kümesini kaydeder)."""
        if not isinstance(given, dict):
            raise AppError("params must be an object.")
        schema = {p["key"]: p for p in info["params"] if engine in p["engines"]}
        unknown = [k for k in given if k not in schema]
        if unknown:
            raise AppError(f"Unrecognised parameter for this method/engine: {', '.join(unknown)}")
        flat: dict = {}
        for k, p in schema.items():
            v = given.get(k, p["default"])
            if v in (None, ""):
                continue
            kind = p["kind"]
            try:
                if kind == "int":
                    f = float(v)
                    if f != int(f):
                        raise ValueError
                    v = int(f)
                elif kind == "float":
                    v = float(v)
                elif kind == "bool":
                    v = bool(v)
                elif kind == "choice":
                    if str(v) not in [c["value"] for c in p["choices"]]:
                        raise ValueError
                    v = str(v)
            except (TypeError, ValueError):
                raise AppError(f"Invalid value for '{p['label']}': {v!r}") from None
            if kind in ("int", "float"):
                if p["min"] is not None and v < p["min"]:
                    raise AppError(f"'{p['label']}' must be at least {p['min']}.")
                if p["max"] is not None and v > p["max"]:
                    raise AppError(f"'{p['label']}' must be at most {p['max']}.")
            flat[k] = v
        out: dict = {}
        for k, v in flat.items():                      # "catboost.depth" -> {"catboost": {"depth": ...}}
            if "." in k:
                a, b = k.split(".", 1)
                out.setdefault(a, {})[b] = v
            else:
                out[k] = v
        return out

    # ============================================================================================ ön kontrol
    def estimate(self, payload: dict, probe: bool = False) -> dict:
        cfg, entry, ds_id = self.parse_config(payload)
        try:
            ds, _ = self._load(ds_id, cfg.max_series)
            prep = prepare(ds, cfg, entry)
            job = build_job(prep, cfg, entry)
        except RunError as exc:
            return {"ok": False, "error": str(exc)}
        eng = get_engine(cfg.engine)
        ok, msg = eng.available()
        if not ok:
            return {"ok": False, "error": f"Engine '{eng.name}' is not available: {msg}"}
        if not eng.supports(cfg.task, cfg.method):
            return {"ok": False, "error": f"Engine '{eng.name}' does not offer '{cfg.method}' right now (the library may not be installed). {msg}"}
        if probe and self.current_job() is not None:
            probe = False
        rep = preflight(job, engine=eng, probe=probe)
        return clean({"ok": True, "n_series_run": len(prep.train), "horizon": prep.horizon, "seasonality": list(prep.seasonality),
                      "integer_conversion": prep.integer_conversion, "lag": job.lag, "missing": prep.missing_info,
                      "excluded": len(prep.excluded), "warnings": list(prep.warnings) + list(rep.warnings),
                      "memory": None if rep.memory is None else asdict(rep.memory),
                      "runtime": None if rep.runtime is None else asdict(rep.runtime),
                      "blocked": rep.blocked, "long_running": rep.long_running, "probed": bool(probe),
                      "mean_length": float(np.mean([len(s) for s in prep.train])), "params": cfg.params})

    # ============================================================================================ koşu işleri
    def current_job(self) -> Job | None:
        with self._lock:
            for j in reversed(self._jobs.values()):
                if j.status == "running" and j.payload.get("kind") != "features":   # feature jobs run alongside
                    return j
        return None

    def start_run(self, payload: dict, confirm_long: bool = False) -> dict:
        with self._lock:
            if self.current_job() is not None:
                raise AppError("Another run is in progress; wait for it to finish or cancel it (this machine is set up for one run at a time).", 409)
            cfg, entry, ds_id = self.parse_config(payload)
            self._seq += 1
            job = Job(id=f"j{int(time.time())}-{self._seq}", payload={**payload, "dataset": ds_id})
            self._jobs[job.id] = job
            while len(self._jobs) > 30:
                self._jobs.popitem(last=False)
        threading.Thread(target=self._run_job, args=(job, cfg, entry, ds_id, confirm_long), daemon=True, name=f"tsfs-{job.id}").start()
        return job.view()

    def _run_job(self, job: Job, cfg: RunConfig, entry, ds_id: str, confirm_long: bool) -> None:
        def progress(stage, done, total):
            job.stage, job.done, job.total = stage, int(done), int(total)

        try:
            progress("loading", 0, 1)
            ds, _ = self._load(ds_id, cfg.max_series)
            if not confirm_long:
                prep = prepare(ds, cfg, entry)
                pre = preflight(build_job(prep, cfg, entry), engine=get_engine(cfg.engine))
                if pre.long_running:
                    raise RunError("Long run: the estimated time exceeds 15 minutes. Read the time estimate, tick the confirmation box and start again.")
            res = run(ds, cfg, entry=entry, dataset_id=ds_id, engine=get_engine(cfg.engine), progress=progress, cancel=job.cancel)
            job.run_id = res.run_dir.name
            job.status = "done"
        except EngineCancelled:
            job.status = "cancelled"
        except (RunError, EngineError) as exc:
            job.status, job.error = "error", str(exc)
        except MemoryError:
            job.status, job.error = "error", "Out of memory. Reduce the number of series (Settings > Number of series) and close other applications."
        except Exception as exc:  # noqa: BLE001
            job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
            try:
                self.runs_dir.mkdir(parents=True, exist_ok=True)
                with open(self.runs_dir / "_app_errors.log", "a", encoding="utf-8") as fh:
                    fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {job.id} {json.dumps(job.payload, ensure_ascii=False, default=str)}\n")
                    fh.write(traceback.format_exc())
            except OSError:
                pass
        finally:
            job.t1 = time.time()

    def job(self, job_id: str) -> dict:
        with self._lock:
            j = self._jobs.get(job_id)
        if j is None:
            raise AppError("Job not found.", 404)
        return j.view()

    def cancel_job(self, job_id: str) -> dict:
        with self._lock:
            j = self._jobs.get(job_id)
        if j is None:
            raise AppError("Job not found.", 404)
        if j.status == "running":
            j.cancel.set()
            j.cancelling = True
        return j.view()

    # ============================================================================================ series features
    def _feat_spec(self, payload: dict) -> tuple[str, tuple[str, ...], bool]:
        from .. import features as F
        ds_id = payload.get("dataset")
        self.resolve(ds_id)
        sets = payload.get("sets") or list(F.FEATURE_SETS)
        if isinstance(sets, str):
            sets = [s for s in sets.split(",") if s]
        bad = [s for s in sets if s not in F.FEATURE_SETS]
        if bad or not sets:
            raise AppError(f"Unknown feature set: {', '.join(bad) or '(none)'}")
        sets = tuple(s for s in F.FEATURE_SETS if s in sets)          # canonical order
        return ds_id, sets, bool(payload.get("study_fft")) and "catch22" in sets

    def _feat_path(self, ds_id: str, sets: tuple[str, ...], study_fft: bool) -> Path:
        tag = "+".join(sets) + ("_studyfft" if study_fft else "")
        return self.runs_dir / "_features" / f"{_safe_name(ds_id.replace(':', '_'))}__{tag}.csv"

    def start_features(self, payload: dict) -> dict:
        ds_id, sets, study_fft = self._feat_spec(payload)
        with self._lock:
            for j in self._jobs.values():
                if j.status == "running" and j.payload.get("kind") == "features":
                    raise AppError("Features are already being computed; wait for that job to finish or cancel it.", 409)
            self._seq += 1
            job = Job(id=f"f{int(time.time())}-{self._seq}",
                      payload={"kind": "features", "dataset": ds_id, "sets": list(sets), "study_fft": study_fft})
            self._jobs[job.id] = job
        threading.Thread(target=self._features_job, args=(job, ds_id, sets, study_fft), daemon=True,
                         name=f"tsfs-{job.id}").start()
        return job.view()

    def _features_job(self, job: Job, ds_id: str, sets: tuple[str, ...], study_fft: bool) -> None:
        from .. import features as F

        def progress(stage, done, total):
            job.stage, job.done, job.total = stage, int(done), int(total)

        try:
            progress("loading", 0, 1)
            ds, _ = self._load(ds_id)
            m = primary_period(seasonality_for(ds.frequency))
            df = F.compute(ds.values, m=m, sets=sets, study_fft=study_fft, names=list(ds.series_names),
                           progress=progress, cancel=job.cancel)
            p = self._feat_path(ds_id, sets, study_fft)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            df.to_csv(tmp, float_format="%.17g")
            os.replace(tmp, p)
            job.status = "done"
        except F.FeaturesCancelled:
            job.status = "cancelled"
        except F.FeatureError as exc:
            job.status, job.error = "error", str(exc)
        except Exception as exc:  # noqa: BLE001
            job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
            try:
                self.runs_dir.mkdir(parents=True, exist_ok=True)
                with open(self.runs_dir / "_app_errors.log", "a", encoding="utf-8") as fh:
                    fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {job.id} features {ds_id}\n{traceback.format_exc()}")
            except OSError:
                pass
        finally:
            job.t1 = time.time()

    def _feat_frame(self, payload: dict) -> tuple[pd.DataFrame | None, Path]:
        ds_id, sets, study_fft = self._feat_spec(payload)
        p = self._feat_path(ds_id, sets, study_fft)
        if not p.is_file():
            return None, p
        return pd.read_csv(p, index_col=0, keep_default_na=True), p

    def features(self, payload: dict, max_rows: int = 5000) -> dict:
        """The stored feature table of a dataset (computed earlier with start_features), with a per-feature summary."""
        from .. import features as F
        df, p = self._feat_frame(payload)
        if df is None:
            return {"status": "none"}
        num = df.select_dtypes(include=[np.number])
        summary = []
        for c in df.columns:
            if c in num.columns:
                v = num[c].to_numpy(float)
                f = v[np.isfinite(v)]
                summary.append({"name": c, "missing": int(v.size - f.size),
                                "mean": f.mean() if f.size else None, "sd": f.std(ddof=1) if f.size > 1 else None,
                                "min": f.min() if f.size else None, "median": np.median(f) if f.size else None,
                                "max": f.max() if f.size else None, "distinct": int(np.unique(f).size)})
            else:
                vc = df[c].value_counts(dropna=True)
                summary.append({"name": c, "missing": int(df[c].isna().sum()), "levels": {str(k): int(v) for k, v in vc.items()}})
        groups = {"basic": [c for c in F.BASIC_NAMES if c in df.columns],
                  "catch22": [c for c in F.CATCH22_NAMES if c in df.columns]}
        rows = df.head(max_rows)
        return clean({"status": "ready", "n_series": int(len(df)), "columns": list(df.columns), "groups": groups,
                      "summary": summary, "updated": p.stat().st_mtime, "truncated": len(df) > max_rows,
                      "index": [str(i) for i in rows.index],
                      "rows": [[None if (isinstance(x, float) and not math.isfinite(x)) else x for x in r]
                               for r in rows.itertuples(index=False, name=None)]})

    def features_file(self, payload: dict) -> tuple[bytes, str]:
        df, p = self._feat_frame(payload)
        if df is None:
            raise AppError("No features computed for this dataset yet.", 404)
        return p.read_bytes(), p.name

    # ============================================================================================ geçmiş koşular
    def _run_dir(self, run_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", run_id or "") or run_id.startswith("."):
            raise AppError("Invalid run id.", 404)
        d = self.runs_dir / run_id
        if not (d / "manifest.json").is_file():
            raise AppError("Run not found.", 404)
        return d

    @staticmethod
    def _read_summary(d: Path) -> pd.DataFrame | None:
        p = d / "summary.csv"
        if p.is_file():
            return pd.read_csv(p)
        x = d / "metrics.xlsx"
        if x.is_file():
            try:
                return pd.read_excel(x, sheet_name="Summary")
            except Exception:  # noqa: BLE001
                return None
        return None

    def _manifest(self, d: Path) -> dict:
        return json.loads((d / "manifest.json").read_text(encoding="utf-8"))

    def runs(self) -> dict:
        out = []
        if self.runs_dir.is_dir():
            for d in self.runs_dir.iterdir():
                mf = d / "manifest.json"
                if not mf.is_file():
                    continue
                mt = mf.stat().st_mtime
                cached = self._runs_index.get(d.name)
                if cached and cached[0] == mt:
                    out.append(cached[1])
                    continue
                try:
                    m = self._manifest(d)
                    s = self._read_summary(d)
                    summ = {} if s is None else {r["Metric"]: {"mean": r["Mean"], "median": r["Median"], "n_valid": r["N_valid"]} for _, r in s.iterrows()}
                    cfg = m.get("config", {})
                    row = clean({
                        "run_id": d.name, "created": m.get("created_utc"), "dataset": m.get("dataset", {}).get("id"),
                        "engine": cfg.get("engine"), "method": cfg.get("method"), "task": cfg.get("task"),
                        "horizon": cfg.get("effective_horizon"), "n_series": m.get("dataset", {}).get("n_series_run"),
                        "fallbacks": m.get("fallbacks", {}).get("count"), "elapsed": m.get("engine", {}).get("elapsed_sec"),
                        "integer_conversion": cfg.get("effective_integer_conversion"), "max_series": cfg.get("max_series"),
                        "n_warnings": len(m.get("warnings") or []), "summary": summ})
                except Exception:  # noqa: BLE001
                    continue
                self._runs_index[d.name] = (mt, row)
                out.append(row)
        out.sort(key=lambda r: r.get("created") or "", reverse=True)
        return {"runs": out[:300], "runs_dir": str(self.runs_dir.resolve())}

    def _run_frames(self, run_id: str) -> dict:
        with self._lock:
            if run_id in self._run_cache:
                self._run_cache.move_to_end(run_id)
                return self._run_cache[run_id]
        d = self._run_dir(run_id)
        fc = pd.read_csv(d / "forecasts.csv")
        ac = pd.read_csv(d / "test_actuals.csv")
        p = d / "errors_per_series.csv"
        if p.is_file():
            er = pd.read_csv(p)
        else:
            er = pd.read_excel(d / "metrics.xlsx", sheet_name="Errors_Per_Series")
        fr = {"fc": fc, "ac": ac, "er": er, "dir": d}
        with self._lock:
            self._run_cache[run_id] = fr
            while len(self._run_cache) > 2:
                self._run_cache.popitem(last=False)
        return fr

    @staticmethod
    def diagnostics(f: np.ndarray, a: np.ndarray) -> dict:
        ok = ~(np.isnan(f) | np.isnan(a))
        n_ok = int(ok.sum())
        if n_ok == 0:
            return {}
        fz = np.where(ok, f == 0, False)
        az = np.where(ok, a == 0, False)
        sum_a = float(np.where(ok, a, 0).sum())
        bias = float(np.where(ok, f - a, 0).sum()) / sum_a * 100 if sum_a > 0 else None
        series_ok = ok.any(axis=1)
        all_zero = [(f[i][ok[i]] == 0).all() for i in range(f.shape[0]) if series_ok[i]]
        return {"zero_forecast_share": float(fz.sum() / n_ok), "zero_actual_share": float(az.sum() / n_ok),
                "series_all_zero_forecast_share": float(np.mean(all_zero)) if all_zero else None,
                "bias_pct": bias, "mean_forecast": float(np.where(ok, f, 0).sum() / n_ok), "mean_actual": float(np.where(ok, a, 0).sum() / n_ok)}

    def run_detail(self, run_id: str) -> dict:
        d = self._run_dir(run_id)
        m = self._manifest(d)
        fr = self._run_frames(run_id)
        f = fr["fc"].drop(columns=["series_name"]).to_numpy(float)
        a = fr["ac"].drop(columns=["series_name"]).to_numpy(float)
        s = self._read_summary(d)
        fb_path = d / "fallbacks.csv"
        fb = pd.read_csv(fb_path) if fb_path.is_file() else pd.DataFrame()
        return clean({
            "run_id": run_id, "manifest": m, "metrics_columns": [c for c in fr["er"].columns if c != "series_name"],
            "summary": [] if s is None else s.to_dict("records"), "diagnostics": self.diagnostics(f, a),
            "n_series": int(f.shape[0]), "horizon": int(f.shape[1]),
            "fallbacks": fb.head(30).to_dict("records") if len(fb) else [],
            "files": [{"key": k, "name": v} for k, v in (m.get("outputs") or {}).items() if (d / v).is_file()] + [{"key": "manifest", "name": "manifest.json"}]})

    def run_errors(self, run_id: str, sort: str | None, desc: bool, q: str, offset: int, limit: int) -> dict:
        er = self._run_frames(run_id)["er"]
        if q:
            er = er[er["series_name"].astype(str).str.contains(q, case=False, regex=False)]
        if sort and sort in er.columns:
            er = er.sort_values(sort, ascending=not desc, na_position="last", kind="stable")
        total = len(er)
        page = er.iloc[offset: offset + max(1, min(limit, 200))]
        return clean({"total": total, "columns": list(er.columns), "rows": page.to_dict("records")})

    def run_hist(self, run_id: str, metric: str, bins: int = 28) -> dict:
        er = self._run_frames(run_id)["er"]
        if metric not in er.columns or metric == "series_name":
            raise AppError("Unknown measure.")
        v = er[metric].to_numpy(float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return {"metric": metric, "counts": [], "edges": [], "n": 0, "clipped": 0}
        lo, hi = float(v.min()), float(np.percentile(v, 98))
        if hi <= lo:
            hi = lo + (abs(lo) * 0.1 or 1.0)
        clipped = int((v > hi).sum())
        counts, edges = np.histogram(v[v <= hi], bins=max(5, min(bins, 60)), range=(lo, hi))     # %98 üstü sayılmaz (clipped olarak bildirilir)
        return clean({"metric": metric, "counts": counts.tolist(), "edges": edges.tolist(), "n": int(v.size), "clipped": clipped,
                      "mean": float(v.mean()), "median": float(np.median(v))})

    def run_series(self, run_id: str, name: str) -> dict:
        fr = self._run_frames(run_id)
        names = fr["fc"]["series_name"].astype(str).tolist()
        if name not in names:
            raise AppError("Series not found.", 404)
        i = names.index(name)
        m = self._manifest(fr["dir"])
        H = int(m["config"]["effective_horizon"])
        f = fr["fc"].drop(columns=["series_name"]).iloc[i].to_numpy(float)
        a = fr["ac"].drop(columns=["series_name"]).iloc[i].to_numpy(float)
        hist: list = []
        hist_note = None
        try:
            ds_id = m["dataset"]["id"]
            ds, _ = self._load(ds_id, m["config"].get("max_series"))
            snames = ds.series_names
            if name in snames:
                y = np.asarray(ds.values[snames.index(name)], float)
                tr = y[:-H] if len(y) > H else y
                k = int(min(len(tr), max(5 * H, 48), 700))
                hist = [None if not math.isfinite(x) else round(float(x), 4) for x in tr[-k:]]
        except Exception as exc:  # noqa: BLE001
            hist_note = f"Past data could not be read ({type(exc).__name__}); showing only the test and forecast."
        errs = fr["er"][fr["er"]["series_name"].astype(str) == name]
        return clean({"name": name, "index": i, "horizon": H, "history": hist, "test": a.tolist(), "forecast": f.tolist(),
                      "errors": {} if errs.empty else {k: v for k, v in errs.iloc[0].to_dict().items() if k != "series_name"}, "note": hist_note,
                      "names_count": len(names)})

    def run_series_names(self, run_id: str, limit: int = 5000) -> list[str]:
        return self._run_frames(run_id)["fc"]["series_name"].astype(str).tolist()[:limit]

    def run_file(self, run_id: str, key: str) -> tuple[Path, str]:
        d = self._run_dir(run_id)
        m = self._manifest(d)
        outs = dict(m.get("outputs") or {})
        outs["manifest"] = "manifest.json"
        if key not in outs:
            raise AppError("File not found.", 404)
        p = d / outs[key]
        if not p.is_file():
            raise AppError("File not found.", 404)
        return p, f"{run_id}__{p.name}"

    # ============================================================================================ yükleme
    def upload(self, filename: str, tmp_path: Path, fill_gaps: bool = False) -> dict:
        ext = Path(filename).suffix.lower()
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        stem = _safe_name(filename)
        target = self.uploads_dir / f"{stem}.tsf"
        k = 2
        while target.exists():
            target = self.uploads_dir / f"{stem}-{k}.tsf"
            k += 1
        if ext == ".tsf":
            try:
                h = read_tsf_header(tmp_path)
                ds = read_tsf(tmp_path, max_series=5)
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "errors": [f"Could not read the TSF file: {exc}"], "warnings": []}
            if not h.frequency:
                return {"ok": False, "errors": ["The header has no @frequency; Forecast Studio requires frequency information."], "warnings": []}
            os.replace(tmp_path, target)
            return {"ok": True, "id": f"{UPLOAD_PREFIX}{target.stem}", "report": "", "errors": [], "warnings": ds.consistency_warnings(),
                    "info": {"frequency": h.frequency, "horizon": h.horizon}}
        if ext in (".xlsx", ".xlsm"):
            from ..excel import excel_to_tsf
            try:
                ds, rep = excel_to_tsf(tmp_path, target, fill_gaps=fill_gaps, lang="en")
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "errors": [f"Could not convert the Excel file: {type(exc).__name__}: {exc}"], "warnings": []}
            if ds is None or rep.errors:
                return {"ok": False, "errors": rep.errors, "warnings": rep.warnings}
            return {"ok": True, "id": f"{UPLOAD_PREFIX}{target.stem}", "report": rep.text(), "errors": [], "warnings": rep.warnings,
                    "info": clean(rep.info)}
        return {"ok": False, "errors": ["Only .tsf and .xlsx files can be uploaded."], "warnings": []}

    @staticmethod
    def template_xlsx() -> bytes:
        from ..excel import build_template
        out = build_template(None)
        return out if isinstance(out, (bytes, bytearray)) else Path(out).read_bytes()
