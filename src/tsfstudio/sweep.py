"""Batch sweep for unattended (HPC) runs: every chosen dataset x every chosen method, one after another.

    python -m tsfstudio sweep --datasets all --methods study --out results/sweep1 --threads 8

One task = one (dataset, method) pair = one `runner.run`. Datasets are worked through strictly one after another:
no task of the next dataset starts before every task of the current one has ended. `--jobs J` runs J tasks of the
current dataset at the same time, each with
`--threads T` threads inside the engine (J x T = the cores of the job). Each task runs in its own Python process, so a task that
crashes, runs out of memory or exceeds `--task-timeout` is recorded and the sweep goes on with the next one. A sweep that
is started again with the same `--out` skips the tasks that already finished, so a job that hit the wall-time limit of the
scheduler is continued by submitting it again.

Layout of the output folder:

    <out>/sweep_runs.jsonl                       one line per invocation: arguments, host, scheduler job, versions
    <out>/tasks/<dataset>/<method>/status.json   state, timings, memory, summary of the measures
    <out>/tasks/<dataset>/<method>/forecasts.csv, test_actuals.csv, errors_per_series.csv, summary.csv,
                                   fallbacks.csv, manifest.json, log.txt
    <out>/report/results.csv                     one row per task: state, measures (mean, median), times, memory
    <out>/report/summary_long.csv                one row per task and measure (mean, median, valid / excluded series)
    <out>/report/errors_per_series.csv.gz        one row per task and series: all measures
    <out>/report/mean_<measure>.csv, engine_sec.csv   dataset x method tables
    <out>/report/report.xlsx                     the tables above without the per-series one

Times: `engine_sec` is the time inside the engine (fitting and forecasting all series; for the C++ engine one library
call), `cpu_sec` the user + system CPU time of the run summed over threads and child processes, `load_sec` the time to
read the data file, `wall_sec` the whole task process. `peak_rss_mb` is the peak resident memory of the task process.
The thread count is fixed by `--threads` (default: the cores the scheduler granted) and recorded with every task.

`--probe N` runs the same plan on N randomly chosen series per dataset into <out>/probe and writes report/estimate.csv,
the projected time of the full sweep; use it to choose the wall-time limit before submitting the real job.
"""
from __future__ import annotations

import gzip
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import __version__
from . import catalog as C
from .estimate import format_duration
from .metrics import ALL_METRICS, DEFAULT_METRICS
from .seasonality import seasonality_for
from .tsf.reader import TSFDataset, read_tsf, read_tsf_header

BUNDLES = ("main", "extra", "m5")
BENCHMARKS = ("naive", "snaive", "drift", "naive2", "ses", "theta", "comb_shd", "historic_average", "window_average",
              "seasonal_window_average", "seasonal_es", "holt_ets", "zero_model")
PRESETS = ("study", "benchmarks", "local", "intermittent", "global", "all")
BAD_STATES = ("failed", "timeout", "crashed")
STATUS = "status.json"


class SweepError(RuntimeError):
    """A problem with the request (unknown dataset or method, missing file) that the user can fix."""


@dataclass
class Target:
    id: str
    path: Path
    entry: C.DatasetEntry | None           # None: a .tsf file that is not in the catalogue


# ------------------------------------------------------------------------------------------------
# Plan
# ------------------------------------------------------------------------------------------------
def select_datasets(spec: Sequence[str], exclude: Sequence[str] = (), data_dir=None) -> tuple[list[Target], list[str]]:
    """Datasets named by `spec` (catalogue ids, 'all', a bundle name, or paths to .tsf files), smallest file first.
    Returns the targets and notes about catalogue datasets that 'all' / a bundle left out because the file is missing."""
    cat = C.load_catalog()
    chosen: dict[str, Target] = {}
    notes: list[str] = []
    for item in spec:
        if item.lower().endswith(".tsf"):
            p = Path(item)
            if not p.is_file():
                raise SweepError(f"Data file not found: {p}")
            chosen[p.stem] = Target(p.stem, p, None)
            continue
        group = item == "all" or item in BUNDLES
        if group:
            entries = [e for e in cat.entries if item == "all" or e.bundle == item]
        else:
            try:
                entries = [cat.get(item)]
            except KeyError:
                raise SweepError(f"Unknown dataset '{item}'. Valid: all, {', '.join(BUNDLES)}, a path to a .tsf file, or one "
                                 f"of: {', '.join(cat.ids())}") from None
        for e in entries:
            state = C.availability(e, data_dir)
            if state == "ok":
                chosen[e.id] = Target(e.id, C._dir(data_dir) / e.file, e)
            elif group:
                notes.append(f"{e.id}: left out, file {e.file} is {state.replace('_', ' ')} in {C._dir(data_dir)}")
            else:
                raise SweepError(f"The data file of '{e.id}' ({e.file}) is {state.replace('_', ' ')} in {C._dir(data_dir)}; "
                                 "run `python -m tsfstudio data-check`.")
    for x in exclude:
        chosen.pop(x, None)
    return sorted(chosen.values(), key=lambda t: (t.path.stat().st_size, t.id)), notes


def methods_for(target: Target, spec: Sequence[str], exclude: Sequence[str], offered: dict[str, tuple[str, ...]],
                lag: int | None = None) -> tuple[list[tuple[str, str]], list[str]]:
    """(task, method) pairs for one dataset, in the order of `spec` (presets and method names), and notes about
    methods that were left out. `offered` is the engine's `methods()`."""
    from .engines.py_sf import INTERMITTENT

    header = read_tsf_header(target.path)
    local, glob = tuple(offered.get("local", ())), tuple(offered.get("global", ()))
    names: list[str] = []
    notes: list[str] = []
    for item in spec:
        if item == "study":
            if target.entry is not None:
                want = target.entry.recommended_local_methods(header) + (target.entry.global_methods or C.GLOBAL_METHODS)
            else:
                freq = header.frequency
                long_season = freq is not None and (len(seasonality_for(freq)) > 1 or freq in ("weekly", "daily"))
                want = ("ses", "ets", "theta", "tbats", "dhr_arima" if long_season else "arima") + C.GLOBAL_METHODS
            missing = [m for m in want if m not in local + glob]
            if missing:
                notes.append(f"not offered by the engine: {', '.join(missing)}")
            names += [m for m in want if m in local + glob]
        elif item == "benchmarks":
            names += [m for m in BENCHMARKS if m in local]
        elif item == "local":
            names += [m for m in local if m not in INTERMITTENT]
        elif item == "intermittent":
            names += [m for m in local if m in INTERMITTENT]
        elif item == "global":
            names += list(glob)
        elif item == "all":
            names += list(local) + list(glob)
        elif item in local + glob:
            names.append(item)
        else:
            raise SweepError(f"The engine does not offer the method '{item}'. Presets: {', '.join(PRESETS)}; "
                             f"methods: {', '.join(local + glob)}")
    eff_lag = lag if lag is not None else (target.entry.lag if target.entry else None)
    out: list[tuple[str, str]] = []
    for m in dict.fromkeys(names):
        if m in exclude:
            continue
        if m in glob and m not in local and eff_lag is None:
            notes.append(f"{m}: left out, a global model needs --lag for a file that is not in the catalogue")
            continue
        out.append(("global" if m in glob and m not in local else "local", m))
    return out, notes


def build_plan(datasets: Sequence[str], methods: Sequence[str], *, engine: str = "cpp", exclude_datasets: Sequence[str] = (),
               exclude_methods: Sequence[str] = (), data_dir=None, lag: int | None = None) -> tuple[list[dict], list[str]]:
    """The ordered task list (dataset by dataset) and the notes to show the user."""
    from .engines.registry import get_engine

    eng = get_engine(engine)
    ok, msg = eng.available()
    if not ok:
        raise SweepError(f"Engine '{engine}' is not available: {msg}")
    offered = eng.methods()
    targets, notes = select_datasets(datasets, exclude_datasets, data_dir)
    if not targets:
        raise SweepError("No dataset to run: none of the requested data files is present. " + " | ".join(notes[:5]))
    plan: list[dict] = []
    for t in targets:
        pairs, m_notes = methods_for(t, methods, exclude_methods, offered, lag)
        notes += [f"{t.id}: {n}" for n in m_notes]
        for task, method in pairs:
            plan.append({"dataset": t.id, "path": str(t.path), "catalog": t.entry is not None, "task": task, "method": method})
    return plan, list(dict.fromkeys(notes))


# ------------------------------------------------------------------------------------------------
# One task
# ------------------------------------------------------------------------------------------------
def task_dir(out: Path, dataset: str, method: str) -> Path:
    return Path(out) / "tasks" / dataset / method


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_status(d: Path) -> dict | None:
    try:
        return json.loads((d / STATUS).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _cpu_seconds() -> float | None:
    """User + system CPU time of this process (all threads) and of its finished child processes."""
    try:
        import resource
    except ImportError:                                     # Windows
        return None
    a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def _peak_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / (1024 * 1024) if sys.platform == "darwin" else kb / 1024      # macOS reports bytes, Linux kilobytes


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sample(ds: TSFDataset, n: int, seed: int) -> TSFDataset:
    """`n` series chosen at random without replacement (file order kept)."""
    if n >= ds.n_series:
        return ds
    idx = np.sort(np.random.default_rng(seed).choice(ds.n_series, size=n, replace=False))
    return TSFDataset(ds.header, ds.attrs.iloc[idx].reset_index(drop=True), [ds.values[i] for i in idx], ds.source)


def run_task(spec: dict) -> dict:
    """Runs one task in this process and writes its folder. `spec`: the plan row plus `opts` (see `sweep`) and `dir`.
    Always returns the status (state 'done' or 'failed'); never raises for a failing model."""
    from .runner import RunConfig, RunError, run

    opts = spec["opts"]
    d = Path(spec["dir"])
    d.mkdir(parents=True, exist_ok=True)
    for old in d.iterdir():                                 # a new attempt starts from an empty folder (the log stays)
        if old.name != "log.txt":
            shutil.rmtree(old) if old.is_dir() else old.unlink()
    threads = int(opts["threads"])
    os.environ["TSFSTUDIO_CPP_THREADS"] = str(threads)
    status = {"schema": 1, "dataset": spec["dataset"], "task": spec["task"], "method": spec["method"], "engine": opts["engine"],
              "state": "running", "started_utc": _now(), "host": socket.gethostname(), "pid": os.getpid(), "threads": threads,
              "sample": opts.get("probe")}
    _write_json(d / STATUS, status)
    t_start = time.time()
    try:
        entry = C.load_catalog().get(spec["dataset"]) if spec.get("catalog") else None
        ds = read_tsf(spec["path"], max_series=opts.get("max_series"))
        n_total = ds.n_series
        if opts.get("probe"):
            ds = _sample(ds, int(opts["probe"]), int(opts.get("seed", 1)))
        status.update(load_sec=time.time() - t_start, n_series_total=n_total)
        cfg = RunConfig(engine=opts["engine"], task=spec["task"], method=spec["method"], lag=opts.get("lag"),
                        horizon=opts.get("horizon"), missing_policy=opts.get("missing_policy", "error"),
                        seed=int(opts.get("seed", 1)), workers=threads, metrics=tuple(opts.get("metrics") or DEFAULT_METRICS),
                        output_dir=d / "_run", force=bool(opts.get("force")), write_xlsx=False,
                        origins=int(opts.get("origins", 1)))
        cpu0, t0 = _cpu_seconds(), time.time()
        res = run(ds, cfg, entry=entry, dataset_id=spec["dataset"])
        run_sec = time.time() - t0
        cpu1 = _cpu_seconds()
        for f in res.run_dir.iterdir():                     # stable paths: tasks/<dataset>/<method>/forecasts.csv ...
            shutil.move(str(f), str(d / f.name))
        shutil.rmtree(d / "_run", ignore_errors=True)
        man = res.manifest
        n = len(res.names)
        eng_sec = float(man["engine"]["elapsed_sec"])
        status.update(
            state="done", n_series=n, n_excluded=man["split"]["n_excluded"], horizon=man["split"]["horizon"],
            frequency=man["dataset"]["frequency"], lag=man["config"]["lag"], seasonality=man["config"]["effective_seasonality"],
            integer_conversion=man["config"]["effective_integer_conversion"], origins=man["split"]["origins"],
            fallbacks=man["fallbacks"]["count"], engine_sec=eng_sec, engine_sec_per_series=eng_sec / n if n else None,
            run_sec=run_sec, cpu_sec=None if cpu0 is None else cpu1 - cpu0, dataset_sha256=man["dataset"]["sha256"],
            engine_versions=man["engine"]["versions"], warnings=man["warnings"],
            summary={r["Metric"]: {"mean": r["Mean"], "median": r["Median"], "n_valid": int(r["N_valid"]),
                                   "n_excluded": int(r["N_excluded"])} for r in res.summary.to_dict("records")})
        if "wrmsse" in man:
            status["wrmsse"] = man["wrmsse"]["value"]
    except (RunError, SweepError, FileNotFoundError, MemoryError) as exc:
        status.update(state="failed", message=f"{type(exc).__name__}: {exc}"[:2000])
    except Exception as exc:  # noqa: BLE001 - one failing model must not stop the sweep
        status.update(state="failed", message=f"{type(exc).__name__}: {exc}"[:2000])
        traceback.print_exc()
    status.update(finished_utc=_now(), total_sec=time.time() - t_start, peak_rss_mb=_peak_rss_mb())
    _write_json(d / STATUS, status)
    return status


# ------------------------------------------------------------------------------------------------
# The sweep
# ------------------------------------------------------------------------------------------------
class _Stop(Exception):
    """SIGTERM / SIGINT: the scheduler is ending the job."""


def default_threads() -> int:
    """Cores granted by the scheduler (SLURM, PBS, SGE slots), else the cores this process may use."""
    for var in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        v = os.environ.get(var, "")
        if v.isdigit() and int(v) > 0:
            return int(v)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _cpu_model() -> str | None:
    try:
        for ln in Path("/proc/cpuinfo").read_text().splitlines():
            if ln.startswith("model name"):
                return ln.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


class _Child:
    """One task running in its own Python process."""

    def __init__(self, spec: dict, index: int):
        self.spec, self.index, self.dir, self.t0 = spec, index, Path(spec["dir"]), time.time()
        self.dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        src = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        self.log = open(self.dir / "log.txt", "ab")
        self.log.write(f"\n=== {_now()} {spec['dataset']} / {spec['method']} ===\n".encode())
        self.log.flush()
        self.proc = subprocess.Popen([sys.executable, "-m", "tsfstudio", "sweep-task", json.dumps(spec)], stdout=self.log,
                                     stderr=subprocess.STDOUT, env=env, start_new_session=(os.name != "nt"))

    def kill(self) -> None:
        try:
            if os.name != "nt":
                os.killpg(self.proc.pid, signal.SIGKILL)    # the task and what it started (the CatBoost program)
            else:
                self.proc.kill()
        except OSError:
            pass
        self.proc.wait()
        self.log.close()

    def poll(self, timeout: float | None) -> tuple[str, str] | None:
        """None while running; else (state override or '', message)."""
        rc = self.proc.poll()
        if rc is None:
            if timeout is not None and time.time() - self.t0 > timeout:
                self.kill()
                return "timeout", f"stopped after the task time limit of {format_duration(timeout)}"
            return None
        self.log.close()
        st = read_status(self.dir)
        if st is not None and st.get("state") in ("done", "failed"):
            return "", ""
        if rc == -signal.SIGTERM:                           # the scheduler ends the job: run it again next time
            return "interrupted", "the task process was terminated (scheduler time limit or cancel)"
        why = f"signal {-rc}" if rc < 0 else f"exit code {rc}"
        hint = " (killed: usually the memory limit of the job)" if rc == -9 else ""
        return "crashed", f"the task process ended with {why}{hint}; see log.txt"


def sweep(datasets: Sequence[str], methods: Sequence[str], out: str | Path, *, engine: str = "cpp",
          exclude_datasets: Sequence[str] = (), exclude_methods: Sequence[str] = (), data_dir=None,
          threads: int | None = None, task_timeout: float | None = None, missing_policy: str = "error",
          metrics: Sequence[str] = DEFAULT_METRICS, max_series: int | None = None, probe: int | None = None,
          seed: int = 1, lag: int | None = None, horizon: int | None = None, origins: int = 1, force: bool = False,
          retry_failed: bool = False, rerun: bool = False, in_process: bool = False, array_index: int | None = None,
          jobs: int = 1,
          report: bool = True, dry_run: bool = False, echo=print) -> int:
    """Runs the plan; returns 0 when every task of this invocation is done, 1 when some failed, 143 when stopped.
    `jobs` tasks run at the same time, each with `threads` threads (default: the granted cores divided by `jobs`)."""
    out = Path(out) / "probe" if probe else Path(out)
    jobs = 1 if in_process else max(1, int(jobs or 1))
    cores = default_threads()
    threads = int(threads or max(1, cores // jobs))
    plan, notes = build_plan(datasets, methods, engine=engine, exclude_datasets=exclude_datasets,
                             exclude_methods=exclude_methods, data_dir=data_dir, lag=lag)
    ds_order = list(dict.fromkeys(p["dataset"] for p in plan))
    if array_index is not None:
        if not 0 <= array_index < len(ds_order):
            echo(f"Array index {array_index} is outside the {len(ds_order)} datasets of this plan (valid: 0..{len(ds_order) - 1}); nothing to do.")
            return 0
        plan = [p for p in plan if p["dataset"] == ds_order[array_index]]
        report = False
    for n in notes:
        echo(f"NOTE: {n}")
    if jobs * threads > cores:
        echo(f"NOTE: {jobs} tasks x {threads} threads = {jobs * threads} exceeds the {cores} cores available; "
             "tasks will share cores and the recorded times will be longer than on free cores.")
    echo(f"Plan: {len(plan)} tasks on {len(dict.fromkeys(p['dataset'] for p in plan))} datasets | engine {engine} | "
         f"{jobs} task(s) at a time x {threads} thread(s) | output {out}" + (f" | probe on {probe} series per dataset" if probe else ""))
    if dry_run:
        for i, ds in enumerate(ds_order):
            ms = [p["method"] for p in plan if p["dataset"] == ds]
            if ms:
                echo(f"  [{i}] {ds}: {', '.join(ms)}")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    opts = {"engine": engine, "threads": threads, "missing_policy": missing_policy, "metrics": list(metrics),
            "max_series": max_series, "probe": probe, "seed": seed, "lag": lag, "horizon": horizon, "origins": origins,
            "force": force}
    with open(out / "sweep_runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "started_utc": _now(), "argv": sys.argv, "datasets": list(datasets), "methods": list(methods),
            "exclude_datasets": list(exclude_datasets), "exclude_methods": list(exclude_methods), "options": opts,
            "task_timeout_sec": task_timeout, "array_index": array_index, "jobs": jobs, "n_tasks": len(plan), "host": socket.gethostname(),
            "cpu": _cpu_model(), "cpu_count": os.cpu_count(), "platform": platform.platform(),
            "scheduler": {k: v for k, v in os.environ.items() if k in (
                "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_PARTITION", "SLURM_CPUS_PER_TASK",
                "SLURM_MEM_PER_NODE", "SLURM_JOB_NODELIST", "PBS_JOBID", "JOB_ID", "SGE_TASK_ID", "NSLOTS", "PE", "QUEUE")},
            "software": {"tsfstudio": __version__, "python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "git_commit": _git_commit()}}, default=str) + "\n")

    def on_signal(signum, _frame):
        raise _Stop(signum)

    # SGE with `-notify`: SIGUSR2 announces the kill (qdel, time limit), SIGUSR1 a suspension (whose default would end us).
    try:
        old = {s: signal.signal(s, on_signal) for s in (signal.SIGTERM, signal.SIGINT)}
        if hasattr(signal, "SIGUSR2"):
            old[signal.SIGUSR2] = signal.signal(signal.SIGUSR2, on_signal)
            old[signal.SIGUSR1] = signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    except ValueError:                                      # not the main thread: no signal handling
        old = {}
    counts = {"done": 0, "skipped": 0, "failed": 0}
    rc = 0
    t_sweep = time.time()
    running: list[_Child] = []

    def head(i: int, p: dict) -> str:
        return f"[{i}/{len(plan)}] {p['dataset']} / {p['method']}"

    def finish(i: int, p: dict, d: Path, state: str, msg: str, t0: float) -> None:
        nonlocal rc
        st = read_status(d) or dict(p, engine=engine, threads=threads, sample=probe)
        if state:                                           # timeout / crash: the child could not write its own end
            st.update(state=state, message=msg, finished_utc=_now())
        st.update(wall_sec=time.time() - t0, isolated=not in_process, concurrent_tasks=jobs)
        _write_json(d / STATUS, st)
        if st["state"] == "done":
            counts["done"] += 1
            s = st.get("summary", {})
            key = next((k for k in ("MASE", "SMAPE") if k in s), next(iter(s), None))
            err = f" | mean {key} {s[key]['mean']:.4f}" if key and s[key]["mean"] is not None else ""
            echo(f"{head(i, p)}: done | {st['n_series']} series | engine {st['engine_sec']:.2f} s{err}"
                 + (f" | {st['fallbacks']} fallbacks" if st.get("fallbacks") else ""))
        else:
            counts["failed"] += 1
            rc = 1
            echo(f"{head(i, p)}: {st['state'].upper()} | {str(st.get('message', ''))[:300]}")

    def reap(block: bool) -> None:
        """Collects finished tasks; with `block`, waits until at least one slot is free."""
        while running:
            for c in list(running):
                res = c.poll(task_timeout)
                if res is not None:
                    running.remove(c)
                    finish(c.index, c.spec, c.dir, res[0], res[1], c.t0)
            if not block or len(running) < jobs:
                return
            time.sleep(0.05)

    current: tuple[dict, Path] | None = None                # the task of an in-process run
    try:
        for i, p in enumerate(plan, 1):
            d = task_dir(out, p["dataset"], p["method"])
            st = read_status(d)
            if st is not None and not rerun and (st.get("state") == "done" or (st.get("state") in BAD_STATES and not retry_failed)):
                counts["skipped"] += 1
                echo(f"{head(i, p)}: already {st['state']}, skipped")
                continue
            spec = dict(p, opts=opts, dir=str(d))
            while running and running[0].spec["dataset"] != p["dataset"]:
                reap(block=False)                           # a dataset is finished before the next one starts
                time.sleep(0.05)
            reap(block=len(running) >= jobs)
            echo(f"{head(i, p)}: started {datetime.now().strftime('%H:%M:%S')}")
            if in_process:
                current, t0 = (p, d), time.time()
                run_task(spec)
                current = None
                finish(i, p, d, "", "", t0)
            else:
                running.append(_Child(spec, i))
        while running:
            reap(block=False)
            time.sleep(0.05)
    except _Stop:
        stopped = [(c.spec, c.dir) for c in running] + ([current] if current else [])
        for c in running:
            c.kill()
        for p, d in stopped:
            st = read_status(d) or dict(p, engine=engine, threads=threads)
            if st.get("state") not in ("done", "failed"):
                st.update(state="interrupted", finished_utc=_now(),
                          message="the sweep was stopped (scheduler time limit or cancel) during this task")
                _write_json(d / STATUS, st)
        echo("Stopped by a signal; the finished tasks are kept. Start the same command again to continue.")
        rc = 143
    finally:
        for s, h in old.items():
            signal.signal(s, h)
    echo(f"Sweep: {counts['done']} done, {counts['failed']} not done, {counts['skipped']} skipped (finished earlier) "
         f"in {format_duration(time.time() - t_sweep)}")
    if report:
        rep = build_report(out)
        echo(f"Report -> {rep}")
        if probe:
            _echo_estimate(rep, echo)
    return rc


# ------------------------------------------------------------------------------------------------
# Report
# ------------------------------------------------------------------------------------------------
def collect(out: Path) -> list[tuple[Path, dict]]:
    rows = []
    for f in sorted((Path(out) / "tasks").glob(f"*/*/{STATUS}")):
        st = read_status(f.parent)
        if st is not None:
            rows.append((f.parent, st))
    return rows


def build_report(out: str | Path) -> Path:
    """Writes <out>/report from the task folders found under <out>/tasks (finished or not). Returns the report folder."""
    out = Path(out)
    rep = out / "report"
    rep.mkdir(parents=True, exist_ok=True)
    tasks = collect(out)
    results, long_rows = [], []
    for d, st in tasks:
        state = st.get("state")
        if state == "running":
            state = "interrupted"                          # nobody is running it once a report is written after the sweep
        row = {"dataset": st.get("dataset"), "task": st.get("task"), "method": st.get("method"), "engine": st.get("engine"),
               "state": state, "n_series": st.get("n_series"), "n_series_total": st.get("n_series_total"),
               "n_excluded": st.get("n_excluded"), "frequency": st.get("frequency"), "horizon": st.get("horizon"),
               "lag": st.get("lag"), "threads": st.get("threads"), "concurrent_tasks": st.get("concurrent_tasks"),
               "fallbacks": st.get("fallbacks")}
        for m, v in (st.get("summary") or {}).items():
            row[f"mean_{m}"], row[f"median_{m}"] = v["mean"], v["median"]
            long_rows.append({"dataset": row["dataset"], "task": row["task"], "method": row["method"], "measure": m, **v})
        if "wrmsse" in st:
            row["WRMSSE"] = st["wrmsse"]
        for k in ("engine_sec", "engine_sec_per_series", "cpu_sec", "run_sec", "load_sec", "wall_sec", "peak_rss_mb"):
            row[k] = st.get(k)
        sample, total = st.get("sample"), st.get("n_series_total")
        if sample and st.get("engine_sec") is not None and st.get("n_series"):
            # local models: time is additive over series; global models are fitted once and do not scale this way
            row["est_full_engine_sec"] = st["engine_sec"] * total / st["n_series"] if st.get("task") == "local" else None
        row.update(started_utc=st.get("started_utc"), host=st.get("host"), message=st.get("message"),
                   warnings=" | ".join(st.get("warnings") or []) or None,
                   task_dir=str(d.relative_to(out)))
        results.append(row)
    res = pd.DataFrame(results)
    _to_csv(res, rep / "results.csv")
    long = pd.DataFrame(long_rows, columns=["dataset", "task", "method", "measure", "mean", "median", "n_valid", "n_excluded"])
    _to_csv(long, rep / "summary_long.csv")

    pivots: dict[str, pd.DataFrame] = {}
    if len(res):
        done = res[res["state"] == "done"]
        ds_order, m_order = list(dict.fromkeys(res["dataset"])), list(dict.fromkeys(res["method"]))
        cols = [c for c in res.columns if c.startswith("mean_")] + ["engine_sec", "cpu_sec", "peak_rss_mb"]
        for c in cols:
            if c in done and done[c].notna().any():
                pv = done.pivot_table(index="dataset", columns="method", values=c, aggfunc="first")
                pivots[c] = pv.reindex(index=[x for x in ds_order if x in pv.index], columns=[x for x in m_order if x in pv.columns])
                _to_csv(pivots[c].reset_index(), rep / f"{c}.csv")
        states = res.pivot_table(index="dataset", columns="method", values="state", aggfunc="first")
        pivots["state"] = states.reindex(index=ds_order, columns=m_order)
        _to_csv(pivots["state"].reset_index(), rep / "state.csv")

    # per series: streamed task by task, so that large sweeps do not have to fit in memory
    files = [(d / "errors_per_series.csv", st) for d, st in tasks if st.get("state") == "done"]
    files = [(f, st) for f, st in files if f.is_file()]
    columns = ["dataset", "task", "method", "series_name"]
    for f, _ in files:                                     # tasks run with different measure sets still line up
        columns += [c for c in pd.read_csv(f, nrows=0).columns[1:] if c not in columns]
    tmp = rep / f"errors_per_series.csv.gz.tmp{os.getpid()}"
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as gz:
        gz.write(",".join(columns) + "\n")
        for f, st in files:
            e = pd.read_csv(f)
            e = e.rename(columns={e.columns[0]: "series_name"})
            for name in ("dataset", "task", "method"):
                e[name] = st[name]
            e.reindex(columns=columns).to_csv(gz, index=False, header=False)
    os.replace(tmp, rep / "errors_per_series.csv.gz")

    if len(res):
        if any(st.get("sample") for _, st in tasks):
            cols = ["dataset", "task", "method", "state", "n_series", "n_series_total", "threads", "engine_sec",
                    "est_full_engine_sec", "peak_rss_mb", "message"]
            _to_csv(res[[c for c in cols if c in res]], rep / "estimate.csv")
        try:
            tmpx = rep / f"report.tmp{os.getpid()}.xlsx"
            with pd.ExcelWriter(tmpx, engine="xlsxwriter") as xw:
                res.to_excel(xw, sheet_name="results", index=False)
                long.to_excel(xw, sheet_name="summary_long", index=False)
                for name, pv in pivots.items():
                    pv.to_excel(xw, sheet_name=name[:31])
            os.replace(tmpx, rep / "report.xlsx")
        except Exception as exc:  # noqa: BLE001 - the CSV files are the record; the workbook is a convenience
            (rep / "report_xlsx_error.txt").write_text(str(exc), encoding="utf-8")
    return rep


def _to_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _echo_estimate(rep: Path, echo) -> None:
    f = rep / "results.csv"
    if not f.is_file():
        return
    r = pd.read_csv(f)
    if "est_full_engine_sec" not in r or not len(r):
        return
    loc = r[(r["task"] == "local") & (r["state"] == "done")]
    total = float(loc["est_full_engine_sec"].sum())
    echo(f"Projected engine time of the full sweep, local models ({len(loc)} tasks, {int(r['threads'].iloc[0])} thread(s)): "
         f"{format_duration(total)}. A projection from a random sample; expect a factor of 2 either way.")
    top = loc.sort_values("est_full_engine_sec", ascending=False).head(10)
    for _, x in top.iterrows():
        echo(f"  {x['dataset']} / {x['method']}: {format_duration(float(x['est_full_engine_sec']))}")
    glob = r[(r["task"] == "global") & (r["state"] == "done")]
    if len(glob):
        echo(f"Global models ({len(glob)} tasks) are fitted once on all series; their time on the sample "
             f"({format_duration(float(glob['engine_sec'].sum()))}) is not projected.")
    bad = r[r["state"] != "done"]
    for _, x in bad.iterrows():
        echo(f"  NOT PROJECTED {x['dataset']} / {x['method']}: {x['state']} | {str(x.get('message'))[:160]}")
