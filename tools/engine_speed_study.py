"""Engine run times for the software paper (manuscript/software_paper): the same method on the C++, Python and R
engines, and the C++ engine on 1 to N threads.

    python tools/engine_speed_study.py --out results/speed --threads 1,2,4,8,16,32
    python tools/engine_speed_study.py --out results/speed --dry-run        # print the plan, run nothing
    python tools/engine_speed_study.py --out results/speed --report-only    # rebuild the tables from runs.csv

On a cluster use the launchers (scripts/hpc/speed.sge, speed.sbatch, run_speed.sh); see docs/HPC_SPEED.md.

What is measured. Every cell (dataset, method, engine, parallel level) is an ordinary run started through the command
line (`python -m tsfstudio run`), so it is what a user gets. The time reported is `timing.engine_wall_sec` of the run
manifest: the engine call that fits and forecasts all series, without reading the data, computing the measures or
writing files. For R it includes starting Rscript. "Parallel level" is the number of threads for C++ and the number of
worker processes for Python and R. The C++ engine is run at every level of --threads, Python and R at 1 and at the
largest level.

Order and limits. For one (dataset, method, engine) the levels run from the largest to 1. When a level took s seconds
on p units, the next level q < p is projected at s * p / q; if that exceeds --timeout-min it is recorded as `skipped`
and not run. A cell that fails or times out ends the lower levels of that engine too.

Continuing. Every repetition is appended to <out>/runs.csv as soon as it ends; starting the same command again skips
what is recorded there.

Forecasts and errors. Every cell keeps its run folder in <out>/runs/<dataset>/<method>/<engine>_x<level>/: forecasts.csv
(one row per series), test_actuals.csv, errors_per_series.csv, summary.csv, fallbacks.csv and manifest.json
(--discard-runs keeps the times only).

Outputs in <out>: runs.csv (one row per repetition), summary.csv (median per cell, speed-up), speed_table.md (the
table for the paper), node.json (host, CPU, cores, software versions).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# dataset -> methods. Local methods that exist on all three engines.
PLAN = {
    "nn5_weekly": ["ses", "theta", "ets", "arima", "tbats"],       # 111 series
    "nn5_daily": ["ses", "theta", "ets", "arima", "tbats"],        # 111 series, 735 observations each
    "m3_monthly": ["ses", "theta", "ets", "arima", "tbats"],       # 1,428 series
    "m4_hourly": ["ses", "theta", "ets", "arima", "tbats"],        # 414 series, two seasonal periods
    "m4_quarterly": ["ses", "theta", "ets", "arima"],              # 24,000 series
    "m4_monthly": ["ses", "theta", "ets", "arima"],                # 48,000 series
}
# Python and R are timed on these; the two large M4 sets are C++ only unless --ref-datasets says otherwise
# (the reference engines would need hours per method there).
REF_DATASETS = ["nn5_weekly", "nn5_daily", "m3_monthly", "m4_hourly"]
ENGINES = ["cpp", "python", "r"]
FIELDS = ["dataset", "method", "engine", "parallel", "rep", "status", "n_series", "horizon", "engine_sec", "wall_sec",
          "fallbacks", "mase_mean", "load1_start", "message"]


def sh(cmd: list[str], env: dict | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)


def base_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("TSFSTUDIO_DATA_DIR", str(ROOT / "data" / "tsf"))
    # one core per process unless the engine itself is told otherwise
    env.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    return env


def engines_available() -> dict[str, str]:
    """engine -> its version line, for the engines `tsfstudio engines` reports as AVAILABLE."""
    p = sh([sys.executable, "-m", "tsfstudio", "engines"], env=base_env())
    out = {}
    for line in p.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) >= 2 and parts[1] == "AVAILABLE":
            out[parts[0]] = parts[2].strip() if len(parts) > 2 else ""
    return out


def engines_of(method: str) -> list[str]:
    sys.path.insert(0, str(ROOT / "src"))
    from tsfstudio.app import models_info as mi
    return list(mi.MODELS[method]["engines"])


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _text(cmd: list[str]) -> str:
    """Output of a system command, or "" when it is not there (qhost outside SGE, lscpu on a minimal system)."""
    if not shutil.which(cmd[0]):
        return ""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return p.stdout if p.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def load1() -> str:
    try:
        return Path("/proc/loadavg").read_text().split()[0]
    except OSError:
        return ""


def hardware(lscpu: str) -> dict:
    """Sockets, physical cores and hardware threads from lscpu: 32 slots may be 16 cores with two threads each."""
    kv = {}
    for line in lscpu.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            kv[k.strip()] = v.strip()
    hw = {k: kv.get(k, "") for k in ("Model name", "CPU(s)", "Socket(s)", "Core(s) per socket", "Thread(s) per core",
                                      "NUMA node(s)", "CPU max MHz", "L3 cache")}
    try:
        hw["physical_cores"] = int(kv["Socket(s)"]) * int(kv["Core(s) per socket"])
    except (KeyError, ValueError):
        hw["physical_cores"] = ""
    try:
        hw["affinity_cpus"] = len(os.sched_getaffinity(0))        # what the scheduler lets this job use
    except (AttributeError, OSError):
        hw["affinity_cpus"] = ""
    try:
        mem = [ln for ln in Path("/proc/meminfo").read_text().splitlines() if ln.startswith("MemTotal")][0]
        hw["mem_total_gb"] = round(int(mem.split()[1]) / 1024 ** 2, 1)
    except (OSError, IndexError, ValueError):
        hw["mem_total_gb"] = ""
    return hw


def node_info(avail: dict[str, str], a, out: Path) -> dict:
    commit = ""
    for f in (ROOT / "VERSION",):
        if f.is_file():
            commit = f.read_text(encoding="utf-8").strip()
    lscpu = _text(["lscpu"])
    qhost = _text(["qhost"])
    host = platform.node()
    qhost_me = _text(["qhost", "-h", host.split(".")[0], "-q"]) if qhost else ""
    # the raw outputs go next to node.json, so the paper can quote them
    for name, txt in (("node_lscpu.txt", lscpu), ("node_qhost.txt", qhost + ("\n--- this host ---\n" + qhost_me if qhost_me else ""))):
        if txt:
            (out / name).write_text(txt, encoding="utf-8")
    return {"host": host, "platform": platform.platform(), "cpu": cpu_model(), "cpu_count": os.cpu_count(),
            "hardware": hardware(lscpu), "load1_start": load1(),
            "granted": os.environ.get("NSLOTS") or os.environ.get("SLURM_CPUS_PER_TASK") or "",
            "job": os.environ.get("JOB_ID") or os.environ.get("SLURM_JOB_ID") or "",
            "queue": os.environ.get("QUEUE") or "", "python": platform.python_version(),
            "engines": avail, "source": commit, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "threads": a.threads, "repeats": a.repeats, "ref_repeats": a.ref_repeats, "timeout_min": a.timeout_min}


def is_global(method: str) -> bool:
    sys.path.insert(0, str(ROOT / "src"))
    from tsfstudio.app import models_info as mi
    return mi.MODELS[method]["task"] == "global"


def one_run(dataset: str, method: str, engine: str, par: int, work: Path, timeout: float, keep: bool,
            max_series: int | None = None) -> dict:
    env = base_env()
    env.pop("TSFSTUDIO_CPP_THREADS", None)
    env["OMP_NUM_THREADS"] = str(par)
    if engine == "cpp":
        env["TSFSTUDIO_CPP_THREADS"] = str(par)        # exact: the library would otherwise use every core of the node
    cmd = [sys.executable, "-m", "tsfstudio", "run", "--dataset", dataset, "--engine", engine, "--method", method,
           "--workers", str(par), "--out", str(work), "--yes"]
    if is_global(method):
        cmd += ["--task", "global"]
    if max_series:
        cmd += ["--max-series", str(max_series)]
    work.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in work.iterdir()}
    t0 = time.perf_counter()
    try:
        p = sh(cmd, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "wall_sec": round(time.perf_counter() - t0, 1), "message": f"> {timeout / 60:.0f} min"}
    wall = round(time.perf_counter() - t0, 2)
    new = [d for d in work.iterdir() if d.name not in before and (d / "manifest.json").is_file()]
    if p.returncode != 0 or not new:
        last = [ln for ln in (p.stderr or p.stdout).strip().splitlines() if ln.strip()][-1:] or ["no output"]
        return {"status": "failed", "wall_sec": wall, "message": last[0][:300]}
    man = json.loads((new[0] / "manifest.json").read_text(encoding="utf-8"))
    mase = ""
    with open(new[0] / "summary.csv", newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            if rec.get("Metric") == "MASE":
                mase = rec.get("Mean", "")
    if keep:
        for x in new[0].glob("*.xlsx"):          # the CSV files hold the same numbers
            x.unlink()
    else:
        shutil.rmtree(new[0], ignore_errors=True)
    return {"status": "ok", "n_series": man["dataset"]["n_series_run"], "horizon": man["split"]["horizon"],
            "engine_sec": round(man["timing"]["engine_wall_sec"], 6), "wall_sec": wall,
            "fallbacks": man["fallbacks"]["count"], "mase_mean": mase, "message": ""}


def read_runs(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append(path: Path, row: dict) -> None:
    new = not path.is_file()
    fields = FIELDS
    if not new:                                          # keep the header of a file written by an earlier version
        with open(path, newline="", encoding="utf-8") as f:
            fields = next(csv.reader(f), FIELDS) or FIELDS
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def levels(engine: str, threads: list[int]) -> list[int]:
    lv = threads if engine == "cpp" else sorted({1, max(threads)})
    return sorted(set(lv), reverse=True)


def fmt(sec: float) -> str:
    return f"{sec:.3f}" if sec < 10 else f"{sec:.1f}" if sec < 1000 else f"{sec:.0f}"


def report(out: Path) -> None:
    rows = read_runs(out / "runs.csv")
    cells: dict[tuple, list[float]] = {}
    info: dict[tuple, dict] = {}
    for r in rows:
        key = (r["dataset"], r["method"], r["engine"], int(r["parallel"]))
        if r["status"] == "ok":
            cells.setdefault(key, []).append(float(r["engine_sec"]))
            info[key] = r
        else:
            info.setdefault(key, r)
    summ = []
    for key, r in info.items():
        ds, m, e, par = key
        t = cells.get(key)
        med = statistics.median(t) if t else None
        cpp1 = cells.get((ds, m, "cpp", 1))
        refs = [statistics.median(v) for (d2, m2, e2, _), v in cells.items() if d2 == ds and m2 == m and e2 != "cpp"]
        summ.append({"dataset": ds, "method": m, "engine": e, "parallel": par, "status": "ok" if t else r["status"],
                     "n_series": r.get("n_series", ""), "repetitions": len(t or []),
                     "engine_sec_median": "" if med is None else round(med, 6),
                     "engine_sec_min": "" if not t else round(min(t), 6), "engine_sec_max": "" if not t else round(max(t), 6),
                     "ms_per_series": "" if med is None or not r.get("n_series") else round(1000 * med / int(r["n_series"]), 5),
                     "speedup_vs_cpp_1_thread": "" if med is None or not cpp1 or e != "cpp" else round(statistics.median(cpp1) / med, 2),
                     "best_reference_sec_over_this": "" if med is None or not refs or e != "cpp" or med <= 0 else round(min(refs) / med, 1),
                     "mase_mean": r.get("mase_mean", ""), "fallbacks": r.get("fallbacks", ""), "message": r.get("message", "")})
    order = {d: i for i, d in enumerate(PLAN)}
    summ.sort(key=lambda s: (order.get(s["dataset"], 99), s["dataset"], s["method"], ENGINES.index(s["engine"]), s["parallel"]))
    if summ:
        with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summ[0]))
            w.writeheader()
            w.writerows(summ)
    # the table for the paper: one row per dataset and method, one column per engine and level
    cols = sorted({(ENGINES.index(s["engine"]), s["engine"], s["parallel"]) for s in summ if s["engine"] != "cpp"}) + \
        sorted({(ENGINES.index(s["engine"]), s["engine"], s["parallel"]) for s in summ if s["engine"] == "cpp"})
    names = {"cpp": "C++", "python": "Python", "r": "R"}
    head = ["Dataset", "Series", "Method"] + [f"{names[e]} ({p})" for _, e, p in cols]
    lines = ["Engine time in seconds (median of the repetitions); in brackets: worker processes (R, Python) or threads (C++).",
             "", "| " + " | ".join(head) + " |", "|:--|--:|:--|" + "--:|" * len(cols)]
    by = {(s["dataset"], s["method"], s["engine"], s["parallel"]): s for s in summ}
    for ds, m in sorted({(s["dataset"], s["method"]) for s in summ}, key=lambda k: (order.get(k[0], 99), k[0], k[1])):
        n = next((s["n_series"] for s in summ if s["dataset"] == ds and s["method"] == m and s["n_series"]), "")
        cellsx = []
        for _, e, p in cols:
            s = by.get((ds, m, e, p))
            cellsx.append("" if s is None else fmt(float(s["engine_sec_median"])) if s["status"] == "ok" else s["status"])
        lines.append(f"| {ds} | {n} | {m} | " + " | ".join(cellsx) + " |")
    node = out / "node.json"
    if node.is_file():
        nd = json.loads(node.read_text(encoding="utf-8"))
        lines += ["", f"Node: {nd.get('host')}, {nd.get('cpu')}, {nd.get('cpu_count')} logical cores"
                      f"{', ' + str(nd['granted']) + ' granted' if nd.get('granted') else ''}"
                      f"{', ' + str(nd['hardware'].get('physical_cores')) + ' physical cores (' + str(nd['hardware'].get('Socket(s)')) + ' sockets x ' + str(nd['hardware'].get('Core(s) per socket')) + ', ' + str(nd['hardware'].get('Thread(s) per core')) + ' threads per core)' if (nd.get('hardware') or {}).get('physical_cores') else ''}"
                      f", load average {nd.get('load1_start', '?')} at the start, {nd.get('load1_end', '?')} at the end. Engines: "
                      + "; ".join(f"{k}: {v}" for k, v in (nd.get("engines") or {}).items())]
    lines += ratios(out, summ, order, names)
    (out / "speed_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written: {out / 'summary.csv'}, {out / 'ratios.csv'}, {out / 'speed_table.md'}", flush=True)


def ratios(out: Path, summ: list[dict], order: dict, names: dict) -> list[str]:
    """Ratios between engines and thread counts, from the medians of summary.csv; written to ratios.csv (one row per
    dataset, method and ratio) and returned as two further tables for speed_table.md:
      A. ms per series, median over the datasets, per method and engine, on 1 unit and on the largest level
         (the layout of the paper's Table 5), with the median over datasets of the per-dataset ratio to C++;
      B. C++ speed-up against one thread and parallel efficiency (speed-up / threads), per dataset and method."""
    med = {(s["dataset"], s["method"], s["engine"], s["parallel"]): float(s["engine_sec_median"])
           for s in summ if s["status"] == "ok" and s["engine_sec_median"] != ""}
    nser = {(s["dataset"], s["method"]): int(s["n_series"]) for s in summ if s.get("n_series")}
    pairs = sorted({(d, m) for d, m, _, _ in med}, key=lambda k: (order.get(k[0], 99), k[0], k[1]))
    cpp_lv = sorted({p for _, _, e, p in med if e == "cpp"})
    top = max([p for *_, p in med] or [1])
    rows = []

    def add(ds, m, name, num, den):
        if num in med and den in med and med[den] > 0:
            rows.append({"dataset": ds, "method": m, "ratio": name, "numerator_sec": med[num],
                         "denominator_sec": med[den], "value": round(med[num] / med[den], 3)})

    for ds, m in pairs:
        for p in cpp_lv:
            if p > 1:
                add(ds, m, f"cpp_speedup_x{p}", (ds, m, "cpp", 1), (ds, m, "cpp", p))
        for e in ("python", "r"):
            add(ds, m, f"{e}_over_cpp_x1", (ds, m, e, 1), (ds, m, "cpp", 1))
            add(ds, m, f"{e}_over_cpp_x{top}", (ds, m, e, top), (ds, m, "cpp", top))
            add(ds, m, f"{e}_speedup_x{top}", (ds, m, e, 1), (ds, m, e, top))
        add(ds, m, f"r_over_python_x1", (ds, m, "r", 1), (ds, m, "python", 1))
    for r in rows:                                   # efficiency of a speed-up: speed-up / parallel units
        if "_speedup_x" in r["ratio"]:
            r["efficiency"] = round(r["value"] / int(r["ratio"].rsplit("x", 1)[1]), 3)
    if rows:
        with open(out / "ratios.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["dataset", "method", "ratio", "numerator_sec", "denominator_sec", "value",
                                              "efficiency"])
            w.writeheader()
            w.writerows(rows)
    by_ratio: dict[tuple, list[float]] = {}
    for r in rows:
        by_ratio.setdefault((r["method"], r["ratio"]), []).append(r["value"])
    methods = sorted({m for _, m in pairs})
    lines = []
    for lvl in sorted({1} | {p for _, _, e, p in med if e != "cpp"}):
        engines = [e for e in ENGINES if any(k[2] == e and k[3] == lvl for k in med)]
        if not engines:
            continue
        lines += ["", f"A. Time per series in ms on {lvl} {'thread / worker' if lvl == 1 else 'threads / workers'}: "
                      f"median over the datasets (number of datasets in brackets); last columns: median over the "
                      f"datasets of the per-dataset ratio to C++.", "",
                  "| Method | " + " | ".join(names[e] for e in engines) + "".join(
                      f" | {names[e]} / C++" for e in engines if e != "cpp") + " |",
                  "|:--|" + "--:|" * (len(engines) + len([e for e in engines if e != "cpp"]))]
        for m in methods:
            cells = []
            for e in engines:
                v = [1000 * med[(d, m, e, lvl)] / nser[(d, m)] for d, m2 in pairs if m2 == m and (d, m, e, lvl) in med]
                cells.append(f"{statistics.median(v):.3g} ({len(v)})" if v else "")
            for e in engines:
                if e != "cpp":
                    v = by_ratio.get((m, f"{e}_over_cpp_x{lvl}"))
                    cells.append(f"{statistics.median(v):.3g}" if v else "")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
    if any(p > 1 for p in cpp_lv):
        hi = [p for p in cpp_lv if p > 1]
        lines += ["", "B. C++ speed-up against one thread (parallel efficiency = speed-up / threads in brackets).", "",
                  "| Dataset | Method | " + " | ".join(f"x{p}" for p in hi) + " |", "|:--|:--|" + "--:|" * len(hi)]
        val = {(r["dataset"], r["method"], r["ratio"]): r for r in rows}
        for ds, m in pairs:
            c = [val.get((ds, m, f"cpp_speedup_x{p}")) for p in hi]
            lines.append(f"| {ds} | {m} | " + " | ".join(f"{r['value']:.2f} ({r['efficiency']:.2f})" if r else ""
                                                         for r in c) + " |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "results" / "speed"))
    ap.add_argument("--threads", default="1,2,4,8,16,32", help="parallel levels, comma separated (default 1,2,4,8,16,32)")
    ap.add_argument("--datasets", default="", help="catalogue ids, comma separated (default: the built-in plan)")
    ap.add_argument("--methods", default="", help="these methods for every dataset, comma separated (any method of the application)")
    ap.add_argument("--engines", default="cpp,python,r", help="engines to time; those not installed are left out")
    ap.add_argument("--ref-datasets", default=",".join(REF_DATASETS),
                    help="datasets on which Python and R are timed too, comma separated, or 'all' "
                         f"(default {','.join(REF_DATASETS)})")
    ap.add_argument("--repeats", type=int, default=1, help="repetitions per C++ cell (default 1; with more, the median is reported)")
    ap.add_argument("--ref-repeats", type=int, default=1, help="repetitions per Python / R cell (default 1)")
    ap.add_argument("--timeout-min", type=float, default=120, help="limit for one run, in minutes (default 120)")
    ap.add_argument("--discard-runs", action="store_true",
                    help="delete the run folders (forecasts, errors per series) and keep the times only")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.report_only:
        report(out)
        return 0
    threads = sorted({int(x) for x in a.threads.split(",") if x.strip()})
    if not threads or threads[0] < 1:
        sys.exit("--threads needs whole numbers >= 1")
    datasets = [d for d in a.datasets.split(",") if d] or list(PLAN)
    only = [m for m in a.methods.split(",") if m]
    avail = engines_available()
    engines = [e for e in ENGINES if e in a.engines.split(",") and e in avail]
    for e in a.engines.split(","):
        if e and e not in avail:
            print(f"engine '{e}' is not available on this machine and is left out", flush=True)
    if not engines:
        sys.exit("no engine available: `python -m tsfstudio engines` shows why")
    data_dir = Path(base_env()["TSFSTUDIO_DATA_DIR"])
    sys.path.insert(0, str(ROOT / "src"))
    from tsfstudio.catalog import availability, load_catalog
    cat = load_catalog()
    entries = {e.id: e for e in (cat.entries.values() if isinstance(cat.entries, dict) else cat.entries)}
    bad = [f"{d}: {entries[d].file} {availability(entries[d], data_dir)}" if d in entries else f"{d}: not in the catalogue"
           for d in datasets if d not in entries or availability(entries[d], data_dir) != "ok"]
    if bad and not a.dry_run:
        sys.exit(f"data files in {data_dir} not usable (expected bytes in src/tsfstudio/data/catalog.yaml):\n  "
                 + "\n  ".join(bad))
    plan = []
    for ds in datasets:
        for m in (only or PLAN.get(ds) or ["ses", "theta", "ets", "arima"]):
            for e in engines:
                if e != "cpp" and a.ref_datasets != "all" and ds not in a.ref_datasets.split(","):
                    continue
                if e in engines_of(m):
                    plan.append((ds, m, e, levels(e, threads)))
    print(f"{len(plan)} (dataset, method, engine) groups; engines: {', '.join(engines)}; levels: {threads}; "
          f"data: {data_dir}", flush=True)
    if a.dry_run:
        for ds, m, e, lv in plan:
            print(f"  {ds:14s} {m:8s} {e:7s} levels {lv}  x {a.repeats if e == 'cpp' else a.ref_repeats}")
        return 0
    node = node_info(avail, a, out)
    (out / "node.json").write_text(json.dumps(node, indent=2), encoding="utf-8")
    runs_csv, work = out / "runs.csv", out / "_work"
    timeout = a.timeout_min * 60
    if "python" in engines:      # statsforecast compiles with numba on first use; do that before anything is timed
        for m in sorted({m for _, m, e, _ in plan if e == "python"}):
            one_run(plan[0][0], m, "python", 1, work, 1800, False, max_series=3)
    done = {(r["dataset"], r["method"], r["engine"], int(r["parallel"]), int(r["rep"])): r for r in read_runs(runs_csv)}
    for ds, m, e, lv in plan:
        prev = None                                     # (level, seconds) of the last level that ran
        stop = ""
        for par in lv:
            reps = a.repeats if e == "cpp" else a.ref_repeats
            for rep in range(1, reps + 1):
                key = (ds, m, e, par, rep)
                if key in done:
                    r = done[key]
                else:
                    if stop:
                        r = {"status": "skipped", "message": stop}
                    elif prev and prev[1] * prev[0] / par > timeout:
                        r = {"status": "skipped", "message": f"projected {prev[1] * prev[0] / par / 60:.0f} min"}
                    else:
                        keep = rep == 1 and not a.discard_runs
                        dest = out / "runs" / ds / m / f"{e}_x{par}" if keep else work
                        l1 = load1()
                        r = {**one_run(ds, m, e, par, dest, timeout, keep), "load1_start": l1}
                    r = {"dataset": ds, "method": m, "engine": e, "parallel": par, "rep": rep, **r}
                    append(runs_csv, r)
                    print(f"{ds:14s} {m:8s} {e:7s} x{par:<3d} rep {rep}: {r['status']:8s} "
                          f"{r.get('engine_sec', '')!s:>10s} s  {r.get('message', '')}", flush=True)
                if r["status"] == "ok":
                    prev = (par, float(r["engine_sec"]))
                elif r["status"] in ("failed", "timeout"):
                    stop = f"level {par}: {r['status']}"
                if r["status"] != "ok":
                    break                               # no further repetitions of a cell that did not run
    shutil.rmtree(work, ignore_errors=True)
    node.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), load1_end=load1())
    (out / "node.json").write_text(json.dumps(node, indent=2), encoding="utf-8")
    report(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
