"""Implementation study: the same method on the R, Python and C++ engines.

Questions:
  1. How far apart are the forecasts of one method across engines (R forecast package, Python statsforecast, C++ port)?
  2. Do the accuracy and the ranking of methods change when only the implementation changes?
  3. How long does each engine take, and how does that grow with the number of series?

Usage (outputs go to runs/implementation_study, which git ignores):
  python tools/implementation_study.py run        # accuracy grid (resumable: finished runs are skipped)
  python tools/implementation_study.py scaling    # timing on the first N series of M4 monthly
  python tools/implementation_study.py analyse    # tables (CSV) from both

Every run is a plain `python -m tsfstudio run ... --workers 1`, so all engines get one core and the same split,
horizon, missing-value policy and error measures. Times are the engine's own elapsed time from manifest.json
(R includes starting Rscript, about one second).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "implementation_study"
PY = sys.executable
RSCRIPT = os.environ.get("TSFSTUDIO_RSCRIPT") or r"C:\Users\michele.cedolin\AppData\Local\Programs\R\R-4.6.2\bin\Rscript.exe"

ENGINES = ["r", "python", "cpp"]
LOCAL = ["snaive", "ses", "theta", "ets", "arima", "dhr_arima", "tbats"]
GLOBAL = ["pooled_regression", "catboost"]
DATASETS = ["m1_yearly", "m1_quarterly", "m1_monthly", "m3_yearly", "m3_quarterly", "m3_monthly", "m3_other",
            "tourism_yearly", "tourism_quarterly", "tourism_monthly", "fred_md", "nn5_weekly", "hospital",
            "car_parts", "m4_weekly", "cif_2016"]
# headers without @horizon: the prediction length of the Monash archive on Hugging Face (Monash-University/monash_tsf).
# cif_2016 gives each series its own horizon (57 series 12, 15 series 6); Hugging Face uses 12 for all, and so do we.
HORIZON = {"fred_md": 12, "cif_2016": 12}
SCALING_N = [100, 1000, 10000, 48000]
SCALING_METHODS = ["ets", "theta", "arima"]
TIMEOUT = 3 * 3600


def available(method: str) -> list[str]:
    sys.path.insert(0, str(ROOT / "src"))
    from tsfstudio.app import models_info as mi
    eng = mi.MODELS[method]["engines"]
    out = [e for e in ENGINES if e in eng and (not ONLY_ENGINES or e in ONLY_ENGINES)]
    # catboost on R needs the catboost R package (not on CRAN): the 1.2.10 binary from the CatBoost GitHub release,
    # the same version as the Python package
    return out


ONLY_ENGINES: list[str] = []


def run_one(folder: Path, dataset: str, method: str, engine: str, max_series: int | None = None) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    done = list(folder.glob("*/manifest.json"))
    if done:
        return {"status": "cached", "manifest": str(done[0])}
    cmd = [PY, "-m", "tsfstudio", "run", "--dataset", dataset, "--engine", engine, "--method", method,
           "--workers", "1", "--out", str(folder), "--yes"]
    if method in GLOBAL:
        cmd += ["--task", "global"]
    if max_series:
        cmd += ["--max-series", str(max_series)]
    if dataset in HORIZON:
        cmd += ["--horizon", str(HORIZON[dataset])]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), TSFSTUDIO_RSCRIPT=RSCRIPT)
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=TIMEOUT)
        status = "ok" if p.returncode == 0 else "error"
        msg = (p.stderr or p.stdout)[-400:].replace("\n", " ") if p.returncode else ""
    except subprocess.TimeoutExpired:
        status, msg = "timeout", f"> {TIMEOUT} s"
    wall = time.perf_counter() - t0
    if status != "ok":
        (folder / "FAILED.txt").write_text(f"{status}\n{msg}\n", encoding="utf-8")
    return {"status": status, "wall_sec": wall, "message": msg}


def log(row: dict, name: str) -> None:
    f = OUT / name
    new = not f.exists()
    with f.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "method", "engine", "n", "status", "wall_sec", "message"])
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in w.fieldnames})


def warm_up() -> None:
    """statsforecast compiles with numba on first use and caches the result; do that before anything is timed."""
    for m in LOCAL + GLOBAL:
        for e in available(m):
            if e == "python":
                run_one(OUT / "_warmup" / m, "nn5_weekly", m, e, max_series=5)


def cmd_run() -> None:
    warm_up()
    for ds in DATASETS:
        for m in LOCAL + GLOBAL:
            for e in available(m):
                folder = OUT / "grid" / ds / m / e
                if (folder / "FAILED.txt").exists():
                    continue
                r = run_one(folder, ds, m, e)
                if r["status"] != "cached":
                    log({"dataset": ds, "method": m, "engine": e, **r}, "grid_log.csv")
                    print(ds, m, e, r["status"], round(r.get("wall_sec", 0), 1), flush=True)


def cmd_scaling() -> None:
    warm_up()
    for m in SCALING_METHODS:
        for e in available(m):
            prev = None
            for n in SCALING_N:
                if prev and prev["sec"] / prev["n"] * n > TIMEOUT:   # would not finish: stop this engine here
                    log({"dataset": "m4_monthly", "method": m, "engine": e, "n": n, "status": "skipped",
                         "message": f"projected {prev['sec'] / prev['n'] * n:.0f} s"}, "scaling_log.csv")
                    break
                folder = OUT / "scaling" / m / e / str(n)
                if (folder / "FAILED.txt").exists():
                    break
                r = run_one(folder, "m4_monthly", m, e, max_series=n)
                if r["status"] != "cached":
                    log({"dataset": "m4_monthly", "method": m, "engine": e, "n": n, **r}, "scaling_log.csv")
                    print("scaling", m, e, n, r["status"], round(r.get("wall_sec", 0), 1), flush=True)
                man = next(folder.glob("*/manifest.json"), None)
                if man is None:
                    break
                prev = {"n": n, "sec": json.loads(man.read_text(encoding="utf-8"))["engine"]["elapsed_sec"]}


# ------------------------------------------------------------------------------------------------ analysis
def load(folder: Path) -> dict | None:
    d = next((p.parent for p in folder.glob("*/manifest.json")), None)
    if d is None:
        return None
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    fc = pd.read_csv(d / "forecasts.csv", index_col=0)
    err = pd.read_csv(d / "errors_per_series.csv", index_col=0)
    return {"man": man, "fc": fc, "err": err, "sec": man["engine"]["elapsed_sec"],
            "fallbacks": man.get("fallbacks", {}).get("count", 0), "n": man["dataset"]["n_series_run"]}


def discrepancy(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """Per series: mean over the horizon of |a-b| / ((|a|+|b|)/2), in percent (0 when both are 0)."""
    a, b = a.align(b, join="inner")
    A, B = a.to_numpy(float), b.to_numpy(float)
    den = (np.abs(A) + np.abs(B)) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, np.abs(A - B) / den, 0.0)
    return pd.Series(100 * np.nanmean(r, axis=1), index=a.index)


def kendall(x: list, y: list) -> float:
    n, s = len(x), 0
    for i, j in itertools.combinations(range(n), 2):
        s += np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
    return s / (n * (n - 1) / 2) if n > 1 else float("nan")


def cmd_analyse() -> None:
    res = {}
    for ds in DATASETS:
        for m in LOCAL + GLOBAL:
            for e in ENGINES:
                r = load(OUT / "grid" / ds / m / e)
                if r:
                    res[(ds, m, e)] = r
    if not res:
        sys.exit("no runs found; run `implementation_study.py run` first")

    # 1. forecasts and accuracy, engine pair by engine pair
    rows = []
    for ds in DATASETS:
        for m in LOCAL + GLOBAL:
            for e1, e2 in itertools.combinations(ENGINES, 2):
                a, b = res.get((ds, m, e1)), res.get((ds, m, e2))
                if not a or not b:
                    continue
                d = discrepancy(a["fc"], b["fc"])
                ma, mb = a["err"]["MASE"].mean(), b["err"]["MASE"].mean()
                sa, sb = a["err"]["SMAPE"].mean(), b["err"]["SMAPE"].mean()
                rows.append({"dataset": ds, "method": m, "pair": f"{e1}-{e2}", "n_series": len(d),
                             "share_identical_1e-6pct": (d < 1e-6).mean(), "share_within_1pct": (d < 1).mean(),
                             "median_disc_pct": d.median(), "p95_disc_pct": d.quantile(0.95), "max_disc_pct": d.max(),
                             f"MASE_{e1}": ma, f"MASE_{e2}": mb, "MASE_diff_pct": 100 * (mb - ma) / ma,
                             "SMAPE_diff_pct": 100 * (sb - sa) / sa})
    pairs = pd.DataFrame(rows)
    pairs.to_csv(OUT / "forecast_differences.csv", index=False)

    # 2. accuracy per engine
    acc = pd.DataFrame([{"dataset": ds, "method": m, "engine": e, "n_series": r["n"], "MASE": r["err"]["MASE"].mean(),
                         "SMAPE": r["err"]["SMAPE"].mean(), "fallbacks": r["fallbacks"], "sec": r["sec"],
                         "ms_per_series": 1000 * r["sec"] / max(r["n"], 1)} for (ds, m, e), r in res.items()])
    acc.to_csv(OUT / "accuracy_and_time.csv", index=False)

    # 3. rankings: methods available on all three engines
    common = [m for m in LOCAL + GLOBAL if all((ds, m, e) in res for ds in DATASETS for e in ENGINES)]
    rk = []
    for ds in DATASETS:
        ms = [m for m in LOCAL + GLOBAL if all((ds, m, e) in res for e in ENGINES)]
        if len(ms) < 3:
            continue
        mase = {(m, e): res[(ds, m, e)]["err"]["MASE"].mean() for m in ms for e in ENGINES}
        order = {e: [mase[(m, e)] for m in ms] for e in ENGINES}
        win = {e: ms[int(np.argmin(order[e]))] for e in ENGINES}
        # mixing implementations, as when results from different papers or packages are compared:
        winners, rank_range = set(), {m: [] for m in ms}
        for combo in itertools.product(ENGINES, repeat=len(ms)):
            v = [mase[(m, e)] for m, e in zip(ms, combo)]
            winners.add(ms[int(np.argmin(v))])
            ranks = pd.Series(v).rank(method="min").to_numpy()
            for m, rnk in zip(ms, ranks):
                rank_range[m].append(rnk)
        # series level: the best method of each series
        errs = {e: pd.concat({m: res[(ds, m, e)]["err"]["MASE"] for m in ms}, axis=1) for e in ENGINES}
        best = {e: errs[e].idxmin(axis=1) for e in ENGINES}
        rk.append({"dataset": ds, "n_methods": len(ms), "methods": " ".join(ms),
                   **{f"winner_{e}": win[e] for e in ENGINES},
                   "tau_r_python": kendall(order["r"], order["python"]), "tau_r_cpp": kendall(order["r"], order["cpp"]),
                   "tau_python_cpp": kendall(order["python"], order["cpp"]),
                   "distinct_winners_when_mixed": len(winners), "winners_when_mixed": " ".join(sorted(winners)),
                   "max_rank_shift_when_mixed": max(max(v) - min(v) for v in rank_range.values()),
                   "series_best_changes_r_python": float((best["r"] != best["python"]).mean()),
                   "series_best_changes_r_cpp": float((best["r"] != best["cpp"]).mean())})
    pd.DataFrame(rk).to_csv(OUT / "rankings.csv", index=False)

    # 4. scaling
    sc = []
    for m in SCALING_METHODS:
        for e in ENGINES:
            for n in SCALING_N:
                r = load(OUT / "scaling" / m / e / str(n))
                if r:
                    sc.append({"method": m, "engine": e, "n": r["n"], "sec": r["sec"], "ms_per_series": 1000 * r["sec"] / r["n"]})
    if sc:
        pd.DataFrame(sc).to_csv(OUT / "scaling.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 4):
        print("methods on all engines for every dataset:", common)
        print(pairs.groupby(["method", "pair"])[["share_identical_1e-6pct", "share_within_1pct", "median_disc_pct",
                                                 "p95_disc_pct", "MASE_diff_pct"]].median())
        print(pd.DataFrame(rk).drop(columns=["methods"]))
        print(acc.groupby(["method", "engine"])["ms_per_series"].median().unstack())
        if sc:
            print(pd.DataFrame(sc).pivot_table(index=["method", "n"], columns="engine", values="sec"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "scaling", "analyse"])
    ap.add_argument("--datasets", nargs="+", help="only these datasets (default: the full list)")
    ap.add_argument("--methods", nargs="+", help="only these methods (run step)")
    ap.add_argument("--engines", nargs="+", help="only these engines (run step)")
    a = ap.parse_args()
    if a.datasets:
        DATASETS[:] = a.datasets
    if a.methods:
        LOCAL[:] = [m for m in LOCAL if m in a.methods]
        GLOBAL[:] = [m for m in GLOBAL if m in a.methods]
    if a.engines:
        ONLY_ENGINES[:] = a.engines
    OUT.mkdir(parents=True, exist_ok=True)
    {"run": cmd_run, "scaling": cmd_scaling, "analyse": cmd_analyse}[a.step]()
