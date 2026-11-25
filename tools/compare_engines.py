"""Forecast agreement of the engines for one method: runs it on each engine and compares the forecasts series by series.

    python tools/compare_engines.py --methods stl_ets,thief --dataset m3_monthly --max-series 40
    python tools/compare_engines.py --methods all-r --dataset nn5_daily --max-series 20 --engines r,cpp,python

For every pair of engines it prints the median and the largest relative difference per forecast value
(|a - b| / max(|a|, |b|), 0 when both are 0), the number of series that differ by more than 1e-6 and by more than 1%
somewhere on the horizon, the fallbacks of each engine, and the two mean MASE values. Runs go through the ordinary
command line (`tsfstudio run`); nothing is kept unless --out is given.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run(dataset: str, method: str, engine: str, max_series: int | None, out: Path, is_global: bool,
        integer_conversion: str = "auto") -> Path | None:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    cmd = [sys.executable, "-m", "tsfstudio", "run", "--dataset", dataset, "--engine", engine, "--method", method,
           "--out", str(out), "--yes", "--integer-conversion", integer_conversion]
    if is_global:
        cmd += ["--task", "global"]
    if max_series:
        cmd += ["--max-series", str(max_series)]
    out.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in out.iterdir()}
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    new = [d for d in out.iterdir() if d.name not in before and (d / "manifest.json").is_file()]
    if p.returncode != 0 or not new:
        last = [ln for ln in (p.stderr or p.stdout).strip().splitlines() if ln.strip()][-1:] or ["no output"]
        print(f"    {engine}: failed: {last[0][:200]}")
        return None
    return new[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--methods", required=True, help="comma separated, or all-r (every local method of the R engine)")
    ap.add_argument("--dataset", default="nn5_weekly")
    ap.add_argument("--engines", default="r,cpp,python")
    ap.add_argument("--max-series", type=int, default=None)
    ap.add_argument("--integer-conversion", default="auto", choices=["auto", "on", "off"],
                    help="off: compare the unrounded forecasts of count data (car_parts, M5)")
    ap.add_argument("--out", default=None, help="keep the run folders here")
    ap.add_argument("--csv", default=None, help="append the comparison rows to this CSV file")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from tsfstudio.app import models_info as mi
    from tsfstudio.engines.r_engine import REngine
    methods = list(REngine.LOCAL_METHODS) if a.methods == "all-r" else a.methods.split(",")
    engines = a.engines.split(",")
    tmp = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="tsfcmp_"))
    rows = []
    for m in methods:
        info = mi.MODELS[m]
        res = {}
        for e in engines:
            if e not in info["engines"]:
                continue
            d = run(a.dataset, m, e, a.max_series, tmp / m / e, info["task"] == "global", a.integer_conversion)
            if d is not None:
                man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
                res[e] = {"fc": pd.read_csv(d / "forecasts.csv", index_col=0),
                          "mase": pd.read_csv(d / "errors_per_series.csv", index_col=0)["MASE"].mean(),
                          "fb": man["fallbacks"]["count"], "sec": man["timing"]["engine_wall_sec"]}
        print(f"{m} on {a.dataset}: " + ", ".join(f"{e} MASE {r['mase']:.4f} ({r['fb']} fallbacks, {r['sec']:.1f} s)"
                                                 for e, r in res.items()))
        for e1, e2 in itertools.combinations(res, 2):
            A, B = res[e1]["fc"].align(res[e2]["fc"], join="inner")
            x, y = A.to_numpy(float), B.to_numpy(float)
            den = np.maximum(np.abs(x), np.abs(y))
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = np.where(den > 0, np.abs(x - y) / den, 0.0)
            worst = np.nanmax(rel, axis=1)
            row = {"dataset": a.dataset, "method": m, "pair": f"{e1}-{e2}", "n_series": len(worst),
                   "median_rel": float(np.nanmedian(rel)), "max_rel": float(np.nanmax(rel)),
                   "series_gt_1e-6": int((worst > 1e-6).sum()), "series_gt_1pct": int((worst > 1e-2).sum()),
                   f"fallbacks_{e1}": res[e1]["fb"], f"fallbacks_{e2}": res[e2]["fb"],
                   "mase_a": res[e1]["mase"], "mase_b": res[e2]["mase"]}
            rows.append(row)
            print(f"    {e1:6s} vs {e2:6s}: median {row['median_rel']:.1e}, max {row['max_rel']:.1e}, "
                  f"> 1e-6: {row['series_gt_1e-6']}/{len(worst)}, > 1%: {row['series_gt_1pct']}/{len(worst)}")
    if a.csv and rows:
        f = Path(a.csv)
        pd.DataFrame(rows).to_csv(f, mode="a", header=not f.exists(), index=False)
    if not a.out:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
