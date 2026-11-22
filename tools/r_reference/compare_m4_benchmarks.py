"""Compare the Python Naive2 / Comb S-H-D (and its SES, Holt, damped Holt parts) with the R reference from m4_benchmarks.R.

    Rscript tools/r_reference/m4_benchmarks.R tests/golden/nn5_weekly/train.csv 52 8 <dir>/weekly
    python tools/r_reference/compare_m4_benchmarks.py --case nn5_weekly --m 52 --ref <dir>/weekly
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tsfstudio.engines import py_models as M
from tsfstudio.metrics import mase, smape

ROOT = Path(__file__).resolve().parents[2]


def load(p: Path) -> np.ndarray:
    for cand in (p, p.with_suffix(p.suffix + ".gz")):
        if cand.is_file():
            return np.loadtxt(cand, delimiter=",", ndmin=2)
    raise FileNotFoundError(p)


def python_parts(y: np.ndarray, h: int, m: int) -> dict[str, np.ndarray]:
    x, si = M._m4_deseasonalise(y, m, h)
    ses, holt, damped = M.ses_forecast(x, h) * si, M.holt_forecast(x, h, False) * si, M.holt_forecast(x, h, True) * si
    return {"naive2": M.naive(x, h) * si, "ses": ses, "holt": holt, "damped": damped, "comb": (ses + holt + damped) / 3,
            "seasonal": M.m4_seasonality_test(y, m)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--ref", required=True, help="output prefix given to m4_benchmarks.R")
    a = ap.parse_args()
    g = ROOT / "tests" / "golden" / a.case
    train, act = load(g / "train.csv"), load(g / "actuals.csv")
    h = act.shape[1]
    parts = [python_parts(r[~np.isnan(r)], h, a.m) for r in train]
    r_seas = np.loadtxt(f"{a.ref}_seasonal.csv", dtype=str)
    py_seas = np.array([p["seasonal"] for p in parts])
    print(f"{a.case}: seasonality test agrees on {int((py_seas == (r_seas == 'TRUE')).sum())}/{len(parts)} series "
          f"(R seasonal {int((r_seas == 'TRUE').sum())})")
    rows = []
    trl = [r[~np.isnan(r)] for r in train]
    for k in ("naive2", "ses", "holt", "damped", "comb"):
        ref = load(Path(f"{a.ref}_{k}.csv"))
        f = np.vstack([p[k] for p in parts])
        rel = np.abs(f - ref) / np.maximum(np.abs(ref), 1e-9)
        per = rel.max(axis=1)
        sm_py, sm_r = np.nanmean(smape(f, act)), np.nanmean(smape(ref, act))
        ma_py, ma_r = np.nanmean(mase(f, act, trl, a.m)), np.nanmean(mase(ref, act, trl, a.m))
        rows.append({"method": k, "median_rel": np.median(rel), "p95_rel": np.quantile(rel, .95), "max_rel": rel.max(),
                     "n_gt_1e-3": int((per > 1e-3).sum()), "SMAPE_py": sm_py, "SMAPE_R": sm_r, "dSMAPE_%": 100 * (sm_py - sm_r) / sm_r,
                     "MASE_py": ma_py, "MASE_R": ma_r, "dMASE_%": 100 * (ma_py - ma_r) / ma_r})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
