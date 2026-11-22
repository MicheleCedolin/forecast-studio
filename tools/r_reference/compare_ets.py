"""Compare the C++ ets (tsfcore) with R forecast::ets (tools/r_reference/ets_reference.R output) and with the study's R
forecasts (tests/golden/<case>/forecasts_ets).

    python tools/r_reference/compare_ets.py --case nn5_daily --m 7 --ref <dir>/ets_daily
"""
from __future__ import annotations

import argparse
import ctypes
import time
from collections import Counter
from pathlib import Path

import numpy as np

from tsfstudio.engines import cpp_engine as C
from tsfstudio.metrics import mase, smape

ROOT = Path(__file__).resolve().parents[2]


def load(p: Path) -> np.ndarray:
    for cand in (p, p.with_suffix(p.suffix + ".gz")):
        if cand.is_file():
            return np.loadtxt(cand, delimiter=",", ndmin=2)
    raise FileNotFoundError(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--ref", required=True)
    a = ap.parse_args()
    g = ROOT / "tests" / "golden" / a.case
    train = [r[~np.isnan(r)] for r in load(g / "train.csv")]
    act = load(g / "actuals.csv")
    h = act.shape[1]
    lib = C.load_lib()
    lib.tsf_ets_fit.restype = ctypes.c_int32
    dp = ctypes.POINTER(ctypes.c_double)
    lib.tsf_ets_fit.argtypes = [dp, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, dp, dp, ctypes.c_char_p, ctypes.c_int32]
    fc = np.full((len(train), h), np.nan)
    info = np.full((len(train), 6), np.nan)
    models = []
    t0 = time.perf_counter()
    for i, y in enumerate(train):
        y = np.ascontiguousarray(y, dtype=np.float64)
        f = np.empty(h)
        inf = np.empty(6)
        buf = ctypes.create_string_buffer(128)
        rc = lib.tsf_ets_fit(y.ctypes.data_as(dp), len(y), a.m, h, f.ctypes.data_as(dp), inf.ctypes.data_as(dp), buf, 128)
        models.append(buf.value.decode() if rc == 0 else "FAIL:" + buf.value.decode())
        if rc == 0:
            fc[i], info[i] = f, inf
    secs = time.perf_counter() - t0
    r_fc = load(Path(f"{a.ref}_forecast.csv"))
    r_mod = Path(f"{a.ref}_model.csv").read_text().split("\n")[: len(train)]
    r_fit = np.genfromtxt(f"{a.ref}_fit.csv", delimiter=",", names=True)
    same = np.array([m == r for m, r in zip(models, r_mod)])
    print(f"{a.case}: C++ {secs:.2f} s for {len(train)} series; same model as R on {same.sum()}/{len(train)}")
    print("   C++ models:", dict(Counter(models)))
    diffs = [(i, models[i], r_mod[i]) for i in np.flatnonzero(~same)]
    for d in diffs[:8]:
        i = d[0]
        print(f"   series {i}: C++ {d[1]} aicc {info[i, 0]:.4f} | R {d[2]} aicc {r_fit['aicc'][i]:.4f}")
    rel = np.abs(fc - r_fc) / np.maximum(np.abs(r_fc), 1e-9)
    per = np.nanmax(rel, axis=1)
    print(f"   forecast vs R 9.0.2: median rel {np.nanmedian(rel):.2e}, same-model median {np.nanmedian(rel[same]):.2e}, "
          f">1e-3 series {int((per > 1e-3).sum())}")
    print(f"   AICc of the chosen model: C++ lower on {int((info[:, 0] < r_fit['aicc'] - 1e-6).sum())}, "
          f"R lower on {int((info[:, 0] > r_fit['aicc'] + 1e-6).sum())}")
    m = a.m
    for label, ref in (("R 9.0.2", r_fc), ("study R", load(g / "forecasts_ets.csv"))):
        d1 = (np.nanmean(smape(fc, act)) - np.nanmean(smape(ref, act))) / np.nanmean(smape(ref, act))
        d2 = (np.nanmean(mase(fc, act, train, m)) - np.nanmean(mase(ref, act, train, m))) / np.nanmean(mase(ref, act, train, m))
        print(f"   vs {label}: dSMAPE {100 * d1:+.3f}%  dMASE {100 * d2:+.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
