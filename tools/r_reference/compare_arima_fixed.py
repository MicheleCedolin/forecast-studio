"""Compare the C++ stats::arima port (tsf_arima_fixed) with arima_fixed_reference.R output.

    python tools/r_reference/compare_arima_fixed.py <reference.csv>
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tsfstudio.engines import cpp_engine as C

ROOT = Path(__file__).resolve().parents[2]


def load(case):
    p = ROOT / "tests" / "golden" / case / "train.csv"
    return np.loadtxt(p if p.is_file() else str(p) + ".gz", delimiter=",", ndmin=2)


def main() -> int:
    ref = pd.read_csv(sys.argv[1])
    lib = C.load_lib()
    dp, ip = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int32)
    lib.tsf_arima_fixed.restype = ctypes.c_int32
    lib.tsf_arima_fixed.argtypes = [dp, ctypes.c_int32, ip, ip, ctypes.c_int32, dp, dp, dp, ctypes.c_char_p, ctypes.c_int32]
    data = {c: load(c) for c in ref["case"].unique()}
    worst = []
    for _, r in ref.iterrows():
        y = data[r["case"]][int(r["series"]) - 1]
        y = np.ascontiguousarray(y[~np.isnan(y)])
        ord_ = np.array([r.p, r.d, r.q, r.P, r.D, r.Q, r.s], dtype=np.int32)
        meth = {"CSS": 0, "CSS-ML": 1, "ML": 2}[r.method]
        flags = np.array([int(r["mean"]), int(r.drift), meth], dtype=np.int32)
        fc, coef, info = np.zeros(8), np.zeros(16), np.zeros(5)
        msg = ctypes.create_string_buffer(256)
        rc = lib.tsf_arima_fixed(y.ctypes.data_as(dp), len(y), ord_.ctypes.data_as(ip), flags.ctypes.data_as(ip), 8,
                                 fc.ctypes.data_as(dp), coef.ctypes.data_as(dp), info.ctypes.data_as(dp), msg, 256)
        label = f"{r['case']}#{int(r.series)} ({int(r.p)},{int(r.d)},{int(r.q)})({int(r.P)},{int(r.D)},{int(r.Q)})[{int(r.s)}]{'+drift' if r.drift else ''} {r.method}"
        if r.status != "OK" or rc != 0:
            print(f"{label}: R {r.status} | C++ {'OK' if rc == 0 else 'ERROR ' + msg.value.decode()}")
            continue
        rf = r[[f"f{i}" for i in range(1, 9)]].astype(float).to_numpy()
        rc_ = r[[f"c{i}" for i in range(1, 17)]].astype(float).to_numpy()
        nc = int(info[4])
        dll = abs(info[0] - r.loglik)
        dfc = np.max(np.abs(fc - rf) / np.maximum(np.abs(rf), 1e-9))
        dcf = np.max(np.abs(coef[:nc] - rc_[:nc])) if nc else 0.0
        worst.append((dfc, label))
        print(f"{label}: dloglik {dll:.2e}  max|dcoef| {dcf:.2e}  max rel dfc {dfc:.2e}  sigma2 {info[2]:.6g}/{r.sigma2:.6g}  "
              f"nanvar C++ {int(info[3] == 0)} R {r.nanvar}")
    worst.sort(reverse=True)
    print("worst forecast differences:", [f"{w:.1e} {l}" for w, l in worst[:5]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
