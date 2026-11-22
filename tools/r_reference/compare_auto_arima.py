"""Compare the C++ auto.arima port (tsf_auto_arima, the study's lambda = 0 -> default -> non-seasonal chain) with
arima_reference.R output.

    python tools/r_reference/compare_auto_arima.py <train.csv[.gz]> <frequency> <ref_prefix>

<ref_prefix>_forecast.csv / _order.csv / _fit.csv come from arima_reference.R with the same data and frequency.
"""
from __future__ import annotations

import ctypes
import sys
import time

import numpy as np
import pandas as pd

from tsfstudio.engines import cpp_engine as C


def main() -> int:
    train, freq, prefix = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    X = np.loadtxt(train, delimiter=",", ndmin=2)
    rf = np.loadtxt(prefix + "_forecast.csv", delimiter=",", ndmin=2)
    ro = pd.read_csv(prefix + "_order.csv")
    rfit = pd.read_csv(prefix + "_fit.csv")
    h = rf.shape[1]
    lib = C.load_lib()
    dp = ctypes.POINTER(ctypes.c_double)
    lib.tsf_auto_arima.restype = ctypes.c_int32
    lib.tsf_auto_arima.argtypes = [dp, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, dp, dp, ctypes.c_char_p, ctypes.c_int32]
    rows, t0 = [], time.perf_counter()
    for i in range(X.shape[0]):
        y = np.ascontiguousarray(X[i][~np.isnan(X[i])])
        fc, info = np.zeros(h), np.zeros(13)
        msg = ctypes.create_string_buffer(256)
        rc = lib.tsf_auto_arima(y.ctypes.data_as(dp), len(y), freq, h, fc.ctypes.data_as(dp), info.ctypes.data_as(dp), msg, 256)
        r = ro.iloc[i]
        rord = (int(r.p), int(r.d), int(r.q), int(r.P), int(r.D), int(r.Q), int(r.drift), int(r["mean"]), int(r.step))
        if rc != 0:
            rows.append(dict(series=i + 1, ok=False, err=msg.value.decode(), r=rord))
            continue
        cord = tuple(int(v) for v in info[[0, 1, 2, 3, 4, 5, 7, 8, 9]])
        dfc = float(np.max(np.abs(fc - rf[i]) / np.maximum(np.abs(rf[i]), 1e-9)))
        rows.append(dict(series=i + 1, ok=True, c=cord, r=rord, same=cord == rord, dfc=dfc,
                         dll=abs(info[10] - rfit.loglik.iloc[i]) if cord == rord else np.nan))
    secs = time.perf_counter() - t0
    ok = [r for r in rows if r["ok"]]
    same = [r for r in ok if r["same"]]
    print(f"{len(rows)} series in {secs:.2f} s (C++, one thread); errors {len(rows) - len(ok)}; "
          f"same order+constant+step {len(same)}/{len(rows)}")
    if same:
        d = np.array([r["dfc"] for r in same])
        ll = np.array([r["dll"] for r in same])
        print(f"  same-order series: forecast max rel diff median {np.median(d):.1e}, max {d.max():.1e}; "
              f"|dloglik| max {np.nanmax(ll):.1e}")
        print("  worst same-order:", sorted(((r["dfc"], r["series"]) for r in same), reverse=True)[:5])
    for r in rows:
        if not r["ok"]:
            print(f"  #{r['series']}: C++ error {r['err']} | R {r['r']}")
        elif not r["same"]:
            print(f"  #{r['series']}: C++ {r['c']} | R {r['r']}  (forecast rel diff {r['dfc']:.2e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
