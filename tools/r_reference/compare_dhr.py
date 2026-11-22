"""Compare the C++ dhr_arima (tsf_dhr_arima) with dhr_reference.R output.

    python tools/r_reference/compare_dhr.py <train.csv[.gz]> <frequency> <K> <ref_prefix>
"""
from __future__ import annotations

import ctypes
import sys
import time

import numpy as np
import pandas as pd

from tsfstudio.engines import cpp_engine as C


def main() -> int:
    train, freq, K, prefix = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    X = np.loadtxt(train, delimiter=",", ndmin=2)
    rf = np.loadtxt(prefix + "_forecast.csv", delimiter=",", ndmin=2)
    ro = pd.read_csv(prefix + "_order.csv")
    h = rf.shape[1]
    lib = C.load_lib()
    dp = ctypes.POINTER(ctypes.c_double)
    lib.tsf_dhr_arima.restype = ctypes.c_int32
    lib.tsf_dhr_arima.argtypes = [dp, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, dp, dp,
                                  ctypes.c_char_p, ctypes.c_int32]
    same, diffs, t0 = 0, [], time.perf_counter()
    for i in range(X.shape[0]):
        y = np.ascontiguousarray(X[i][~np.isnan(X[i])])
        fc, info = np.zeros(h), np.zeros(13)
        msg = ctypes.create_string_buffer(256)
        rc = lib.tsf_dhr_arima(y.ctypes.data_as(dp), len(y), freq, K, h, fc.ctypes.data_as(dp), info.ctypes.data_as(dp), msg, 256)
        r = ro.iloc[i]
        if int(r.ok) == 0 or rc != 0:
            print(f"  #{i + 1}: R ok={int(r.ok)} | C++ rc={rc} {msg.value.decode()}")
            continue
        rord = (int(r.p), int(r.d), int(r.q), int(r.drift), int(r["mean"]))
        cord = tuple(int(v) for v in info[[0, 1, 2, 7, 8]])
        d = float(np.max(np.abs(fc - rf[i]) / np.maximum(np.abs(rf[i]), 1e-9)))
        if cord == rord:
            same += 1
            diffs.append(d)
        else:
            print(f"  #{i + 1}: C++ {cord} | R {rord}  (forecast rel diff {d:.2e})")
    d = np.array(diffs)
    print(f"{X.shape[0]} series in {time.perf_counter() - t0:.2f} s; same order+constant {same}/{X.shape[0]}; "
          f"forecast rel diff median {np.median(d):.1e}, max {d.max():.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
