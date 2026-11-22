"""Kullanıcının NN5 R çıktılarından küçük altın (golden) test setleri üretir.

Girdi  : NN5 klasörü (NN5weekly ve NN5daily; txt tahminleri, *_dataset.txt, *_results.txt, "R Errors/*.xlsx")
Çıktı  : tests/golden/nn5_weekly/ ve nn5_daily/  (train, actuals, forecasts_<model>, r_errors_<model>, meta.json)
Kullanım: python tools/make_nn5_fixture.py <NN5 klasörü>
"""
import glob
import gzip
import json
import os
import sys

import numpy as np
import openpyxl

src = sys.argv[1]
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden")

SPECS = {
    "nn5_weekly": dict(folder="NN5weekly", prefix="nn5_weekly", frequency="weekly", horizon=8, lag=65, gz=False),
    "nn5_daily": dict(folder="NN5daily", prefix="nn5_daily", frequency="daily", horizon=56, lag=9, gz=True),
}
LOCAL = ("ses", "ets", "theta", "tbats", "arima", "dhr_arima")


def read_txt(path, has_name):
    rows = [ln.strip().split(",") for ln in open(path, encoding="utf-8") if ln.strip()]
    return np.array([[float(x) for x in (r[1:] if has_name else r)] for r in rows])


for name, sp in SPECS.items():
    base = os.path.join(src, sp["folder"])
    dst = os.path.join(root, name)
    os.makedirs(dst, exist_ok=True)
    ext = ".csv.gz" if sp["gz"] else ".csv"

    def save(fname, arr):
        path = os.path.join(dst, fname + ext)
        opener = gzip.open if sp["gz"] else open
        with opener(path, "wt") as fh:
            np.savetxt(fh, arr, delimiter=",", fmt="%.17g")

    save("train", read_txt(f"{base}/txt/{sp['prefix']}_dataset.txt", False))
    save("actuals", read_txt(f"{base}/txt/{sp['prefix']}_results.txt", False))
    have = []
    for m in LOCAL + (f"pooled_regression_lag_{sp['lag']}",):
        p = f"{base}/txt/{sp['prefix']}_{m}.txt"
        if os.path.exists(p):
            save(f"forecasts_{m}", read_txt(p, True))
            have.append(m)
    for f in glob.glob(f"{base}/R Errors/*_errors.xlsx") + glob.glob(f"{base}/R errors/*_errors.xlsx"):
        m = os.path.basename(f).replace(sp["prefix"] + "_", "").replace("_errors.xlsx", "")
        ws = openpyxl.load_workbook(f, data_only=True).active
        rows = list(ws.iter_rows(values_only=True))[1:]
        save(f"r_errors_{m}", np.array([[float(v) for v in r[:5]] for r in rows]))
    json.dump({"name": name, "frequency": sp["frequency"], "horizon": sp["horizon"], "lag": sp["lag"],
               "source": "Kullanıcının R çıktıları (NN5 yarışma verisi)", "methods": have},
              open(os.path.join(dst, "meta.json"), "w"), indent=2, ensure_ascii=False)
    print(name, "->", sorted(os.listdir(dst)))
