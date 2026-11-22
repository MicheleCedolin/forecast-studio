"""Build Monash-style .tsf files from the raw M5 CSV release.

Usage:
    python3 -I build_m5_tsf.py --data-dir <raw M5 folder> --out-dir <empty output folder>

Design (confirmed with the user):
  * evaluation phase: sales_train_evaluation (d_1..d_1941) + sales_test_evaluation (d_1942..d_1969)
  * two outputs: bottom level (item x store, 30,490 series) and level 11 (item x state, 9,147 series)
  * each series trimmed to start at the first week that has a sell price (pre-launch zeros removed)
  * target only: no covariates in the .tsf (prices are used only to find the launch week)
  * frequency daily, horizon 28, @missing false, @equallength false

This M5 release has no `id` column and calendar.csv has no `d` column:
series are keyed by (item_id, store_id) and d_i is row i (1-based) of calendar.csv.
"""
import argparse
import hashlib
import os
import time

import numpy as np
import pandas as pd

HORIZON = 28
CATS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
CITATION = (
    "# Makridakis, S., Spiliotis, E., Assimakopoulos, V., 2022. M5 accuracy competition: "
    "Results, findings, and conclusions. International Journal of Forecasting 38 (4)."
)


def tsf_date(date_str):
    """Format a YYYY-MM-DD string the way the Monash loader expects (%Y-%m-%d %H-%M-%S)."""
    return f"{date_str} 00-00-00"


def write_tsf(path, relation, comments, attr_defs, rows, horizon):
    """rows: iterable of (attribute values (list of str), start_date_str, 1-D int array)."""
    with open(path, "w", encoding="ascii", newline="\n") as f:
        for c in comments:
            f.write(c + "\n")
        f.write(f"@relation {relation}\n")
        for name, typ in attr_defs:
            f.write(f"@attribute {name} {typ}\n")
        f.write("@attribute start_timestamp date\n")
        f.write("@frequency daily\n")
        f.write(f"@horizon {horizon}\n")
        f.write("@missing false\n")
        f.write("@equallength false\n")
        f.write("@data\n")
        for attrs, start_date, values in rows:
            for a in attrs:
                assert ":" not in a and " " not in a, a
            f.write(":".join(attrs) + ":" + tsf_date(start_date) + ":" + ",".join(map(str, values.tolist())) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    data_dir = args.data_dir.rstrip("/")
    out_dir = args.out_dir.rstrip("/")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # ---- calendar: d_i is row i; weeks used to find launch dates
    cal = pd.read_csv(f"{data_dir}/calendar.csv", usecols=["date", "wm_yr_wk"])
    dates = cal["date"].to_numpy()
    n_days = len(cal)
    assert dates[0] == "2011-01-29" and dates[-1] == "2016-06-19", (dates[0], dates[-1])
    assert (pd.to_datetime(cal["date"]).diff().dropna() == pd.Timedelta(days=1)).all()
    week_first_day = cal.assign(i=np.arange(n_days)).groupby("wm_yr_wk")["i"].min()

    # ---- sales: evaluation train + evaluation test, aligned on the key columns
    tr = pd.read_csv(f"{data_dir}/sales_train_evaluation.csv")
    te = pd.read_csv(f"{data_dir}/sales_test_evaluation.csv")
    assert (tr[CATS].values == te[CATS].values).all(), "train/test rows differ"
    d_tr = [c for c in tr.columns if c.startswith("d_")]
    d_te = [c for c in te.columns if c.startswith("d_")]
    assert [int(c[2:]) for c in d_tr + d_te] == list(range(1, n_days + 1)), "d columns not 1..N"
    assert len(d_te) == HORIZON, len(d_te)
    Y = np.concatenate([tr[d_tr].to_numpy(), te[d_te].to_numpy()], axis=1).astype(np.int32)
    assert Y.min() >= 0
    meta = tr[CATS].copy()
    del tr, te
    n_series = len(meta)
    names = (meta["item_id"] + "_" + meta["store_id"]).to_numpy()
    assert len(set(names)) == n_series

    # ---- launch day per series = first day of the first priced week (never later than first sale)
    pr = pd.read_csv(f"{data_dir}/sell_prices.csv", usecols=["store_id", "item_id", "wm_yr_wk"],
                     dtype={"store_id": "category", "item_id": "category"})
    first_week = pr.groupby(["store_id", "item_id"], observed=True)["wm_yr_wk"].min()
    del pr
    idx = pd.MultiIndex.from_arrays([meta["store_id"], meta["item_id"]])
    s_price = week_first_day.reindex(first_week.reindex(idx).to_numpy()).to_numpy()
    assert not np.isnan(s_price).any(), "series without any price row"
    s_price = s_price.astype(int)
    first_nz = (Y > 0).argmax(axis=1)
    assert (Y > 0).any(axis=1).all(), "all-zero series present"
    n_viol = int((first_nz < s_price).sum())
    start = np.minimum(s_price, first_nz)
    print(f"[load] series={n_series} days={n_days} | sales before first price (start moved earlier): {n_viol}")

    manifest = []

    # ---- level 12: item x store
    comments12 = [
        "# Dataset Information",
        f"# This dataset contains {n_series} daily time series of unit sales (item x store, level 12 of the M5 hierarchy) from the M5 forecasting competition.",
        f"# Days d_1 to d_{n_days} (2011-01-29 to 2016-06-19) are the concatenation of sales_train_evaluation (d_1-d_1941) and sales_test_evaluation (d_1942-d_{n_days}).",
        "# Each series starts in the first week for which a sell price exists; zeros before that week (product not yet on sale) were removed.",
        f"# The last {HORIZON} days are the M5 evaluation-phase test period. Values are non-negative integers and no values are missing.",
        "# Attributes item_id, dept_id, cat_id, store_id and state_id give the product and store hierarchy.",
        "#",
        "# For more details, please refer to",
        CITATION,
        "#",
    ]
    attr12 = [("series_name", "string")] + [(c, "string") for c in CATS]

    def rows12():
        for i in range(n_series):
            m = meta.iloc[i]
            v = Y[i, start[i]:]
            manifest.append(("L12_item_store", names[i], int(start[i]) + 1, dates[start[i]], len(v), int(start[i]), float((v == 0).mean())))
            yield [names[i]] + [m[c] for c in CATS], dates[start[i]], v

    p12 = f"{out_dir}/m5_daily_dataset.tsf"
    write_tsf(p12, "M5", comments12, attr12, rows12(), HORIZON)
    print(f"[write] {p12} ({os.path.getsize(p12)/1e6:.1f} MB) in {time.time()-t0:.0f}s")

    # ---- level 11: item x state (sum of member stores; starts at earliest member launch)
    grp = meta["item_id"] + "|" + meta["state_id"]
    codes, uniques = pd.factorize(grp, sort=True)
    for c in ["dept_id", "cat_id"]:
        assert (meta.groupby(codes)[c].nunique() == 1).all()
    order = np.argsort(codes, kind="stable")
    bounds = np.r_[0, np.flatnonzero(np.diff(codes[order])) + 1]
    agg = np.add.reduceat(Y[order], bounds, axis=0)
    start_agg = np.minimum.reduceat(start[order], bounds)
    rep = meta.iloc[order[bounds]].reset_index(drop=True)
    n_agg = len(uniques)
    assert agg.shape == (n_agg, n_days)
    assert agg.sum() == Y.sum(), "aggregation lost units"
    print(f"[agg] level 11 series={n_agg}")

    comments11 = [
        "# Dataset Information",
        f"# This dataset contains {n_agg} daily time series of unit sales (item x state, level 11 of the M5 hierarchy) from the M5 forecasting competition.",
        "# Each series is the sum of the item x store series of that item over the stores of one state (CA, TX or WI).",
        f"# Days d_1 to d_{n_days} (2011-01-29 to 2016-06-19) come from sales_train_evaluation (d_1-d_1941) and sales_test_evaluation (d_1942-d_{n_days}).",
        "# Each series starts in the week in which the first store of that state has a sell price for the item; earlier zeros were removed.",
        f"# The last {HORIZON} days are the M5 evaluation-phase test period. Values are non-negative integers and no values are missing.",
        "# Attributes item_id, dept_id, cat_id and state_id give the product and state.",
        "#",
        "# For more details, please refer to",
        CITATION,
        "#",
    ]
    attr11 = [("series_name", "string"), ("item_id", "string"), ("dept_id", "string"), ("cat_id", "string"), ("state_id", "string")]

    def rows11():
        for j in range(n_agg):
            r = rep.iloc[j]
            v = agg[j, start_agg[j]:]
            nm = r["item_id"] + "_" + r["state_id"]
            manifest.append(("L11_item_state", nm, int(start_agg[j]) + 1, dates[start_agg[j]], len(v), int(start_agg[j]), float((v == 0).mean())))
            yield [nm, r["item_id"], r["dept_id"], r["cat_id"], r["state_id"]], dates[start_agg[j]], v

    p11 = f"{out_dir}/m5_item_state_daily_dataset.tsf"
    write_tsf(p11, "M5_item_state", comments11, attr11, rows11(), HORIZON)
    print(f"[write] {p11} ({os.path.getsize(p11)/1e6:.1f} MB) in {time.time()-t0:.0f}s")

    # ---- manifest + checksums
    pm = f"{out_dir}/m5_tsf_manifest.csv"
    pd.DataFrame(manifest, columns=["level", "series_name", "start_d", "start_date", "length", "trimmed_days", "zero_share"]).to_csv(pm, index=False)
    sums = f"{out_dir}/SHA256SUMS.txt"
    with open(sums, "w") as f:
        for p in (p12, p11, pm):
            f.write(f"{sha256(p)}  {os.path.basename(p)}\n")
    L = np.array([m[4] for m in manifest if m[0] == "L12_item_store"])
    print(f"[summary] L12 lengths: min={L.min()} median={int(np.median(L))} max={L.max()} | trimmed cells={int(start.sum())}")
    print(f"[done] total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
