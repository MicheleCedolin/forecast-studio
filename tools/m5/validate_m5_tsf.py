"""Round-trip validation of the M5 .tsf files against the raw CSVs.

Usage: python3 -I validate_m5_tsf.py --data-dir <raw M5 folder> --tsf-dir <folder with the .tsf files>

1. Parses each .tsf with a replica of the Monash reference loader (convert_tsf_to_dataframe,
   opened as cp1252, same checks; strtobool replaced by a local shim because distutils was
   removed in Python 3.12).
2. Compares every series, value by value, with the CSV data (start day taken from the
   start_timestamp attribute, not from the builder), checks that all trimmed prefixes are zero,
   that all series end on 2016-06-19, and that level 12 and level 11 sum to the CSV daily totals.
"""
import argparse
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

CATS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
FAILS = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def strtobool(v):
    v = v.lower()
    if v in ("y", "yes", "t", "true", "on", "1"):
        return 1
    if v in ("n", "no", "f", "false", "off", "0"):
        return 0
    raise ValueError(v)


def convert_tsf_to_dataframe(full_file_path_and_name, replace_missing_vals_with="NaN", value_column_name="series_value"):
    """Replica of the Monash TSForecasting reference loader (logic as published in monash_tsf/utils.py)."""
    col_names, col_types, all_data = [], [], {}
    line_count = 0
    frequency = forecast_horizon = contain_missing_values = contain_equal_length = None
    found_data_tag = found_data_section = started_reading_data_section = False
    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            line = line.strip()
            if line:
                if line.startswith("@"):
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if len(line_content) != 3:
                                raise ValueError("Invalid meta-data specification.")
                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if len(line_content) != 2:
                                raise ValueError("Invalid meta-data specification.")
                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(strtobool(line_content[1]))
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))
                    else:
                        if len(col_names) == 0:
                            raise ValueError("Missing attribute section. Attribute section must come before data.")
                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise ValueError("Missing attribute section. Attribute section must come before data.")
                    elif not found_data_tag:
                        raise ValueError("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []
                            for col in col_names:
                                all_data[col] = []
                        full_info = line.split(":")
                        if len(full_info) != (len(col_names) + 1):
                            raise ValueError("Missing attributes/values in series.")
                        series = full_info[len(full_info) - 1].split(",")
                        if len(series) == 0:
                            raise ValueError("Empty series.")
                        numeric_series = []
                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))
                        if numeric_series.count(replace_missing_vals_with) == len(numeric_series):
                            raise ValueError("All series values are missing.")
                        all_series.append(np.array(numeric_series, dtype=np.float32))
                        for i in range(len(col_names)):
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(full_info[i], "%Y-%m-%d %H-%M-%S")
                            else:
                                raise ValueError("Invalid attribute type.")
                            all_data[col_names[i]].append(att_val)
                        line_count += 1
    if line_count == 0:
        raise ValueError("Empty file.")
    if not found_data_section:
        raise ValueError("Missing series information under data section.")
    all_data[value_column_name] = all_series
    return pd.DataFrame(all_data), frequency, forecast_horizon, contain_missing_values, contain_equal_length


def compare_level(label, df, truth, truth_row, d1, n_days, end_date):
    """Compare every parsed series with the truth matrix; return the reconstructed (n x n_days) matrix."""
    n = len(df)
    full = np.zeros((n, n_days), dtype=np.float64)
    bad_values = bad_prefix = bad_end = 0
    for k in range(n):
        name = df["series_name"].iat[k]
        v = df["series_value"].iat[k]
        s = (df["start_timestamp"].iat[k] - d1).days
        if s + len(v) != n_days or (d1 + pd.Timedelta(days=s + len(v) - 1)) != end_date:
            bad_end += 1
        row = truth[truth_row[name]]
        if not np.array_equal(v, row[s:].astype(np.float32)):
            bad_values += 1
        if row[:s].sum() != 0:
            bad_prefix += 1
        full[k, s:] = v
    check(bad_values == 0, f"{label}: all {n} series equal the CSV values from their start date (mismatches: {bad_values})")
    check(bad_prefix == 0, f"{label}: no non-zero sales were trimmed (series with non-zero trimmed prefix: {bad_prefix})")
    check(bad_end == 0, f"{label}: every series ends on {end_date.date()} with consistent length (violations: {bad_end})")
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--tsf-dir", required=True)
    a = ap.parse_args()
    data_dir, tsf_dir = a.data_dir.rstrip("/"), a.tsf_dir.rstrip("/")
    t0 = time.time()

    cal = pd.read_csv(f"{data_dir}/calendar.csv", usecols=["date"])
    n_days = len(cal)
    d1 = pd.Timestamp(cal["date"].iat[0])
    end_date = pd.Timestamp(cal["date"].iat[-1])
    tr = pd.read_csv(f"{data_dir}/sales_train_evaluation.csv")
    te = pd.read_csv(f"{data_dir}/sales_test_evaluation.csv")
    d_tr = [c for c in tr.columns if c.startswith("d_")]
    d_te = [c for c in te.columns if c.startswith("d_")]
    Y = np.concatenate([tr[d_tr].to_numpy(), te[d_te].to_numpy()], axis=1).astype(np.int64)
    meta = tr[CATS].copy()
    del tr, te

    # ---- level 12
    df12, freq, hor, miss, eq = convert_tsf_to_dataframe(f"{tsf_dir}/m5_daily_dataset.tsf")
    check(len(df12) == 30490, f"L12: {len(df12)} series parsed (expected 30490)")
    check((freq, hor, miss, eq) == ("daily", 28, False, False), f"L12 metadata = {(freq, hor, miss, eq)}")
    check(df12["series_name"].is_unique, "L12: series_name unique")
    truth_row12 = {n: i for i, n in enumerate((meta["item_id"] + "_" + meta["store_id"]).to_numpy())}
    check(set(df12["series_name"]) == set(truth_row12), "L12: series names match the CSV keys")
    attrs_ok = all((df12[c].to_numpy() == meta.set_index(meta["item_id"] + "_" + meta["store_id"]).loc[df12["series_name"], c].to_numpy()).all() for c in CATS)
    check(attrs_ok, "L12: item/dept/cat/store/state attributes match the CSV")
    full12 = compare_level("L12", df12, Y, truth_row12, d1, n_days, end_date)

    # ---- level 11 (aggregation via a different code path: pandas groupby)
    df11, freq, hor, miss, eq = convert_tsf_to_dataframe(f"{tsf_dir}/m5_item_state_daily_dataset.tsf")
    check(len(df11) == 9147, f"L11: {len(df11)} series parsed (expected 9147)")
    check((freq, hor, miss, eq) == ("daily", 28, False, False), f"L11 metadata = {(freq, hor, miss, eq)}")
    check(df11["series_name"].is_unique, "L11: series_name unique")
    g = pd.DataFrame(Y).groupby([meta["item_id"].to_numpy(), meta["state_id"].to_numpy()]).sum()
    truth11 = g.to_numpy()
    truth_row11 = {f"{i}_{s}": r for r, (i, s) in enumerate(g.index)}
    check(set(df11["series_name"]) == set(truth_row11), "L11: series names match item x state pairs")
    full11 = compare_level("L11", df11, truth11, truth_row11, d1, n_days, end_date)

    # ---- consistency across levels and against the raw totals
    daily_csv = Y.sum(axis=0)
    check(np.array_equal(full12.sum(axis=0), daily_csv), "L12 daily totals equal the CSV daily totals")
    check(np.array_equal(full11.sum(axis=0), daily_csv), "L11 daily totals equal the CSV daily totals")
    check(int(full12.sum()) == int(Y.sum()), f"total units preserved: {int(Y.sum())}")

    print(f"\n{'ALL CHECKS PASSED' if not FAILS else str(len(FAILS)) + ' CHECK(S) FAILED'} in {time.time()-t0:.0f}s")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
