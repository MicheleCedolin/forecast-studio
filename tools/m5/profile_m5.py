"""Profile the raw M5 CSV files before building TSF output.

Usage: python3 -I profile_m5.py <data_dir>
Read-only: prints structure facts that the TSF build depends on.
Note: this M5 release has no `id` column and no `d` column in calendar.csv;
rows are keyed by (item_id, store_id) and d_i is row i (1-based) of calendar.csv.
"""
import sys
import numpy as np
import pandas as pd

data_dir = sys.argv[1].rstrip("/")
CATS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

cal = pd.read_csv(f"{data_dir}/calendar.csv", usecols=["date", "wm_yr_wk"])
print("calendar rows:", len(cal), "| first:", cal.iloc[0].to_dict(), "| last:", cal.iloc[-1].to_dict())
dates = pd.to_datetime(cal["date"])
print("calendar contiguous daily:", bool((dates.diff().dropna() == pd.Timedelta(days=1)).all()))
print("weeks: distinct wm_yr_wk =", cal["wm_yr_wk"].nunique(), "| every week has 7 days:", bool((cal.groupby("wm_yr_wk").size() == 7).all()))

tr = pd.read_csv(f"{data_dir}/sales_train_evaluation.csv")
te = pd.read_csv(f"{data_dir}/sales_test_evaluation.csv")
print("\ntrain_eval shape:", tr.shape, "| last col:", tr.columns[-1])
print("test_eval  shape:", te.shape, "| first/last d col:", te.columns[5], te.columns[-1])
print("key columns equal (same rows, same order):", bool((tr[CATS].values == te[CATS].values).all()))
key_tr = tr["item_id"] + "_" + tr["store_id"]
print("unique (item,store) keys:", key_tr.nunique(), "of", len(tr))
print("unique items/depts/cats/stores/states:", [tr[c].nunique() for c in CATS])

d_tr = [c for c in tr.columns if c.startswith("d_")]
d_te = [c for c in te.columns if c.startswith("d_")]
print("train d range:", d_tr[0], "->", d_tr[-1], f"({len(d_tr)} days)", "| test d range:", d_te[0], "->", d_te[-1], f"({len(d_te)} days)")
print("d numbering contiguous:", [int(c[2:]) for c in d_tr + d_te] == list(range(1, len(d_tr) + len(d_te) + 1)))
print("train+test days == calendar rows:", len(d_tr) + len(d_te) == len(cal))

Y = np.concatenate([tr[d_tr].to_numpy(), te[d_te].to_numpy()], axis=1)
print("\nsales matrix:", Y.shape, Y.dtype, "| min:", Y.min(), "| max:", Y.max(),
      "| NaN:", int(pd.isna(pd.DataFrame(Y)).to_numpy().sum()), "| zero share overall: %.3f" % (Y == 0).mean())
first_nz = np.where((Y > 0).any(axis=1), (Y > 0).argmax(axis=1), -1)
print("series with no sales at all:", int((first_nz == -1).sum()))
print("series with leading zeros (first sale after d_1):", int((first_nz > 0).sum()), "| share: %.3f" % ((first_nz > 0).mean()))

pr = pd.read_csv(f"{data_dir}/sell_prices.csv", dtype={"store_id": "category", "item_id": "category"})
print("\nsell_prices rows:", len(pr), "| cols:", list(pr.columns), "| NaN prices:", int(pr["sell_price"].isna().sum()))
first_week = pr.groupby(["store_id", "item_id"], observed=True)["wm_yr_wk"].min()
print("priced (store,item) pairs:", len(first_week), "| series in sales:", len(tr))

week_first_day = cal.assign(i=np.arange(len(cal))).groupby("wm_yr_wk")["i"].min()
idx = pd.MultiIndex.from_arrays([tr["store_id"], tr["item_id"]])
fw = first_week.reindex(idx).to_numpy()
s_price = week_first_day.reindex(fw).to_numpy()
has_price = ~np.isnan(s_price)
print("series with at least one price row:", int(has_price.sum()))
sp = np.where(has_price, s_price, -1).astype(int)
both = has_price & (first_nz >= 0)
diff = first_nz[both] - sp[both]
print("first priced day vs first sale (series having both):")
print("  sale BEFORE first priced day (violations):", int((diff < 0).sum()))
print("  priced start > d_1 (late launch):", int((sp[both] > 0).sum()))
print("  pre-launch days that would be trimmed (sum of priced start index):", int(sp[both].sum()))
print("  first sale more than 7 days after first priced day:", int((diff > 7).sum()))
print("  priced-but-unsold leading days kept (sum of positive diff):", int(diff[diff > 0].sum()))
print("share of all cells that are trimmed: %.3f" % (sp[both].sum() / Y.size))
