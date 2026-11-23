"""WRMSSE, the accuracy measure of the M5 competition (Makridakis, Spiliotis & Assimakopoulos, 2022).

The bottom series (item x store) are summed into the 12 levels of the M5 hierarchy: total, state, store, category,
department, state x category, state x department, store x category, store x department, item, item x state,
item x store. For every aggregated series

    RMSSE = sqrt( mean_h (y - f)^2 / mean_{t>=2} (y_t - y_{t-1})^2 ),

the denominator taken over its training part from its first non-zero value on. Each level weighs its series by their
share of the level's sales over the last 28 training days, and

    WRMSSE = (1/12) * sum_levels sum_i w_i * RMSSE_i.

The official weights are dollar sales (units x sell price). When M5's sell_prices.csv and calendar.csv are given they are
used; otherwise the weights are unit sales and the result is reported as "unit-weighted", which is not the official
figure. Series are aligned on their last observation (all M5 series end on the same day).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

HIERARCHY = ("item_id", "dept_id", "cat_id", "store_id", "state_id")
LEVELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Total", ()), ("State", ("state_id",)), ("Store", ("store_id",)), ("Category", ("cat_id",)),
    ("Department", ("dept_id",)), ("State x Category", ("state_id", "cat_id")),
    ("State x Department", ("state_id", "dept_id")), ("Store x Category", ("store_id", "cat_id")),
    ("Store x Department", ("store_id", "dept_id")), ("Item", ("item_id",)), ("Item x State", ("item_id", "state_id")),
    ("Item x Store", ("item_id", "store_id")),
)


def has_hierarchy(attrs: pd.DataFrame) -> bool:
    return all(c in attrs.columns for c in HIERARCHY)


def _right_align(train: Sequence[np.ndarray]) -> np.ndarray:
    T = max(len(t) for t in train)
    out = np.zeros((len(train), T))
    for i, t in enumerate(train):
        out[i, T - len(t):] = np.nan_to_num(np.asarray(t, float))
    return out


def dollar_weights_source(m5_dir: str | Path | None) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """(sell_prices, calendar) from an M5 folder, or None when the files are not there."""
    if not m5_dir:
        return None
    d = Path(m5_dir)
    sp, cal = d / "sell_prices.csv", d / "calendar.csv"
    if not (sp.is_file() and cal.is_file()):
        return None
    return pd.read_csv(sp), pd.read_csv(cal, usecols=["date", "wm_yr_wk"])


def wrmsse(train: Sequence[np.ndarray], test: np.ndarray, forecasts: np.ndarray, attrs: pd.DataFrame,
           prices: tuple[pd.DataFrame, pd.DataFrame] | None = None, last_train_date: str | None = None) -> dict:
    """WRMSSE and its value per level. attrs: one row per series with the HIERARCHY columns (same order as train).

    prices = (sell_prices, calendar) gives dollar weights; last_train_date (YYYY-MM-DD, the last training day) is then
    needed to price the last 28 training days."""
    Y = _right_align(train)
    test = np.asarray(test, float)
    fc = np.asarray(forecasts, float)
    T = Y.shape[1]
    units = Y[:, T - 28:]
    weighting = "unit sales"
    if prices is not None and last_train_date:
        sp, cal = prices
        cal = cal.assign(date=pd.to_datetime(cal["date"]))
        days = pd.date_range(end=pd.Timestamp(last_train_date), periods=28, freq="D")
        wk = cal.set_index("date").loc[days, "wm_yr_wk"].to_numpy()
        price = sp.pivot_table(index=["store_id", "item_id"], columns="wm_yr_wk", values="sell_price")
        key = pd.MultiIndex.from_frame(attrs[["store_id", "item_id"]].astype(str))
        P = price.reindex(key).reindex(columns=wk).to_numpy(float)
        units = units * np.nan_to_num(P)
        weighting = "dollar sales (sell prices)"
    a = attrs[list(HIERARCHY)].astype(str).reset_index(drop=True)
    per_level = []
    for name, keys in LEVELS:
        groups = a.groupby(list(keys), sort=True).indices if keys else {"all": np.arange(len(a))}
        tot_w = 0.0
        acc = 0.0
        for _, idx in groups.items():
            y_tr = Y[idx].sum(axis=0)
            nz = np.flatnonzero(y_tr != 0)
            if nz.size == 0:
                continue
            y_tr = y_tr[nz[0]:]
            if y_tr.size < 2:
                continue
            den = np.mean(np.diff(y_tr) ** 2)
            num = np.nanmean((test[idx].sum(axis=0) - fc[idx].sum(axis=0)) ** 2)
            w = float(units[idx].sum())
            if den > 0 and np.isfinite(num):
                acc += w * np.sqrt(num / den)
                tot_w += w
        per_level.append({"level": name, "n_series": len(groups), "WRMSSE": acc / tot_w if tot_w > 0 else np.nan})
    vals = [r["WRMSSE"] for r in per_level]
    return {"WRMSSE": float(np.nanmean(vals)), "levels": per_level, "weights": weighting}
