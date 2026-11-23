"""Where do accuracy measures disagree: inside series, or when series are averaged? (2026-11-22)

`reliability.py` compares models across tasks from task-mean losses. This module goes one level down and uses the losses
of each single series. It implements Section 5.5 of the manuscript "How Reliable Are Forecasting-Model Rankings?" and
reproduces that section's per-series values (`tests/test_instability.py`).

Input: a long table with the columns series, metric, model, loss (the loss of one model on one series; smaller = better).

  series_instability     per series: 1 - Kendall's W of the metric rankings, and mean cross-metric selection regret
  aggregation_effect     per group of measures that are equivalent inside a series: Kendall's W of their TASK-level rankings
  series_features        six descriptive features of one series (numpy only)
  feature_table          those features for every series of a TSF data set
  association            Spearman correlation between series features and series instability (one task)
  pool_over_tasks        pools within-task correlations; tasks, not series, are the units of inference
  losses_from_runs / losses_from_wide   build the input table from run folders or from a wide per-series error file

Two facts drive the design. (1) Inside one series MAE, nMAE, WAPE and MASE differ by a constant factor, so they rank the
models identically; so do MSE, RMSE, nRMSE and RMSSE. Each group is ONE judge at the series level (`EQUIVALENT`).
(2) The same measures can rank models differently at the task level, because they weight the series differently in the
task mean. `aggregation_effect` measures exactly that part of the disagreement.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata

from .reliability import METRIC_FAMILY, kendalls_w

# Lower-case metric name -> class of measures that are monotone transformations of one another INSIDE a series.
EQUIVALENT: dict[str, str] = {
    "mae": "absolute error", "nmae": "absolute error", "mase": "absolute error", "wape": "absolute error",
    "mse": "squared error", "rmse": "squared error", "nrmse": "squared error", "nrmse_mean": "squared error",
    "nrmse_range": "squared error", "rmsse": "squared error",
}
_COLS = ("series", "metric", "model", "loss")
FEATURES = ("trend", "seasonal_strength", "entropy", "x_acf1", "cv", "log_level")


def _class_of(metric: str) -> str:
    return EQUIVALENT.get(metric.lower(), metric)


def judges_of(metrics: Sequence[str]) -> list[str]:
    """One representative per equivalence class, in the order given.

    The representative does not affect rankings or Kendall's W. It can affect the normalized regret in one case: MSE is
    a non-linear transformation of RMSE, nRMSE and RMSSE (which are linear in one another). A member on the RMSE scale
    therefore represents the squared-error class whenever one is present, so results do not depend on column order."""
    out, pos = [], {}
    for m in metrics:
        c = _class_of(m)
        if c not in pos:
            pos[c] = len(out)
            out.append(m)
        elif out[pos[c]].lower() == "mse" and m.lower() != "mse":
            out[pos[c]] = m
    return out


def _kendall_w_rows(r: np.ndarray) -> np.ndarray:
    """r: series x judges x models (ranks with average ties). Tie-corrected W per series."""
    n_s, m, k = r.shape
    s = ((r.sum(axis=1) - m * (k + 1) / 2.0) ** 2).sum(axis=1)
    t = np.zeros(n_s)
    for i in range(n_s):
        for j in range(m):
            _, c = np.unique(r[i, j], return_counts=True)
            t[i] += (c ** 3 - c).sum()
    den = m ** 2 * (k ** 3 - k) - m * t
    return np.where(den > 0, 12.0 * s / np.where(den > 0, den, 1.0), np.nan)


def _check(losses: pd.DataFrame) -> None:
    missing = [c for c in _COLS if c not in losses.columns]
    if missing:
        raise ValueError(f"losses needs the columns {list(_COLS)}; missing: {missing}")


def series_instability(losses: pd.DataFrame, metrics: Sequence[str] | None = None,
                       models: Sequence[str] | None = None) -> pd.DataFrame:
    """Per series: disagreement = 1 - Kendall's W over the judges, regret = mean normalized selection regret over ordered
    pairs of judges. Only series with a finite loss for every judge and model are kept.

    `metrics` / `models` restrict and order the evidence (default: all present). The judges used are in `.attrs["judges"]`."""
    _check(losses)
    metrics = list(pd.unique(losses["metric"])) if metrics is None else list(metrics)
    models = list(pd.unique(losses["model"])) if models is None else list(models)
    judges = judges_of(metrics)
    if len(judges) < 2:
        raise ValueError("At least two measures that are not equivalent inside a series are needed "
                         f"(got {metrics}; MAE/MASE/WAPE count as one, MSE/RMSE/RMSSE as one).")
    if len(models) < 3:
        raise ValueError("At least three models are needed for a concordance coefficient.")
    d = losses[losses["metric"].isin(judges) & losses["model"].isin(models)]
    wide = d.pivot_table(index="series", columns=["metric", "model"], values="loss", aggfunc="first", sort=False)
    wide = wide.reindex(columns=pd.MultiIndex.from_product([judges, models]))
    cube = wide.to_numpy(float).reshape(len(wide), len(judges), len(models))
    ok = np.isfinite(cube).all(axis=(1, 2))
    x = cube[ok]
    ranks = np.apply_along_axis(rankdata, 2, x)
    w = _kendall_w_rows(ranks)
    lo, hi = x.min(axis=2), x.max(axis=2)
    span = np.where(hi > lo, hi - lo, np.nan)
    chosen = x.argmin(axis=2)
    rows = np.arange(len(x))
    tot, cnt = np.zeros(len(x)), np.zeros(len(x))
    for a in range(len(judges)):
        for b in range(len(judges)):
            if a == b:
                continue
            v = (x[rows, b, chosen[:, a]] - lo[:, b]) / span[:, b]
            good = np.isfinite(v)
            tot[good] += v[good]
            cnt[good] += 1
    out = pd.DataFrame({"disagreement": 1.0 - w, "regret": np.where(cnt > 0, tot / np.where(cnt > 0, cnt, 1), np.nan)},
                       index=wide.index[ok])
    out.attrs.update(judges=judges, models=models, n_series=int(len(wide)), n_dropped=int((~ok).sum()))
    return out


def aggregation_effect(losses: pd.DataFrame, metrics: Sequence[str] | None = None,
                       models: Sequence[str] | None = None) -> pd.DataFrame:
    """Task-level rankings (models ranked by mean loss over series) and their concordance inside each equivalence class.

    Inside a series the members of a class rank the models identically, so any W below 1 in a class row is produced by
    averaging over series alone. The row "all measures" gives the usual task-level W for comparison."""
    _check(losses)
    metrics = list(pd.unique(losses["metric"])) if metrics is None else list(metrics)
    models = list(pd.unique(losses["model"])) if models is None else list(models)
    d = losses[losses["metric"].isin(metrics) & losses["model"].isin(models)]
    d = d[np.isfinite(d["loss"].astype(float))]
    mean = d.groupby(["metric", "model"], sort=False)["loss"].mean().unstack("model").reindex(index=metrics, columns=models)
    mean = mean.loc[:, np.isfinite(mean.to_numpy()).all(axis=0)]
    ranks = mean.rank(axis=1, method="average")
    rows = [{"measures": "all measures", "n_measures": len(ranks), "members": ", ".join(ranks.index), "kendall_w": kendalls_w(ranks)}]
    for cls in dict.fromkeys(_class_of(m) for m in ranks.index):
        members = [m for m in ranks.index if _class_of(m) == cls]
        if len(members) >= 2:
            rows.append({"measures": f"{cls} (equivalent inside a series)", "n_measures": len(members),
                         "members": ", ".join(members), "kendall_w": kendalls_w(ranks.loc[members])})
    return pd.DataFrame(rows).set_index("measures")


# ---------------------------------------------------------------- features

def _moving_average(y: np.ndarray, w: int) -> np.ndarray:
    """Centered moving average (2 x w for even w, as in classical decomposition); NaN where the window is incomplete."""
    n = len(y)
    out = np.full(n, np.nan)
    if w < 2 or n < w + 1:
        return out
    if w % 2:
        k = np.ones(w) / w
        h = w // 2
    else:
        k = np.r_[0.5, np.ones(w - 1), 0.5] / w
        h = w // 2
    out[h: n - h] = np.convolve(y, k, mode="valid")
    return out


def series_features(y: Sequence[float], period: int = 1) -> dict[str, float]:
    """Six features of one series. Missing values are dropped first.

    trend, seasonal_strength  1 - Var(remainder) / Var(component + remainder), floored at 0 (Wang, Smith & Hyndman 2006),
                              from a classical additive decomposition with a centered moving-average trend. For period 1
                              the window is about a tenth of the series. seasonal_strength is NaN without three full periods.
    entropy                   Shannon entropy of the normalized periodogram, divided by its maximum (0 = one frequency,
                              1 = white noise)
    x_acf1                    lag-one autocorrelation
    cv                        standard deviation / |mean|
    log_level                 log10 |mean|

    These follow the definitions of the R package tsfeatures but not its estimators (STL, smoothed spectrum). Checked on
    the 111 NN5 weekly series against tsfeatures: x_acf1, cv and log_level are identical; entropy has rank correlation
    0.91 and trend 0.64 with the R values (`tests/test_instability.py`). For results that must match R, pass R's own
    feature table to `association`."""
    x = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    out = dict.fromkeys(FEATURES, float("nan"))
    if n < 8:
        return out
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    if abs(mean) > 0:
        out["cv"] = sd / abs(mean)
        out["log_level"] = float(np.log10(abs(mean)))
    if sd == 0:
        return out
    z = x - mean
    out["x_acf1"] = float((z[1:] * z[:-1]).sum() / (z * z).sum())
    spec = np.abs(np.fft.rfft(z)[1:]) ** 2
    if spec.sum() > 0 and len(spec) > 1:
        p = spec / spec.sum()
        p = p[p > 0]
        out["entropy"] = float(-(p * np.log(p)).sum() / np.log(len(spec)))
    period = int(period) if period and period > 1 else 1
    # With only two periods every seasonal mean rests on two points and the seasonal strength comes out near 1
    # (seen on NN5 weekly: 0.94 against 0.57 from R's STL), so three full periods are required.
    seasonal_ok = period > 1 and n >= 3 * period
    w = period if seasonal_ok else max(3, (n // 10) | 1)
    trend = _moving_average(x, w)
    valid = np.isfinite(trend)
    if valid.sum() < 4:
        return out
    detr = x - trend
    if seasonal_ok:
        idx = np.arange(n) % period
        seas_mean = np.array([np.nanmean(detr[idx == i]) if np.isfinite(detr[idx == i]).any() else 0.0 for i in range(period)])
        seas = seas_mean[idx] - np.nanmean(seas_mean)
    else:
        seas = np.zeros(n)
    rem = (detr - seas)[valid]
    v_rem = float(np.var(rem, ddof=1))
    v_tr = float(np.var((trend + detr - seas)[valid], ddof=1))
    if v_tr > 0:
        out["trend"] = max(0.0, 1.0 - v_rem / v_tr)
    if seasonal_ok:
        v_se = float(np.var(detr[valid], ddof=1))
        if v_se > 0:
            out["seasonal_strength"] = max(0.0, 1.0 - v_rem / v_se)
    return out


def feature_table(values: Sequence[np.ndarray], names: Sequence[str], period: int = 1, drop_last: int = 0) -> pd.DataFrame:
    """Features for every series. `drop_last` removes the forecast horizon so that features describe the training part only."""
    rows = [series_features(v[: len(v) - drop_last] if drop_last else v, period) for v in values]
    return pd.DataFrame(rows, index=pd.Index(list(names), name="series"))[list(FEATURES)]


# ---------------------------------------------------------------- association

def association(outcomes: pd.DataFrame, features: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """Spearman correlation of every feature with disagreement and with regret, inside one task. Columns: outcome, feature, n, rho, p."""
    d = outcomes.join(features, how="inner")
    rows = []
    for outcome in ("disagreement", "regret"):
        for col in features.columns:
            x, y = d[col].to_numpy(float), d[outcome].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < min_n or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
                continue
            r = stats.spearmanr(x[ok], y[ok])
            rows.append({"outcome": outcome, "feature": col, "n": int(ok.sum()), "rho": float(r.statistic), "p": float(r.pvalue)})
    return pd.DataFrame(rows, columns=["outcome", "feature", "n", "rho", "p"])


def pool_over_tasks(corr: pd.DataFrame, min_tasks: int = 5) -> pd.DataFrame:
    """Pools within-task correlations (columns task, outcome, feature, rho). The task correlations are the observations:
    mean Fisher z back-transformed, t interval and t-test across tasks, number of tasks with a positive correlation.

    Series of one task are not independent forecasting problems, so p-values of single tasks must not be multiplied up."""
    rows = []
    for (outcome, feat), g in corr.groupby(["outcome", "feature"], sort=False):
        z = np.arctanh(np.clip(g["rho"].to_numpy(float), -0.999999, 0.999999))
        z = z[np.isfinite(z)]
        n = len(z)
        row = {"outcome": outcome, "feature": feat, "n_tasks": n, "rho": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan,
               "n_positive": int((z > 0).sum()), "min_rho": float(g["rho"].min()), "max_rho": float(g["rho"].max())}
        if n >= min_tasks:
            mean, se = z.mean(), z.std(ddof=1) / np.sqrt(n)
            half = stats.t.ppf(0.975, n - 1) * se
            row.update(rho=float(np.tanh(mean)), lo=float(np.tanh(mean - half)), hi=float(np.tanh(mean + half)),
                       p=float(2 * stats.t.sf(abs(mean / se), n - 1)) if se > 0 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- input builders

def losses_from_wide(table: pd.DataFrame, series_col: str = "series_id", model_col: str = "model",
                     metrics: Sequence[str] | None = None) -> pd.DataFrame:
    """Long loss table from a wide one (one row per series and model, one column per measure).
    `metrics=None` keeps every column that is a known loss measure."""
    cols = [c for c in table.columns if c.lower() in METRIC_FAMILY] if metrics is None else [m for m in metrics if m in table.columns]
    long = table.melt(id_vars=[series_col, model_col], value_vars=cols, var_name="metric", value_name="loss")
    long = long.rename(columns={series_col: "series", model_col: "model"})
    long["loss"] = pd.to_numeric(long["loss"], errors="coerce")
    return long[list(_COLS)]


def losses_from_runs(runs: Mapping[str, str], metrics: Sequence[str] | None = None) -> pd.DataFrame:
    """Long per-series loss table for ONE task. Key: model name; value: run folder holding metrics.xlsx."""
    parts = []
    for model, d in runs.items():
        df = pd.read_excel(Path(d) / "metrics.xlsx", sheet_name="Errors_Per_Series")
        df["model"] = model
        parts.append(losses_from_wide(df, "series_name", "model", metrics))
    return pd.concat(parts, ignore_index=True)
