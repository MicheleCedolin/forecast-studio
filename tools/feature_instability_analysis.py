"""Do time-series characteristics explain where forecasting-model rankings are unstable?

Relates per-task summaries of series features (R packages tsfeatures and Rcatch22, one CSV per task with one row per
series, as written by reference/experiments/feature_experiments.R) to the two task-level instability measures of the
manuscript: metric disagreement (1 - Kendall's W) and mean cross-metric selection regret.

There are 15 tasks, so this is a small-sample, descriptive analysis. Two layers are kept apart on purpose:
  confirmatory   six features and four design covariates fixed in PRESPECIFIED below before any result was seen;
                 Spearman correlation, permutation p-value, Holm correction within each outcome
  exploratory    every feature of both packages (task median); Benjamini-Hochberg q-values across the whole screen

Usage:
  python tools/feature_instability_analysis.py --tsfeatures DIR --catch22 DIR \
         --agreement manuscript/analysis/task_agreement.csv --out manuscript/analysis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

OUTCOMES = {"disagreement": "1 - Kendall's W", "mean_regret": "mean selection regret"}
# Chosen for their link to why accuracy measures can disagree, before looking at any correlation:
#   trend, seasonal_strength, entropy, x_acf1  the standard descriptors of signal structure and forecastability
#   cv                                         within-series variability relative to level (percentage errors, zeros)
#   scale_spread                               spread of series levels across the task (scale-dependent measures)
PRESPECIFIED = ["trend", "seasonal_strength", "entropy", "x_acf1", "cv", "scale_spread"]
DESIGN = ["n_metrics", "n_models", "horizon", "log10_n_series"]
SKIP = {"time_kl_shift", "time_level_shift", "time_var_shift", "nperiods", "seasonal_period", "peak", "trough", "mean", "var"}


def spearman_perm(x: np.ndarray, y: np.ndarray, n_perm: int, rng: np.random.Generator) -> tuple[float, float]:
    """Spearman rho and a two-sided permutation p-value, (exceedances + 1) / (n_perm + 1)."""
    rx, ry = rankdata(x), rankdata(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return float("nan"), float("nan")
    rho = float(np.corrcoef(rx, ry)[0, 1])
    perm = np.array([rng.permutation(ry) for _ in range(n_perm)])
    rxc = (rx - rx.mean()) / rx.std()
    sims = (perm - ry.mean()) / ry.std() @ rxc / len(rx)
    return rho, float((np.sum(np.abs(sims) >= abs(rho) - 1e-12) + 1) / (n_perm + 1))


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m = len(order)
    adj = pd.Series(index=p.index, dtype=float)
    running = 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[k]))
        adj[k] = running
    return adj


def bh(p: pd.Series) -> pd.Series:
    ok = p.dropna().sort_values()
    m = len(ok)
    q = (ok * m / np.arange(1, m + 1))[::-1].cummin()[::-1].clip(upper=1.0)
    return q.reindex(p.index)


def task_summaries(folder: Path, tasks: list[str], prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per task: median of every numeric feature, and the number of series."""
    med, extra = {}, {}
    for t in tasks:
        df = pd.read_csv(folder / f"{t}.csv")
        num = df.select_dtypes("number").replace([np.inf, -np.inf], np.nan)
        med[t] = num.median()
        row = {"n_rows": len(df)}
        if {"mean", "var"} <= set(num.columns):
            level = num["mean"].abs()
            row["cv"] = float((np.sqrt(num["var"]) / level.where(level > 0)).median())
            lg = np.log10(level.where(level > 0)).dropna()
            row["scale_spread"] = float(lg.quantile(0.75) - lg.quantile(0.25))
        extra[t] = row
    m = pd.DataFrame(med).T
    m = m[[c for c in m.columns if c not in SKIP]].add_prefix(prefix)
    return m, pd.DataFrame(extra).T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsfeatures", type=Path, required=True)
    ap.add_argument("--catch22", type=Path, required=True)
    ap.add_argument("--agreement", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--perm", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=20261122)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    ag = pd.read_csv(a.agreement).set_index("task")
    tasks = list(ag.index)
    ag["disagreement"] = 1.0 - ag["kendall_W"]
    ag["log10_n_series"] = np.log10(ag["n_series"])

    ts, ts_extra = task_summaries(a.tsfeatures, tasks, "ts_")
    c22, c22_extra = task_summaries(a.catch22, tasks, "c22_")
    for name, extra in (("tsfeatures", ts_extra), ("catch22", c22_extra)):
        bad = extra["n_rows"].astype(int) != ag["n_series"]
        if bad.any():
            raise SystemExit(f"{name}: row count differs from the number of series for {list(extra.index[bad])}")
    # non-seasonal tasks have no seasonal_strength column: the strength of a seasonal component that is absent is 0
    ts["ts_seasonal_strength"] = ts.get("ts_seasonal_strength", pd.Series(dtype=float)).reindex(tasks).fillna(0.0)

    conf = pd.DataFrame({"trend": ts["ts_trend"], "seasonal_strength": ts["ts_seasonal_strength"], "entropy": ts["ts_entropy"],
                         "x_acf1": ts["ts_x_acf1"], "cv": ts_extra["cv"], "scale_spread": ts_extra["scale_spread"]})[PRESPECIFIED]
    features = pd.concat([conf, ag[DESIGN], ts, c22], axis=1)
    a.out.mkdir(parents=True, exist_ok=True)
    features.join(ag[["kendall_W", "disagreement", "mean_regret"]]).to_csv(a.out / "task_feature_summary.csv")

    rows = []
    for outcome in OUTCOMES:
        y = ag[outcome].to_numpy(float)
        for col in features.columns:
            x = features[col].to_numpy(float)
            ok = np.isfinite(x)
            layer = "confirmatory" if col in PRESPECIFIED else "design" if col in DESIGN else "exploratory"
            if ok.sum() < 8:
                continue
            rho, p = spearman_perm(x[ok], y[ok], a.perm, rng)
            rows.append({"outcome": outcome, "layer": layer, "feature": col, "n_tasks": int(ok.sum()), "spearman": rho, "p_perm": p})
    res = pd.DataFrame(rows)
    res["p_holm"] = np.nan
    res["q_bh"] = np.nan
    for outcome in OUTCOMES:
        for layer in ("confirmatory", "design"):
            sel = (res["outcome"] == outcome) & (res["layer"] == layer)
            res.loc[sel, "p_holm"] = holm(res.loc[sel, "p_perm"])
        sel = (res["outcome"] == outcome) & (res["layer"] == "exploratory")
        res.loc[sel, "q_bh"] = bh(res.loc[sel, "p_perm"])
    res.to_csv(a.out / "feature_instability_correlations.csv", index=False)

    pd.set_option("display.width", 200)
    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    for outcome, label in OUTCOMES.items():
        print(f"\n=== {label} ({len(tasks)} tasks) ===")
        for layer in ("confirmatory", "design"):
            t = res[(res["outcome"] == outcome) & (res["layer"] == layer)][["feature", "spearman", "p_perm", "p_holm"]]
            print(f"\n{layer}:\n" + t.to_string(index=False, float_format=fmt))
        e = res[(res["outcome"] == outcome) & (res["layer"] == "exploratory")].sort_values("p_perm")
        print(f"\nexploratory: {len(e)} features; smallest BH q = {e['q_bh'].min():.3f}; "
              f"{int((e['p_perm'] < 0.05).sum())} with unadjusted p < 0.05 (expected by chance: {0.05 * len(e):.1f}); top 8:")
        print(e.head(8)[["feature", "n_tasks", "spearman", "p_perm", "q_bh"]].to_string(index=False, float_format=fmt))
    print("\nWritten:", a.out / "task_feature_summary.csv", "and", a.out / "feature_instability_correlations.csv")


if __name__ == "__main__":
    main()
