"""Series-level version of tools/feature_instability_analysis.py.

For every series, the common-support models of its task are ranked under each Primary measure of that task, using the
series' own losses. Two instability measures follow per series: metric disagreement (1 - Kendall's W) and the mean
cross-metric selection regret. They are related to the features of the same series (tsfeatures, catch22).

Design choices, fixed before any result was seen:
  * Evidence set. Primary measures and common-support models per task are read from the study's own ranking file
    (ALL_common_support_metric_rankings.csv), so the series-level analysis uses the evidence of the task-level one.
  * Equivalent measures. Within one series, MAE, nMAE and MASE differ by a constant factor and rank models identically,
    as do MSE, RMSE, nRMSE and RMSSE. Each group counts as ONE judge (checked on the data, see "equivalence check").
  * Units of inference. Series within a task are not independent replications of a forecasting problem. Correlations are
    therefore computed within each task and the 15 task-level correlations are the observations: pooled value =
    mean of Fisher z (back-transformed), with a t interval and a sign test across tasks.
  * Layers. Six pre-specified characteristics (Holm within outcome) and an exploratory screen of all features (BH).

Feature files have no series identifier; row i is series T(i) in archive order. This alignment is checked by correlating
the feature `mean` with the test-period mean in the error file, against a shuffled alignment.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsfstudio import instability as INST  # noqa: E402

CLASSES = {"MAE": "abs", "nMAE": "abs", "MASE": "abs", "MSE": "sq", "RMSE": "sq", "nRMSE": "sq", "RMSSE": "sq"}
OUTCOMES = {"disagreement": "1 - Kendall's W", "regret": "mean selection regret"}
PRESPECIFIED = ["trend", "seasonal_strength", "entropy", "x_acf1", "cv", "log_level"]
DESIGN = ["log_n_train"]
SKIP = {"time_kl_shift", "time_level_shift", "time_var_shift", "nperiods", "seasonal_period", "peak", "trough", "mean", "var"}
ALIAS = {"N-BEATS": "nbeats", "Pooled Regression": "pooledregression"}


def norm_model(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def kendall_w_rows(r: np.ndarray) -> np.ndarray:
    """r: series x judges x models (ranks, average ties). Tie-corrected W per series."""
    n_s, m, k = r.shape
    s = ((r.sum(axis=1) - m * (k + 1) / 2.0) ** 2).sum(axis=1)
    t = np.zeros(n_s)
    for i in range(n_s):
        for j in range(m):
            _, c = np.unique(r[i, j], return_counts=True)
            t[i] += (c ** 3 - c).sum()
    den = m ** 2 * (k ** 3 - k) - m * t
    return np.where(den > 0, 12.0 * s / np.where(den > 0, den, 1.0), np.nan)


def series_outcomes(err: pd.DataFrame, metrics: list[str], models: list[str]) -> tuple[pd.DataFrame, dict]:
    err = err.assign(_m=err["model"].map(norm_model))
    keep = [norm_model(m) for m in models]
    err = err[err["_m"].isin(keep)]
    judges = INST.judges_of(metrics)        # one judge per equivalence class (RMSE scale for the squared class)
    cube = np.stack([err.pivot(index="series_id", columns="_m", values=m).reindex(columns=keep).to_numpy(float) for m in judges], axis=1)
    ids = err.pivot(index="series_id", columns="_m", values=judges[0]).index
    ok = np.isfinite(cube).all(axis=(1, 2))
    info = {"judges": judges, "n_models": len(keep), "n_series": len(ids), "n_complete": int(ok.sum())}
    # equivalence check on the data: share of series where a dropped measure ranks the models as its representative does
    eq = []
    for m in metrics:
        rep = next(j for j in judges if CLASSES.get(j, j) == CLASSES.get(m, m))
        if rep != m:
            a = err.pivot(index="series_id", columns="_m", values=m).reindex(columns=keep).to_numpy(float)
            b = cube[:, judges.index(rep), :]
            both = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
            same = [(rankdata(np.round(x, 10)) == rankdata(np.round(y, 10))).all() for x, y in zip(a[both], b[both])]
            eq.append(float(np.mean(same)) if same else np.nan)
    info["equivalence_min_share"] = float(np.nanmin(eq)) if eq else float("nan")
    # the computation itself lives in the application package, so the paper and the tool share one implementation
    long = INST.losses_from_wide(err, "series_id", "_m", metrics)
    out = INST.series_instability(long, metrics, keep)
    out.index.name = "series_id"
    return out, info


def natural_ids(n: int) -> list[str]:
    return [f"T{i}" for i in range(1, n + 1)]


def load_features(folder: Path, task: str, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(folder / f"{task}.csv").select_dtypes("number").replace([np.inf, -np.inf], np.nan)
    df.index = natural_ids(len(df))
    return df.add_prefix(prefix)


def pooled(zs: np.ndarray) -> dict:
    """Tasks are the observations: mean Fisher z, t interval, sign test."""
    zs = zs[np.isfinite(zs)]
    n = len(zs)
    if n < 5:
        return {"n_tasks": n, "rho": np.nan, "lo": np.nan, "hi": np.nan, "p_t": np.nan, "n_pos": int((zs > 0).sum()), "p_sign": np.nan}
    mean, se = zs.mean(), zs.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf(0.975, n - 1) * se
    return {"n_tasks": n, "rho": float(np.tanh(mean)), "lo": float(np.tanh(mean - half)), "hi": float(np.tanh(mean + half)),
            "p_t": float(2 * stats.t.sf(abs(mean / se), n - 1)) if se > 0 else np.nan, "n_pos": int((zs > 0).sum()),
            "p_sign": float(stats.binomtest(int((zs > 0).sum()), n, 0.5).pvalue)}


def adjust(p: pd.Series, how: str) -> pd.Series:
    ok = p.dropna().sort_values()
    m = len(ok)
    if how == "holm":
        adj = (ok * (m - np.arange(m))).cummax().clip(upper=1.0)
    else:
        adj = (ok * m / np.arange(1, m + 1))[::-1].cummin()[::-1].clip(upper=1.0)
    return adj.reindex(p.index)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--errors", type=Path, required=True, help="folder with <task>.csv per-series error files (or <task>/ parts)")
    ap.add_argument("--tsfeatures", type=Path, required=True)
    ap.add_argument("--catch22", type=Path, required=True)
    ap.add_argument("--rankings", type=Path, required=True, help="ALL_common_support_metric_rankings.csv")
    ap.add_argument("--agreement", type=Path, required=True, help="task list (manuscript/analysis/task_agreement.csv)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    tasks = list(pd.read_csv(a.agreement)["task"])
    rk = pd.read_csv(a.rankings)
    rk["task"] = rk["dataset_key"].str.lower()
    per_task, info_rows, corr_rows = [], [], []
    for task in tasks:
        g = rk[rk["task"] == task]
        metrics, models = list(pd.unique(g["metric"])), list(pd.unique(g["model"]))
        f = a.errors / f"{task}.csv"
        err = pd.read_csv(f, low_memory=False) if f.exists() else pd.concat(
            [pd.read_csv(p, low_memory=False) for p in sorted((a.errors / task).glob("*.csv"))], ignore_index=True)
        out, info = series_outcomes(err, metrics, models)
        ts = load_features(a.tsfeatures, task, "ts_")
        c22 = load_features(a.catch22, task, "c22_")
        level = ts["ts_mean"].abs()
        conf = pd.DataFrame({"trend": ts["ts_trend"], "seasonal_strength": ts.get("ts_seasonal_strength", np.nan),
                             "entropy": ts["ts_entropy"], "x_acf1": ts["ts_x_acf1"],
                             "cv": np.sqrt(ts["ts_var"]) / level.where(level > 0), "log_level": np.log10(level.where(level > 0))})
        ntrain = err.drop_duplicates("series_id").set_index("series_id")["n_train"].astype(float)
        feats = pd.concat([conf, np.log10(ntrain.where(ntrain > 0)).rename("log_n_train"),
                           ts[[c for c in ts.columns if c[3:] not in SKIP]], c22], axis=1)
        d = out.join(feats, how="inner")
        # alignment check: feature mean (whole series) against test-period mean from the error file
        tm = err.drop_duplicates("series_id").set_index("series_id")["actual_mean"].astype(float).reindex(ts.index)
        okm = (tm > 0) & (level > 0)
        r_al = stats.spearmanr(np.log(tm[okm]), np.log(level[okm])).statistic
        r_sh = stats.spearmanr(np.log(tm[okm]), np.log(level[okm]).sample(frac=1, random_state=1).to_numpy()).statistic
        info.update(task=task, n_metrics=len(metrics), n_judges=len(info["judges"]), n_used=len(d), align_rho=r_al, align_rho_shuffled=r_sh,
                    median_disagreement=float(d["disagreement"].median()), median_regret=float(d["regret"].median()))
        info["judges"] = " ".join(info["judges"])
        info_rows.append(info)
        d.insert(0, "task", task)
        per_task.append(d)
        for outcome in OUTCOMES:
            for col in feats.columns:
                x, y = d[col].to_numpy(float), d[outcome].to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 30 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
                    continue
                rho = stats.spearmanr(x[ok], y[ok]).statistic
                corr_rows.append({"task": task, "outcome": outcome, "feature": col, "n": int(ok.sum()), "rho": float(rho)})

    a.out.mkdir(parents=True, exist_ok=True)
    info = pd.DataFrame(info_rows).set_index("task")
    info.to_csv(a.out / "series_level_task_info.csv")
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(a.out / "series_level_within_task_correlations.csv", index=False)
    pd.concat(per_task).reset_index(names="series_id")[["task", "series_id", "disagreement", "regret"] + PRESPECIFIED + DESIGN].to_csv(
        a.out / "series_level_outcomes.csv", index=False)

    rows = []
    for (outcome, feat), g in corr.groupby(["outcome", "feature"]):
        z = np.arctanh(np.clip(g["rho"].to_numpy(), -0.999999, 0.999999))
        layer = "confirmatory" if feat in PRESPECIFIED else "design" if feat in DESIGN else "exploratory"
        rows.append({"outcome": outcome, "layer": layer, "feature": feat, "min_rho": g["rho"].min(), "max_rho": g["rho"].max(), **pooled(z)})
    res = pd.DataFrame(rows)
    res["p_adj"] = np.nan
    for outcome in OUTCOMES:
        s = (res["outcome"] == outcome) & (res["layer"] == "confirmatory")
        res.loc[s, "p_adj"] = adjust(res.loc[s, "p_t"], "holm")
        s = (res["outcome"] == outcome) & (res["layer"] == "exploratory")
        res.loc[s, "p_adj"] = adjust(res.loc[s, "p_t"], "bh")
    res.to_csv(a.out / "series_level_pooled_correlations.csv", index=False)

    pd.set_option("display.width", 220)
    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    print(info[["n_series", "n_complete", "n_used", "n_models", "n_metrics", "n_judges", "equivalence_min_share", "align_rho",
                "align_rho_shuffled", "median_disagreement", "median_regret"]].to_string(float_format=fmt))
    print("total series used:", int(info["n_used"].sum()))
    cols = ["feature", "n_tasks", "rho", "lo", "hi", "min_rho", "max_rho", "n_pos", "p_t", "p_sign", "p_adj"]
    for outcome, label in OUTCOMES.items():
        print(f"\n=== {label}: within-task Spearman, pooled over tasks ===")
        for layer in ("confirmatory", "design"):
            print(f"\n{layer} (p_adj = Holm):\n" + res[(res.outcome == outcome) & (res.layer == layer)][cols].to_string(index=False, float_format=fmt))
        e = res[(res.outcome == outcome) & (res.layer == "exploratory")].sort_values("p_t")
        print(f"\nexploratory: {len(e)} features, {int((e.p_adj < 0.05).sum())} with BH q < 0.05; top 10 (p_adj = BH q):")
        print(e.head(10)[cols].to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
