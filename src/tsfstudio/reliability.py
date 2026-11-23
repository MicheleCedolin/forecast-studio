"""Task-level ranking reliability under incomplete model coverage (2026-11-22).

`ranking.py` works inside one task (blocks = series, complete blocks only). This module works across tasks
(blocks = dataset-frequency tasks) and keeps tasks on which some models are missing. It implements the framework of the
manuscript "How Reliable Are Forecasting-Model Rankings?" and is validated against that study's stored outputs
(`tests/test_reliability.py`).

Input: a long table `losses` with columns task, metric, model, loss (mean loss of the model on the task; smaller = better).

  common_support_ranks     per task: keep the models with a finite loss under every metric, rank them per metric
  family_balanced_scores   normalized ranks averaged within metric family, then across families -> task x model table
  fit_bradley_terry        task-balanced Bradley-Terry log-abilities (each task has total pair weight 1; ties = half wins)
  bootstrap_bradley_terry  resamples tasks: interval for the log-ability, P(first), P(top k)
  skillings_mack           Skillings-Mack statistic (Friedman for incomplete blocks) with within-task permutation p-value
  kendalls_w               concordance of the metric rankings of one task (tie-corrected)
  selection_regret         normalized loss of choosing with metric a and evaluating with metric b, all ordered pairs
  agreement_by_task        Kendall's W and regret summary for every task
  leave_one_family_out     Bradley-Terry ordering after dropping one metric family at a time

References: Bradley & Terry (1952); Skillings & Mack (1981); Chatfield & Mander (2009); Kendall & Smith (1939).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

# Metric -> broad family. Keys are lower-case; both this package's names and the manuscript's names are covered.
# ME and PBias are signed bias measures, not losses, and are deliberately absent.
METRIC_FAMILY: dict[str, str] = {
    "mae": "absolute", "medae": "absolute", "mdae": "absolute",
    "mse": "squared", "rmse": "squared",
    "wape": "normalized", "nmae": "normalized", "nrmse": "normalized", "nrmse_mean": "normalized", "nrmse_range": "normalized",
    "mase": "scaled", "rmsse": "scaled",
    "smape": "percentage", "msmape": "percentage", "mape": "percentage", "mdape": "percentage", "rmspe": "percentage",
    "rmdspe": "percentage", "maape": "percentage",
    "rmsle": "logarithmic", "msle": "logarithmic",
}
_COLS = ("task", "metric", "model", "loss")


def _family_of(metric: str, families: Mapping[str, str] | None) -> str:
    if families is not None:
        if metric in families:
            return families[metric]
        raise ValueError(f"metric '{metric}' has no family in the mapping that was passed")
    try:
        return METRIC_FAMILY[metric.lower()]
    except KeyError:
        raise ValueError(f"metric '{metric}' has no known family (it may not be a loss); pass `families`") from None


def common_support_ranks(losses: pd.DataFrame) -> pd.DataFrame:
    """Adds rank, n_models and normalized_rank = (rank - 1) / (n_models - 1) on each task's common model support.

    A model stays in a task only if its loss is finite under every metric present for that task. Ties get average ranks."""
    missing = [c for c in _COLS if c not in losses.columns]
    if missing:
        raise ValueError(f"losses needs the columns {list(_COLS)}; missing: {missing}")
    if losses.duplicated(["task", "metric", "model"]).any():
        raise ValueError("losses has duplicate (task, metric, model) rows")
    parts = []
    for task, g in losses.groupby("task", sort=True):
        wide = g.pivot(index="metric", columns="model", values="loss").astype(float)
        wide = wide.loc[:, np.isfinite(wide.to_numpy()).all(axis=0)]
        k = wide.shape[1]
        if k < 2:
            continue
        ranks = wide.rank(axis=1, method="average")
        long = wide.stack().rename("loss").reset_index()
        long["rank"] = ranks.stack().to_numpy()
        long.insert(0, "task", task)
        long["n_models"] = k
        long["normalized_rank"] = (long["rank"] - 1.0) / (k - 1.0)
        parts.append(long)
    if not parts:
        raise ValueError("No task has at least two models with finite losses under all of its metrics.")
    return pd.concat(parts, ignore_index=True)


def family_balanced_scores(ranks: pd.DataFrame, families: Mapping[str, str] | None = None,
                           omit_family: str | None = None) -> pd.DataFrame:
    """Task x model table of family-balanced scores (0 = best, 1 = worst; NaN = model not observed on the task)."""
    d = ranks.copy()
    d["family"] = [_family_of(m, families) for m in d["metric"]]
    if omit_family is not None:
        d = d[d["family"] != omit_family]
        if d.empty:
            raise ValueError(f"nothing is left after omitting the family '{omit_family}'")
    fam = d.groupby(["task", "model", "family"], as_index=False)["normalized_rank"].mean()
    return fam.groupby(["task", "model"])["normalized_rank"].mean().unstack("model")


def _pair_statistics(scores: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per task: win credit per model (T x K) and comparison weight per pair (T x K x K); each task sums to weight 1."""
    s = scores.to_numpy(float)
    n_tasks, k = s.shape
    wins = np.zeros((n_tasks, k))
    nij = np.zeros((n_tasks, k, k))
    for t in range(n_tasks):
        idx = np.flatnonzero(np.isfinite(s[t]))
        n_pairs = len(idx) * (len(idx) - 1) / 2
        if n_pairs <= 0:
            continue
        w = 1.0 / n_pairs
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                nij[t, i, j] += w
                nij[t, j, i] += w
                if np.isclose(s[t, i], s[t, j]):
                    wins[t, i] += 0.5 * w
                    wins[t, j] += 0.5 * w
                elif s[t, i] < s[t, j]:
                    wins[t, i] += w
                else:
                    wins[t, j] += w
    return wins, nij


def _fit_bt(wins: np.ndarray, nij: np.ndarray, max_iter: int = 5000, tol: float = 1e-11) -> np.ndarray:
    """MM iterations (Hunter 2004 form) for weighted Bradley-Terry; returns log-abilities centered at mean 0.

    A model with no wins would have ability 0 (log = -inf); it is floored at 1e-12 so that the ordering stays defined."""
    k = len(wins)
    p = np.ones(k)
    for _ in range(max_iter):
        ps = p[:, None] + p[None, :]
        denom = np.where(nij > 0, nij / ps, 0.0).sum(axis=1)
        q = np.where(denom > 0, wins / np.where(denom > 0, denom, 1.0), 1e-12)
        q = np.maximum(q, 1e-12)
        q = q / np.exp(np.log(q).mean())
        done = np.max(np.abs(np.log(q / p))) < tol
        p = q
        if done:
            break
    return np.log(p)


def fit_bradley_terry(scores: pd.DataFrame) -> pd.Series:
    """Task-balanced Bradley-Terry log-abilities (higher = better), sorted from best to worst."""
    wins, nij = _pair_statistics(scores)
    return pd.Series(_fit_bt(wins.sum(axis=0), nij.sum(axis=0)), index=scores.columns, name="log_ability").sort_values(ascending=False)


def bootstrap_bradley_terry(scores: pd.DataFrame, n_boot: int = 5000, top_k: int = 3, seed: int = 1) -> pd.DataFrame:
    """Resamples the tasks with replacement and refits. Columns: coverage, mean_score, log_ability, bt_rank, lo, hi, p_first, p_top_k.

    With few tasks the frequencies describe sensitivity to benchmark composition; they are not population probabilities."""
    wins, nij = _pair_statistics(scores)
    n_tasks, k = wins.shape
    base = _fit_bt(wins.sum(axis=0), nij.sum(axis=0))
    rng = np.random.default_rng(seed)
    samples = np.empty((n_boot, k))
    first = np.zeros(k)
    top = np.zeros(k)
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, n_tasks, size=n_tasks), minlength=n_tasks).astype(float)
        ab = _fit_bt(np.tensordot(counts, wins, axes=(0, 0)), np.tensordot(counts, nij, axes=(0, 0)))
        samples[b] = ab
        order = np.argsort(-ab, kind="stable")
        first[order[0]] += 1
        top[order[: min(top_k, k)]] += 1
    out = pd.DataFrame({
        "coverage": scores.notna().sum(axis=0).to_numpy(),
        "mean_score": scores.mean(axis=0).to_numpy(),
        "log_ability": base,
        "bt_rank": rankdata(-base, method="min").astype(int),
        "lo": np.quantile(samples, 0.025, axis=0),
        "hi": np.quantile(samples, 0.975, axis=0),
        "p_first": first / n_boot,
        "p_top_k": top / n_boot,
    }, index=scores.columns)
    out.attrs.update(top_k=min(top_k, k), n_boot=n_boot, n_tasks=n_tasks)
    return out.sort_values("bt_rank")


@dataclass
class SkillingsMackResult:
    statistic: float
    p_value: float          # (exceedances + 1) / (n_perm + 1)
    exceedances: int
    n_perm: int
    n_tasks: int
    n_models: int


def skillings_mack(scores: pd.DataFrame, n_perm: int = 20000, seed: int = 1, chunk: int = 5000) -> SkillingsMackResult:
    """Skillings-Mack test of "no systematic rank differences" on incomplete blocks; equals Friedman when all blocks are complete.

    The null distribution comes from permuting the observed scores within each task (keeps every task's support)."""
    s = scores.to_numpy(float)
    k = s.shape[1]
    comps = []
    present = []
    a_obs = np.zeros(k)
    for row in s:
        idx = np.flatnonzero(np.isfinite(row))
        kd = len(idx)
        if kd < 2:
            continue
        vec = np.sqrt(12.0 / (kd + 1.0)) * (rankdata(row[idx], method="average") - (kd + 1.0) / 2.0)
        a_obs[idx] += vec
        comps.append((idx, vec))
        col = np.zeros(k)
        col[idx] = 1.0
        present.append(col)
    if not comps:
        raise ValueError("No task has at least two models.")
    m = np.column_stack(present)
    cov = -(m @ m.T)
    np.fill_diagonal(cov, 0.0)
    np.fill_diagonal(cov, -cov.sum(axis=0))
    cplus = np.linalg.pinv(cov)
    obs = float(a_obs @ cplus @ a_obs)
    rng = np.random.default_rng(seed)
    ge, remaining = 0, n_perm
    while remaining > 0:
        n = min(chunk, remaining)
        a = np.zeros((n, k))
        for idx, vec in comps:
            a[:, idx] += vec[np.argsort(rng.random((n, len(idx))), axis=1)]
        ge += int(np.sum(np.einsum("bi,ij,bj->b", a, cplus, a) >= obs - 1e-12))
        remaining -= n
    return SkillingsMackResult(obs, (ge + 1) / (n_perm + 1), ge, n_perm, len(comps), k)


def kendalls_w(rank_table: pd.DataFrame) -> float:
    """Kendall's coefficient of concordance. Rows = judges (metrics), columns = models, cells = ranks; tie-corrected."""
    r = rank_table.to_numpy(float)
    m, n = r.shape
    if m < 2 or n < 2:
        return float("nan")
    s = float(((r.sum(axis=0) - m * (n + 1) / 2.0) ** 2).sum())
    t = 0.0
    for row in r:
        _, counts = np.unique(row, return_counts=True)
        t += float((counts ** 3 - counts).sum())
    den = m ** 2 * (n ** 3 - n) - m * t
    return 12.0 * s / den if den > 0 else float("nan")


def selection_regret(loss_table: pd.DataFrame) -> pd.DataFrame:
    """Rows = metrics, columns = models, cells = losses. Returns the metric x metric table R[a, b]: the model chosen by
    metric a (first one on a tie), scored under b as (loss - best) / (worst - best). Diagonal = 0; NaN if b has zero range."""
    x = loss_table.to_numpy(float)
    chosen = np.argmin(x, axis=1)
    lo, hi = x.min(axis=1), x.max(axis=1)
    rng_ = np.where(hi > lo, hi - lo, np.nan)
    reg = (x[:, chosen].T - lo[None, :]) / rng_[None, :]
    return pd.DataFrame(reg, index=loss_table.index, columns=loss_table.index)


def agreement_by_task(ranks: pd.DataFrame) -> pd.DataFrame:
    """Per task: n_metrics, n_models, kendall_w, mean / max regret over ordered pairs a != b, and the zero-regret share."""
    rows = []
    for task, g in ranks.groupby("task", sort=True):
        rk = g.pivot(index="metric", columns="model", values="rank")
        m = rk.shape[0]
        row = {"task": task, "n_metrics": m, "n_models": rk.shape[1], "kendall_w": kendalls_w(rk),
               "mean_regret": np.nan, "max_regret": np.nan, "zero_regret_share": np.nan}
        if m >= 2:
            reg = selection_regret(g.pivot(index="metric", columns="model", values="loss")).to_numpy()
            off = reg[~np.eye(m, dtype=bool)]
            off = off[np.isfinite(off)]
            if off.size:
                row.update(mean_regret=float(off.mean()), max_regret=float(off.max()),
                           zero_regret_share=float(np.mean(np.isclose(off, 0.0))))
        rows.append(row)
    return pd.DataFrame(rows).set_index("task").sort_values("kendall_w")


def leave_one_family_out(ranks: pd.DataFrame, families: Mapping[str, str] | None = None, top_k: int = 3) -> pd.DataFrame:
    """Bradley-Terry ordering after dropping each metric family: Spearman correlation with the full ordering and the top k."""
    full = fit_bradley_terry(family_balanced_scores(ranks, families))
    fams = sorted({_family_of(m, families) for m in ranks["metric"].unique()})
    rows = []
    for fam in fams:
        if len(fams) < 2:
            break
        fit = fit_bradley_terry(family_balanced_scores(ranks, families, omit_family=fam))
        common = full.index.intersection(fit.index)
        rho = spearmanr(full.loc[common].rank(ascending=False), fit.loc[common].rank(ascending=False)).statistic
        rows.append({"omitted_family": fam, "spearman_vs_full": float(rho), "top": ", ".join(map(str, fit.index[:top_k]))})
    return pd.DataFrame(rows, columns=["omitted_family", "spearman_vs_full", "top"]).set_index("omitted_family")


def losses_from_runs(runs: Mapping[tuple[str, str], str], metrics: list[str] | None = None) -> pd.DataFrame:
    """Long loss table from run folders. Key: (task, model); value: run folder holding metrics.xlsx.

    The loss of a model on a task is the mean of the finite per-series values (the same convention as the Summary sheet).
    `metrics=None` uses every metric that has a family in METRIC_FAMILY."""
    rows = []
    for (task, model), d in runs.items():
        df = pd.read_excel(Path(d) / "metrics.xlsx", sheet_name="Errors_Per_Series").set_index("series_name")
        cols = [c for c in df.columns if c.lower() in METRIC_FAMILY] if metrics is None else list(metrics)
        for c in cols:
            if c not in df.columns:
                continue
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            v = v[np.isfinite(v)]
            rows.append({"task": task, "metric": c, "model": model, "loss": float(v.mean()) if v.size else np.nan})
    return pd.DataFrame(rows, columns=list(_COLS))
