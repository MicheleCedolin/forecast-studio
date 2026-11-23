"""Is the incomplete model coverage a property of the benchmark or of the common-support rule?

The study drops a model from a task when one of the task's Primary measures is undefined for it on any series
(common support). This script audits which measure and how many series cause each exclusion, and re-estimates the
global ordering under two alternative rules that keep the model instead:

  baseline       the study's rule: Primary measures as screened, models restricted to the common support
  drop_measure   a Primary measure that is undefined for any model of a task is removed from that task; all models stay
  drop_series    all Primary measures stay; each task is restricted to the series on which every model has every
                 Primary measure defined (lenient towards the models that caused the exclusion: their offending
                 series are removed for everybody)

The baseline is recomputed from the per-series error files and checked against the study's stored mean losses and
family-balanced scores before anything else is reported. All ranking computations are `tsfstudio.reliability`.

Example:
  python tools/coverage_sensitivity.py --errors data/ranking_study/errors \
      --rankings data/ranking_study/ALL_common_support_metric_rankings.csv --out manuscript/analysis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfstudio import reliability as RL  # noqa: E402

GOLDEN = ROOT / "tests" / "golden" / "ranking_reliability" / "competition_family_balanced_task_scores.csv"
VARIANTS = ("baseline", "drop_measure", "drop_series")


def load_errors(folder: Path, task: str) -> pd.DataFrame | None:
    f = folder / f"{task}.csv"
    if f.exists():
        return pd.read_csv(f, low_memory=False)
    parts = sorted((folder / task).glob("*.csv")) if (folder / task).is_dir() else []
    return pd.concat([pd.read_csv(p, low_memory=False) for p in parts], ignore_index=True) if parts else None


def series_cube(err: pd.DataFrame, metrics: list[str]) -> dict[str, pd.DataFrame]:
    """metric -> series x model table of per-series losses (non-finite -> NaN)."""
    out = {}
    for m in metrics:
        t = err.pivot(index="series_id", columns="model", values=m).apply(pd.to_numeric, errors="coerce")
        out[m] = t.where(np.isfinite(t))
    return out


def mean_losses(task: str, cube: dict[str, pd.DataFrame], rows: pd.Index | None = None) -> pd.DataFrame:
    """Long table task, metric, model, loss. A model's loss is NaN if the measure is undefined on any series used."""
    recs = []
    for m, t in cube.items():
        t = t if rows is None else t.loc[rows]
        for model, v in t.mean(axis=0, skipna=False).items():
            recs.append({"task": task, "metric": m, "model": model, "loss": v})
    return pd.DataFrame(recs)


def analyse(losses: pd.DataFrame, n_boot: int, n_perm: int) -> dict:
    ranks = RL.common_support_ranks(losses)
    scores = RL.family_balanced_scores(ranks)
    bt = RL.bootstrap_bradley_terry(scores, n_boot=n_boot, top_k=3, seed=1)
    sm = RL.skillings_mack(scores, n_perm=n_perm, seed=1)
    complete = scores.dropna(axis=0)
    fr = stats.friedmanchisquare(*[complete[c] for c in complete.columns]) if len(complete) >= 3 else None
    return {"ranks": ranks, "scores": scores, "bt": bt, "sm": sm, "agreement": RL.agreement_by_task(ranks),
            "friedman": fr, "n_complete_tasks": len(complete)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--errors", type=Path, required=True, help="folder with <task>.csv per-series error files (or <task>/ parts)")
    ap.add_argument("--rankings", type=Path, required=True, help="ALL_common_support_metric_rankings.csv")
    ap.add_argument("--golden", type=Path, default=GOLDEN, help="stored family-balanced scores of the study (baseline check)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--perm", type=int, default=20000)
    a = ap.parse_args()

    rk = pd.read_csv(a.rankings)
    rk = rk[rk["dataset_group"] != "Domain-specific"]          # the 15 competition tasks
    group = rk.groupby("dataset_key")["dataset_group"].first()
    losses = {v: [] for v in VARIANTS}
    audit, design = [], []
    max_rel = 0.0
    for task in sorted(rk["dataset_key"].unique()):
        g = rk[rk["dataset_key"] == task]
        primary, common = list(pd.unique(g["metric"])), set(g["model"])
        err = load_errors(a.errors, task.lower())
        if err is None:
            raise SystemExit(f"no error file for {task}")
        cube = series_cube(err, primary)
        n_series, models = err["series_id"].nunique(), list(pd.unique(err["model"]))
        base = mean_losses(task, cube)

        # baseline check 1: the common support and the mean losses stored by the study
        mine = base.dropna(subset=["loss"]).groupby("model")["metric"].nunique()
        if set(mine[mine == len(primary)].index) != common:
            raise SystemExit(f"{task}: common support differs from the study's")
        chk = base.merge(g[["metric", "model", "mean_loss"]], on=["metric", "model"])
        max_rel = max(max_rel, float((chk["loss"] / chk["mean_loss"] - 1).abs().max()))

        # audit: which measure removes which model, and on how many series
        neg = err.assign(_neg=err["has_negative_forecast"].astype(str).eq("True")).groupby("model")["_neg"].sum()
        blocking = []
        for m in primary:
            bad = cube[m].isna().sum(axis=0)
            for model, nb in bad[bad > 0].items():
                audit.append({"task": task, "model": model, "measure": m, "series_undefined": int(nb), "n_series": n_series,
                              "share_undefined": nb / n_series, "series_with_negative_forecast": int(neg.get(model, 0))})
            if (bad > 0).any():
                blocking.append(m)
        ok_series = pd.concat([cube[m].notna().all(axis=1) for m in primary], axis=1).all(axis=1)
        keep = ok_series.index[ok_series]

        losses["baseline"].append(base)
        losses["drop_measure"].append(base[~base["metric"].isin(blocking)])
        losses["drop_series"].append(mean_losses(task, cube, keep))
        design.append({"task": task, "n_series": n_series, "models_in_file": len(models), "primary": len(primary),
                       "baseline_models": len(common), "measures_removed": " ".join(blocking) or "-",
                       "drop_measure_primary": len(primary) - len(blocking), "drop_series_kept": len(keep),
                       "drop_series_removed": n_series - len(keep)})

    res = {v: analyse(pd.concat(losses[v], ignore_index=True), a.boot, a.perm) for v in VARIANTS}

    # baseline check 2: the study's stored scores and Table 4
    gold = pd.read_csv(a.golden).pivot(index="task", columns="model", values="score")
    mine = res["baseline"]["scores"].reindex(index=gold.index, columns=gold.columns)
    if not (mine.notna() == gold.notna()).all().all():
        raise SystemExit("baseline: observed cells differ from the study's stored scores")
    score_diff = float((mine - gold).abs().max().max())

    a.out.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(audit)
    audit.to_csv(a.out / "coverage_exclusion_audit.csv", index=False)
    design = pd.DataFrame(design).set_index("task")
    design.to_csv(a.out / "coverage_sensitivity_design.csv")
    order = res["baseline"]["bt"].index
    wide = pd.concat({v: res[v]["bt"].reindex(order)[["coverage", "bt_rank", "log_ability", "lo", "hi", "p_first", "p_top_k"]]
                      for v in VARIANTS}, axis=1)
    wide.to_csv(a.out / "coverage_sensitivity_orderings.csv")
    pd.concat({v: res[v]["agreement"] for v in VARIANTS}, names=["variant"]).to_csv(a.out / "coverage_sensitivity_agreement.csv")

    pd.concat({v: res[v]["scores"].stack() for v in VARIANTS}, axis=1).rename_axis(["task", "model"]).to_csv(
        a.out / "coverage_sensitivity_scores.csv")

    # how do the models score on the tasks they were excluded from? (drop_measure scores, split by baseline membership)
    was_in = res["baseline"]["scores"].reindex_like(res["drop_measure"]["scores"]).notna()
    dm = res["drop_measure"]["scores"]
    restored = pd.DataFrame({"tasks_kept": was_in.sum(), "score_on_kept": dm.where(was_in).mean(),
                             "tasks_restored": (dm.notna() & ~was_in).sum(), "score_on_restored": dm.where(~was_in).mean()}).reindex(order)
    restored.to_csv(a.out / "coverage_sensitivity_restored_tasks.csv")

    # leave-one-competition-out under each rule
    loco = []
    for v in VARIANTS:
        s = res[v]["scores"]
        full = RL.fit_bradley_terry(s)
        for comp in sorted(group.unique()):
            sub = s.loc[[t for t in s.index if group[t] != comp]].dropna(axis=1, how="all")
            fit = RL.fit_bradley_terry(sub)
            rho = stats.spearmanr(full.reindex(fit.index).rank(ascending=False), fit.rank(ascending=False)).statistic
            loco.append({"variant": v, "omitted": comp, "tasks_left": len(sub), "spearman_vs_full": rho,
                         "top3": ", ".join(fit.index[:3]), "leader": fit.index[0]})
    loco = pd.DataFrame(loco)
    loco.to_csv(a.out / "coverage_sensitivity_leave_one_competition_out.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    f3 = lambda x: f"{x:.3f}"  # noqa: E731
    print(f"baseline check: max relative difference of mean losses = {max_rel:.1e}; max |score - stored score| = {score_diff:.1e}")
    print("\n=== design ===\n" + design.to_string())
    print(f"\n=== exclusion audit: {len(audit)} model-measure cells in {audit.groupby(['task', 'model']).ngroups} model-task pairs ===")
    print("measures involved:", audit["measure"].value_counts().to_dict())
    print("undefined series per excluded pair: median %.0f, max %d; share of the task's series: median %.2f%%, max %.2f%%" % (
        audit["series_undefined"].median(), audit["series_undefined"].max(),
        100 * audit["share_undefined"].median(), 100 * audit["share_undefined"].max()))
    print("pairs excluded for 1 series: %d, for <= 3 series: %d" % ((audit["series_undefined"] == 1).sum(), (audit["series_undefined"] <= 3).sum()))
    print("all undefined cells have series_undefined <= series_with_negative_forecast:",
          bool((audit["series_undefined"] <= audit["series_with_negative_forecast"]).all()))
    print(audit.to_string(index=False, float_format=f3))
    for v in VARIANTS:
        r = res[v]
        print(f"\n=== {v}: {int(r['scores'].notna().sum().sum())} model-task cells, {r['n_complete_tasks']} complete tasks ===")
        print(r["bt"].to_string(float_format=f3))
        print(f"Skillings-Mack {r['sm'].statistic:.2f}, permutation p = {r['sm'].p_value:.1e} ({r['sm'].n_tasks} tasks)")
        if r["friedman"] is not None:
            print(f"Friedman on the {r['n_complete_tasks']} complete tasks: {r['friedman'].statistic:.2f}, asymptotic p = {r['friedman'].pvalue:.1e}")
        ag = r["agreement"]
        print(f"Kendall W: median {ag['kendall_w'].median():.3f}, min {ag['kendall_w'].min():.3f} ({ag['kendall_w'].idxmin()}); "
              f"mean regret: median {ag['mean_regret'].median():.3f}, max {ag['mean_regret'].max():.3f}")
        if v != "baseline":
            b = res["baseline"]["bt"]["log_ability"]
            rho = stats.spearmanr(b.rank(ascending=False), r["bt"]["log_ability"].reindex(b.index).rank(ascending=False)).statistic
            print(f"Spearman with the baseline ordering: {rho:.3f}")
    ag = pd.concat({v: res[v]["agreement"][["n_metrics", "n_models", "kendall_w", "mean_regret", "max_regret"]] for v in VARIANTS}, axis=1)
    print("\n=== agreement by task ===\n" + ag.sort_index().to_string(float_format=f3))
    print("\n=== drop_measure scores on the tasks a model kept / was restored to ===\n" + restored.to_string(float_format=f3))
    print("\n=== leave one competition out ===\n" + loco.to_string(index=False, float_format=f3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
