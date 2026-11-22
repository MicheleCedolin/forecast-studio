"""Run a set of Python-engine methods on the NN5 golden data (tests/golden) and rank them.

    python tools/nn5_benchmark.py --case nn5_weekly --max-series 8 --workers 4      # timing probe
    python tools/nn5_benchmark.py --case nn5_weekly --case nn5_daily --workers 4    # full run

Each method is one normal run (forecasts, metrics.xlsx, manifest) under <out>/<case>/. A timing table and a ranking report
(mean ranks, Friedman, Nemenyi CD, bootstrap P(first)/P(top 3), Kendall tau) are written per case and metric.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tsfstudio import ranking as RK
from tsfstudio.engines.registry import get_engine
from tsfstudio.metrics import ALL_METRICS
from tsfstudio.runner import RunConfig, run
from tsfstudio.tsf.reader import read_tsf
from tsfstudio.tsf.writer import write_tsf

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ["naive", "snaive", "drift", "naive2", "ses", "theta", "comb_shd", "ets", "auto_ces", "dotm", "arima", "scum"]
GLOBAL = ["pooled_regression"]
RANK_METRICS = ["MASE", "SMAPE", "RMSSE"]


def md_table(df: pd.DataFrame) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    rows = [[str(i)] + [f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in r]
            for i, r in zip(df.index, df.itertuples(index=False))]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in rows])


def build_tsf(case: str, out: Path) -> tuple[Path, dict]:
    d = ROOT / "tests" / "golden" / case
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))

    def load(stem):
        p = d / f"{stem}.csv"
        return np.loadtxt(p if p.is_file() else d / f"{stem}.csv.gz", delimiter=",", ndmin=2)

    full = np.hstack([load("train"), load("actuals")])
    p = out / f"{case}.tsf"
    write_tsf(p, relation=case, attributes=[("series_name", "string")], attr_rows=[(f"T{i + 1}",) for i in range(len(full))],
              values=list(full), frequency=meta["frequency"], horizon=int(meta["horizon"]), missing=False, equal_length=True)
    return p, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--max-series", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "runs" / "nn5_benchmark"))
    a = ap.parse_args()
    eng = get_engine("python")
    avail = eng.methods()
    for case in a.case or ["nn5_weekly"]:
        out = Path(a.out) / (case + (f"_first{a.max_series}" if a.max_series else ""))
        out.mkdir(parents=True, exist_ok=True)
        tsf, meta = build_tsf(case, out)
        ds = read_tsf(tsf, max_series=a.max_series)
        timing, runs = [], {}
        for m in a.methods or (LOCAL + GLOBAL):
            task = "global" if m in GLOBAL else "local"
            if m not in avail[task]:
                print(f"[{case}] {m}: not available in this installation, skipped", flush=True)
                continue
            cfg = RunConfig(engine="python", task=task, method=m, lag=int(meta["lag"]) if task == "global" else None,
                            workers=a.workers, max_series=a.max_series, metrics=ALL_METRICS, output_dir=out, force=False)
            t0 = time.time()
            res = run(ds, cfg, dataset_id=case, engine=eng)
            sec = time.time() - t0
            runs[m] = res.run_dir
            s = res.summary.set_index("Metric")
            timing.append({"method": m, "sec": round(sec, 2), "fallbacks": len(res.fallbacks),
                           "MASE_mean": s.loc["MASE", "Mean"], "SMAPE_mean": s.loc["SMAPE", "Mean"]})
            print(f"[{case}] {m:18s} {sec:8.1f} s  fallbacks={len(res.fallbacks):3d}  "
                  f"MASE={s.loc['MASE', 'Mean']:.4f}  SMAPE={s.loc['SMAPE', 'Mean']:.4f}", flush=True)
        tdf = pd.DataFrame(timing)
        tdf.to_csv(out / "timing_and_means.csv", index=False)
        lines = [f"# {case} ({ds.n_series} series, h = {meta['horizon']})", "",
                 md_table(tdf.set_index("method")), ""]
        for metric in RANK_METRICS:
            e = pd.DataFrame({m: pd.read_csv(Path(d) / "errors_per_series.csv").set_index("series_name")[metric]
                              for m, d in runs.items()})
            mr = RK.mean_ranks(e)
            fr = RK.friedman(e)
            cd = RK.nemenyi_cd(fr.n_models, fr.n_blocks)
            st = RK.rank_stability(e, n_boot=a.boot, top_k=3, seed=1)
            tab = pd.DataFrame({"mean_rank": mr, "mean_error": e.mean()[mr.index], "P(first)": st.p_first,
                                "P(top3)": st.p_top_k}).join(st.mean_rank_ci).sort_values("mean_rank")
            tab.to_csv(out / f"ranking_{metric}.csv")
            lines += [f"## {metric}", "",
                      f"Friedman chi2 = {fr.statistic:.2f}, p = {fr.p_value:.2e}, blocks = {fr.n_blocks} (dropped {fr.n_dropped}); "
                      f"Nemenyi CD (0.05) = {cd:.3f}; bootstrap {st.n_boot}: Kendall tau median {np.median(st.kendall_tau):.3f}, "
                      f"5th pct {np.percentile(st.kendall_tau, 5):.3f}", "", md_table(tab), ""]
        (out / "RANKING.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"[{case}] written: {out / 'RANKING.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
