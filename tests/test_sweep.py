"""`sweep`: the batch runner for HPC jobs (plan, isolation, time limit, continuing, report, probe)."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsfstudio import sweep as S
from tsfstudio.cli import main
from tsfstudio.tsf.writer import write_tsf

G = Path(__file__).parent / "golden" / "nn5_weekly"


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """Two small weekly files cut from NN5 weekly (not in the catalogue)."""
    full = np.hstack([np.loadtxt(G / "train.csv", delimiter=","), np.loadtxt(G / "actuals.csv", delimiter=",")])
    d = tmp_path_factory.mktemp("sweepdata")
    out = []
    for name, rows in (("small_a", full[:12]), ("small_b", full[12:20])):
        p = d / f"{name}.tsf"
        write_tsf(p, relation=name, attributes=[("series_name", "string")], attr_rows=[(f"T{i + 1}",) for i in range(len(rows))],
                  values=list(rows), frequency="weekly", horizon=8, missing=False, equal_length=True)
        out.append(str(p))
    return out


def _sweep(data, out, methods=("naive", "snaive"), **kw):
    lines = []
    rc = S.sweep(data, list(methods), out, engine="python-baseline", threads=1, echo=lines.append, **kw)
    return rc, lines


def test_plan_presets_and_unknown_names(data):
    plan, notes = S.build_plan(data, ["snaive", "naive", "snaive"], engine="python-baseline")
    assert [(p["dataset"], p["method"]) for p in plan] == [("small_b", "snaive"), ("small_b", "naive"),
                                                           ("small_a", "snaive"), ("small_a", "naive")]   # smallest file first
    assert all(p["task"] == "local" and not p["catalog"] for p in plan)
    with pytest.raises(S.SweepError, match="does not offer"):
        S.build_plan(data, ["no_such_method"], engine="python-baseline")
    with pytest.raises(S.SweepError, match="Unknown dataset"):
        S.build_plan(["no_such_dataset"], ["naive"], engine="python-baseline")
    plan, _ = S.build_plan(data, ["naive", "snaive"], engine="python-baseline", exclude_methods=["naive"], exclude_datasets=["small_a"])
    assert [(p["dataset"], p["method"]) for p in plan] == [("small_b", "snaive")]


def test_sweep_writes_per_series_outputs_times_and_report(data, tmp_path):
    out = tmp_path / "sw"
    rc, lines = _sweep(data, out, methods=("snaive",))
    assert rc == 0 and any("2 done, 0 not done" in ln for ln in lines)
    d = out / "tasks" / "small_a" / "snaive"
    for f in ("forecasts.csv", "test_actuals.csv", "errors_per_series.csv", "summary.csv", "manifest.json", "status.json", "log.txt"):
        assert (d / f).is_file(), f
    assert pd.read_csv(d / "forecasts.csv").shape == (12, 9)
    st = json.loads((d / "status.json").read_text())
    assert st["state"] == "done" and st["isolated"] and st["threads"] == 1 and st["n_series"] == 12
    assert st["engine_sec"] >= 0 and st["wall_sec"] > 0
    if os.name != "nt":                                      # CPU time and peak memory come from `resource` (POSIX)
        assert st["peak_rss_mb"] > 0 and st["cpu_sec"] >= 0

    res = pd.read_csv(out / "report" / "results.csv")
    assert len(res) == 2 and (res["state"] == "done").all()
    assert {"mean_MASE", "median_MASE", "mean_SMAPE", "engine_sec", "cpu_sec", "wall_sec", "peak_rss_mb"} <= set(res.columns)
    per = pd.read_csv(out / "report" / "errors_per_series.csv.gz")
    assert len(per) == 12 + 8 and list(per.columns[:4]) == ["dataset", "task", "method", "series_name"]
    one = per[(per["dataset"] == "small_a") & (per["method"] == "snaive")]
    own = pd.read_csv(d / "errors_per_series.csv")
    assert np.allclose(one["MASE"].to_numpy(), own["MASE"].to_numpy())
    row = res[(res["dataset"] == "small_a") & (res["method"] == "snaive")].iloc[0]
    assert np.isclose(row["mean_MASE"], own["MASE"].mean())
    piv = pd.read_csv(out / "report" / "mean_MASE.csv").set_index("dataset")
    assert np.isclose(piv.loc["small_a", "snaive"], row["mean_MASE"]) and set(piv.columns) == {"snaive"}
    assert (out / "report" / "report.xlsx").is_file() and (out / "sweep_runs.jsonl").is_file()

    rc, lines = _sweep(data, out, methods=("naive", "snaive"))               # continue: only the new method runs
    assert rc == 0 and sum("already done, skipped" in ln for ln in lines) == 2
    assert any("2 done, 0 not done, 2 skipped" in ln for ln in lines)
    assert len(pd.read_csv(out / "report" / "results.csv")) == 4


def test_failed_task_is_recorded_and_the_sweep_goes_on(data, tmp_path):
    """A horizon longer than the series fails in `prepare` for both datasets; both are reported, and a failed task is
    not run again unless asked."""
    out = tmp_path / "sw"
    rc, lines = _sweep(data, out, methods=("naive",), horizon=500)
    assert rc == 1 and sum("FAILED" in ln for ln in lines) == 2
    res = pd.read_csv(out / "report" / "results.csv")
    assert (res["state"] == "failed").all() and res["message"].str.contains("RunError").all()
    rc, lines = _sweep(data, out, methods=("naive",))
    assert rc == 0 and sum("already failed, skipped" in ln for ln in lines) == 2
    rc, lines = _sweep(data, out, methods=("naive",), retry_failed=True)
    assert rc == 0 and (pd.read_csv(out / "report" / "results.csv")["state"] == "done").all()


def test_time_limit_and_crash_do_not_stop_the_sweep(data, tmp_path, monkeypatch):
    out = tmp_path / "sw"
    rc, lines = _sweep(data[:1], out, methods=("naive",), task_timeout=0.001)
    st = json.loads((out / "tasks" / "small_a" / "naive" / "status.json").read_text())
    assert rc == 1 and st["state"] == "timeout" and "time limit" in st["message"]

    if os.name == "nt":
        return
    monkeypatch.setattr(S.sys, "executable", "false")        # the task process dies without writing its state
    rc, lines = _sweep(data[:1], out, methods=("snaive",))
    st = json.loads((out / "tasks" / "small_a" / "snaive" / "status.json").read_text())
    assert rc == 1 and st["state"] == "crashed" and "exit code 1" in st["message"]
    assert set(pd.read_csv(out / "report" / "results.csv")["state"]) == {"timeout", "crashed"}


def test_probe_projects_the_full_time(data, tmp_path):
    out = tmp_path / "sw"
    rc, lines = _sweep(data[:1], out, probe=4)
    assert rc == 0 and not (out / "tasks").exists()
    est = pd.read_csv(out / "probe" / "report" / "estimate.csv")
    assert (est["n_series"] == 4).all() and (est["n_series_total"] == 12).all()
    assert np.allclose(est["est_full_engine_sec"], est["engine_sec"] * 3)
    assert any("Projected engine time" in ln for ln in lines)


def test_cli_dry_run_array_index_and_in_process(data, tmp_path, capsys):
    args = ["sweep", "--out", str(tmp_path / "sw"), "--engine", "python-baseline", "--datasets", ",".join(data), "--methods", "naive,snaive"]
    assert main(args + ["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "[0] small_b: naive, snaive" in out and "[1] small_a: naive, snaive" in out and not (tmp_path / "sw").exists()
    assert main(args + ["--array-index", "1", "--in-process", "--threads", "1"]) == 0
    assert not (tmp_path / "sw" / "report").exists() and not (tmp_path / "sw" / "tasks" / "small_b").exists()
    assert main(args + ["--array-index", "0", "--in-process", "--threads", "1"]) == 0
    assert main(["sweep", "--out", str(tmp_path / "sw"), "--report-only"]) == 0
    res = pd.read_csv(tmp_path / "sw" / "report" / "results.csv")
    assert len(res) == 4 and (res["state"] == "done").all()


def test_cpp_thread_count_is_fixed_by_the_environment(monkeypatch):
    from tsfstudio.engines import cpp_engine as C
    from tsfstudio.engines.base import EngineError
    monkeypatch.delenv("TSFSTUDIO_CPP_THREADS", raising=False)
    assert C.fixed_threads() is None
    monkeypatch.setenv("TSFSTUDIO_CPP_THREADS", "3")
    assert C.fixed_threads() == 3
    monkeypatch.setenv("TSFSTUDIO_CPP_THREADS", "0")
    with pytest.raises(EngineError):
        C.fixed_threads()


def test_parallel_jobs_give_the_same_results_as_one_at_a_time(data, tmp_path):
    rc1, _ = _sweep(data, tmp_path / "seq")
    rc4, lines = _sweep(data, tmp_path / "par", jobs=4)
    assert rc1 == 0 and rc4 == 0 and any("4 task(s) at a time x 1 thread(s)" in ln for ln in lines)
    a = pd.read_csv(tmp_path / "seq" / "report" / "errors_per_series.csv.gz")
    b = pd.read_csv(tmp_path / "par" / "report" / "errors_per_series.csv.gz")
    pd.testing.assert_frame_equal(a, b)
    res = pd.read_csv(tmp_path / "par" / "report" / "results.csv")
    assert len(res) == 4 and (res["state"] == "done").all() and (res["concurrent_tasks"] == 4).all()


def test_a_dataset_is_finished_before_the_next_one_starts(data, tmp_path):
    """With several tasks at a time, the tasks of the second dataset start only after all tasks of the first ended."""
    rc, _ = _sweep(data, tmp_path / "sw", jobs=4)
    assert rc == 0
    st = {(d, m): json.loads((tmp_path / "sw" / "tasks" / d / m / "status.json").read_text())
          for d in ("small_a", "small_b") for m in ("naive", "snaive")}
    first_end = max(st["small_b", m]["finished_utc"] for m in ("naive", "snaive"))      # small_b is the smaller file
    second_start = min(st["small_a", m]["started_utc"] for m in ("naive", "snaive"))
    assert first_end <= second_start
