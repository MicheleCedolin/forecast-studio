"""Koşucu, tahminci ve manifest testleri (gerçek R gerekmez: referans motor kullanılır)."""
import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsfstudio.engines import EngineJob, PythonBaselineEngine
from tsfstudio.engines.registry import ENGINES, engine_status, get_engine
from tsfstudio.estimate import (
    available_memory_bytes, estimate_global_memory, format_duration, global_rows, preflight, runtime_prior)
from tsfstudio.metrics import smape
from tsfstudio.runner import RunConfig, RunError, prepare, run
from tsfstudio.tsf.reader import TSFDataset, TSFHeader, read_tsf
from tsfstudio.tsf.writer import write_tsf


def make_ds(tmp_path, series, frequency="weekly", horizon=4, name="toy"):
    p = tmp_path / f"{name}.tsf"
    write_tsf(p, relation=name, attributes=[("series_name", "string")],
              attr_rows=[(f"T{i + 1}",) for i in range(len(series))], values=series,
              frequency=frequency, horizon=horizon, missing=any(np.isnan(s).any() for s in series),
              equal_length=len({len(s) for s in series}) == 1)
    return read_tsf(p)


def saw(n, period=7, level=10.0, amp=3.0):
    t = np.arange(n)
    return level + amp * np.sin(2 * np.pi * t / period)


# ------------------------------------------------------------------------------ prepare
def test_prepare_split_and_defaults(tmp_path):
    ds = make_ds(tmp_path, [saw(40), saw(30)], frequency="daily", horizon=5)
    prep = prepare(ds, RunConfig(method="snaive"))
    assert prep.horizon == 5 and prep.seasonality == (7.0,) and prep.test.shape == (2, 5)
    assert [len(t) for t in prep.train] == [35, 25]
    np.testing.assert_array_equal(prep.test[0], saw(40)[-5:])


def test_prepare_excludes_short_series_and_reports(tmp_path):
    ds = make_ds(tmp_path, [saw(40), saw(5)], horizon=4)
    prep = prepare(ds, RunConfig(method="naive"))
    assert prep.names == ["T1"] and prep.excluded[0]["name"] == "T2" and "too short" in prep.excluded[0]["reason"]
    assert any("excluded" in w for w in prep.warnings)


def test_missing_policy_error_names_the_series(tmp_path):
    y = saw(40); y[10] = np.nan
    ds = make_ds(tmp_path, [saw(40), y])
    with pytest.raises(RunError, match="T2.*1 missing"):
        prepare(ds, RunConfig(method="naive"))


@pytest.mark.parametrize("policy,expected", [("zero", 0.0), ("locf", None), ("interpolate", None)])
def test_missing_policies_fill_train_only(tmp_path, policy, expected):
    y = np.arange(1.0, 41.0); y[10] = np.nan; y[-1] = np.nan      # biri eğitimde, biri testte
    ds = make_ds(tmp_path, [y], frequency="daily", horizon=4)
    prep = prepare(ds, RunConfig(method="naive", missing_policy=policy))
    assert not np.isnan(prep.train[0]).any()
    assert prep.missing_info["imputed_values"] == 1 and prep.missing_info["test_missing_values"] == 1
    assert np.isnan(prep.test[0, -1])                                # test eksiği korunur
    if policy == "zero": assert prep.train[0][10] == expected
    if policy == "locf": assert prep.train[0][10] == 10.0
    if policy == "interpolate": assert prep.train[0][10] == pytest.approx(11.0)


def test_horizon_override_and_missing_horizon(tmp_path):
    ds = make_ds(tmp_path, [saw(40)], horizon=None)
    with pytest.raises(RunError, match="could not be determined"):
        prepare(ds, RunConfig(method="naive"))
    assert prepare(ds, RunConfig(method="naive", horizon=6)).horizon == 6


# ------------------------------------------------------------------------------ run
def test_run_end_to_end_writes_outputs_and_manifest(tmp_path):
    ds = make_ds(tmp_path, [saw(60, 7), saw(70, 7, level=50)], frequency="daily", horizon=7)
    seen = []
    res = run(ds, RunConfig(engine="python-baseline", method="snaive", output_dir=tmp_path / "out"),
              dataset_id="toy", progress=lambda s, d, t: seen.append(s))
    d = res.run_dir
    for f in ("forecasts.csv", "forecasts.xlsx", "test_actuals.csv", "metrics.xlsx", "fallbacks.csv", "manifest.json"):
        assert (d / f).is_file(), f
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert m["dataset"]["sha256"] and len(m["dataset"]["sha256"]) == 64
    assert m["config"]["effective_horizon"] == 7 and m["engine"]["name"] == "python-baseline"
    assert m["software"]["tsfstudio"] and m["fallbacks"]["count"] == 0
    # mevsimsel veri + SNaive (periyot 7, gürültüsüz) => SMAPE ~ 0
    assert res.forecasts.shape == (2, 7) and np.nanmax(res.per_series["SMAPE"]) < 1e-9
    assert {"prepare", "metrics", "writing", "done"} <= set(seen)
    xl = pd.read_excel(d / "metrics.xlsx", sheet_name=None)
    assert {"Errors_Per_Series", "Summary", "Run_Info"} <= set(xl)
    fc = pd.read_csv(d / "forecasts.csv")
    assert list(fc.columns) == ["series_name"] + [f"h{k}" for k in range(1, 8)]


def test_run_metrics_agree_with_direct_computation(tmp_path):
    rng = np.random.default_rng(0)
    ds = make_ds(tmp_path, [saw(60) + rng.normal(0, .5, 60) for _ in range(5)], frequency="daily", horizon=7)
    res = run(ds, RunConfig(engine="python-baseline", method="naive", output_dir=tmp_path / "o"))
    np.testing.assert_allclose(res.per_series["SMAPE"].to_numpy(), smape(res.forecasts, res.test))


def test_run_unknown_or_unsupported_engine_method(tmp_path):
    ds = make_ds(tmp_path, [saw(40)])
    with pytest.raises(ValueError, match="Unknown engine"):
        get_engine("yok")
    with pytest.raises(RunError, match="does not support"):
        run(ds, RunConfig(engine="python-baseline", method="ets", output_dir=tmp_path / "o"))


def test_run_unavailable_engine_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("TSFSTUDIO_RSCRIPT", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    ds = make_ds(tmp_path, [saw(40)])
    with pytest.raises(RunError, match="is not available"):
        run(ds, RunConfig(engine="r", method="ets", output_dir=tmp_path / "o"))


def test_global_requires_lag(tmp_path):
    class FakeGlobal(PythonBaselineEngine):
        GLOBAL_METHODS = ("pooled_regression",)
    ds = make_ds(tmp_path, [saw(40)])
    with pytest.raises(RunError, match="lag"):
        run(ds, RunConfig(task="global", method="pooled_regression", output_dir=tmp_path / "o"), engine=FakeGlobal())


def test_cancel_propagates(tmp_path):
    class Slow(PythonBaselineEngine):
        def run(self, job, progress=None, cancel=None):
            from tsfstudio.engines import EngineCancelled
            raise EngineCancelled("iptal")
    ds = make_ds(tmp_path, [saw(40)])
    from tsfstudio.engines import EngineCancelled
    with pytest.raises(EngineCancelled):
        run(ds, RunConfig(method="naive", output_dir=tmp_path / "o"), engine=Slow(), cancel=threading.Event())


def test_registry_lists_engines():
    assert {"r", "python-baseline"} <= set(ENGINES)
    st = engine_status()
    assert st["python-baseline"][0] is True


# ------------------------------------------------------------------------------ estimate
def test_global_rows_and_window():
    assert global_rows([100, 50], lag=10) == 90 + 40
    assert global_rows([100, 50], lag=10, max_obs=30) == 20 + 20
    assert global_rows([5], lag=10) == 0


def test_memory_estimate_block_warn_ok():
    big = estimate_global_memory([1941] * 30490, lag=12, method="pooled_regression", available=8 * 1024 ** 3)
    assert big.rows == 30490 * (1941 - 12) and big.verdict == "block" and "max_obs" in big.message
    win = estimate_global_memory([1941] * 30490, lag=12, method="pooled_regression", max_obs=200, available=8 * 1024 ** 3)
    assert win.rows == 30490 * 188 and win.verdict in ("ok", "warn")
    small = estimate_global_memory([105] * 111, lag=65, method="pooled_regression", available=8 * 1024 ** 3)
    assert small.verdict == "ok"
    unk = estimate_global_memory([100], lag=5, method="catboost", available=None)
    assert unk.verdict == "unknown"


def test_available_memory_is_positive_or_none():
    v = available_memory_bytes()
    assert v is None or v > 0


def test_runtime_prior_reproduces_nn5_measurements():
    r = runtime_prior("tbats", n_series=111, mean_length=105)
    assert r.seconds == pytest.approx(11.759 * 60, rel=1e-6) and r.basis == "prior"
    d = runtime_prior("arima", n_series=111, mean_length=735)
    assert d.seconds == pytest.approx(30.026 * 60, rel=1e-6)
    m5 = runtime_prior("tbats", n_series=30490, mean_length=1941)
    assert m5.seconds > 86400 and m5.high > 3 * m5.seconds / 2        # M5 + TBATS (tek çekirdek): günler sürer
    assert runtime_prior("bilinmeyen", 10, 100).seconds is None
    assert format_duration(30) == "30 s" and format_duration(7200).endswith("h")


def test_preflight_local_warns_on_long_run():
    train = [np.ones(1941)] * 3000
    job = EngineJob(task="local", method="tbats", train=train, horizon=28, seasonality=(7.0,))
    rep = preflight(job)
    assert rep.long_running and any("Long-run" in w for w in rep.warnings)


def test_preflight_probe_scales_linearly():
    class Timed(PythonBaselineEngine):
        def run(self, job, progress=None, cancel=None):
            res = super().run(job, progress, cancel)
            res.elapsed_sec = 0.5 * len(job.train)               # seri başına 0,5 sn
            return res
    job = EngineJob(task="local", method="snaive", train=[np.arange(50.0)] * 100, horizon=3, seasonality=(7.0,))
    rep = preflight(job, engine=Timed(), probe=True)
    assert rep.runtime.basis == "probe" and rep.runtime.seconds == pytest.approx(50.0, rel=0.05)


def test_windows_memory_helper_is_inert_off_windows_and_linux_still_reads_memory():
    import os

    from tsfstudio.estimate import _windows_available_memory, available_memory_bytes
    if os.name != "nt":
        assert _windows_available_memory() is None
        assert available_memory_bytes() is None or available_memory_bytes() > 0


def test_integer_rounding_with_intermittent_method_warns_and_can_be_switched_off(tmp_path):
    """Gerçek olay: car_parts (katalogda yuvarlama AÇIK) + croston_sba -> tüm tahminler 0 -> SMAPE=2, WAPE=%100."""
    ds = make_ds(tmp_path, [saw(40), saw(30)], frequency="monthly", horizon=4)
    on = prepare(ds, RunConfig(method="croston_sba", integer_conversion=True))
    assert on.integer_conversion and any("Rounding to integers is ON" in w and "--integer-conversion off" in w for w in on.warnings)
    off = prepare(ds, RunConfig(method="croston_sba", integer_conversion=False))
    assert not off.integer_conversion and not any("Rounding to integers" in w for w in off.warnings)
    other = prepare(ds, RunConfig(method="theta", integer_conversion=True))            # aralıklı olmayan yöntemde uyarı yok
    assert not any("yuvarlama" in w for w in other.warnings)


def test_same_second_runs_get_distinct_directories(tmp_path, monkeypatch):
    """Aynı saniyede başlayan iki koşu aynı klasöre yazıp birbirini ezmemeli."""
    import tsfstudio.runner as R

    monkeypatch.setattr(R, "_run_id", lambda label, cfg: "ayni__python-naive__20260101-000000")
    ds = make_ds(tmp_path, [saw(40), saw(40)], horizon=4)
    cfg = RunConfig(engine="python-baseline", method="naive", output_dir=tmp_path / "runs")
    a = run(ds, cfg, engine=get_engine("python-baseline"))
    b = run(ds, cfg, engine=get_engine("python-baseline"))
    assert a.run_dir != b.run_dir and a.run_dir.is_dir() and b.run_dir.is_dir()
    assert b.run_dir.name == "ayni__python-naive__20260101-000000-2"
    assert (a.run_dir / "manifest.json").is_file() and (b.run_dir / "manifest.json").is_file()
