"""Web edition (web/): the page's JavaScript and the WebAssembly build of tsfcore against the Python package and the
study's R forecasts. Node runs tools/web/check.mjs, which uses the same modules as the page (core.js, engine.js,
tsfcore.mjs). Skipped when Node or the WebAssembly build (cpp/build_wasm.sh) is missing.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tsfstudio import metrics as M
from tsfstudio.engines.cpp_engine import CppEngine, forecast_batch, load_lib
from tsfstudio.seasonality import primary_period, seasonality_for
from tsfstudio.tsf import read_tsf

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
G = ROOT / "tests" / "golden" / "nn5_weekly"
SAMPLE = WEB / "samples" / "nn5_weekly.tsf"
# method -> (the study's R forecasts, largest relative difference allowed); the bounds are those of the native library
R_REFERENCE = {"ets": ("forecasts_ets.csv", 1e-10), "tbats": ("forecasts_tbats.csv", 1e-8), "arima": ("forecasts_arima.csv", 1e-4),
               "dhr_arima": ("forecasts_dhr_arima.csv", 1e-6), "pooled_regression": ("forecasts_pooled_regression_lag_65.csv", 1e-10)}
NATIVE = ["naive", "snaive", "drift", "naive2", "ses", "theta", "croston_sba", "dotm", "auto_ces", "mstl", "holt_ets"]

pytestmark = pytest.mark.skipif(shutil.which("node") is None or not (WEB / "tsfcore.wasm").is_file(),
                                reason="needs Node and web/tsfcore.wasm (cpp/build_wasm.sh)")


@pytest.fixture(scope="module")
def web():
    methods = ",".join(dict.fromkeys([*R_REFERENCE, *NATIVE]))
    out = subprocess.run(["node", str(ROOT / "tools" / "web" / "check.mjs"), str(G), str(SAMPLE), methods],
                         capture_output=True, text=True, timeout=900, check=True).stdout
    return json.loads(out)


def _arr(x) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in x], dtype=np.float64)


def test_sample_is_read_and_split_like_the_python_reader(web):
    ds = read_tsf(SAMPLE)
    d = web["data"]
    assert (d["frequency"], d["horizon"], d["n"]) == (ds.frequency, ds.horizon, ds.n_series) == ("weekly", 8, 111)
    assert d["names"] == ds.series_names[:3] and d["lengths"] == ds.lengths.tolist()
    assert np.array_equal(_arr(d["train0"]), ds.values[0][:-8]) and np.array_equal(_arr(d["test0"]), ds.values[0][-8:])
    assert web["periods"] == list(seasonality_for("weekly")) and web["lag"] == 65       # the study's lag for weekly data


def test_table_with_decimal_commas_and_gaps(web):
    assert web["csv"]["names"] == ["a", "b"]
    a, b = (_arr(s) for s in web["csv"]["series"])
    assert np.array_equal(a, [1.5, 2.5, np.nan, 4.5, 5.5], equal_nan=True) and np.array_equal(b, [10, 11, 12])
    f = web["filled"]                              # training part of [NaN, 1, NaN, NaN, 4, 5 | 6, 7]
    assert f["interpolate"] == [1, 1, 2, 3, 4, 5] and f["locf"] == [1, 1, 1, 1, 4, 5] and f["zero"] == [0, 1, 0, 0, 4, 5]


def test_measures_equal_the_python_measures(web):
    ds = read_tsf(SAMPLE)
    train = [v[:-8] for v in ds.values]
    actual = np.vstack([v[-8:] for v in ds.values])
    f = np.loadtxt(G / "forecasts_ets.csv", delimiter=",", ndmin=2)
    seas = seasonality_for("weekly")
    assert primary_period(seas) == 52
    expected = {"SMAPE": M.smape(f, actual), "mSMAPE": M.msmape(f, actual), "MAE": M.mae(f, actual), "RMSE": M.rmse(f, actual),
                "WAPE": M.wape(f, actual), "MASE": M.mase(f, actual, train, seas), "RMSSE": M.rmsse(f, actual, train)}
    for key, want in expected.items():
        got = _arr([r[key] for r in web["perSeries"]])
        np.testing.assert_allclose(got, want, rtol=1e-12, err_msg=key)
        assert web["summary"][key]["mean"] == pytest.approx(np.nanmean(want), rel=1e-12)
        assert web["summary"][key]["median"] == pytest.approx(np.nanmedian(want), rel=1e-12)
        assert web["summary"][key]["excluded"] == int(np.isnan(want).sum())
    # two methods, the second better exactly on the series with MAE <= 15
    better = int((expected["MAE"] <= 15).sum())
    assert web["ranks"]["series"] == 111
    assert web["ranks"]["ranks"]["b"] == pytest.approx((better + 2 * (111 - better)) / 111)


@pytest.mark.parametrize("method", list(R_REFERENCE))
def test_webassembly_core_reproduces_the_r_forecasts(web, method):
    ref, tol = R_REFERENCE[method]
    want = np.loadtxt(G / ref, delimiter=",", ndmin=2)
    got = _arr(web["forecasts"][method]["values"]).reshape(want.shape)
    assert web["forecasts"][method]["fallbacks"] == 0
    assert np.max(np.abs(got - want) / np.abs(want)) < tol


@pytest.mark.skipif(load_lib() is None, reason="native tsfcore library not built")
@pytest.mark.parametrize("method", NATIVE)
def test_webassembly_core_equals_the_native_library(web, method):
    ds = read_tsf(SAMPLE)
    want, flags, _ = forecast_batch(method, [v[:-8] for v in ds.values], 8, seasonality_for("weekly"), threads=1)
    got = _arr(web["forecasts"][method]["values"]).reshape(want.shape)
    assert web["forecasts"][method]["fallbacks"] == int(flags.sum())
    np.testing.assert_allclose(got, want, rtol=1e-9)


def test_method_list_matches_the_engine_and_the_generator():
    text = (WEB / "methods.js").read_text(encoding="utf-8")
    listed = json.loads(text[text.index("["): text.rindex("]") + 1])
    ids = [m["id"] for m in listed]
    assert len(ids) == len(set(ids))
    assert set(ids) <= set(CppEngine.LOCAL_METHODS) | {"pooled_regression"}
    assert {m["id"] for m in listed if m["task"] == "global"} == {"pooled_regression"}
    before = text
    subprocess.run(["python", str(ROOT / "tools" / "web" / "make_methods.py")], check=True, capture_output=True, cwd=ROOT)
    assert (WEB / "methods.js").read_text(encoding="utf-8") == before, "web/methods.js is stale: run tools/web/make_methods.py"
