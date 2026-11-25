"""R engine, added methods (r/lib/extra_models.R): registration is checked without R; the forecasts are compared with
the C++ engine when both R (with the forecast package) and tsfcore are available on this machine."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from tsfstudio.app import models_info as mi
from tsfstudio.engines.base import EngineJob
from tsfstudio.engines.r_engine import REngine

R_FILE = Path(mi.__file__).resolve().parents[1] / "r" / "lib" / "extra_models.R"


def _r_vector(name: str, text: str) -> list[str]:
    body = re.search(name + r"\s*<-\s*c\((.*?)\)", text, re.S).group(1)
    return re.findall(r'"([^"]+)"', body)


def test_python_and_r_tables_agree():
    text = R_FILE.read_text(encoding="utf-8")
    assert tuple(_r_vector("extra_methods", text)) == REngine.EXTRA_LOCAL
    body = re.search(r"extra_packages\s*<-\s*list\((.*?)\)\n", text, re.S).group(1)      # one line: name = "pkg" | c("a", "b")
    pk = {k: tuple(re.findall(r'"([^"]+)"', v)) for k, v in re.findall(r'(\w+)\s*=\s*(c\([^)]*\)|"[^"]+")', body)}
    assert pk == REngine.PACKAGES
    for m in REngine.EXTRA_LOCAL:
        assert f"get_{m}_forecasts <- function(time_series, forecast_horizon, params = list())" in text


def test_every_added_method_is_wired_for_the_interface():
    for m in REngine.EXTRA_LOCAL:
        info = mi.MODELS[m]
        assert info["task"] == "local"
        assert info["engines"]["r"] == mi.R_NOTES[m]
        for p in info["params"]:           # a parameter is offered for R only when the R function honours it
            assert ("r" in p["engines"]) == (p["key"] in mi.R_PARAMS.get(m, set()))
    assert set(mi.R_PARAMS) <= set(REngine.EXTRA_LOCAL)


def test_methods_leaves_out_what_the_r_library_lacks(monkeypatch):
    eng = REngine(rscript="Rscript-for-test")
    monkeypatch.setitem(REngine._installed, "Rscript-for-test", frozenset({"forecast", "thief"}))
    local = eng.methods()["local"]
    assert "thief" in local and "stl_ets" in local and "mapa" not in local and "auto_ces" not in local
    monkeypatch.setitem(REngine._installed, "Rscript-for-test", None)      # R could not be asked: offer everything
    assert eng.methods()["local"] == REngine.LOCAL_METHODS


def _live_engines():
    r = REngine()
    ok = r.available()[0] and "forecast" in (r._packages() or ())
    try:
        from tsfstudio.engines.cpp_engine import CppEngine
        cpp = CppEngine()
        ok = ok and cpp.available()[0]
    except Exception:  # noqa: BLE001
        return None
    return (r, cpp) if ok else None


def _series(n_series: int, n: int, m: int, seed: int = 3) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return [100 + 0.3 * t + 12 * np.sin(2 * np.pi * (t + k) / m) + rng.normal(0, 4, n) for k in range(n_series)]


# methods whose C++ version is a port of the same R call: forecasts must agree to rounding
EXACT = [("naive", 1e-12), ("drift", 1e-12), ("historic_average", 1e-12), ("naive2", 1e-10), ("holt_ets", 1e-8),
         ("holt_winters", 1e-8), ("stl_ets", 1e-8), ("stl_arima", 1e-5), ("bagged_ets", 1e-8), ("thief", 1e-8),
         ("mapa", 1e-8), ("tslm", 1e-12), ("struct_ts", 1e-10), ("nnetar", 1e-10)]


@pytest.mark.parametrize("method,tol", EXACT)
def test_r_method_equals_the_cpp_port(method, tol):
    live = _live_engines()
    if live is None:
        pytest.skip("needs R with the forecast package and the C++ library")
    r, cpp = live
    if method not in r.methods()["local"]:
        pytest.skip(f"the R package for {method} is not installed")
    train = _series(4, 96, 12)
    params = {"n_boot": 10} if method == "bagged_ets" else {}
    out = {}
    for eng in (r, cpp):
        res = eng.run(EngineJob(task="local", method=method, train=train, horizon=6, seasonality=[12.0], params=params))
        assert not res.fallbacks, res.fallbacks
        out[eng.name] = res.forecasts
    rel = np.abs(out["r"] - out["cpp"]) / np.maximum(np.abs(out["r"]), np.abs(out["cpp"]))
    assert rel.max() <= tol, f"{method}: R and C++ differ by {rel.max():.2e}"


def test_bats_equals_the_cpp_port_where_the_same_model_is_chosen():
    """bats: the C++ fit of each candidate equals R's to about 1e-11, and the forecasts to rounding when both engines
    keep the same candidate. The first of the four test series (a noisy sine fitted without a season) is a known
    exception, kept here on purpose: auto.arima on the residuals fits an ARMA(2, 2) that is nearly non-invertible (MA
    root of modulus 1.0002, rejected below 1.01), and the 5e-11 difference in the residuals lets the C++ search accept
    it (C++ adds ARMA(2, 2) errors, R does not; forecasts differ by up to 10%). M3 monthly T295 is the same case the
    other way round (cpp/README.md, row 8c). The other series must agree."""
    live = _live_engines()
    if live is None:
        pytest.skip("needs R with the forecast package and the C++ library")
    train = _series(4, 96, 12)
    out = {}
    for eng in live:
        res = eng.run(EngineJob(task="local", method="bats", train=train, horizon=6, seasonality=[12.0]))
        assert not res.fallbacks, res.fallbacks
        out[eng.name] = res.forecasts
    rel = (np.abs(out["r"] - out["cpp"]) / np.maximum(np.abs(out["r"]), np.abs(out["cpp"]))).max(axis=1)
    assert rel[1:].max() <= 1e-8, f"bats: R and C++ differ by {rel[1:].max():.2e} on a series outside the known exception"


def test_dshw_equals_the_cpp_port():
    """dshw needs two seasonal periods, one a multiple of the other, and positive data: with periods 4 and 12 the R
    call and the C++ port agree to rounding; with one period both engines give every series the fallback."""
    live = _live_engines()
    if live is None:
        pytest.skip("needs R with the forecast package and the C++ library")
    train = _series(3, 120, 12)
    out = {}
    for eng in live:
        res = eng.run(EngineJob(task="local", method="dshw", train=train, horizon=8, seasonality=[4.0, 12.0]))
        assert not res.fallbacks, res.fallbacks
        out[eng.name] = res.forecasts
        one = eng.run(EngineJob(task="local", method="dshw", train=train, horizon=8, seasonality=[12.0]))
        assert len(one.fallbacks) == len(train)
    rel = np.abs(out["r"] - out["cpp"]) / np.maximum(np.abs(out["r"]), np.abs(out["cpp"]))
    assert rel.max() <= 1e-10, f"dshw: R and C++ differ by {rel.max():.2e}"


@pytest.mark.parametrize("method", ["comb_shd", "comb_eat", "comb_eat_median", "auto_ces", "croston_classic", "croston_sba",
                                    "croston_optimized", "tsb", "imapa", "dotm", "scum", "stm", "otm", "dstm"])
def test_r_library_method_runs(method):
    live = _live_engines()
    if live is None:
        pytest.skip("needs R with the forecast package and the C++ library")
    r, _ = live
    if method not in r.methods()["local"]:
        pytest.skip(f"the R package for {method} is not installed")
    train = _series(3, 96, 12)
    res = r.run(EngineJob(task="local", method=method, train=train, horizon=6, seasonality=[12.0]))
    assert res.forecasts.shape == (3, 6) and np.isfinite(res.forecasts).all() and not res.fallbacks
