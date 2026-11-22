"""Gerçek R ile uçtan uca testler. `Rscript` yoksa atlanır; Docker imajında çalıştırılır:
    docker compose run --rm studio pytest tests/test_r_engine_live.py -q

1) Tüm yöntemler küçük sentetik veride çalışır ve doğru biçimde sonuç verir.
2) NN5 haftalık altın karşılaştırma: R işçisinin tahmini, kullanıcının kendi R çıktısı ile aynı olmalı.
   Deterministik modeller (ses, ets, theta, arima, pooled_regression) sıkı toleransla, tbats / dhr_arima daha gevşek
   toleransla karşılaştırılır (paket sürümü farkı); sapma raporlanır.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

from tsfstudio.engines import EngineJob, REngine

pytestmark = pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript yok (testler Docker imajında çalışır)")

G = Path(__file__).parent / "golden" / "nn5_weekly"


@pytest.fixture(scope="module")
def engine():
    e = REngine()
    ok, msg = e.available()
    assert ok, msg
    return e


def synthetic(n=4, length=120, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    return [50 + 5 * np.sin(2 * np.pi * t / 12) + 0.05 * t + rng.normal(0, 1, length) + 10 * i for i in range(n)]


@pytest.mark.parametrize("method", ["ses", "ets", "theta", "tbats", "arima", "dhr_arima", "snaive"])
def test_local_methods_run_on_synthetic(engine, method):
    res = engine.run(EngineJob(task="local", method=method, train=synthetic(), horizon=6, seasonality=(12.0,)))
    assert res.forecasts.shape == (4, 6) and np.isfinite(res.forecasts).all()
    assert res.versions.get("R")


@pytest.mark.parametrize("method", ["pooled_regression", "catboost"])
def test_global_methods_run_on_synthetic(engine, method):
    res = engine.run(EngineJob(task="global", method=method, train=synthetic(), horizon=6, seasonality=(12.0,), lag=12))
    assert res.forecasts.shape == (4, 6) and np.isfinite(res.forecasts).all()


def test_per_series_horizons_pad_with_nan(engine):
    res = engine.run(EngineJob(task="local", method="ets", train=synthetic(3), horizon=6, horizons=[6, 4, 2], seasonality=(12.0,)))
    assert np.isnan(res.forecasts[1, 4:]).all() and np.isfinite(res.forecasts[1, :4]).all()
    assert np.isnan(res.forecasts[2, 2:]).all()


def test_integer_conversion_rounds(engine):
    res = engine.run(EngineJob(task="local", method="ses", train=synthetic(2), horizon=3, seasonality=(12.0,), integer_conversion=True))
    np.testing.assert_array_equal(res.forecasts, np.round(res.forecasts))


GOLD_LOCAL = [("ses", 1e-6), ("ets", 1e-6), ("theta", 1e-6), ("arima", 1e-4), ("tbats", 1e-3), ("dhr_arima", 1e-3)]


@pytest.mark.parametrize("method,rtol", GOLD_LOCAL)
def test_nn5_weekly_matches_user_r_forecasts(engine, method, rtol):
    train = np.loadtxt(G / "train.csv", delimiter=",")
    ref = np.loadtxt(G / f"forecasts_{method}.csv", delimiter=",")
    res = engine.run(EngineJob(task="local", method=method, train=list(train), horizon=8, seasonality=(365.25 / 7,)))
    rel = np.max(np.abs(res.forecasts - ref) / np.maximum(np.abs(ref), 1e-9))
    print(f"[altın] {method}: en büyük bağıl fark {rel:.2e}; yedek={len(res.fallbacks)}; R={res.versions.get('R')}")
    np.testing.assert_allclose(res.forecasts, ref, rtol=rtol)


def test_nn5_weekly_pooled_regression_matches(engine):
    train = np.loadtxt(G / "train.csv", delimiter=",")
    ref = np.loadtxt(G / "forecasts_pooled_regression_lag_65.csv", delimiter=",")
    res = engine.run(EngineJob(task="global", method="pooled_regression", train=list(train), horizon=8,
                               seasonality=(365.25 / 7,), lag=65))
    np.testing.assert_allclose(res.forecasts, ref, rtol=1e-6)
