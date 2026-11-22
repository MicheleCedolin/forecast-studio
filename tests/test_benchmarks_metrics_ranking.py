"""2026-11-22 eklemeleri: M3/M4 kıyas yöntemleri (drift, naive2, comb_shd, dotm, scum), ek metrikler, sıralama modülü."""
import numpy as np
import pandas as pd
import pytest

from tsfstudio import metrics as MT
from tsfstudio import ranking as R
from tsfstudio.engines import EngineJob, PythonEngine
from tsfstudio.engines import py_models as M
from tsfstudio.engines import py_sf as S


def seasonal_series(L=96, m=12, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(L)
    return 100 + 0.5 * t + 20 * np.sin(2 * np.pi * t / m) + rng.normal(0, 1, L)


# ------------------------------------------------------------------ yerleşik kıyaslar
def test_drift_hand_example():
    np.testing.assert_allclose(M.drift(np.array([1.0, 3.0, 5.0]), 3), [7.0, 9.0, 11.0])
    np.testing.assert_allclose(M.drift(np.array([4.0]), 2), [4.0, 4.0])


def test_drift_matches_statsforecast_rwd():
    sm = pytest.importorskip("statsforecast.models")
    y = seasonal_series()
    ref = sm.RandomWalkWithDrift().forecast(y=y, h=10)["mean"]
    np.testing.assert_allclose(M.drift(y, 10), ref, rtol=1e-10)


def test_m4_seasonality_test_and_naive2():
    y = seasonal_series()
    assert M.m4_seasonality_test(y, 12)
    f = M.naive2(y, 24, 12)
    assert f.shape == (24,) and np.isfinite(f).all()
    np.testing.assert_allclose(f[:12], f[12:])                     # yeniden mevsimselleştirilmiş naif: dönemlik tekrar
    assert np.ptp(f) > 10                                          # mevsim deseni geri konmuş
    rng = np.random.default_rng(3)
    w = 50 + np.cumsum(rng.normal(size=80))                        # mevsimsiz rastgele yürüyüş -> düz naif
    assert not M.m4_seasonality_test(w, 12)
    np.testing.assert_allclose(M.naive2(w, 5, 12), np.full(5, w[-1]))
    assert not M.m4_seasonality_test(y[:30], 12)                   # n < 3m


def test_seasonality_test_long_period_is_non_seasonal_like_r_acf_default():
    t = np.arange(400)
    y = 10 + np.sin(2 * np.pi * t / 52)                            # floor(10*log10(400)) = 26 < 52 -> R'da NA -> FALSE
    assert not M.m4_seasonality_test(y, 52)


def test_holt_recovers_a_line_and_damped_flattens():
    y = 5.0 + 2.0 * np.arange(40)
    np.testing.assert_allclose(M.holt_forecast(y, 5), 5.0 + 2.0 * np.arange(40, 45), rtol=1e-4)
    fd = M.holt_forecast(y, 50, damped=True)
    assert np.all(np.diff(fd) > 0) and np.diff(fd)[-1] < np.diff(fd)[0]   # artıyor ama sönümleniyor


def test_comb_shd_shape_and_level():
    y = seasonal_series()
    f = M.comb_shd(y, 12, 12)
    assert f.shape == (12,) and np.isfinite(f).all()
    assert abs(f.mean() - y[-12:].mean()) < 25


@pytest.mark.parametrize("method", ["drift", "naive2", "comb_shd"])
def test_new_native_methods_through_engine(method):
    train = [seasonal_series(seed=i) for i in range(4)]
    res = PythonEngine().run(EngineJob(task="local", method=method, train=train, horizon=6, seasonality=(12.0,)))
    assert res.forecasts.shape == (4, 6) and np.isfinite(res.forecasts).all() and not res.fallbacks


# ------------------------------------------------------------------ statsforecast tabanlı
def _need_sf():
    if not S.sf_available():
        pytest.skip("statsforecast is not installed")


def test_dotm_and_scum_are_listed_and_run():
    _need_sf()
    eng = PythonEngine()
    assert {"dotm", "scum"} <= set(eng.methods()["local"])
    train = [seasonal_series(L=72, seed=i) for i in range(3)]
    for m in ("dotm", "scum"):
        res = eng.run(EngineJob(task="local", method=m, train=train, horizon=6, seasonality=(12.0,)))
        assert res.forecasts.shape == (3, 6) and np.isfinite(res.forecasts).all() and not res.fallbacks, (m, res.notes)


def test_scum_is_median_of_components():
    _need_sf()
    y, h, m = seasonal_series(L=72), 6, 12
    comps = np.vstack([S.forecast_ets(y, h, m), S.forecast_spec("auto_ces", y, h, m, (12.0,), None),
                       S.forecast_arima(y, h, m), S.forecast_spec("dotm", y, h, m, (12.0,), None)])
    np.testing.assert_allclose(S.forecast_scum(y, h, m, (12.0,), {}), np.median(comps, axis=0), rtol=1e-8)


def test_scum_survives_one_failing_component(monkeypatch):
    _need_sf()
    monkeypatch.setattr(S, "forecast_arima", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no arima")))
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        f = S.forecast_scum(seasonal_series(L=72), 6, 12, (12.0,), {})
    assert np.isfinite(f).all() and any("arima" in str(x.message) for x in w)


# ------------------------------------------------------------------ ek metrikler (elle hesap)
F = np.array([[2.0, 4.0, 0.0], [1.0, 1.0, 1.0]])
A = np.array([[1.0, 5.0, 0.0], [0.0, 2.0, 4.0]])


def test_extra_metrics_hand_computed():
    np.testing.assert_allclose(MT.mse(F, A), [(1 + 1 + 0) / 3, (1 + 1 + 9) / 3])
    np.testing.assert_allclose(MT.mape(F, A), [100 * (1 + 0.2) / 2, 100 * (0.5 + 0.75) / 2])      # a=0 noktaları dışarıda
    np.testing.assert_allclose(MT.maape(F, A), [(np.arctan(1) + np.arctan(0.2) + 0) / 3,
                                                (np.pi / 2 + np.arctan(0.5) + np.arctan(0.75)) / 3])
    np.testing.assert_allclose(MT.medae(F, A), [1.0, 1.0])
    np.testing.assert_allclose(MT.me(F, A), [0.0, (1 - 1 - 3) / 3])
    np.testing.assert_allclose(MT.pbias(F, A), [0.0, 100 * (-3) / 6])
    r = MT.rmsle(F, A)
    assert np.isclose(r[0], np.sqrt(((np.log(3) - np.log(2)) ** 2 + (np.log(5) - np.log(6)) ** 2) / 3))
    assert np.isnan(MT.rmsle(np.array([[-1.0, 1.0]]), np.array([[1.0, 1.0]]))[0])
    assert np.isnan(MT.pbias(np.array([[1.0]]), np.array([[0.0]]))[0])


def test_rmsse_hand_computed_and_zero_scale():
    train = [np.array([1.0, 2.0, 4.0]), np.array([3.0, 3.0, 3.0])]       # ölçek: mean([1, 4]) = 2.5 ; 0
    out = MT.rmsse(F, A, train)
    assert np.isclose(out[0], np.sqrt((2 / 3) / 2.5)) and np.isnan(out[1])


def test_error_table_supports_all_metrics_and_default_set_is_unchanged():
    assert MT.DEFAULT_METRICS == ("SMAPE", "mSMAPE", "MASE", "MAE", "RMSE", "WAPE", "nRMSE_Mean", "nRMSE_Range")
    train = [np.arange(1.0, 10.0), np.arange(2.0, 11.0)]
    tab = MT.compute_error_table(F, A, train, 1, metrics=MT.ALL_METRICS)
    assert list(tab.columns) == list(MT.ALL_METRICS) and len(tab) == 2


def test_ui_registry_covers_all_metrics():
    from tsfstudio.app import models_info as MI
    assert set(MT.ALL_METRICS) <= set(MI.METRICS_INFO)


# ------------------------------------------------------------------ sıralama modülü
def test_mean_ranks_and_friedman_match_scipy():
    from scipy import stats
    rng = np.random.default_rng(0)
    e = pd.DataFrame({"a": rng.random(30), "b": rng.random(30) + 0.3, "c": rng.random(30) + 0.6})
    mr = R.mean_ranks(e)
    assert list(mr.index) == ["a", "b", "c"] and np.isclose(mr.sum(), 6.0)
    fr = R.friedman(e)
    ref = stats.friedmanchisquare(e.a, e.b, e.c)
    assert np.isclose(fr.statistic, ref.statistic) and np.isclose(fr.p_value, ref.pvalue) and fr.n_blocks == 30


def test_incomplete_blocks_are_dropped():
    e = pd.DataFrame({"a": [1.0, np.nan, 1.0], "b": [2.0, 2.0, 2.0], "c": [3.0, 3.0, np.inf]})
    assert R.mean_ranks(e).to_dict() == {"a": 1.0, "b": 2.0, "c": 3.0}
    with pytest.raises(ValueError):
        R.mean_ranks(pd.DataFrame({"a": [np.nan], "b": [1.0]}))


def test_nemenyi_critical_values_match_demsar_table():
    # Demšar (2006), Tablo 5(a), α = 0.05: q = 1.960 (k=2), 2.343 (k=3), 2.728 (k=5), 3.164 (k=10)
    for k, q in ((2, 1.960), (3, 2.343), (5, 2.728), (10, 3.164)):
        cd = R.nemenyi_cd(k, 1)
        assert np.isclose(cd / np.sqrt(k * (k + 1) / 6.0), q, atol=2e-3), k


def test_avg_rel_mae_and_owa():
    mae = pd.DataFrame({"bench": [2.0, 4.0], "m": [1.0, 8.0]})
    out = R.avg_rel_mae(mae, "bench")
    assert np.isclose(out["m"], np.exp((np.log(0.5) + np.log(2.0)) / 2)) and np.isclose(out["bench"], 1.0)
    o = R.owa({"naive2": 0.2, "x": 0.1}, {"naive2": 2.0, "x": 3.0})
    assert np.isclose(o["naive2"], 1.0) and np.isclose(o["x"], 0.5 * (0.5 + 1.5))


def test_rank_stability_clear_winner_and_reproducible():
    rng = np.random.default_rng(1)
    e = pd.DataFrame({"best": rng.random(50), "mid": rng.random(50) + 1, "worst": rng.random(50) + 2})
    s = R.rank_stability(e, n_boot=200, top_k=2, seed=7)
    assert s.p_first["best"] == 1.0 and s.p_top_k["worst"] == 0.0 and np.allclose(s.kendall_tau, 1.0)
    s2 = R.rank_stability(e, n_boot=200, top_k=2, seed=7)
    pd.testing.assert_series_equal(s.p_first, s2.p_first)
