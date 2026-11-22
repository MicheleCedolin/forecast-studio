"""PythonEngine: orkestrasyon, yedek sayımı, çok süreçli çalışma, iptal, statsforecast adaptörleri (sahte modülle)."""
import sys
import threading
import types

import numpy as np
import pytest

from tsfstudio.engines import EngineCancelled, EngineError, EngineJob, PythonEngine
from tsfstudio.engines import py_engine, py_sf


def series(n=5, L=100, seed=0, m=7):
    rng = np.random.default_rng(seed)
    t = np.arange(L)
    return [30 + 4 * i + 5 * np.sin(2 * np.pi * t / m) + 0.05 * t + rng.normal(0, 0.5, L) for i in range(n)]


def job(method="ses", task="local", **kw):
    base = dict(task=task, method=method, train=series(), horizon=7, seasonality=(7.0,))
    base.update(kw)
    return EngineJob(**base)


# ------------------------------------------------------------------ yerleşik yöntemler
@pytest.mark.parametrize("method", ["naive", "snaive", "ses", "theta"])
def test_native_local_methods(method):
    res = PythonEngine().run(job(method))
    assert res.forecasts.shape == (5, 7) and np.isfinite(res.forecasts).all() and not res.fallbacks


def test_global_pooled_regression_and_missing_lag():
    res = PythonEngine().run(job("pooled_regression", task="global", lag=14))
    assert res.forecasts.shape == (5, 7) and np.isfinite(res.forecasts).all()
    with pytest.raises(EngineError, match="lag"):
        PythonEngine().run(job("pooled_regression", task="global"))
    with pytest.raises(EngineError, match="too short"):
        PythonEngine().run(job("pooled_regression", task="global", lag=500))


def test_unavailable_methods_are_not_listed_and_rejected():
    eng = PythonEngine()
    if not py_sf.sf_available():
        assert "ets" not in eng.methods()["local"]
        with pytest.raises(EngineError, match="does not support"):
            eng.run(job("ets"))
    ok, msg = eng.available()
    assert ok and "numpy" in msg


def test_per_series_horizons_and_integer_conversion():
    res = PythonEngine().run(job("ses", horizons=[7, 4, 2, 7, 7], integer_conversion=True))
    assert np.isnan(res.forecasts[1, 4:]).all() and np.isnan(res.forecasts[2, 2:]).all()
    np.testing.assert_array_equal(res.forecasts[0], np.round(res.forecasts[0]))


def test_progress_and_versions():
    seen = []
    res = PythonEngine().run(job("theta"), progress=lambda s, d, t: seen.append((s, d, t)))
    assert seen[0][0] == "forecasting" and seen[-1][0] == "finished"
    assert res.versions["numpy"] and res.versions["python"]


def test_cancel_before_start_in_serial_mode():
    ev = threading.Event(); ev.set()
    with pytest.raises(EngineCancelled):
        PythonEngine().run(job("ses"), cancel=ev)


def test_parallel_workers_give_identical_results():
    train = series(n=120, L=60)
    a = PythonEngine().run(EngineJob(task="local", method="theta", train=train, horizon=5, seasonality=(7.0,), workers=1))
    b = PythonEngine().run(EngineJob(task="local", method="theta", train=train, horizon=5, seasonality=(7.0,), workers=2))
    np.testing.assert_allclose(a.forecasts, b.forecasts)


# ------------------------------------------------------------------ yedek (fallback) semantiği
def test_model_error_falls_back_to_snaive_and_is_counted(monkeypatch):
    real = py_engine.forecast_one

    def flaky(method, y, h, seasonality, params):
        if y[0] > 40:
            raise RuntimeError("uydurma hata")
        return real(method, y, h, seasonality, params)

    monkeypatch.setattr(py_engine, "forecast_one", flaky)
    train = [np.arange(50.0) + 10 * i for i in range(6)]       # y[0] = 0,10,...,50 -> son iki seri (50) hata verir
    res = PythonEngine().run(EngineJob(task="local", method="ses", train=train, horizon=3, seasonality=(7.0,)))
    assert [d["index"] for d in res.fallbacks] == [5] and "uydurma hata" in res.fallbacks[0]["message"]
    np.testing.assert_array_equal(res.forecasts[5], train[5][-7:][:3])      # SNaive (m=7): son mevsimin ilk 3 değeri
    assert any("fallback" in n for n in res.notes) and res.na_replaced == 0


def test_nonfinite_model_output_is_a_fallback_not_zero(monkeypatch):
    monkeypatch.setattr(py_engine, "forecast_one", lambda *a, **k: np.array([np.nan] * 3))
    res = PythonEngine().run(EngineJob(task="local", method="ses", train=[np.arange(30.0)], horizon=3, seasonality=(7.0,)))
    assert len(res.fallbacks) == 1 and "non-finite" in res.fallbacks[0]["message"]
    assert np.isfinite(res.forecasts).all()


# ------------------------------------------------------------------ statsforecast adaptörleri (sahte modül)
@pytest.fixture
def fake_sf(monkeypatch):
    calls = []

    class _Base:
        def __init__(self, **kw): self.kw = kw

        def forecast(self, y, h, X=None, X_future=None, **_):
            calls.append((type(self).__name__, self.kw, X, X_future, np.asarray(y)))
            if self.kw.get("fail"):
                raise RuntimeError("fit başarısız")
            return {"mean": np.full(h, float(np.mean(y)))}

    class AutoETS(_Base):
        def __init__(self, season_length=1, model="ZZZ", fail=False): super().__init__(season_length=season_length, model=model, fail=fail)

    class AutoARIMA(_Base):
        def __init__(self, season_length=1, allowdrift=False, stepwise=True, approximation=False, fail=False):
            super().__init__(season_length=season_length, allowdrift=allowdrift, stepwise=stepwise,
                             approximation=approximation, fail=fail)

    class AutoTBATS(_Base):
        # Gerçek imza (nixtlaverse belgesi): season_length ZORUNLU; int ya da int listesi.
        def __init__(self, season_length, use_boxcox=None, bc_lower_bound=0.0, bc_upper_bound=1.0, use_trend=None,
                     use_damped_trend=None, use_arma_errors=True, alias="AutoTBATS"):
            super().__init__(season_length=season_length, use_boxcox=use_boxcox, use_arma_errors=use_arma_errors)

    # --- C3 sınıfları: __init__ imzaları resmi belgeden (nixtlaverse) ya da teyit edilmiş kaynaklardan
    def _simple(name):
        def __init__(self, alias=name, prediction_intervals=None): _Base.__init__(self)
        return type(name, (_Base,), {"__init__": __init__})

    class TSB(_Base):
        def __init__(self, alpha_d, alpha_p, alias="TSB", prediction_intervals=None):
            super().__init__(alpha_d=alpha_d, alpha_p=alpha_p)

    class AutoTheta(_Base):
        def __init__(self, season_length=1, decomposition_type="multiplicative", model=None, alias="AutoTheta",
                     prediction_intervals=None, distribution="normal"):
            super().__init__(season_length=season_length, decomposition_type=decomposition_type, model=model)

    class AutoCES(_Base):
        def __init__(self, season_length=1, model="Z", alias="CES", prediction_intervals=None, distribution="normal"):
            super().__init__(season_length=season_length, model=model)

    class HoltWinters(_Base):
        def __init__(self, season_length=1, error_type="A", alias="HoltWinters", prediction_intervals=None):
            super().__init__(season_length=season_length, error_type=error_type)

    class MSTL(_Base):
        def __init__(self, season_length, trend_forecaster=None, stl_kwargs=None, alias="MSTL"):
            super().__init__(season_length=season_length)

    class AutoMFLES(_Base):
        def __init__(self, test_size, season_length=None, n_windows=2, config=None, step_size=None, metric="smape",
                     verbose=False, prediction_intervals=None, alias="AutoMFLES"):
            super().__init__(test_size=test_size, season_length=season_length, n_windows=n_windows, metric=metric)

    pkg, mod = types.ModuleType("statsforecast"), types.ModuleType("statsforecast.models")
    pkg.__version__ = "0.0-fake"
    mod.AutoETS, mod.AutoARIMA, mod.AutoTBATS = AutoETS, AutoARIMA, AutoTBATS
    for _n in ("CrostonClassic", "CrostonSBA", "CrostonOptimized", "ADIDA", "IMAPA"):
        setattr(mod, _n, _simple(_n))
    mod.TSB, mod.AutoTheta, mod.AutoCES, mod.HoltWinters, mod.MSTL, mod.AutoMFLES = TSB, AutoTheta, AutoCES, HoltWinters, MSTL, AutoMFLES
    pkg.models = mod
    monkeypatch.setitem(sys.modules, "statsforecast", pkg)
    monkeypatch.setitem(sys.modules, "statsforecast.models", mod)
    py_sf.DROPPED.clear()
    return calls


def test_methods_list_follows_installed_statsforecast(fake_sf):
    ms = PythonEngine().methods()["local"]
    assert {"ets", "arima", "dhr_arima"} <= set(ms)
    assert "tbats" not in ms                    # kullanıcı kararıyla devre dışı (2026-11-03)


def test_ets_ignores_seasonality_above_24_like_r(fake_sf):
    with pytest.warns(py_sf.AdapterNote, match="52 > 24"):
        py_sf.forecast_ets(np.arange(1.0, 60.0), 3, 52)
    py_sf.forecast_ets(np.arange(1.0, 60.0), 3, 12)
    assert fake_sf[0][1]["season_length"] == 1 and fake_sf[1][1]["season_length"] == 12


def test_arima_chain_log_then_raw_then_nonseasonal(fake_sf):
    y = np.arange(1.0, 40.0)
    f = py_sf.forecast_arima(y, 2, 7)
    assert np.allclose(f, np.exp(np.mean(np.log(y))))                        # 1. adım: log dönüşümü + exp geri dönüşümü
    assert fake_sf[0][1]["allowdrift"] is True and fake_sf[0][1]["approximation"] is False   # n=39, m=7
    fake_sf.clear()
    y0 = np.arange(0.0, 40.0)                                                 # sıfır içerir -> log adımı atlanır
    py_sf.forecast_arima(y0, 2, 7)
    assert len(fake_sf) == 1 and fake_sf[0][4].min() == 0.0
    fake_sf.clear()
    py_sf.forecast_arima(np.arange(1.0, 200.0), 2, 13)
    assert fake_sf[0][1]["approximation"] is True                              # n > 150 -> R gibi approximation


def test_arima_all_steps_fail_raises_with_all_messages(fake_sf, monkeypatch):
    import statsforecast.models as m
    orig = m.AutoARIMA
    monkeypatch.setattr(m, "AutoARIMA", lambda **kw: orig(**kw, fail=True) if "fail" in orig.__init__.__code__.co_varnames else orig(**kw))
    with pytest.raises(RuntimeError, match="log:.*raw:.*non-seasonal:"):
        py_sf.forecast_arima(np.arange(1.0, 40.0), 2, 7)


def test_arima_large_season_goes_nonseasonal(fake_sf):
    py_sf.forecast_arima(np.arange(1.0, 400.0), 2, 365)
    assert all(c[1]["season_length"] == 1 for c in fake_sf)


def test_dhr_arima_passes_fourier_regressors(fake_sf):
    y = np.arange(1.0, 51.0)
    py_sf.forecast_dhr_arima(y, 4, (7.0,), K=1)
    name, kw, X, Xf, _ = fake_sf[0]
    assert kw["season_length"] == 1 and X.shape == (50, 2) and Xf.shape == (4, 2)
    fake_sf.clear()
    py_sf.forecast_dhr_arima(y, 4, (7.0, 52.0), K=1)       # çok mevsimli: skaler K her periyoda uygulanır
    assert fake_sf[0][2].shape == (50, 4)


def test_tbats_gets_all_periods_and_unknown_kwargs_are_reported(fake_sf):
    py_sf.forecast_tbats(np.arange(1.0, 30.0), 2, (24.0, 168.0))
    assert fake_sf[0][1]["season_length"] == [24, 168]                       # çok mevsimli: tam sayı listesi
    py_sf.forecast_tbats(np.arange(1.0, 30.0), 2, (52.1786,))
    assert fake_sf[1][1]["season_length"] == 52                              # tek dönem: tam sayıya yuvarlanır
    assert not any(k.startswith("AutoTBATS.") for k in py_sf.DROPPED)        # hiçbir anahtar sessizce atılmadı
    py_sf.make_model(sys.modules["statsforecast.models"].AutoETS, season_length=1, bogus_option=3)
    assert "AutoETS.bogus_option" in py_sf.DROPPED


def test_engine_end_to_end_with_fake_statsforecast_and_failing_series(fake_sf, monkeypatch):
    res = PythonEngine().run(job("ets", train=series(4, 100)))
    assert res.forecasts.shape == (4, 7) and not res.fallbacks
    import statsforecast.models as m
    monkeypatch.setattr(m.AutoETS, "forecast", lambda self, **k: (_ for _ in ()).throw(RuntimeError("kaboom")))
    res = PythonEngine().run(job("ets", train=series(2, 100)))
    assert len(res.fallbacks) == 2 and "kaboom" in res.fallbacks[0]["message"] and np.isfinite(res.forecasts).all()


# ------------------------------------------------------------------ Windows benzeri 'spawn' ile çok süreçli çalışma
def test_cli_run_with_spawn_workers_in_subprocess(tmp_path):
    """Windows'ta çok süreç 'spawn' kullanır: ana modül yeniden içe aktarılır. __main__ koruması ve seçilebilirlik sınanır."""
    import os
    import subprocess
    from pathlib import Path

    from tsfstudio.tsf.writer import write_tsf

    train = series(n=130, L=70)
    p = tmp_path / "toy.tsf"
    write_tsf(p, relation="toy", attributes=[("series_name", "string")], attr_rows=[(f"T{i + 1}",) for i in range(130)],
              values=train, frequency="daily", horizon=7, missing=False, equal_length=True)
    env = dict(os.environ, TSFSTUDIO_MP_START="spawn", PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    cp = subprocess.run([sys.executable, "-m", "tsfstudio", "run", "--file", str(p), "--engine", "python", "--method", "theta",
                         "--workers", "2", "--out", str(tmp_path / "out")], env=env, capture_output=True, text=True, timeout=180)
    assert cp.returncode == 0, cp.stdout[-800:] + cp.stderr[-800:]
    assert "Done ->" in cp.stdout and cp.stdout.count("Done ->") == 1      # işçiler komutu yeniden çalıştırmadı


def test_required_kwarg_dropped_by_version_gives_clear_error(fake_sf):
    """Gerçek hata: gerekli parametre adı sürümde farklıysa 'anahtar tanınmıyor' demeli, sadece 'eksik argüman' değil."""
    with pytest.raises(TypeError, match="does not recognise"):
        py_sf.make_model(sys.modules["statsforecast.models"].AutoTBATS, seasonal_periods=[7])


# ------------------------------------------------------------------ uyarı yakalama ve yedek nedeni görünürlüğü
def test_library_convergence_warnings_are_counted_not_printed(fake_sf, monkeypatch, capsys):
    import warnings

    import statsforecast.models as m

    def noisy(self, y, h, **k):
        warnings.warn("possible convergence problem: minimize gave code 2]", UserWarning)
        return {"mean": np.full(h, float(np.mean(y)))}

    monkeypatch.setattr(m.AutoETS, "forecast", noisy)
    res = PythonEngine().run(job("ets", train=series(3, 100)))
    assert not res.fallbacks
    assert any("3 series" in n and "convergence" in n for n in res.notes)
    assert "convergence" not in capsys.readouterr().err                 # ekranı doldurmaz


def test_fallback_reason_is_in_notes_with_exception_type(fake_sf, monkeypatch):
    import statsforecast.models as m
    monkeypatch.setattr(m.AutoETS, "forecast", lambda self, **k: (_ for _ in ()).throw(ValueError("kaboom")))
    res = PythonEngine().run(job("ets", train=series(2, 100)))
    note = next(n for n in res.notes if "fallback" in n)
    assert "ValueError" in note and "kaboom" in note


def test_golden_reports_fallback_reason_instead_of_hiding_it(fake_sf, monkeypatch):
    """Gerçek olay: TBATS 111/111 yedeğe düştü ama tabloda yalnızca 'fallbacks' sayısı görünüyordu."""
    from tsfstudio.golden import compare, load_case
    import statsforecast.models as m
    monkeypatch.setattr(m.AutoETS, "forecast", lambda self, **k: (_ for _ in ()).throw(RuntimeError("ets yok")))
    df = compare(PythonEngine(), load_case("nn5_weekly"), ["ets"])
    row = df.iloc[0]
    assert row["fallbacks"] == 111 and "ets yok" in row["fb_reason"]


def test_cli_does_not_crash_on_cp1252_pipe(tmp_path):
    """Gerçek olay: `golden ... | Tee-Object` -> stdout cp1252 -> 'Yazıldı' UnicodeEncodeError ile çöküyordu."""
    import os
    import subprocess
    from pathlib import Path

    env = dict(os.environ, PYTHONIOENCODING="cp1252", PYTHONUTF8="0",
               PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    cp = subprocess.run([sys.executable, "-m", "tsfstudio", "golden", "--case", "nn5_weekly", "--methods", "ses",
                         "--csv", str(tmp_path / "g.csv")], env=env, capture_output=True, cwd=tmp_path, timeout=180)
    assert cp.returncode == 0, cp.stderr[-600:].decode("cp1252", "replace")
    assert (tmp_path / "g.csv").is_file() and b"Written" in cp.stdout


def test_tbats_is_not_offered_but_r_reference_row_reports_missing(fake_sf):
    from tsfstudio.golden import compare, load_case
    assert "tbats" not in PythonEngine().methods()["local"]
    with pytest.raises(EngineError, match="does not support"):
        PythonEngine().run(job("tbats"))
    df = compare(PythonEngine(), load_case("nn5_weekly"), ["tbats"])
    assert df.iloc[0]["status"] == "not in engine"



# ------------------------------------------------------------------ C3: aralıklı talep (A) ve ek klasik (B) modeller
NEW_A = ("croston_classic", "croston_sba", "croston_optimized", "tsb", "adida", "imapa")
NEW_B = ("auto_theta", "auto_ces", "holt_winters", "mstl", "auto_mfles")


def intermittent(n=3, L=120, seed=0):
    rng = np.random.default_rng(seed)
    return [np.where(rng.random(L) < 0.3, rng.poisson(4, L) + 1.0, 0.0) for _ in range(n)]


def test_all_new_methods_are_listed_and_run_end_to_end(fake_sf):
    import importlib.util
    new = [m for m in NEW_A + NEW_B if m != "auto_mfles" or importlib.util.find_spec("sklearn") is not None]
    ms = set(PythonEngine().methods()["local"])
    assert set(new) <= ms
    for meth in new:
        res = PythonEngine().run(job(meth, train=series(3, 120), horizon=7, seasonality=(7.0,)))
        assert res.forecasts.shape == (3, 7) and np.isfinite(res.forecasts).all(), meth
        assert not res.fallbacks, (meth, res.fallbacks)


def test_new_methods_absent_without_statsforecast(monkeypatch):
    monkeypatch.setattr(py_sf, "sf_available", lambda: False)
    assert not (set(NEW_A + NEW_B) & set(PythonEngine().methods()["local"]))


@pytest.mark.parametrize("meth", ["croston_classic", "croston_sba", "croston_optimized", "adida", "imapa"])
def test_intermittent_models_take_no_arguments(fake_sf, meth):
    py_engine.forecast_one(meth, intermittent(1)[0], 5, (7.0,), {})
    assert fake_sf[-1][1] == {}


def test_tsb_alphas_default_override_and_validation(fake_sf):
    y = intermittent(1)[0]
    py_engine.forecast_one("tsb", y, 5, (7.0,), {})
    assert fake_sf[-1][1] == {"alpha_d": 0.1, "alpha_p": 0.1}
    py_engine.forecast_one("tsb", y, 5, (7.0,), {"alpha_d": 0.3, "alpha_p": 0.05, "bilinmeyen": 1})
    assert fake_sf[-1][1] == {"alpha_d": 0.3, "alpha_p": 0.05}
    res = PythonEngine().run(job("tsb", train=[y], params={"alpha_d": 1.5}))
    assert len(res.fallbacks) == 1 and "alpha_d" in res.fallbacks[0]["message"]


def test_all_zero_series_forecast_is_zero_without_calling_the_library(fake_sf):
    res = PythonEngine().run(job("croston_sba", train=[np.zeros(60)]))
    assert not res.fallbacks and (res.forecasts == 0).all() and not fake_sf


def test_auto_theta_seasonality_rule_and_additive_switch(fake_sf):
    y = np.arange(1.0, 101.0)
    py_engine.forecast_one("auto_theta", y, 3, (7.0,), {})
    assert fake_sf[-1][1]["season_length"] == 7 and fake_sf[-1][1]["decomposition_type"] == "multiplicative"
    with pytest.warns(py_sf.AdapterNote, match="at least 2 periods"):
        py_engine.forecast_one("auto_theta", np.arange(1.0, 11.0), 3, (7.0,), {})
    assert fake_sf[-1][1]["season_length"] == 1
    with pytest.warns(py_sf.AdapterNote, match="additive"):
        py_engine.forecast_one("auto_theta", np.arange(0.0, 100.0), 3, (7.0,), {})
    assert fake_sf[-1][1]["decomposition_type"] == "additive"


@pytest.mark.parametrize("meth", ["auto_ces", "holt_winters"])
def test_max_season_rule_is_a_parameter(fake_sf, meth):
    y = np.arange(1.0, 400.0)
    with pytest.warns(py_sf.AdapterNote, match="max_season"):
        py_engine.forecast_one(meth, y, 3, (52.0,), {})
    assert fake_sf[-1][1]["season_length"] == 1
    py_engine.forecast_one(meth, y, 3, (52.0,), {"max_season": 52})
    assert fake_sf[-1][1]["season_length"] == 52


def test_holt_winters_multiplicative_requires_positive_values(fake_sf):
    res = PythonEngine().run(job("holt_winters", train=[np.arange(0.0, 60.0)], params={"error_type": "M"}))
    assert len(res.fallbacks) == 1 and "> 0" in res.fallbacks[0]["message"]


def test_mstl_periods_are_rounded_filtered_and_required(fake_sf):
    y = np.arange(1.0, 736.0)
    py_engine.forecast_one("mstl", y, 3, (7.0, 365.25), {})
    assert fake_sf[-1][1]["season_length"] == [7, 365]
    py_engine.forecast_one("mstl", np.arange(1.0, 401.0), 3, (7.0, 365.25), {})        # 2*365 > 400 -> yalnızca 7
    assert fake_sf[-1][1]["season_length"] == 7
    res = PythonEngine().run(job("mstl", train=[np.arange(1.0, 50.0)], seasonality=(1.0,)))      # yıllık: dönem yok
    assert len(res.fallbacks) == 1 and "no valid seasonal period" in res.fallbacks[0]["message"]


def test_auto_mfles_test_size_is_the_horizon(fake_sf):
    py_engine.forecast_one("auto_mfles", np.arange(1.0, 201.0), 12, (7.0,), {"n_windows": 3})
    kw = fake_sf[-1][1]
    assert kw["test_size"] == 12 and kw["season_length"] == 7 and kw["n_windows"] == 3 and kw["metric"] == "smape"


def test_adapter_notes_are_counted_in_engine_notes_not_printed(fake_sf, capsys):
    res = PythonEngine().run(job("auto_ces", train=series(4, 120, m=52), seasonality=(52.0,)))
    assert any(n.startswith("4 series:") and "max_season" in n for n in res.notes)
    assert "AdapterNote" not in capsys.readouterr().err


def test_signature_mismatch_in_new_model_is_a_clear_fallback_reason(fake_sf, monkeypatch):
    import statsforecast.models as m

    class OldTSB:                       # kütüphane sürümü alpha_d/alpha_p tanımıyor
        def __init__(self, alias="TSB"): pass

    monkeypatch.setattr(m, "TSB", OldTSB)
    res = PythonEngine().run(job("tsb", train=[intermittent(1)[0]]))
    assert len(res.fallbacks) == 1 and "does not recognise" in res.fallbacks[0]["message"]


def test_auto_mfles_hidden_until_scikit_learn_is_installed_and_hint_is_given(fake_sf, monkeypatch):
    """Gerçek olay: kullanıcı makinesinde 'MFLES requires scikit-learn' ImportError'ı seriyi yedeğe düşürdü."""
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name == "sklearn" else real(name, *a, **k))
    eng = PythonEngine()
    assert "auto_mfles" not in eng.methods()["local"] and "auto_theta" in eng.methods()["local"]
    ok, msg = eng.available()
    assert "pip install scikit-learn" in msg
    monkeypatch.setattr(importlib.util, "find_spec", real)
    if real("sklearn") is not None:
        assert "auto_mfles" in PythonEngine().methods()["local"]
