"""Optional deep-learning / foundation models (neuralforecast, GluonTS, Chronos-Bolt) in the Python engine."""
import numpy as np
import pytest

from tsfstudio.engines import py_deep as D
from tsfstudio.engines import py_gluonts as G
from tsfstudio.engines.base import EngineJob
from tsfstudio.engines.registry import get_engine


def _series(n=20, length=60, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    return [50 + 10 * np.sin(2 * np.pi * t / 12 + i) + rng.normal(scale=1, size=length) for i in range(n)]


@pytest.mark.skipif(not D.nhits_available(), reason="torch / neuralforecast not installed")
def test_nhits_global_forecast_is_deterministic():
    eng = get_engine("python")
    assert "nhits" in eng.methods()["global"]
    job = EngineJob(task="global", method="nhits", train=_series(), horizon=6, seasonality=(12.0,), lag=24,
                    params={"nhits": {"max_steps": 30}}, seed=3)
    a = eng.run(job).forecasts
    b = eng.run(job).forecasts
    assert a.shape == (20, 6) and np.isfinite(a).all()
    np.testing.assert_allclose(a, b, rtol=1e-5)


@pytest.mark.skipif(not D.chronos_available(), reason="torch / chronos-forecasting not installed")
def test_chronos_zero_shot_follows_a_seasonal_pattern():
    import os
    if os.environ.get("TSFSTUDIO_SKIP_NETWORK"):         # the first run downloads the tiny model from Hugging Face
        pytest.skip("network disabled")
    train = _series(5, 72)
    f = D.chronos_forecast(train, 12, {"model": "amazon/chronos-bolt-tiny"}, 1)
    assert f.shape == (5, 12) and np.isfinite(f).all()
    truth = np.array([50 + 10 * np.sin(2 * np.pi * np.arange(72, 84) / 12 + i) for i in range(5)])
    assert np.mean(np.abs(f - truth)) < 4                       # a flat forecast would be about 6.4 off
    with pytest.raises(ValueError):
        D.chronos_forecast(train, 3, {"model": "nope"}, 1)


@pytest.mark.skipif(not D.nhits_available(), reason="torch / neuralforecast not installed")
def test_nbeats_global_forecast():
    eng = get_engine("python")
    assert "nbeats" in eng.methods()["global"]
    job = EngineJob(task="global", method="nbeats", train=_series(), horizon=6, seasonality=(12.0,), lag=24,
                    params={"nbeats": {"max_steps": 30}}, seed=3)
    f = eng.run(job).forecasts
    assert f.shape == (20, 6) and np.isfinite(f).all()


@pytest.mark.skipif(not D.nhits_available(), reason="torch / neuralforecast not installed")
@pytest.mark.parametrize("method,steps", [("lstm", 300), ("rnn", 300), ("gru", 300), ("deepar", 150)])
def test_recurrent_global_forecast(method, steps):
    """LSTM, RNN, GRU and DeepAR (neuralforecast): shape, determinism for a seed, and they learn the seasonal pattern."""
    eng = get_engine("python")
    assert method in eng.methods()["global"]
    job = EngineJob(task="global", method=method, train=_series(), horizon=6, seasonality=(12.0,), lag=24,
                    params={method: {"max_steps": steps}}, seed=3)
    res = eng.run(job)
    f = res.forecasts
    assert f.shape == (20, 6) and np.isfinite(f).all()
    assert any("neuralforecast" in n for n in res.notes)
    truth = np.array([50 + 10 * np.sin(2 * np.pi * np.arange(60, 66) / 12 + i) for i in range(20)])
    assert np.mean(np.abs(f - truth)) < 2.5                    # a flat forecast at the mean is about 6.4 off
    if method != "deepar":                                     # DeepAR samples paths; checked for shape and accuracy only
        np.testing.assert_allclose(f, eng.run(job).forecasts, rtol=1e-5)


@pytest.mark.skipif(not G.available(), reason="gluonts[torch] not installed")
@pytest.mark.parametrize("method", G.METHODS)
def test_gluonts_global_forecast(method):
    """DeepAR, FFNN and WaveNet through GluonTS (the Monash study's library): shape, determinism, learning the pattern."""
    eng = get_engine("python")
    assert method in eng.methods()["global"]
    job = EngineJob(task="global", method=method, train=_series(), horizon=6, seasonality=(12.0,), lag=24,
                    params={method: {"epochs": 30}}, seed=3)
    res = eng.run(job)
    f = res.forecasts
    assert f.shape == (20, 6) and np.isfinite(f).all()
    assert any("GluonTS" in n and "frequency 1M" in n for n in res.notes)
    truth = np.array([50 + 10 * np.sin(2 * np.pi * np.arange(60, 66) / 12 + i) for i in range(20)])
    assert np.mean(np.abs(f - truth)) < 2.5                    # a flat forecast at the mean is about 6.4 off
    np.testing.assert_allclose(f, eng.run(job).forecasts, rtol=1e-5)


def test_gluonts_frequency_follows_the_monash_maps():
    assert G.freq_for((24.0, 168.0)) == ("1h", True)
    assert G.freq_for((7.0,)) == ("1D", True)
    assert G.freq_for((365.25 / 7,)) == ("1W", True)
    assert G.freq_for((12.0,)) == ("1M", True)
    assert G.freq_for((4.0,)) == ("1Q", True)
    assert G.freq_for((1.0,)) == ("1Y", True)
    assert G.freq_for((144.0, 1008.0)) == ("10min", True)
    assert G.freq_for((5.0,)) == ("1D", False)                 # no Monash frequency: daily, and the run says so


def test_gluonts_models_are_described():
    from tsfstudio.app.models_info import MODELS, REFS
    for m in G.METHODS:
        info = MODELS[m]
        assert info["task"] == "global" and set(info["engines"]) == {"python"} and "GluonTS" in info["name"]
        assert all(r in REFS for r in info["refs"])
        assert {p["key"] for p in info["params"]} == {f"{m}.epochs", f"{m}.learning_rate"}


def test_recurrent_models_are_described():
    from tsfstudio.app.models_info import MODELS, REFS
    for m in ("lstm", "rnn", "gru", "deepar"):
        info = MODELS[m]
        assert info["task"] == "global" and set(info["engines"]) == {"python"}
        assert all(r in REFS for r in info["refs"])
        assert {p["key"] for p in info["params"]} >= {f"{m}.max_steps", f"{m}.hidden_size", f"{m}.n_layers"}
