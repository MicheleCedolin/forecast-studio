"""Optional deep-learning / foundation models (N-HiTS, Chronos-Bolt) in the Python engine."""
import numpy as np
import pytest

from tsfstudio.engines import py_deep as D
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
