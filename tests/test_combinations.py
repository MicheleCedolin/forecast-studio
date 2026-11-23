"""ETS + ARIMA + Theta combinations (comb_eat, comb_eat_median) in the Python and C++ engines."""
import numpy as np
import pytest

from tsfstudio.engines import cpp_engine
from tsfstudio.engines.base import EngineJob
from tsfstudio.engines.registry import get_engine


def _series(n=8, length=60, seed=4):
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    return [100 + 0.3 * t + 8 * np.sin(2 * np.pi * t / 12 + i) + rng.normal(scale=2, size=length) for i in range(n)]


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_combinations_equal_mean_and_median_of_components(engine):
    eng = get_engine(engine)
    ok, _ = eng.available()
    if not ok or "comb_eat" not in eng.methods()["local"]:
        pytest.skip(f"{engine} engine or its components not available")
    train = _series()
    fc = {}
    for m in ("ets", "arima", "theta", "comb_eat", "comb_eat_median"):
        res = eng.run(EngineJob(task="local", method=m, train=train, horizon=6, seasonality=(12.0,)))
        assert not res.fallbacks
        fc[m] = res.forecasts
    st = np.stack([fc["ets"], fc["arima"], fc["theta"]])
    np.testing.assert_allclose(fc["comb_eat"], st.mean(axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(fc["comb_eat_median"], np.median(st, axis=0), rtol=0, atol=1e-12)


def test_combination_survives_a_failing_component(monkeypatch):
    from tsfstudio.engines import py_engine as P
    from tsfstudio.engines import py_sf as S
    if not S.sf_available():
        pytest.skip("statsforecast not installed")

    def boom(*a, **k):
        raise RuntimeError("arima failed")

    monkeypatch.setattr(S, "forecast_arima", boom)
    y = _series(1)[0]
    f = P.forecast_comb_eat(y, 6, 12, {}, median=False)
    expected = (np.asarray(S.forecast_ets(y, 6, 12)) + np.asarray(P.M.theta_forecast(y, 6, 12))) / 2
    np.testing.assert_allclose(f, expected)
    monkeypatch.setattr(P.M, "theta_forecast", boom)
    with pytest.raises(RuntimeError):
        P.forecast_comb_eat(y, 6, 12, {}, median=True)
