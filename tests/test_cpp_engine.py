"""C++ engine (tsfcore) against the Python engine and the R reference fixtures.

Skipped when the native library has not been built (cpp/build.ps1).
  * naive, snaive, drift, naive2, ses, theta: same algorithm and optimiser as Python -> agreement to ~1e-9.
  * Holt / damped Holt / Comb S-H-D: same model and closed-form initial state; the optimiser differs (Nelder-Mead instead of
    scipy L-BFGS-B), so the check is fit quality (C++ SSE <= Python SSE + tiny) and decision-level agreement with R.
"""
from pathlib import Path

import numpy as np
import pytest

from tsfstudio.engines import EngineJob
from tsfstudio.engines import cpp_engine as C
from tsfstudio.engines import py_models as M
from tsfstudio.engines.py_engine import forecast_one
from tsfstudio.metrics import mase, smape

pytestmark = pytest.mark.skipif(C.load_lib() is None, reason="native library tsfcore not built")
G = Path(__file__).parent / "golden"
CASES = [("nn5_weekly", (52.17857142857143,), 8), ("nn5_daily", (7.0,), 56)]


def _load(d: Path, stem: str) -> np.ndarray:
    for ext in (".csv", ".csv.gz"):
        if (d / f"{stem}{ext}").is_file():
            return np.loadtxt(d / f"{stem}{ext}", delimiter=",", ndmin=2)
    raise FileNotFoundError(d / stem)


@pytest.fixture(scope="module", params=CASES, ids=[c[0] for c in CASES])
def case(request):
    name, seas, h = request.param
    d = G / name
    return name, seas, h, [r[~np.isnan(r)] for r in _load(d, "train")], _load(d, "actuals")


@pytest.mark.parametrize("method", ["naive", "snaive", "drift", "naive2", "ses", "theta"])
def test_exact_methods_match_python(case, method):
    # SES/Theta use the same Brent minimiser (xatol 1e-10) as scipy; a different floating-point summation order can end
    # it at a slightly different alpha inside that tolerance, which moves forecasts by up to ~3e-8 relative on NN5 daily.
    name, seas, h, train, _ = case
    f, flags, _ = C.forecast_batch(method, train, h, seas)
    ref = np.vstack([forecast_one(method, y, h, seas, {}) for y in train])
    assert not flags.any()
    rtol = 1e-6 if method in ("ses", "theta") else 1e-10
    np.testing.assert_allclose(f, ref, rtol=rtol, atol=1e-10, err_msg=f"{name}/{method}")


def test_holt_fit_at_least_as_good_as_python(case):
    name, seas, h, train, _ = case
    m = M.r_frequency(seas)
    worse = 0
    for y in train[:40]:
        x, _ = M._m4_deseasonalise(y, m, 1)
        for damped in (False, True):
            best = [np.inf]
            orig = M._holt_profile

            def spy(yy, a, b, p, _o=orig):
                r = _o(yy, a, b, p)
                best[0] = min(best[0], r[0])
                return r
            M._holt_profile = spy
            try:
                M.holt_forecast(x, 1, damped)
            finally:
                M._holt_profile = orig
            xs = np.ascontiguousarray(x, dtype=np.float64)
            sse_c = C.load_lib().tsf_holt_sse(C._ptr(xs, C.ctypes.c_double), len(xs), int(damped))
            if sse_c > best[0] * (1 + 1e-7):
                worse += 1
    assert worse == 0, f"{name}: C++ Holt fit worse than Python on {worse} fits"


def test_comb_shd_matches_r_at_decision_level(case):
    name, seas, h, train, act = case
    ref = _load(G / name, "r_m4_comb")
    f, flags, _ = C.forecast_batch("comb_shd", train, h, seas)
    assert not flags.any()
    m = M.r_frequency(seas)
    for a, b in ((smape(f, act), smape(ref, act)), (mase(f, act, train, m), mase(ref, act, train, m))):
        assert abs(np.nanmean(a) - np.nanmean(b)) / np.nanmean(b) < 0.01, name


def test_ets_reproduces_r_forecast_ets(case):
    """C++ ets is a port of R forecast::ets (9.x). The study's R forecasts (tests/golden/<case>/forecasts_ets) are the
    reference: on NN5 weekly (non-seasonal, m > 24) and daily (m = 7) the same model is chosen and forecasts agree to
    machine precision, except where R's Nelder-Mead and ours drift apart over a long run (1 daily series)."""
    name, seas, h, train, act = case
    ref = _load(G / name, "forecasts_ets")
    f, flags, _ = C.forecast_batch("ets", train, h, seas)
    assert not flags.any()
    rel = np.abs(f - ref) / np.maximum(np.abs(ref), 1e-9)
    assert np.median(rel) < 1e-9, f"{name}: median relative difference {np.median(rel):.2e}"
    assert int((rel.max(axis=1) > 1e-3).sum()) <= 2, name
    m = M.r_frequency(seas)
    d = abs(np.nanmean(mase(f, act, train, m)) - np.nanmean(mase(ref, act, train, m))) / np.nanmean(mase(ref, act, train, m))
    assert d < 1e-3, name


def test_arima_reproduces_r_auto_arima(case):
    """C++ arima is a port of R forecast::auto.arima (9.0.2) with the study's call chain auto.arima(lambda = 0) ->
    auto.arima() -> auto.arima(seasonal = FALSE). The study's R forecasts are the reference; the same model is chosen
    on all NN5 series and forecasts agree to ~1e-12 (worst ~1e-4). Daily uses the first 12 series to keep the suite
    fast; tools/r_reference/compare_auto_arima.py checks all of them."""
    name, seas, h, train, act = case
    ref = _load(G / name, "forecasts_arima")
    k = len(train) if name == "nn5_weekly" else 12
    f, flags, _ = C.forecast_batch("arima", train[:k], h, seas)
    assert not flags.any()
    rel = np.abs(f - ref[:k]) / np.maximum(np.abs(ref[:k]), 1e-9)
    assert np.median(rel) < 1e-9, f"{name}: median relative difference {np.median(rel):.2e}"
    assert rel.max() < 1e-3, f"{name}: worst relative difference {rel.max():.2e}"


def test_arima_q0_methods_agree():
    """Gardner's Q0 (as R's getQ0) and the autocovariance form used for large state dimension give the same matrix."""
    lib = C.load_lib()
    dp = C.ctypes.POINTER(C.ctypes.c_double)
    rng = np.random.default_rng(3)
    for p, q in [(1, 0), (2, 1), (5, 5), (8, 2), (15, 7), (19, 14)]:
        phi = np.r_[rng.uniform(-0.5, 0.5, 1), np.zeros(p - 1)] if p > 1 else rng.uniform(-0.9, 0.9, 1)
        phi[-1] = rng.uniform(-0.6, 0.6)                 # (1 - a B)(1 - b B^p)-like: sparse, stationary
        phi = np.ascontiguousarray(phi)
        th = np.ascontiguousarray(rng.uniform(-0.6, 0.6, q))
        r = max(p, q + 1)
        out = [np.zeros(r * r), np.zeros(r * r)]
        for m in (0, 1):
            assert lib.tsf_arima_q0(phi.ctypes.data_as(dp), p, th.ctypes.data_as(dp), q, m, out[m].ctypes.data_as(dp)) == 0
        np.testing.assert_allclose(out[1], out[0], rtol=1e-9, atol=1e-11 * np.abs(out[0]).max(), err_msg=f"({p},{q})")


def test_engine_run_horizons_threads_and_fallback():
    eng = C.CppEngine()
    assert eng.available()[0]
    rng = np.random.default_rng(0)
    train = [50 + np.cumsum(rng.normal(size=60)) for _ in range(30)] + [np.array([3.0])]   # last one: too short for theta's fit
    hz = [7] * 30 + [3]
    r1 = eng.run(EngineJob(task="local", method="theta", train=train, horizon=7, seasonality=(7.0,), horizons=hz, workers=1))
    r4 = eng.run(EngineJob(task="local", method="theta", train=train, horizon=7, seasonality=(7.0,), horizons=hz, workers=4))
    np.testing.assert_array_equal(r1.forecasts, r4.forecasts)                       # threading does not change results
    assert np.isnan(r1.forecasts[30, 3:]).all() and np.isfinite(r1.forecasts[30, :3]).all()
    with pytest.raises(Exception, match="does not support"):
        eng.run(EngineJob(task="local", method="tbats", train=train, horizon=3, seasonality=(7.0,)))
