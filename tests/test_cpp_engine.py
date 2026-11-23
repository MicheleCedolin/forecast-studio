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
GOLDEN_LOESS = G / "r_loess"
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


def test_dhr_arima_reproduces_r():
    """C++ dhr_arima = auto.arima(y, xreg = fourier(y, K = 1), seasonal = FALSE), as the study's R code. Against the
    study's weekly R forecasts: same model on all 111 series, forecasts to ~1e-8 (daily, checked with
    tools/r_reference/compare_dhr.py: 111/111, worst 4e-6)."""
    d = G / "nn5_weekly"
    train = [r[~np.isnan(r)] for r in _load(d, "train")]
    ref = _load(d, "forecasts_dhr_arima")
    f, flags, _ = C.forecast_batch("dhr_arima", train, ref.shape[1], (52.17857142857143,))
    assert not flags.any()
    rel = np.abs(f - ref) / np.maximum(np.abs(ref), 1e-9)
    assert np.median(rel) < 1e-9 and rel.max() < 1e-5, f"median {np.median(rel):.2e}, worst {rel.max():.2e}"


def test_dhr_arima_r_failure_modes():
    rng = np.random.default_rng(5)
    train = [100 + np.cumsum(rng.normal(size=120)) for _ in range(3)]
    # two seasonal periods with a scalar K: R's fourier() stops, the app falls back to snaive
    f, flags, msg = C.forecast_batch("dhr_arima", train, 7, (7.0, 30.5))
    assert flags.all() and "Number of periods" in msg
    # period 1: 2K > period, fourier() stops
    _, flags, msg = C.forecast_batch("dhr_arima", train, 3, (1.0,))
    assert flags.all() and "K must" in msg
    with pytest.raises(Exception, match="K = 1"):
        C.CppEngine().run(EngineJob(task="local", method="dhr_arima", train=train, horizon=3, seasonality=(7.0,),
                                    params={"fourier_k": 2}))


def test_tbats_reproduces_r_tbats(case):
    """C++ tbats is a port of R forecast::tbats (9.0.2) with its BATS alternative, ARMA errors, harmonic search and
    Guerrero lambda. Against the study's R forecasts (tests/golden/<case>/forecasts_tbats): all 111 weekly and 111 daily
    series agree to <= 1.3e-9 (median ~2e-15); a subset keeps the suite fast."""
    name, seas, h, train, _ = case
    ref = _load(G / name, "forecasts_tbats")
    k = 30 if name == "nn5_weekly" else 10
    f, flags, _ = C.forecast_batch("tbats", train[:k], h, seas)
    assert not flags.any()
    rel = np.abs(f - ref[:k]) / np.maximum(np.abs(ref[:k]), 1e-9)
    assert rel.max() < 1e-6, f"{name}: worst relative difference {rel.max():.2e}"


def test_pooled_regression_matches_r_and_python():
    """Global pooled regression (mean-normalised lag rows of all series, OLS without intercept, R glm's aliasing,
    recursive forecasts): equal to the study's R forecasts and to the Python engine to ~1e-14."""
    from tsfstudio.engines import py_global as PG
    for name, lag, h in (("nn5_weekly", 65, 8), ("nn5_daily", 9, 56)):
        train = [r[~np.isnan(r)] for r in _load(G / name, "train")]
        f, aliased = C.pooled_regression(train, lag, h)
        ref = _load(G / name, f"forecasts_pooled_regression_lag_{lag}")
        assert not aliased
        np.testing.assert_allclose(f, ref, rtol=1e-12, atol=0, err_msg=name)
    # collinear lags: two series that are exact geometric sequences make Lag2.. linear in Lag1 -> aliased, as R glm
    train = [10.0 * 1.01 ** np.arange(40), 5.0 * 1.01 ** np.arange(30)]
    f, aliased = C.pooled_regression(train, 4, 3)
    ref, info = PG.pooled_regression_forecast(train, 4, 3)
    assert aliased == info.aliased and aliased
    np.testing.assert_allclose(f, ref, rtol=1e-10)
    eng = C.CppEngine()
    r = eng.run(EngineJob(task="global", method="pooled_regression", train=train, horizon=3, seasonality=(1.0,), lag=4))
    np.testing.assert_allclose(r.forecasts, ref, rtol=1e-10)
    assert any("linearly dependent" in n for n in r.notes)


@pytest.mark.skipif(C.catboost_paths() is None, reason="CatBoost binaries (third_party/catboost) not downloaded")
def test_catboost_bit_identical_to_python():
    """C++ catboost (same data pipeline, CatBoost CLI training, model-library prediction) gives the Python engine's
    CatBoost forecasts bit for bit (NN5 weekly and daily, all series: identical)."""
    from tsfstudio.engines import py_global as PG
    train = [r[~np.isnan(r)] for r in _load(G / "nn5_weekly", "train")][:30]
    f = C.catboost(train, 12, 8, threads=4)
    ref, _ = PG.catboost_forecast(train, 12, 8, workers=4)
    np.testing.assert_array_equal(f, ref)
    r = C.CppEngine().run(EngineJob(task="global", method="catboost", train=train, horizon=8, seasonality=(1.0,), lag=12, workers=4))
    np.testing.assert_array_equal(r.forecasts, ref)
    with pytest.raises(Exception, match="defaults"):
        C.CppEngine().run(EngineJob(task="global", method="catboost", train=train, horizon=8, seasonality=(1.0,), lag=12,
                                    params={"catboost": {"depth": 4}}))


@pytest.mark.parametrize("method", ["stl_ets", "stl_arima"])
def test_stl_models_reproduce_r_stlf(case, method):
    """C++ stl_ets / stl_arima = R forecast(mstl(msts(y, periods)), method = "ets" / "arima") (forecast 9.0.2,
    tools/r_reference/stlf_reference.R). stl_ets: all 222 series <= 1e-9. stl_arima: 221 of 222 series <= 1e-6; one
    daily series (108), whose auto.arima estimate is ill-conditioned, differs by 0.26%."""
    name, seas, h, train, _ = case
    ref = _load(G / name, f"r_stlf_{method.split('_')[1]}")
    f, flags, _ = C.forecast_batch(method, train, h, seas)
    assert not flags.any()
    rel = (np.abs(f - ref) / np.maximum(np.abs(ref), 1e-9)).max(axis=1)
    tol = 1e-10 if method == "stl_ets" else 1e-6
    assert (rel > tol).sum() <= (0 if method == "stl_ets" else 1) and rel.max() < 5e-3, f"{name}: worst {rel.max():.2e}"


def test_stl_decomposition_python_equals_cpp():
    """The Python engine's R stl / mstl port (r_stl.py) and the C++ one give the same decomposition bit for bit, for
    one and several periods (periods >= n/2 dropped as in R); R errors are raised in both."""
    import ctypes
    from tsfstudio.engines import py_sf as S
    lib = C.load_lib()
    dp = ctypes.POINTER(ctypes.c_double)
    lib.tsf_stlf_parts.restype = ctypes.c_int32
    lib.tsf_stlf_parts.argtypes = [dp, ctypes.c_int32, dp, ctypes.c_int32, ctypes.c_int32, dp, dp]
    train = [r[~np.isnan(r)] for r in _load(G / "nn5_daily", "train")][:12]
    for per in [(7.0,), (7.0, 30.4375), (7.0, 30.4375, 365.25)]:
        for y in train:
            y = np.ascontiguousarray(y)
            pp = np.asarray(per)
            xa, sa = np.zeros(len(y)), np.zeros(20)
            fr = lib.tsf_stlf_parts(y.ctypes.data_as(dp), len(y), pp.ctypes.data_as(dp), len(pp), 20,
                                    xa.ctypes.data_as(dp), sa.ctypes.data_as(dp))
            xp, sp, frp = S.r_mstl_parts(y, 20, per)
            assert fr == frp and np.array_equal(xa, xp) and np.array_equal(sa, sp)
    _, flags, msg = C.forecast_batch("stl_ets", [np.arange(14.0)], 3, (7.0,))     # n <= 2 * period
    assert flags.all() and "less than two periods" in msg
    _, flags, msg = C.forecast_batch("stl_arima", [np.arange(30.0)], 3, (1.0,))   # not seasonal
    assert flags.all() and "not a seasonal" in msg
    with pytest.raises(ValueError, match="less than two periods"):
        S.r_mstl_parts(np.arange(14.0), 3, (7.0,))


def test_bagged_ets_reproduces_r_bagged_ets():
    """C++ bagged_ets = R forecast(baggedETS(msts(y, 52.18)), 8)$mean after set.seed(0) (forecast 9.0.2,
    tools/r_reference/bagged_reference.R): R's Mersenne-Twister and sample() are reproduced, so all 111 NN5 weekly
    series agree to <= 1e-10 (median ~3e-15)."""
    train = [r[~np.isnan(r)] for r in _load(G / "nn5_weekly", "train")]
    ref = _load(G / "nn5_weekly", "r_bagged_ets")
    f, flags, _ = C.forecast_batch("bagged_ets", train, ref.shape[1], (52.17857142857143,))
    assert not flags.any()
    rel = np.abs(f - ref) / np.maximum(np.abs(ref), 1e-9)
    assert rel.max() < 1e-10 and np.median(rel) < 1e-13, f"worst {rel.max():.2e}"
    # Daily (R: 5473 s, C++ ~200 s on 6 cores): 97 of 111 series <= 1e-6; the rest inherit the known ETS
    # multiplicative-season differences (worst: series 73, 0.3%). Slow, so only checked against the stored output
    # of a few series here.
    tr = [r[~np.isnan(r)] for r in _load(G / "nn5_daily", "train")]
    refd = _load(G / "nn5_daily", "r_bagged_ets")
    idx = [0, 1, 2, 3]
    fd, flags, _ = C.forecast_batch("bagged_ets", [tr[i] for i in idx], refd.shape[1], (7.0,))
    assert not flags.any()
    np.testing.assert_allclose(fd, refd[idx], rtol=1e-6)
    with pytest.raises(Exception, match="n_boot = 100"):
        C.CppEngine().run(EngineJob(task="local", method="bagged_ets", train=train[:2], horizon=3, seasonality=(52.0,),
                                    params={"n_boot": 20}))


def test_bagged_bootstrap_python_equals_cpp_and_r_rng():
    """The Python engine's bootstrap (r_boot.py) and the C++ one are identical bit for bit, and so are their R RNG
    draws (both were checked against R 4.6.2's set.seed / sample and bld.mbb.bootstrap output during development)."""
    import ctypes
    from tsfstudio.engines import r_boot as B
    lib = C.load_lib()
    dp = ctypes.POINTER(ctypes.c_double)
    lib.tsf_r_rng.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32, dp]
    o = np.zeros(20)
    lib.tsf_r_rng(0, 10, 100, o.ctypes.data_as(dp))
    g = B.RRng(0)
    py = [g.sample1(100) for _ in range(10)]
    assert list(o[10:].astype(int)) == py
    lib.tsf_bld_mbb.restype = ctypes.c_int32
    lib.tsf_bld_mbb.argtypes = [dp, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_uint32, dp, ctypes.c_char_p,
                                ctypes.c_int32]
    for ds, fr, num in [("nn5_daily", 7, 30), ("nn5_weekly", 52, 10)]:
        for y in [r[~np.isnan(r)] for r in _load(G / ds, "train")][:3]:
            y = np.ascontiguousarray(y)
            out = np.zeros(num * len(y))
            msg = ctypes.create_string_buffer(256)
            assert lib.tsf_bld_mbb(y.ctypes.data_as(dp), len(y), fr, num, 7, out.ctypes.data_as(dp), msg, 256) == 0
            assert np.array_equal(out.reshape(num, -1), np.vstack(B.bld_mbb_bootstrap(y, fr, num, 7)))


def _rows(path):
    import gzip
    with gzip.open(path, "rt") as f:
        return [np.array([np.nan if v == "NA" else float(v) for v in line.strip().split(",")]) for line in f if line.strip()]


def test_loess_equals_r_bit_for_bit():
    """loess(y ~ seq_along(y), span = 6/n, degree = 1)$fitted (R 4.6.2, tools/r_reference/loess_reference.R) for 128
    series of length 8..735 (random walks and integer counts): the C++ and Python ports (k-d tree, vertex fits with
    LINPACK QR/SVD, Hermite interpolation) give R's values bit for bit."""
    import ctypes
    from tsfstudio.engines import r_loess as RL
    lib = C.load_lib()
    dp = ctypes.POINTER(ctypes.c_double)
    lib.tsf_loess.restype = ctypes.c_int32
    lib.tsf_loess.argtypes = [dp, dp, ctypes.c_int32, ctypes.c_double, dp]
    X, R = _rows(GOLDEN_LOESS / "loess_in.csv.gz"), _rows(GOLDEN_LOESS / "r_loess_fitted.csv.gz")
    for i, (r, ref) in enumerate(zip(X, R)):
        y = np.ascontiguousarray(r[~np.isnan(r)])
        n = len(y)
        x = np.arange(1, n + 1, dtype=float)
        out = np.zeros(n)
        assert lib.tsf_loess(x.ctypes.data_as(dp), y.ctypes.data_as(dp), n, 6.0 / n, out.ctypes.data_as(dp)) == 0
        assert np.array_equal(out, ref), f"C++ row {i} (n = {n})"
        if i % 8 == 0:
            assert np.array_equal(np.array(RL.loess_fit_1d(list(x), list(y), 6.0 / n)), ref), f"Python row {i}"


def test_bagged_ets_non_seasonal_reproduces_r():
    """Non-seasonal bagged_ets (R's loess path) on 41 synthetic yearly series (n 13..47, with zeros and integer
    data): C++ = R baggedETS after set.seed(0) to <= 1e-10 (R 31 s, C++ 0.5 s); bootstrap series bit-identical to
    R's in C++ and Python (needs the 64-bit x87 precision R runs with, see rstats.hpp)."""
    import ctypes
    from tsfstudio.engines import r_boot as B
    train = [r[~np.isnan(r)] for r in _rows(GOLDEN_LOESS / "yearly_train.csv.gz")]
    ref = np.vstack(_rows(GOLDEN_LOESS / "r_bagged_ets_yearly.csv.gz"))
    f, flags, _ = C.forecast_batch("bagged_ets", train, ref.shape[1], (1.0,))
    assert not flags.any()
    assert (np.abs(f - ref) / np.abs(ref)).max() < 1e-10
    boot = np.vstack(_rows(GOLDEN_LOESS / "r_bld_mbb_yearly2_seed0.csv.gz"))
    lib = C.load_lib()
    dp = ctypes.POINTER(ctypes.c_double)
    lib.tsf_bld_mbb.restype = ctypes.c_int32
    lib.tsf_bld_mbb.argtypes = [dp, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_uint32, dp, ctypes.c_char_p,
                                ctypes.c_int32]
    y = np.ascontiguousarray(train[1])
    out = np.zeros(30 * len(y))
    msg = ctypes.create_string_buffer(256)
    assert lib.tsf_bld_mbb(y.ctypes.data_as(dp), len(y), 1, 30, 0, out.ctypes.data_as(dp), msg, 256) == 0
    assert np.array_equal(out.reshape(30, -1), boot)
    assert np.array_equal(np.vstack(B.bld_mbb_bootstrap(y, 1, 30, 0)), boot)


@pytest.mark.skipif(C.lightgbm_lib_path() is None, reason="lightgbm package (lib_lightgbm) not installed")
@pytest.mark.parametrize("dataset,lag,h", [("nn5_weekly", 12, 8), ("nn5_daily", 21, 56)])
def test_lightgbm_bit_identical_to_python(dataset, lag, h):
    """C++ lightgbm (same data pipeline and parameter string, LightGBM C API from C++) gives the Python engine's
    forecasts bit for bit, with default and custom parameters, on all NN5 series."""
    from tsfstudio.engines import py_global as PG
    train = [r[~np.isnan(r)] for r in _load(G / dataset, "train")]
    ref, _ = PG.lightgbm_forecast(train, lag, h)
    np.testing.assert_array_equal(C.lightgbm(train, lag, h), ref)
    custom = {"num_iterations": 150, "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 10, "seed": 3}
    ref2, _ = PG.lightgbm_forecast(train, lag, h, max_obs=200, params=custom, workers=4)
    job = EngineJob(task="global", method="lightgbm", train=train, horizon=h, seasonality=(1.0,), lag=lag, workers=4,
                    params={"max_obs": 200, "lightgbm": custom})
    np.testing.assert_array_equal(C.CppEngine().run(job).forecasts, ref2)
    assert not np.array_equal(ref, ref2)
    with pytest.raises(Exception, match="Unknown LightGBM parameter"):
        C.CppEngine().run(EngineJob(task="global", method="lightgbm", train=train, horizon=h, seasonality=(1.0,), lag=lag,
                                    params={"lightgbm": {"max_depth": 4}}))


@pytest.mark.parametrize("method", ["historic_average", "window_average", "seasonal_window_average", "seasonal_es", "zero_model"])
def test_simple_benchmarks_match_python_bit_for_bit(case, method):
    """Ports of statsforecast's HistoricAverage, WindowAverage, SeasonalWindowAverage,
    SeasonalExponentialSmoothingOptimized and ZeroModel (numpy summation order kept): identical forecasts."""
    name, seas, h, train, _ = case
    f, flags, _ = C.forecast_batch(method, train, h, seas)
    ref = np.vstack([forecast_one(method, y, h, seas, {}) for y in train])
    assert not flags.any()
    np.testing.assert_array_equal(f, ref, err_msg=f"{name}/{method}")


def test_holt_ets_reproduces_r_ets_aan():
    """C++ holt_ets = R ets(y, model = "AAN") through the ets port (undamped and damped trend by AICc). Reference: R 9.0.2
    on NN5 daily (tests/golden/nn5_daily/r_ets_aan), 34 ETS(A,A,N) and 77 ETS(A,Ad,N) fits: equal to ~1e-12.
    statsforecast's Holt (Python engine) differs from R by ~0.3%."""
    train = [r[~np.isnan(r)] for r in _load(G / "nn5_daily", "train")]
    ref = _load(G / "nn5_daily", "r_ets_aan")
    f, flags, _ = C.forecast_batch("holt_ets", train, ref.shape[1], (7.0,))
    assert not flags.any()
    np.testing.assert_allclose(f, ref, rtol=1e-9, atol=0)


def test_holt_winters_reproduces_r_ets_aaa():
    """C++ holt_winters = R ets(y, model = "AAA") (damped / undamped by AICc) through the C++ ets port. Reference:
    R 9.0.2 forecasts on NN5 daily (tests/golden/nn5_daily/r_ets_aaa). statsforecast's HoltWinters, which the
    Python engine uses, differs from R by ~1% here; the C++ version follows R, as the C++ ets does."""
    d = G / "nn5_daily"
    train = [r[~np.isnan(r)] for r in _load(d, "train")]
    ref = _load(d, "r_ets_aaa")
    f, flags, _ = C.forecast_batch("holt_winters", train, ref.shape[1], (7.0,))
    assert not flags.any()
    rel = np.abs(f - ref) / np.maximum(np.abs(ref), 1e-9)
    assert rel.max() < 1e-9, f"worst {rel.max():.2e}"            # exact since the damped-forecast fix
    # weekly: period 52 > 24, so no usable season -> "Nonseasonal data" and the snaive fallback, as statsforecast
    w = [r[~np.isnan(r)] for r in _load(G / "nn5_weekly", "train")][:5]
    _, flags, msg = C.forecast_batch("holt_winters", w, 8, (52.17857142857143,))
    assert flags.all() and "Nonseasonal" in msg


def test_dotm_matches_python_engine(case):
    """C++ dotm ports statsforecast 2.1.1's DynamicOptimizedTheta (theta recursion, its Nelder-Mead, statsmodels'
    seasonal test and decomposition): forecasts equal the Python engine's to machine precision (~1e-15)."""
    import warnings
    name, seas, h, train, _ = case
    sub = train if name == "nn5_weekly" else train[:20]
    f, flags, _ = C.forecast_batch("dotm", sub, h, seas)
    assert not flags.any()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = np.vstack([forecast_one("dotm", y, h, seas, {}) for y in sub])
    np.testing.assert_allclose(f, ref, rtol=1e-11, atol=0, err_msg=name)


def test_auto_theta_matches_python_engine(case):
    """C++ auto_theta = statsforecast AutoTheta (STM / OTM / DSTM / DOTM by in-sample mse) on the theta port used by
    dotm: equal to the Python engine to machine precision."""
    import warnings
    name, seas, h, train, _ = case
    sub = train if name == "nn5_weekly" else train[:15]
    f, flags, _ = C.forecast_batch("auto_theta", sub, h, seas)
    assert not flags.any()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = np.vstack([forecast_one("auto_theta", y, h, seas, {}) for y in sub])
    np.testing.assert_allclose(f, ref, rtol=1e-11, atol=0, err_msg=name)


@pytest.mark.parametrize("periods", [(7.0,), (7.0, 30.4)])
def test_mstl_decomposition_matches_statsforecast(periods):
    """C++ mstl: the decomposition (trend + remainder, seasonal naive forecasts) equals statsforecast's mstl() with
    statsmodels' STL to machine precision; the trend is then forecast with the C++ ets (R's ets, model "ZZN")."""
    import warnings
    from statsforecast.mstl import mstl
    from statsforecast.models import _predict_mstl_seas
    train = [r[~np.isnan(r)] for r in _load(G / "nn5_daily", "train")][:12]
    lib, dp, h = C.load_lib(), C.ctypes.POINTER(C.ctypes.c_double), 14
    pp = np.ascontiguousarray(periods, dtype=float)
    for y in train:
        pv = sorted({int(round(p)) for p in periods})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mo = mstl(y, pv if len(pv) > 1 else pv[0])
            sf_ref = _predict_mstl_seas(mo, h, pv if len(pv) > 1 else pv[0])
        xsa_ref = mo[["trend", "remainder"]].sum(axis=1).values
        yy, xsa, sf = np.ascontiguousarray(y), np.zeros(len(y)), np.zeros(h)
        assert lib.tsf_mstl_parts(yy.ctypes.data_as(dp), len(y), pp.ctypes.data_as(dp), len(pp), h,
                                  xsa.ctypes.data_as(dp), sf.ctypes.data_as(dp)) == 0
        np.testing.assert_allclose(xsa, xsa_ref, rtol=0, atol=1e-12 * np.abs(y).max())
        np.testing.assert_allclose(sf, sf_ref, rtol=0, atol=1e-12 * np.abs(y).max())
    f, flags, _ = C.forecast_batch("mstl", train, h, periods)
    assert not flags.any() and np.isfinite(f).all()
    _, flags, msg = C.forecast_batch("mstl", [np.arange(10.0)], 3, (7.0,))   # 2 * 7 > 10: no valid period
    assert flags.all() and "no valid seasonal period" in msg


def test_auto_mfles_matches_python_engine():
    """C++ auto_mfles ports statsforecast 2.1.1's AutoMFLES (MFLES boosting, configuration grid, rolling-origin
    selection) with scikit-learn's Lasso coordinate descent: forecasts equal the Python engine's to machine precision,
    including short series whose validation windows are shorter than the season."""
    import warnings
    rng = np.random.default_rng(4)
    items = [(r, (52.17857142857143,), 8) for r in [r[~np.isnan(r)] for r in _load(G / "nn5_weekly", "train")][:20]]
    for k in range(24):
        n = int(rng.integers(12, 120))
        m = [1, 4, 7, 12][k % 4]
        t = np.arange(n)
        y = 50 + 0.1 * t + 8 * np.sin(2 * np.pi * t / max(m, 2)) + rng.normal(0, 3, n)
        items.append((y - (60 if k % 6 == 0 else 0), (float(m),), 10))
    for y, seas, h in items:
        f, flags, _ = C.forecast_batch("auto_mfles", [y], h, seas)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = np.asarray(forecast_one("auto_mfles", y, h, seas, {}), dtype=float)
        assert not flags[0]
        np.testing.assert_allclose(f[0], ref, rtol=1e-10, atol=1e-10 * np.abs(ref).max())


def test_lasso_matches_sklearn():
    import warnings
    from statsforecast.mfles import get_basis, lasso_nb
    lib, dp = C.load_lib(), C.ctypes.POINTER(C.ctypes.c_double)
    rng = np.random.default_rng(9)
    for _ in range(30):
        n = int(rng.integers(20, 300))
        r = np.cumsum(rng.normal(size=n))
        X = get_basis(r, max(1, int(0.1 * n)), -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w_ref = lasso_nb(X, r, 1.0)
        Xf, out = np.asfortranarray(X), np.zeros(X.shape[1])
        lib.tsf_lasso(Xf.ctypes.data_as(dp), n, X.shape[1], np.ascontiguousarray(r).ctypes.data_as(dp),
                      C.ctypes.c_double(1.0), out.ctypes.data_as(dp))
        np.testing.assert_allclose(X @ out, X @ w_ref, rtol=0, atol=1e-12 * np.abs(X @ w_ref).max())


def test_auto_ces_matches_python_engine(case):
    """C++ auto_ces ports statsforecast 2.1.1's AutoCES (model "Z": N / S / P / F by AICc, three-pass fit, its
    Nelder-Mead): forecasts equal the Python engine's (weekly bit for bit, daily to ~1e-10)."""
    import warnings
    name, seas, h, train, _ = case
    sub = train if name == "nn5_weekly" else train[:25]
    f, flags, _ = C.forecast_batch("auto_ces", sub, h, seas)
    assert not flags.any()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = np.vstack([forecast_one("auto_ces", y, h, seas, {}) for y in sub])
    np.testing.assert_allclose(f, ref, rtol=1e-8, atol=0, err_msg=name)


def test_scum_is_median_of_cpp_components():
    """C++ scum = pointwise median of the C++ ets, auto_ces, arima and dotm forecasts (the Python engine's rule); the
    components are the validated C++ ports (ets/arima reproduce R, auto_ces/dotm statsforecast)."""
    d = G / "nn5_weekly"
    train = [r[~np.isnan(r)] for r in _load(d, "train")][:40]
    seas = (52.17857142857143,)
    s, flags, _ = C.forecast_batch("scum", train, 8, seas)
    assert not flags.any()
    comps = [C.forecast_batch(mth, train, 8, seas)[0] for mth in ("ets", "auto_ces", "arima", "dotm")]
    np.testing.assert_array_equal(s, np.median(np.stack(comps), axis=0))


def test_scum_component_failures():
    # short series: some components fail (ets needs > 6 observations, dotm > 3, auto_ces > 2); scum takes the median
    # of those that succeed, and falls back (error) when fewer than two do
    base = [10.0, 12, 11, 13, 12, 14]
    seen_fallback = seen_partial = False
    for n in range(2, 7):
        y = [np.array(base[:n])]
        s, flags, msg = C.forecast_batch("scum", y, 3, (1.0,))
        comps = []
        for mth in ("ets", "auto_ces", "arima", "dotm"):
            f, fl, _ = C.forecast_batch(mth, y, 3, (1.0,))
            if not fl[0]:
                comps.append(f[0])
        if len(comps) < 2:
            assert flags[0] and "fewer than two" in msg, n
            seen_fallback = True
        else:
            assert not flags[0], n
            np.testing.assert_array_equal(s[0], np.median(np.stack(comps), axis=0))
            seen_partial |= len(comps) < 4
    assert seen_fallback and seen_partial


INTERMITTENT = ["croston_classic", "croston_sba", "croston_optimized", "tsb", "adida", "imapa"]


@pytest.mark.parametrize("method", INTERMITTENT)
def test_intermittent_methods_match_python_bit_for_bit(method):
    """Ports of statsforecast 2.1.1 (src/ses.cpp, models.py) with numpy's pairwise mean and round-half-to-even:
    the forecasts are identical to the Python engine's, including the error cases."""
    rng = np.random.default_rng(11)
    train = []
    for k in range(80):
        n = int(rng.integers(10, 300))
        y = np.where(rng.uniform(size=n) < rng.uniform(0.4, 0.95), 0.0, np.round(rng.gamma(2.0, 5.0, n)))
        if k % 20 == 0:
            y[:] = 0.0
            y[int(rng.integers(n))] = 3.0
        train.append(y)
    train += [np.array([0., 0, 5, 0, 0, 0, 7, 0]), np.array([0., 4, 0, 4]), np.array([2.0]), np.zeros(9),
              np.array([1., 0, 1, 0, 0, 1, 0, 0, 0, 1]), np.array([0., -2, 0, -1, 0])]   # last: no positive demand
    f, flags, _ = C.forecast_batch(method, train, 5, (7.0,), threads=2)
    for i, y in enumerate(train):
        try:
            ref = np.asarray(forecast_one(method, y, 5, (7.0,), {}), dtype=float)
        except Exception:
            assert flags[i] == 1, f"{method} #{i}: Python fails, C++ did not"
            continue
        assert flags[i] == 0, f"{method} #{i}: C++ failed"
        np.testing.assert_array_equal(f[i], ref, err_msg=f"{method} #{i}")


def test_tsb_refuses_non_default_alphas():
    with pytest.raises(Exception, match="alpha_d"):
        C.CppEngine().run(EngineJob(task="local", method="tsb", train=[np.array([0., 1, 0, 2])], horizon=2,
                                    seasonality=(1.0,), params={"alpha_d": 0.2}))


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
        eng.run(EngineJob(task="local", method="no_such_model", train=train, horizon=3, seasonality=(7.0,)))
