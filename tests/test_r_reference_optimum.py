"""Direct comparisons with R for methods whose R counterpart is not the study's own output (docs/VALIDATION.md 1e).

Fixtures were written on 2026-11-23 with R 4.3.3, forecast 8.21.1 and tsintermittent 2.0 by the scripts in tools/r_reference
(basic_reference.R, cran_reference.R, ses_optimum_reference.R, holt_theta_optimum_reference.R):
  * snaive, naive, drift equal forecast::snaive / naive / rwf(drift = TRUE);
  * croston_classic, croston_sba, tsb equal forecast::croston and tsintermittent's crost / tsb with w = 0.1, init = "naive";
  * ses and theta do NOT equal R on every series. Where they differ, the optimum of the SSE lies at the lower bound of
    alpha and R's fit has a higher SSE; the app's forecasts equal the exact optimum (`r_ses_optimum`, `r_opt_theta`).
The C++ tests are skipped when the native library has not been built.
"""
from pathlib import Path

import numpy as np
import pytest

from tsfstudio.engines import cpp_engine as C
from tsfstudio.engines.py_engine import forecast_one

G = Path(__file__).parent / "golden"
CASES = [("nn5_weekly", (52.17857142857143,), 8), ("nn5_daily", (7.0,), 56)]
needs_cpp = pytest.mark.skipif(C.load_lib() is None, reason="native library tsfcore not built")
ENGINES = [pytest.param("cpp", marks=needs_cpp), "python"]


def _load(d: Path, stem: str, **kw) -> np.ndarray:
    for ext in (".csv", ".csv.gz"):
        if (d / f"{stem}{ext}").is_file():
            return np.loadtxt(d / f"{stem}{ext}", delimiter=",", ndmin=2, **kw)
    raise FileNotFoundError(d / stem)


def _forecast(engine: str, method: str, train, h: int, seas) -> np.ndarray:
    if engine == "cpp":
        f, flags, _ = C.forecast_batch(method, train, h, seas)
        assert not flags.any()
        return f
    return np.vstack([forecast_one(method, y, h, seas, {}) for y in train])


def _rel(f: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.abs(f - ref) / np.maximum(np.abs(ref), 1e-9)


@pytest.fixture(scope="module", params=CASES, ids=[c[0] for c in CASES])
def case(request):
    name, seas, h = request.param
    return name, seas, h, [r[~np.isnan(r)] for r in _load(G / name, "train")]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("method", ["snaive", "naive", "drift"])
def test_closed_form_benchmarks_equal_r(case, engine, method):
    name, seas, h, train = case
    np.testing.assert_allclose(_forecast(engine, method, train, h, seas), _load(G / name, f"r_{method}"), rtol=1e-12,
                               err_msg=f"{name}/{method}/{engine}")


@needs_cpp
@pytest.mark.parametrize("method,ref", [("croston_classic", "r_forecast_croston"), ("croston_classic", "r_crost"),
                                        ("croston_sba", "r_crost_sba"), ("tsb", "r_tsb")])
def test_intermittent_methods_equal_r_with_fixed_parameters(method, ref):
    d = G / "r_intermittent"
    train = [r[~np.isnan(r)] for r in _load(d, "train")]
    r = _load(d, ref)
    f, flags, _ = C.forecast_batch(method, train, r.shape[1], (1.0,))
    assert not flags.any()
    np.testing.assert_allclose(f, r, rtol=1e-12, err_msg=method)


@pytest.mark.parametrize("engine", ENGINES)
def test_ses_is_the_exact_optimum_and_r_deviates_only_where_its_fit_is_worse(case, engine):
    name, seas, h, train = case
    # columns: sse_r, alpha_opt, sse_opt, fc_opt (sse_r_fixed is NA where ets() could not fit with alpha fixed)
    sse_r, alpha_opt, sse_opt, fc_opt = _load(G / name, "r_ses_optimum", skiprows=1, usecols=(2, 5, 6, 7)).T
    f = _forecast(engine, "ses", train, h, seas)[:, 0]
    np.testing.assert_allclose(f, fc_opt, rtol=1e-6, err_msg=f"{name}/{engine}: not the SSE optimum")
    dev = _rel(f, _load(G / name, "forecasts_ses")[:, 0]) > 1e-3
    assert 0 < dev.sum() <= 0.08 * len(train)
    assert (alpha_opt[dev] <= 1e-3).all(), "a deviation from R away from the lower bound of alpha"
    assert (sse_r[dev] > sse_opt[dev]).all(), "R deviates with a fit that is not worse than the optimum"
    assert (sse_r >= sse_opt * (1 - 1e-9)).all(), "R's SSE below the 'optimum': the reference optimum is wrong"


@pytest.mark.parametrize("engine", ENGINES)
def test_theta_equals_the_exact_ses_optimum(case, engine):
    name, seas, h, train = case
    f = _forecast(engine, "theta", train, h, seas)
    rel = _rel(f, _load(G / name, "r_opt_theta")).max(axis=1)
    assert rel.max() < 1e-4 and (rel > 1e-6).sum() <= 1, f"{name}/{engine}: max {rel.max():.1e}"


@needs_cpp
def test_comb_shd_is_close_to_the_multistart_optimum(case):
    # Holt and damped Holt have no closed-form optimum: the reference is a multi-start search of the same SSE, so a few
    # series end in a slightly different point (one per dataset by up to 0.5%). R's own fits are further away: its SSE
    # exceeds this optimum by more than 0.1% on 71 / 83 (weekly) and 26 / 19 (daily) of the 111 Holt / damped fits.
    name, seas, h, train = case
    f, flags, _ = C.forecast_batch("comb_shd", train, h, seas)
    assert not flags.any()
    rel = _rel(f, _load(G / name, "r_opt_comb")).max(axis=1)
    assert np.median(rel) < 1e-6 and (rel > 1e-3).sum() <= 1 and rel.max() < 1e-2, f"{name}: max {rel.max():.1e}"
    st = _load(G / name, "r_holt_optimum", skiprows=1)
    for r_col, opt_col in ((4, 5), (6, 7)):             # holt_r, holt_opt, damped_r, damped_opt
        assert (st[:, r_col] >= st[:, opt_col] * (1 - 1e-9)).all()
