"""Series features: catch22 (native) against the ranking study's Rcatch22 values and against a plain numpy ACF."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsfstudio.engines import cpp_engine

pytestmark = pytest.mark.skipif(cpp_engine.load_lib() is None, reason="native library not built")

GOLD = Path(__file__).parent / "golden" / "catch22"


def _gold():
    s = pd.read_csv(GOLD / "series.csv.gz")
    ref = pd.read_csv(GOLD / "rcatch22_study.csv.gz")
    series = [s.loc[s.id == i, "value"].to_numpy() for i in ref.id]
    return series, ref


def test_catch22_reproduces_study_rcatch22_values():
    from tsfstudio.features import CATCH22_NAMES, catch22
    series, ref = _gold()
    assert tuple(ref.columns[1:]) == CATCH22_NAMES
    out = catch22(series, study_fft=True)
    R = ref[list(CATCH22_NAMES)].to_numpy(float)
    np.testing.assert_allclose(out, R, rtol=1e-12, atol=1e-12)


def test_catch22_threads_do_not_change_results():
    from tsfstudio.features import catch22
    series, _ = _gold()
    np.testing.assert_array_equal(catch22(series, threads=1), catch22(series, threads=4))


def test_catch22_correct_fft_matches_numpy_autocorrelation():
    from tsfstudio.features import CATCH22_NAMES, catch22
    rng = np.random.default_rng(3)
    series = [np.cumsum(rng.normal(size=n)) + 5 * np.sin(np.arange(n) / 3) for n in (30, 64, 101, 250)]
    out = catch22(series)
    j_f1, j_min = CATCH22_NAMES.index("CO_f1ecac"), CATCH22_NAMES.index("CO_FirstMin_ac")
    for y, row in zip(series, out):
        z = y - y.mean()
        n = len(z)
        ac = np.array([np.dot(z[: n - k], z[k:]) for k in range(n)]) / np.dot(z, z)
        thr = 1 / np.e
        f1 = next((i + 1 for i in range(n - 1) if (ac[i] - thr) * (ac[i + 1] - thr) < 0), n)
        fmin = next((i for i in range(1, n - 1) if ac[i] < ac[i - 1] and ac[i] < ac[i + 1]), n)
        assert row[j_f1] == f1 and row[j_min] == fmin


def test_catch22_degenerate_series_give_nan_rows():
    from tsfstudio.features import catch22
    out = catch22([np.arange(5.0), np.full(30, 2.0), np.r_[np.arange(20.0), np.nan], np.arange(30.0) % 7])
    assert np.isnan(out[:3]).all()
    assert np.isfinite(out[3]).all()
