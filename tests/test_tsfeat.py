"""tsfeatures as the ranking study computed it (tsfstudio.tsfeat) against the study's stored R values."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsfstudio.engines import cpp_engine

pytestmark = pytest.mark.skipif(cpp_engine.load_lib() is None, reason="native library not built")

GOLD = Path(__file__).parent / "golden"


def test_study_tsfeatures_reproduce_r():
    from tsfstudio import tsfeat as T
    meta = pd.read_csv(GOLD / "tsfeatures" / "meta.csv")
    ref = pd.read_csv(GOLD / "tsfeatures" / "r_tsfeatures_study.csv.gz")
    ser = pd.read_csv(GOLD / "catch22" / "series.csv.gz")
    bad = []
    for _, m in meta.iterrows():
        y = ser.loc[ser.id == m.id, "value"].to_numpy()
        per = T.study_periods(m.frequency if isinstance(m.frequency, str) else None)
        got = T.study_features(y, per, stl=True, startx=T.study_startx(m.start_timestamp, per))
        for _, r in ref[ref.id == m.id].iterrows():
            a, b = got.get(r.feature, np.nan), r.value
            if not ((np.isnan(a) and np.isnan(b)) or abs(a - b) <= 1e-7 * max(1.0, abs(b))):
                bad.append((m.id, r.feature, a, b))
    assert not bad, bad[:10]


def test_tsfeatures_table_drops_partial_columns():
    from tsfstudio.features import tsfeatures_table
    rng = np.random.default_rng(2)
    long = 50 + np.cumsum(rng.normal(size=60)) + 5 * np.sin(np.arange(60) * 2 * np.pi / 12)
    short = long[:20]                                   # too short for seasonal features at period 12
    df = tsfeatures_table([long, short], (12,))
    assert "seasonal_strength" not in df.columns and "trend" in df.columns
    assert df.shape[0] == 2 and np.isfinite(df.loc[0, "hurst"])


def test_native_stl_robust_and_supsmu_match_r():
    from tsfstudio.tsfeat import _native_stl
    x = np.array([5, 3, 8, 6, 4, 9, 7, 5, 10, 8, 6, 12, 9, 7, 11], float)
    s, t = _native_stl(x, 3, True, 7, True)
    # stl(ts(x, frequency = 3), "periodic", robust = TRUE) in R 4.6.2
    np.testing.assert_allclose(t[-3:], [9.0, 8.833333, 8.666666], atol=1e-6)
    np.testing.assert_allclose(s[:3], [0.0, -2.333333, 2.333334], atol=1e-6)