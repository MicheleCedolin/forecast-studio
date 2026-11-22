"""Naive2 and Comb S-H-D against R (forecast 9.0.2, R 4.6.2), following the official M4 benchmark code.

Fixtures: tests/golden/<case>/r_m4_naive2.csv[.gz] and r_m4_comb.csv[.gz], produced by tools/r_reference/m4_benchmarks.R
(NN5 weekly: period 52, h 8; NN5 daily: period 7, h 56). See docs/VALIDATION.md, section 1d.

  * Naive2: same seasonality test, classical decomposition and naive -> identical to machine precision.
  * Comb S-H-D: same model (mean of SES, Holt and damped Holt on the deseasonalised series). Python's least-squares fit is at
    least as good as R's on every series (R's Nelder-Mead can stop early), so forecasts differ slightly per series;
    the check is at decision level.
"""
from pathlib import Path

import numpy as np
import pytest

from tsfstudio.engines import py_models as M
from tsfstudio.metrics import mase, smape

G = Path(__file__).parent / "golden"
CASES = [("nn5_weekly", 52), ("nn5_daily", 7)]


def _load(d: Path, stem: str) -> np.ndarray:
    for ext in (".csv", ".csv.gz"):
        if (d / f"{stem}{ext}").is_file():
            return np.loadtxt(d / f"{stem}{ext}", delimiter=",", ndmin=2)
    raise FileNotFoundError(d / stem)


@pytest.fixture(scope="module", params=CASES, ids=[c for c, _ in CASES])
def case(request):
    name, m = request.param
    d = G / name
    train = [r[~np.isnan(r)] for r in _load(d, "train")]
    return name, m, train, _load(d, "actuals"), _load(d, "r_m4_naive2"), _load(d, "r_m4_comb")


def test_naive2_identical_to_r(case):
    _, m, train, act, ref, _ = case
    f = np.vstack([M.naive2(y, act.shape[1], m) for y in train])
    np.testing.assert_allclose(f, ref, rtol=1e-9, atol=1e-9)


def test_comb_shd_matches_r_at_decision_level(case):
    name, m, train, act, _, ref = case
    f = np.vstack([M.comb_shd(y, act.shape[1], m) for y in train])
    rel = np.abs(f - ref) / np.maximum(np.abs(ref), 1e-9)
    assert np.median(rel) < 5e-3, f"{name}: median relative difference {np.median(rel):.2e}"
    d_smape = (np.nanmean(smape(f, act)) - np.nanmean(smape(ref, act))) / np.nanmean(smape(ref, act))
    d_mase = (np.nanmean(mase(f, act, train, m)) - np.nanmean(mase(ref, act, train, m))) / np.nanmean(mase(ref, act, train, m))
    assert abs(d_smape) < 0.01 and abs(d_mase) < 0.01, f"{name}: dSMAPE {100 * d_smape:.2f}%, dMASE {100 * d_mase:.2f}%"
