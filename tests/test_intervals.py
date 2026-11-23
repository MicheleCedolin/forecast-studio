"""Split-conformal prediction intervals and their measures (tsfstudio.intervals, runner integration)."""
import numpy as np
import pandas as pd
import pytest

from tsfstudio import intervals as IV
from tsfstudio.runner import RunConfig, RunError, run
from tsfstudio.tsf.reader import read_tsf
from tsfstudio.tsf.writer import write_tsf


def test_measures_by_hand():
    f = np.array([[10.0, 10.0]])
    y = np.array([[12.0, 7.0]])
    b = {0.025: np.array([[8.0, 8.0]]), 0.975: np.array([[13.0, 13.0]])}
    m = IV.interval_metrics(f, y, np.array([2.0]), [95], b, ["a"]).loc["a"]
    assert m["Coverage95"] == 0.5
    # widths 5, 5; the second point misses below by 1: 5 + 2/0.05 * 1 = 45 -> mean 25, / scale 2
    assert m["MSIS"] == pytest.approx(12.5)
    pin = lambda p, d: max(p * d, (p - 1) * d)  # noqa: E731
    wql = np.mean([2 * sum(pin(p, yy - bb) for yy, bb in zip([12, 7], bnd)) / 19
                   for p, bnd in ((0.5, [10, 10]), (0.025, [8, 8]), (0.975, [13, 13]))])
    assert m["WQL"] == pytest.approx(wql)


def test_probs_and_quantiles():
    assert IV.probs_for([80, 95]) == [0.025, 0.1, 0.9, 0.975]
    rng = np.random.default_rng(0)
    e = rng.normal(size=(4000, 3))
    q = IV.conformal_quantiles(np.zeros((4000, 3)), e, np.ones(4000), [0.025, 0.975])
    np.testing.assert_allclose(q, [[-1.96] * 3, [1.96] * 3], atol=0.12)


def _data(path, n=300, length=80):
    rng = np.random.default_rng(7)
    vals = [100 + np.cumsum(rng.normal(size=length)) for _ in range(n)]
    write_tsf(path, relation="x", attributes=[("series_name", "string")], attr_rows=[(f"S{i + 1}",) for i in range(n)],
              values=vals, frequency="monthly", horizon=6, missing=False, equal_length=True)
    return read_tsf(path)


def test_run_with_intervals_is_calibrated(tmp_path):
    ds = _data(tmp_path / "x.tsf")
    r = run(ds, RunConfig(engine="python", method="naive", intervals=(80, 95), output_dir=tmp_path / "runs",
                          write_xlsx=False))
    s = r.summary.set_index("Metric")["Mean"]
    assert abs(s["Coverage80"] - 0.80) < 0.04 and abs(s["Coverage95"] - 0.95) < 0.03
    assert np.isfinite(s["MSIS"]) and np.isfinite(s["WQL"])
    iv = pd.read_csv(r.run_dir / "intervals.csv")
    assert sorted(iv["quantile"].unique()) == [0.025, 0.1, 0.9, 0.975] and len(iv) == 4 * 300
    lo = iv[iv["quantile"] == 0.025].iloc[:, 2:].to_numpy()
    hi = iv[iv["quantile"] == 0.975].iloc[:, 2:].to_numpy()
    assert (lo <= r.forecasts + 1e-9).all() and (hi >= r.forecasts - 1e-9).all()
    assert r.manifest["split"]["interval_levels"] == [80, 95]


def test_intervals_with_rolling_origin_and_bad_levels(tmp_path):
    ds = _data(tmp_path / "x.tsf", n=40)
    r = run(ds, RunConfig(engine="python", method="naive", intervals=(90,), origins=2, output_dir=tmp_path / "runs",
                          write_xlsx=False))
    eo = pd.read_csv(r.run_dir / "errors_by_origin.csv")
    assert "Coverage90" in eo.columns and set(eo.origin) == {1, 2}
    with pytest.raises(RunError):
        run(ds, RunConfig(engine="python", method="naive", intervals=(100,), output_dir=tmp_path / "runs"))
