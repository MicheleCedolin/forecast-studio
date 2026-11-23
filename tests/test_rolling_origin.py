"""Rolling-origin evaluation: each origin is an ordinary run on the data before it; series are scored by their mean."""
import numpy as np
import pandas as pd
import pytest

from tsfstudio.runner import RunConfig, RunError, run
from tsfstudio.tsf.reader import read_tsf
from tsfstudio.tsf.writer import write_tsf


def _write(path, vals):
    write_tsf(path, relation="x", attributes=[("series_name", "string")],
              attr_rows=[(f"S{i + 1}",) for i in range(len(vals))], values=vals, frequency="monthly", horizon=6,
              missing=False, equal_length=False)
    return read_tsf(path)


@pytest.fixture()
def data(tmp_path):
    rng = np.random.default_rng(1)
    vals = [60 + np.cumsum(rng.normal(size=50)) for _ in range(6)] + [np.arange(19.0)]
    return tmp_path, vals


def test_origins_match_truncated_runs_and_average(data):
    d, vals = data
    r3 = run(_write(d / "x.tsf", vals), RunConfig(engine="python", method="ses", origins=3, output_dir=d / "runs",
                                                  write_xlsx=False))
    sp = r3.manifest["split"]
    assert sp["origins"] == 3 and sp["origin_step"] == 6 and sp["origin_cuts"] == [12, 6, 0]
    assert sp["n_excluded"] == 1                              # 19 < 6 + 2 + 2 * 6
    eo = pd.read_csv(r3.run_dir / "errors_by_origin.csv")
    assert eo.groupby("origin").size().to_dict() == {1: 6, 2: 6, 3: 6}
    for cut, origin in ((12, 1), (6, 2), (0, 3)):
        ds = _write(d / f"y{cut}.tsf", [v[: len(v) - cut] for v in vals[:6]])
        r1 = run(ds, RunConfig(engine="python", method="ses", output_dir=d / "runs", write_xlsx=False))
        np.testing.assert_allclose(eo[eo.origin == origin]["MASE"].to_numpy(), r1.per_series["MASE"].to_numpy())
    mean = eo.groupby("series_name", sort=False)["MASE"].mean().to_numpy()
    np.testing.assert_allclose(r3.per_series["MASE"].to_numpy(), mean)
    # the usual files keep the latest origin
    fc = pd.read_csv(r3.run_dir / "forecasts.csv")
    fo = pd.read_csv(r3.run_dir / "forecasts_by_origin.csv")
    np.testing.assert_allclose(fc.iloc[:, 1:].to_numpy(), fo[fo.origin == 3].iloc[:, 3:].to_numpy())
    assert set(pd.read_csv(r3.run_dir / "summary_by_origin.csv").origin) == {1, 2, 3}


def test_custom_step_and_single_origin_unchanged(data):
    d, vals = data
    ds = _write(d / "x.tsf", vals[:6])
    r = run(ds, RunConfig(engine="python", method="naive", origins=4, origin_step=2, output_dir=d / "runs",
                          write_xlsx=False))
    assert r.manifest["split"]["origin_cuts"] == [6, 4, 2, 0]
    one = run(ds, RunConfig(engine="python", method="naive", output_dir=d / "runs", write_xlsx=False))
    assert one.manifest["split"]["origins"] == 1 and "errors_by_origin_csv" not in one.manifest["outputs"]
    with pytest.raises(RunError):
        run(ds, RunConfig(engine="python", method="naive", origins=0, output_dir=d / "runs", write_xlsx=False))
