"""WRMSSE over the M5 hierarchy (tsfstudio.wrmsse) and its runner integration."""
import numpy as np
import pandas as pd
import pytest

from tsfstudio import wrmsse as W


def _toy(seed=0, T=60, H=4):
    rng = np.random.default_rng(seed)
    rows, train, test = [], [], []
    for state in ("CA", "TX"):
        for s in (1, 2):
            for cat in ("FOODS", "HOBBIES"):
                for item in (1, 2):
                    rows.append({"item_id": f"{cat}_{item}", "dept_id": f"{cat}_1", "cat_id": cat,
                                 "store_id": f"{state}_{s}", "state_id": state})
                    y = rng.poisson(3 + item, size=T + H).astype(float)
                    train.append(y[:T])
                    test.append(y[T:])
    return pd.DataFrame(rows), train, np.array(test)


def test_perfect_forecast_is_zero_and_twelve_levels():
    attrs, train, test = _toy()
    r = W.wrmsse(train, test, test, attrs)
    assert r["WRMSSE"] == 0 and len(r["levels"]) == 12 and r["weights"] == "unit sales"
    assert [lv["n_series"] for lv in r["levels"]] == [1, 2, 4, 2, 2, 4, 4, 8, 8, 4, 8, 16]


def test_total_level_by_hand():
    attrs, train, test = _toy(1)
    fc = test + 1.0
    r = W.wrmsse(train, test, fc, attrs)
    y = np.sum(train, axis=0)
    y = y[np.flatnonzero(y)[0]:]
    expected = np.sqrt(np.mean((test.sum(0) - fc.sum(0)) ** 2) / np.mean(np.diff(y) ** 2))
    assert r["levels"][0]["WRMSSE"] == pytest.approx(expected)
    assert r["WRMSSE"] == pytest.approx(np.mean([lv["WRMSSE"] for lv in r["levels"]]))


def test_dollar_weights_change_the_bottom_levels():
    attrs, train, test = _toy(2)
    fc = test * 1.3
    days = pd.date_range("2016-01-01", periods=70, freq="D")
    cal = pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "wm_yr_wk": 11600 + np.arange(70) // 7})
    sp = pd.DataFrame([{"store_id": r.store_id, "item_id": r.item_id, "wm_yr_wk": w,
                        "sell_price": 10.0 if r.cat_id == "HOBBIES" else 1.0}
                       for r in attrs.itertuples() for w in cal.wm_yr_wk.unique()])
    unit = W.wrmsse(train, test, fc, attrs)
    dollar = W.wrmsse(train, test, fc, attrs, (sp, cal), "2016-02-29")
    assert dollar["weights"].startswith("dollar")
    assert dollar["levels"][0]["WRMSSE"] == unit["levels"][0]["WRMSSE"]        # one series: weights do not matter
    assert dollar["levels"][11]["WRMSSE"] != pytest.approx(unit["levels"][11]["WRMSSE"])


def test_runner_writes_wrmsse_for_hierarchical_data(tmp_path):
    from tsfstudio.runner import RunConfig, run
    from tsfstudio.tsf.reader import read_tsf
    from tsfstudio.tsf.writer import write_tsf
    attrs, train, test = _toy(3)
    vals = [np.r_[a, b] for a, b in zip(train, test)]
    cols = ["series_name", *W.HIERARCHY]
    rows = [(f"S{i}", *[r[c] for c in W.HIERARCHY]) for i, r in attrs.iterrows()]
    write_tsf(tmp_path / "m.tsf", relation="m", attributes=[(c, "string") for c in cols], attr_rows=rows, values=vals,
              frequency="daily", horizon=4, missing=False, equal_length=True)
    r = run(read_tsf(tmp_path / "m.tsf"), RunConfig(engine="python", method="naive", output_dir=tmp_path / "runs",
                                                    write_xlsx=False))
    assert r.manifest["wrmsse"]["complete"] and np.isfinite(r.manifest["wrmsse"]["value"])
    assert len(pd.read_csv(r.run_dir / "wrmsse.csv")) == 12


def test_official_m5_seasonal_naive_benchmark():
    """With M5's prices (data/m5) the seasonal naive scores the published sNaive WRMSSE, 0.847 (evaluation phase)."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    tsf, m5 = root / "data" / "tsf" / "m5_daily_dataset.tsf", root / "data" / "m5"
    prices = W.dollar_weights_source(m5)
    if prices is None or not tsf.is_file():
        pytest.skip("M5 data or price files not present")
    from tsfstudio.tsf.reader import read_tsf
    ds = read_tsf(tsf)
    train = [np.asarray(v[:-28], float) for v in ds.values]
    test = np.array([v[-28:] for v in ds.values])
    fc = np.array([np.tile(t[-7:], 4) for t in train])
    last = (pd.Timestamp(ds.attrs["start_timestamp"].iloc[0]) + pd.Timedelta(days=len(train[0]) - 1)).strftime("%Y-%m-%d")
    r = W.wrmsse(train, test, fc, ds.attrs, prices, last)
    assert r["weights"].startswith("dollar") and r["WRMSSE"] == pytest.approx(0.847, abs=5e-4)
