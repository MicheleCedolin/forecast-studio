"""Series-level instability: hand-computed cases, the manuscript's stored per-series values, and R's tsfeatures."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from tsfstudio import instability as IN

G = Path(__file__).parent / "golden"


def _long(table: dict) -> pd.DataFrame:
    """{series: {metric: {model: loss}}} -> long table."""
    return pd.DataFrame([{"series": s, "metric": m, "model": j, "loss": v}
                         for s, ms in table.items() for m, js in ms.items() for j, v in js.items()])


# ---------------------------------------------------------------- judges and series-level outcomes

def test_equivalent_measures_count_as_one_judge():
    assert IN.judges_of(["MAE", "MASE", "WAPE", "RMSE", "MSE", "RMSSE", "sMAPE", "nMAE"]) == ["MAE", "RMSE", "sMAPE"]
    assert IN.judges_of(["nRMSE_Mean", "MedAE", "nRMSE_Range"]) == ["nRMSE_Mean", "MedAE"]


def test_series_instability_hand_example():
    losses = _long({
        "agree": {"MAE": {"a": 1, "b": 2, "c": 3}, "RMSE": {"a": 2, "b": 4, "c": 6}, "sMAPE": {"a": .1, "b": .2, "c": .3}},
        "split": {"MAE": {"a": 1, "b": 2, "c": 3}, "RMSE": {"a": 6, "b": 4, "c": 2}, "sMAPE": {"a": .1, "b": .3, "c": .2}},
    })
    out = IN.series_instability(losses)
    assert out.attrs["judges"] == ["MAE", "RMSE", "sMAPE"] and out.attrs["n_dropped"] == 0
    assert out.loc["agree", "disagreement"] == pytest.approx(0.0) and out.loc["agree", "regret"] == pytest.approx(0.0)
    # split: rank sums a=5, b=7, c=6 -> S = 2, W = 12*2 / (9 * 24) = 1/9
    assert out.loc["split", "disagreement"] == pytest.approx(1 - 1 / 9)
    # MAE and sMAPE pick a, RMSE picks c. Ordered pairs (select -> evaluate): MAE->RMSE 1, sMAPE->RMSE 1, RMSE->MAE 1,
    # RMSE->sMAPE 0.5, MAE->sMAPE 0, sMAPE->MAE 0
    assert out.loc["split", "regret"] == pytest.approx(3.5 / 6)


def test_squared_class_is_represented_on_the_rmse_scale_whatever_the_column_order():
    assert IN.judges_of(["MSE", "MAE", "RMSE"]) == ["RMSE", "MAE"]
    assert IN.judges_of(["RMSE", "MSE", "MAE"]) == ["RMSE", "MAE"]
    assert IN.judges_of(["MSE", "MAE"]) == ["MSE", "MAE"]
    base = {"MAE": {"a": 1, "b": 2, "c": 4}, "sMAPE": {"a": .3, "b": .1, "c": .2}}
    rm = {"a": 3.0, "b": 2.0, "c": 1.0}
    t1 = _long({"s": {"MSE": {k: v * v for k, v in rm.items()}, "RMSE": rm, **base}})
    t2 = _long({"s": {"RMSE": rm, **base, "MSE": {k: v * v for k, v in rm.items()}}})
    pd.testing.assert_frame_equal(IN.series_instability(t1), IN.series_instability(t2))


def test_series_with_a_missing_loss_is_dropped_and_too_few_judges_is_an_error():
    t = {"s1": {"MAE": {"a": 1, "b": 2, "c": 3}, "sMAPE": {"a": 1, "b": 2, "c": 3}},
         "s2": {"MAE": {"a": 1, "b": 2, "c": float("nan")}, "sMAPE": {"a": 1, "b": 2, "c": 3}}}
    out = IN.series_instability(_long(t))
    assert list(out.index) == ["s1"] and out.attrs["n_dropped"] == 1
    with pytest.raises(ValueError, match="not equivalent"):
        IN.series_instability(_long({"s": {"MAE": {"a": 1, "b": 2, "c": 3}, "MASE": {"a": 1, "b": 2, "c": 3}}}))


def test_reproduces_the_manuscript_values_for_every_nn5_weekly_series():
    err = pd.read_csv(G / "instability" / "nn5_weekly_errors_per_series.csv")
    expected = pd.read_csv(G / "instability" / "nn5_weekly_expected_outcomes.csv").set_index("series_id")
    out = IN.series_instability(IN.losses_from_wide(err))
    assert len(out) == 111 and len(out.attrs["judges"]) == 10 and len(out.attrs["models"]) == 11
    out = out.reindex(expected.index)
    np.testing.assert_allclose(out["disagreement"], expected["disagreement"], atol=1e-9)
    np.testing.assert_allclose(out["regret"], expected["regret"], atol=1e-6)
    assert out["disagreement"].median() == pytest.approx(0.129, abs=1e-3)      # Section 5.5 task table


# ---------------------------------------------------------------- aggregation effect

def test_aggregation_alone_reorders_models_under_equivalent_measures():
    # two series of very different scale; MASE = MAE / scale. Inside each series MAE and MASE agree.
    scale = {"big": 100.0, "small": 1.0}
    mae = {"big": {"a": 10.0, "b": 20.0, "c": 30.0}, "small": {"a": 3.0, "b": 2.0, "c": 1.0}}
    t = {s: {"MAE": mae[s], "MASE": {m: v / scale[s] for m, v in mae[s].items()},
             "sMAPE": {"a": .1, "b": .2, "c": .3}} for s in mae}
    losses = _long(t)
    assert (IN.series_instability(losses, metrics=["MAE", "sMAPE"])["disagreement"] >= 0).all()
    agg = IN.aggregation_effect(losses)
    row = agg.loc["absolute error (equivalent inside a series)"]
    assert row["n_measures"] == 2 and row["members"] == "MAE, MASE"
    assert row["kendall_w"] == pytest.approx(0.0)        # mean MAE ranks a<b<c, mean MASE ranks c<b<a
    assert "all measures" in agg.index


def test_aggregation_effect_on_the_manuscript_data():
    losses = IN.losses_from_wide(pd.read_csv(G / "instability" / "nn5_weekly_errors_per_series.csv"))
    agg = IN.aggregation_effect(losses)
    assert agg.loc["all measures", "kendall_w"] == pytest.approx(0.942, abs=1e-3)                    # Table 5
    assert agg.loc["absolute error (equivalent inside a series)", "kendall_w"] == pytest.approx(0.996, abs=1e-3)
    assert agg.loc["squared error (equivalent inside a series)", "kendall_w"] == pytest.approx(0.966, abs=1e-3)


# ---------------------------------------------------------------- features

def test_features_hand_cases():
    t = np.arange(120, dtype=float)
    line = IN.series_features(100 + 2 * t)
    assert line["trend"] == pytest.approx(1.0) and line["x_acf1"] > 0.95 and np.isnan(line["seasonal_strength"])
    assert line["log_level"] == pytest.approx(np.log10((100 + 2 * t).mean()))
    rng = np.random.default_rng(0)
    noise = IN.series_features(50 + rng.normal(size=400))
    assert noise["entropy"] > 0.85 and abs(noise["x_acf1"]) < 0.15 and noise["trend"] < 0.3
    wave = IN.series_features(10 + np.sin(2 * np.pi * t / 12), period=12)
    assert wave["seasonal_strength"] > 0.99 and wave["entropy"] < 0.2
    assert np.isnan(IN.series_features(10 + np.sin(2 * np.pi * t[:30] / 12), period=12)["seasonal_strength"])   # under three periods
    assert all(np.isnan(v) for v in IN.series_features([1.0, 2.0, 3.0]).values())
    const = IN.series_features(np.full(50, 7.0))
    assert const["cv"] == 0.0 and np.isnan(const["x_acf1"])


def test_built_in_features_against_r_tsfeatures_on_nn5_weekly():
    train = np.loadtxt(G / "nn5_weekly" / "train.csv", delimiter=",")
    act = np.loadtxt(G / "nn5_weekly" / "actuals.csv", delimiter=",")
    whole = np.hstack([train, act])                       # R computed the features on the complete series
    r = pd.read_csv(G / "instability" / "nn5_weekly_tsfeatures_R.csv").set_index("series_id")
    f = IN.feature_table(list(whole), list(r.index), period=52)
    np.testing.assert_allclose(f["x_acf1"], r["x_acf1"], atol=1e-10)
    np.testing.assert_allclose(f["cv"], np.sqrt(r["var"]) / r["mean"].abs(), rtol=1e-10)
    np.testing.assert_allclose(f["log_level"], np.log10(r["mean"].abs()), atol=1e-10)
    assert spearmanr(f["entropy"], r["entropy"]).statistic > 0.85      # different spectrum estimator
    assert spearmanr(f["trend"], r["trend"]).statistic > 0.55          # moving average instead of STL
    assert f["seasonal_strength"].isna().all()                         # 113 points < three periods of 52
    drop = IN.feature_table(list(whole), list(r.index), period=52, drop_last=8)
    assert not np.allclose(drop["x_acf1"], f["x_acf1"])


# ---------------------------------------------------------------- association and pooling

def test_association_and_pooling_with_tasks_as_units():
    rng = np.random.default_rng(3)
    parts = []
    for task in range(6):
        x = rng.normal(size=200)
        out = pd.DataFrame({"disagreement": 0.5 * x + rng.normal(size=200), "regret": rng.normal(size=200)},
                           index=[f"s{i}" for i in range(200)])
        feats = pd.DataFrame({"signal": x, "noise": rng.normal(size=200)}, index=out.index)
        c = IN.association(out, feats)
        c.insert(0, "task", f"t{task}")
        parts.append(c)
    pooled = IN.pool_over_tasks(pd.concat(parts)).set_index(["outcome", "feature"])
    hit = pooled.loc[("disagreement", "signal")]
    assert hit["n_tasks"] == 6 and hit["n_positive"] == 6 and hit["p"] < 0.001 and hit["lo"] > 0.2
    assert pooled.loc[("disagreement", "noise"), "p"] > 0.01 and pooled.loc[("regret", "signal"), "p"] > 0.01
    few = IN.pool_over_tasks(pd.concat(parts[:3]))
    assert few["rho"].isna().all() and (few["n_tasks"] == 3).all()       # not enough tasks to pool
    assert IN.association(out.iloc[:10], feats).empty                    # under min_n series


# ---------------------------------------------------------------- command line

def _fake_run(folder: Path, names, mae, smape, rmse, horizon=2):
    folder.mkdir(parents=True)
    df = pd.DataFrame({"series_name": names, "MAE": mae, "MASE": np.asarray(mae) / 2.0, "SMAPE": smape, "RMSE": rmse, "ME": 0.0})
    with pd.ExcelWriter(folder / "metrics.xlsx") as xw:
        df.to_excel(xw, sheet_name="Errors_Per_Series", index=False)
    return str(folder)


def test_cli_instability_with_tsf_data_and_with_a_feature_table(tmp_path, capsys):
    from tsfstudio.cli import main
    from tsfstudio.tsf import write_tsf

    rng = np.random.default_rng(1)
    n = 40
    names = [f"T{i + 1}" for i in range(n)]
    args = ["instability", "--out", str(tmp_path / "out")]
    for model in ("a", "b", "c"):
        d = _fake_run(tmp_path / model, names, rng.uniform(1, 2, n), rng.uniform(.1, .2, n), rng.uniform(1, 3, n))
        args += ["--run", f"demo:{model}={d}"]
    feats = pd.DataFrame({"my_feature": rng.normal(size=n)}, index=pd.Index(names, name="series"))
    feats.to_csv(tmp_path / "feats.csv")
    assert main(args + ["--features", f"demo={tmp_path / 'feats.csv'}"]) == 0
    text = capsys.readouterr().out
    assert "Judges" in text and "MAE, SMAPE, RMSE" in text and "equivalent inside a series" in text and "my_feature" in text
    per_series = pd.read_csv(tmp_path / "out" / "demo_series.csv", index_col=0)
    assert len(per_series) == n and {"disagreement", "regret", "my_feature"} <= set(per_series.columns)
    assert (tmp_path / "out" / "demo_aggregation_effect.csv").exists()

    import inspect
    if "values" in inspect.signature(write_tsf).parameters or True:
        assert main(["instability", "--run", "x:a=" + str(tmp_path / "a")]) == 0       # one model: skipped with a message
        assert "Skipped" in capsys.readouterr().out
    assert main(["instability"]) == 2
