"""Task-level ranking reliability: hand-computed cases and the stored outputs of the manuscript's reanalysis."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tsfstudio import reliability as RL

GOLDEN = Path(__file__).parent / "golden" / "ranking_reliability" / "competition_family_balanced_task_scores.csv"


def _long(table: dict) -> pd.DataFrame:
    """{task: {metric: {model: loss}}} -> long table."""
    return pd.DataFrame([{"task": t, "metric": m, "model": j, "loss": v}
                         for t, ms in table.items() for m, js in ms.items() for j, v in js.items()])


def _paper_scores() -> pd.DataFrame:
    return pd.read_csv(GOLDEN).pivot(index="task", columns="model", values="score")


# ---------------------------------------------------------------- ranks and scores

def test_common_support_drops_models_missing_under_any_metric():
    losses = _long({"t": {"MAE": {"a": 1.0, "b": 2.0, "c": 3.0}, "MAPE": {"a": 5.0, "b": float("nan"), "c": 4.0}}})
    r = RL.common_support_ranks(losses)
    assert sorted(r["model"].unique()) == ["a", "c"]
    assert set(r["n_models"]) == {2}
    got = r.set_index(["metric", "model"])["normalized_rank"]
    assert got[("MAE", "a")] == 0.0 and got[("MAE", "c")] == 1.0
    assert got[("MAPE", "a")] == 1.0 and got[("MAPE", "c")] == 0.0


def test_family_balancing_stops_repeated_votes():
    # three percentage metrics prefer a, one absolute metric prefers b: equal metric weights would favour a, families tie
    losses = _long({"t": {"MAPE": {"a": 1, "b": 2}, "sMAPE": {"a": 1, "b": 2}, "MAAPE": {"a": 1, "b": 2}, "MAE": {"a": 2, "b": 1}}})
    s = RL.family_balanced_scores(RL.common_support_ranks(losses))
    assert s.loc["t", "a"] == pytest.approx(0.5) and s.loc["t", "b"] == pytest.approx(0.5)
    s2 = RL.family_balanced_scores(RL.common_support_ranks(losses), omit_family="absolute")
    assert s2.loc["t", "a"] == 0.0 and s2.loc["t", "b"] == 1.0


def test_unknown_metric_and_bias_measures_are_rejected():
    r = RL.common_support_ranks(_long({"t": {"ME": {"a": 1, "b": 2}}}))
    with pytest.raises(ValueError, match="no known family"):
        RL.family_balanced_scores(r)


# ---------------------------------------------------------------- Bradley-Terry

def test_bradley_terry_two_models_closed_form():
    # a beats b on 3 of 4 tasks -> ability ratio 3 -> log-ability difference log(3)
    s = pd.DataFrame({"a": [0, 0, 0, 1], "b": [1, 1, 1, 0]}, index=list("wxyz"), dtype=float)
    fit = RL.fit_bradley_terry(s)
    assert fit["a"] - fit["b"] == pytest.approx(np.log(3.0), abs=1e-8)
    assert fit.sum() == pytest.approx(0.0, abs=1e-9)


def test_bradley_terry_is_task_balanced_and_splits_ties():
    wins, nij = RL._pair_statistics(pd.DataFrame(
        {"a": [0.0, 0.2], "b": [0.5, 0.2], "c": [1.0, np.nan], "d": [0.7, np.nan]}, index=["big", "small"]))
    assert wins.sum(axis=1) == pytest.approx([1.0, 1.0])            # 6 pairs vs 1 pair, same total weight
    assert wins[1] == pytest.approx([0.5, 0.5, 0.0, 0.0])           # exact tie -> half wins
    assert nij[1, 0, 1] == pytest.approx(1.0) and nij[0, 0, 1] == pytest.approx(1 / 6)


def test_bradley_terry_reproduces_the_manuscript_ordering():
    fit = RL.fit_bradley_terry(_paper_scores())
    expected = {"TBATS": 1.171, "ETS": 1.051, "N-BEATS": 0.845, "DeepAR": 0.540, "Theta": 0.521, "FFNN": -0.144,
                "Pooled Regression": -0.194, "ARIMA": -0.542, "WaveNet": -0.792, "Transformer": -1.140, "SES": -1.315}
    assert list(fit.index) == list(expected)
    for model, value in expected.items():
        assert fit[model] == pytest.approx(value, abs=6e-4), model


def test_bootstrap_matches_manuscript_coverage_and_is_reproducible():
    s = _paper_scores()
    a = RL.bootstrap_bradley_terry(s, n_boot=300, seed=7)
    b = RL.bootstrap_bradley_terry(s, n_boot=300, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert a.loc["TBATS", "coverage"] == 8 and a.loc["ETS", "coverage"] == 9 and a.loc["ARIMA", "coverage"] == 15
    assert a.loc["TBATS", "mean_score"] == pytest.approx(0.247, abs=5e-4)
    assert a["p_first"].sum() == pytest.approx(1.0) and a["p_top_k"].sum() == pytest.approx(3.0)
    assert (a["lo"] <= a["hi"]).all()
    # the leading group of the paper, with a leader that is not certain
    assert set(a["p_top_k"].sort_values(ascending=False).index[:3]) == {"TBATS", "ETS", "N-BEATS"}
    assert 0.3 < a.loc["TBATS", "p_first"] < 0.7


# ---------------------------------------------------------------- Skillings-Mack

def test_skillings_mack_equals_friedman_on_complete_blocks():
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(9, 5)) + np.arange(5) * 0.4, columns=list("abcde"))
    sm = RL.skillings_mack(x, n_perm=200, seed=1)
    fr = stats.friedmanchisquare(*[x[c] for c in x.columns]).statistic
    assert sm.statistic == pytest.approx(fr, rel=1e-10)
    assert sm.n_tasks == 9 and sm.n_models == 5


def test_skillings_mack_reproduces_the_manuscript_statistic():
    sm = RL.skillings_mack(_paper_scores(), n_perm=4000, seed=3)
    assert sm.statistic == pytest.approx(45.941534, abs=1e-5)
    assert sm.n_tasks == 15 and sm.n_models == 11
    assert sm.p_value < 0.005 and sm.p_value == (sm.exceedances + 1) / (sm.n_perm + 1)


def test_skillings_mack_permutation_is_calibrated_under_the_null():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(12, 6))
    x[rng.random(x.shape) < 0.25] = np.nan
    sm = RL.skillings_mack(pd.DataFrame(x), n_perm=2000, seed=2)
    assert sm.p_value > 0.01


# ---------------------------------------------------------------- agreement and regret

def test_kendalls_w_extremes_and_ties():
    assert RL.kendalls_w(pd.DataFrame([[1, 2, 3, 4]] * 3)) == pytest.approx(1.0)
    assert RL.kendalls_w(pd.DataFrame([[1, 2, 3, 4], [4, 3, 2, 1]])) == pytest.approx(0.0)
    # hand value: rank sums 2.5, 3.5, 6 -> S = 6.5, tie term 6, W = 12*6.5 / (4*24 - 2*6)
    assert RL.kendalls_w(pd.DataFrame([[1.5, 1.5, 3], [1, 2, 3]])) == pytest.approx(78 / 84)


def test_selection_regret_hand_example():
    t = pd.DataFrame({"a": [1.0, 10.0], "b": [2.0, 4.0], "c": [5.0, 6.0]}, index=["m1", "m2"])
    reg = RL.selection_regret(t)
    assert reg.loc["m1", "m1"] == 0.0 and reg.loc["m2", "m2"] == 0.0
    assert reg.loc["m1", "m2"] == pytest.approx(1.0)        # m1 picks a, the worst model under m2
    assert reg.loc["m2", "m1"] == pytest.approx(0.25)       # m2 picks b: (2 - 1) / (5 - 1)


def test_agreement_by_task_and_leave_one_family_out():
    losses = _long({
        "agree": {"MAE": {"a": 1, "b": 2, "c": 3}, "RMSE": {"a": 1, "b": 2, "c": 3}, "MASE": {"a": 1, "b": 2, "c": 3}},
        "split": {"MAE": {"a": 1, "b": 2, "c": 3}, "RMSE": {"a": 3, "b": 2, "c": 1}, "MASE": {"a": 1, "b": 3, "c": 2}},
    })
    ranks = RL.common_support_ranks(losses)
    ag = RL.agreement_by_task(ranks)
    assert ag.loc["agree", "kendall_w"] == pytest.approx(1.0) and ag.loc["agree", "mean_regret"] == 0.0
    assert ag.loc["agree", "zero_regret_share"] == 1.0
    assert ag.loc["split", "kendall_w"] < 0.5 and ag.loc["split", "max_regret"] == pytest.approx(1.0)
    assert ag.index[0] == "split"                              # sorted from least to most agreement
    loo = RL.leave_one_family_out(ranks)
    assert list(loo.index) == ["absolute", "scaled", "squared"]
    assert loo["spearman_vs_full"].between(-1, 1).all()


# ---------------------------------------------------------------- run folders and CLI

def _fake_run(folder: Path, mae, mase):
    folder.mkdir(parents=True)
    df = pd.DataFrame({"series_name": [f"T{i}" for i in range(len(mae))], "MAE": mae, "MASE": mase, "ME": 0.0})
    with pd.ExcelWriter(folder / "metrics.xlsx") as xw:
        df.to_excel(xw, sheet_name="Errors_Per_Series", index=False)
    return str(folder)


def test_losses_from_runs_uses_finite_means_and_skips_non_losses(tmp_path):
    runs = {("t1", "a"): _fake_run(tmp_path / "a", [1.0, 3.0, np.nan], [0.5, 0.7, 0.9]),
            ("t1", "b"): _fake_run(tmp_path / "b", [4.0, 4.0, 4.0], [1.0, 1.0, 1.0])}
    losses = RL.losses_from_runs(runs)
    assert set(losses["metric"]) == {"MAE", "MASE"}
    got = losses.set_index(["model", "metric"])["loss"]
    assert got[("a", "MAE")] == pytest.approx(2.0) and got[("a", "MASE")] == pytest.approx(0.7)


def test_cli_reliability_from_runs_and_from_csv(tmp_path, capsys):
    from tsfstudio.cli import main

    args = ["reliability", "--boot", "50", "--perm", "200"]
    for task, shift in (("t1", 0.0), ("t2", 0.1), ("t3", -0.1)):
        for model, level in (("a", 1.0), ("b", 2.0), ("c", 3.0)):
            d = _fake_run(tmp_path / f"{task}_{model}", [level + shift] * 3, [level / 2 + shift] * 3)
            args += ["--run", f"{task}:{model}={d}"]
    out_dir = tmp_path / "out"
    assert main(args + ["--out", str(out_dir)]) == 0
    text = capsys.readouterr().out
    assert "Bradley-Terry" in text and "Skillings-Mack" in text and "Kendall" in text
    bt = pd.read_csv(out_dir / "bradley_terry.csv", index_col=0)
    assert list(bt.index) == ["a", "b", "c"]
    assert (out_dir / "agreement_by_task.csv").exists() and (out_dir / "family_balanced_scores.csv").exists()

    losses = pd.read_csv(out_dir / "losses.csv")
    assert main(["reliability", "--losses", str(out_dir / "losses.csv"), "--boot", "20", "--perm", "50"]) == 0
    assert len(losses) == 18
    assert main(["reliability", "--boot", "5"]) == 2           # neither --run nor --losses
