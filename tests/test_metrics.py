"""Metrik testleri: elle hesaplanmış beklenen değerler (R tanımlarına göre)."""
import numpy as np
import pytest

from tsfstudio import metrics as M
from tsfstudio.seasonality import primary_period

F = np.array([[2.0, 4.0], [1.0, 1.0], [0.0, 0.0]])
A = np.array([[1.0, 5.0], [0.0, 0.0], [0.0, 0.0]])


def test_smape_hand_computed_and_zero_zero_is_nan():
    s = M.smape(F, A)
    assert s[0] == pytest.approx((2 / 3 + 2 / 9) / 2)
    assert s[1] == pytest.approx(2.0)          # |f-a|=1, |f|+|a|=1 -> 2
    assert np.isnan(s[2])                      # 0/0 -> NaN (R: rowMeans na.rm -> NaN)


def test_msmape_epsilon():
    s = M.msmape(F, A)
    assert s[0] == pytest.approx((2 / 3.1 + 2 / 9.1) / 2)
    assert s[1] == pytest.approx(2 / 1.1)
    assert s[2] == pytest.approx(0.0)          # payda max(0.6, 0.1) = 0.6 -> 0


def test_mae_rmse():
    assert M.mae(F, A).tolist() == [1.0, 1.0, 0.0]
    assert M.rmse(F, A).tolist() == [1.0, 1.0, 0.0]


def test_wape_and_nrmse_denominator_zero_gives_nan():
    w = M.wape(F, A)
    assert w[0] == pytest.approx(2 / 6 * 100)
    assert np.isnan(w[1]) and np.isnan(w[2])
    nm, nr = M.nrmse_mean(F, A), M.nrmse_range(F, A)
    assert nm[0] == pytest.approx(1 / 3)
    assert nr[0] == pytest.approx(1 / 4)
    assert np.isnan(nm[1]) and np.isnan(nr[1])


def test_nan_padding_is_ignored():
    f = np.array([[1.0, 2.0, np.nan]])
    a = np.array([[2.0, 2.0, np.nan]])
    assert M.mae(f, a)[0] == pytest.approx(0.5)
    assert M.rmse(f, a)[0] == pytest.approx(np.sqrt(0.5))
    assert M.wape(f, a)[0] == pytest.approx(1 / 4 * 100)


def test_mase_seasonal_and_lag1():
    train = [np.arange(1.0, 7.0)]              # 1..6
    f, a = np.array([[2.0, 4.0]]), np.array([[1.0, 5.0]])   # MAE = 1
    assert M.mase(f, a, train, 1)[0] == pytest.approx(1.0)  # ölçek = 1
    assert M.mase(f, a, train, 4)[0] == pytest.approx(0.25)  # lag 4: |5-1|,|6-2| -> 4


def test_mase_falls_back_to_lag1_when_series_shorter_than_season():
    train = [np.array([1.0, 2.0, 3.0])]
    f, a = np.array([[2.0]]), np.array([[1.0]])
    assert M.mase(f, a, train, 12)[0] == pytest.approx(1.0)


def test_mase_zero_over_zero_triggers_lag1_like_R():
    # mevsimsel ölçek 0 ve hata 0 -> R: NaN -> lag-1 ölçeğiyle yeniden: 0/1 = 0
    train = [np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])]
    f = a = np.array([[3.0]])
    assert M.mase(f, a, train, 2)[0] == pytest.approx(0.0)


def test_mase_constant_train_is_nan_not_inf():
    train = [np.array([5.0, 5.0, 5.0, 5.0])]
    f, a = np.array([[6.0]]), np.array([[5.0]])
    assert np.isnan(M.mase(f, a, train, 1)[0])


def test_mase_fractional_and_vector_seasonality():
    assert primary_period(365.25 / 7) == 52
    assert primary_period((24.0, 168.0, 8766.0)) == 24
    assert primary_period(np.float64(7)) == 7
    assert primary_period(0.4) == 1


def test_table_alignment_and_summary_counts():
    train = [np.arange(1.0, 7.0), np.array([5.0, 5.0, 5.0]), np.arange(1.0, 7.0)]
    tab = M.compute_error_table(F, A, train, 1, ["a", "b", "c"])
    assert list(tab.index) == ["a", "b", "c"]
    assert list(tab.columns) == list(M.DEFAULT_METRICS)
    assert len(tab) == 3                       # MASE vektörü kısalmaz (orijinalde kısalıyordu)
    sm = M.summarize(tab).set_index("Metric")
    assert sm.loc["SMAPE", "N_valid"] == 2 and sm.loc["SMAPE", "N_excluded"] == 1
    assert sm.loc["SMAPE", "Mean"] == pytest.approx((((2 / 3 + 2 / 9) / 2) + 2.0) / 2)
    assert sm.loc["WAPE", "N_valid"] == 1


def test_unknown_metric_and_shape_errors():
    with pytest.raises(ValueError):
        M.compute_error_table(F, A, [np.ones(3)] * 3, 1, metrics=["FOO"])
    with pytest.raises(ValueError):
        M.mae(F, A[:2])


def test_excel_export_roundtrip(tmp_path):
    import pandas as pd
    train = [np.arange(1.0, 7.0)] * 3
    tab = M.compute_error_table(F, A, train, 1, ["a", "b", "c"])
    p = tmp_path / "m.xlsx"
    M.write_metrics_xlsx(p, tab, M.summarize(tab), {"seed": 1, "engine": "r"})
    x = pd.ExcelFile(p)
    assert x.sheet_names == ["Errors_Per_Series", "Summary", "Run_Info"]
    assert list(pd.read_excel(p, "Errors_Per_Series")["series_name"]) == ["a", "b", "c"]
