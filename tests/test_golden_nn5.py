"""Altın test: tsfstudio.metrics, kullanıcının gerçek R hata çıktılarını (NN5 haftalık, 111 seri) yeniden üretir.

Kaynak: kullanıcının R kodu ile ürettiği `R Errors/nn5_weekly_<model>_errors.xlsx` (smape, msmape, mase, mae, rmse).
Gerçek veri: test seti 8 adım; eğitim 105 hafta; mevsimsellik = 365.25/7 -> MASE gecikmesi 52.
"""
from pathlib import Path

import numpy as np
import pytest

from tsfstudio.metrics import mae, mase, msmape, rmse, smape
from tsfstudio.seasonality import seasonality_for

G = Path(__file__).parent / "golden" / "nn5_weekly"
MODELS = ["ses", "ets", "theta", "tbats", "arima", "dhr_arima"]


@pytest.fixture(scope="module")
def data():
    train = np.loadtxt(G / "train.csv", delimiter=",")
    act = np.loadtxt(G / "actuals.csv", delimiter=",")
    return train, act


def test_fixture_shapes(data):
    train, act = data
    assert train.shape == (111, 105) and act.shape == (111, 8)


@pytest.mark.parametrize("model", MODELS)
def test_metrics_match_user_r_errors(data, model):
    train, act = data
    f = np.loadtxt(G / f"forecasts_{model}.csv", delimiter=",")
    ref = np.loadtxt(G / f"r_errors_{model}.csv", delimiter=",")
    seas = seasonality_for("weekly")
    mine = np.column_stack([smape(f, act), msmape(f, act), mase(f, act, list(train), seas), mae(f, act), rmse(f, act)])
    assert mine.shape == ref.shape
    np.testing.assert_allclose(mine, ref, rtol=1e-10, atol=1e-12)
