"""py_global: havuzlanmış regresyon (akış hâlinde QR) ve veri hattı."""
import numpy as np
import pytest

from tsfstudio.engines import py_global as G


def make(n=6, L=80, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(L)
    return [(20 + 5 * i) * (1 + 0.3 * np.sin(2 * np.pi * t / 7 + i)) + rng.normal(0, 0.5, L) for i in range(n)]


def test_normalise_means_window_and_zero_mean():
    s = [np.array([2.0, 4.0, 6.0, 8.0, 10.0]), np.zeros(5)]
    norm, means = G.normalise(s, lag=2, max_obs=4)
    assert means[0] == pytest.approx(7.0)                       # son 4 değerin ortalaması (4,6,8,10)
    np.testing.assert_allclose(norm[0], np.array([4, 6, 8, 10]) / 7.0)
    assert means[1] == 1.0                                      # ortalama 0 -> 1
    with pytest.raises(ValueError, match="Series 1.*too short"):
        G.normalise([np.arange(3.0)], lag=5)


def test_embed_matches_r_embed_layout():
    x = np.arange(1.0, 7.0)                                     # 1..6
    e = G.embed(x, 2)                                           # R: embed(1:6, 3) -> [3 2 1; 4 3 2; 5 4 3; 6 5 4]
    np.testing.assert_array_equal(e, [[3, 2, 1], [4, 3, 2], [5, 4, 3], [6, 5, 4]])
    np.testing.assert_array_equal(G.final_lags([x], 2), [[6, 5]])


def test_streaming_qr_equals_full_lstsq_for_any_chunking():
    series = make()
    norm, _ = G.normalise(series, lag=5)
    X = np.vstack([G.embed(x, 5)[:, 1:] for x in norm]); y = np.concatenate([G.embed(x, 5)[:, 0] for x in norm])
    beta_full, *_ = np.linalg.lstsq(X, y, rcond=None)
    for chunk in (50, 300, 10_000_000):
        beta, info = G.fit_pooled_regression(norm, 5, chunk_rows=chunk)
        np.testing.assert_allclose(beta, beta_full, rtol=1e-9, atol=1e-11)
        assert info.rank == 5 and not info.aliased and info.n_rows == X.shape[0]


def test_forecast_recursion_matches_manual_ar():
    # AR(2) kesin: y_t = 0.6 y_{t-1} + 0.3 y_{t-2}  (normalize edilmiş uzayda)
    rng = np.random.default_rng(1)
    x = [1.0, 1.2]
    for _ in range(200):
        x.append(0.6 * x[-1] + 0.3 * x[-2] + rng.normal(0, 0.01) + 0.4)   # kesişimsiz model değil ama deneyin yönü yeterli
    f, info = G.pooled_regression_forecast([np.array(x)], lag=2, horizon=3)
    beta, _ = G.fit_pooled_regression(*[G.normalise([np.array(x)], 2)[0]], 2)
    norm, means = G.normalise([np.array(x)], 2)
    l1, l2 = norm[0][-1], norm[0][-2]
    p1 = beta[0] * l1 + beta[1] * l2
    p2 = beta[0] * p1 + beta[1] * l1
    np.testing.assert_allclose(f[0, :2], np.array([p1, p2]) * means[0])


def test_aliased_columns_follow_r_rule():
    # İki özdeş seriyi tek seride birleştirip Lag sütunlarını bağımlı yapmak güç; sabit seri tüm Lag'leri aynı yapar.
    const = [np.full(40, 5.0), np.full(40, 5.0)]
    f, info = G.pooled_regression_forecast(const, lag=3, horizon=2)
    assert info.rank == 1 and info.aliased == [2, 3]            # Lag1 tutulur; Lag2, Lag3 NA gibi düşer
    np.testing.assert_allclose(f, 5.0)                          # normalize edilmiş seri 1; beta1 = 1
    assert any("NA coefficients" in n for n in info.notes)


def test_too_few_rows_raises():
    with pytest.raises(ValueError, match="Not enough rows"):
        G.fit_pooled_regression([np.array([1.0, 2.0, 3.0])], 2)


def test_max_obs_window_changes_training_rows():
    series = make(3, 100)
    _, info_full = G.pooled_regression_forecast(series, 5, 2)
    _, info_win = G.pooled_regression_forecast(series, 5, 2, max_obs=30)
    assert info_full.n_rows == 3 * 95 and info_win.n_rows == 3 * 25
