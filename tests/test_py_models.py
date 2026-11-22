"""py_models: R `forecast` basit modellerinin numpy karşılıkları (birim testleri)."""
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from tsfstudio.engines import py_models as M


def test_r_frequency_is_floor_of_max():
    assert M.r_frequency((365.25 / 7,)) == 52
    assert M.r_frequency((24.0, 168.0, 8766.0)) == 8766
    assert M.r_frequency(7.0) == 7 and M.r_frequency((0.5,)) == 1


def test_naive_and_snaive_phase():
    y = np.arange(1.0, 11.0)                      # 1..10
    np.testing.assert_array_equal(M.naive(y, 3), [10, 10, 10])
    np.testing.assert_array_equal(M.snaive(y, 5, 4), [7, 8, 9, 10, 7])      # mevsim 4: son 4 değer tekrarlanır
    np.testing.assert_array_equal(M.snaive(y[:3], 2, 4), [3, 3])            # seri m'den kısa -> naif


def test_ses_profile_matches_brute_force_recursion():
    rng = np.random.default_rng(0)
    y = 20 + np.cumsum(rng.normal(0, 1, 60))
    for alpha in (0.05, 0.3, 0.9):
        sse, l0, ln = M._ses_profile(y, alpha)

        def run(l0_):
            lvl, e2 = l0_, 0.0
            for v in y:
                e = v - lvl
                e2 += e * e
                lvl += alpha * e
            return e2, lvl

        brute = minimize_scalar(lambda l: run(l)[0], bounds=(-1e3, 1e3), method="bounded", options={"xatol": 1e-12})
        assert sse == pytest.approx(brute.fun, rel=1e-9)
        assert l0 == pytest.approx(brute.x, abs=1e-5)
        assert ln == pytest.approx(run(l0)[1], rel=1e-10)


def test_ses_constant_and_random_walk_limits():
    a, lvl = M.ses_fit(np.full(30, 7.0))
    assert lvl == 7.0
    rw = np.cumsum(np.random.default_rng(1).normal(0, 1, 400))
    alpha, level = M.ses_fit(rw)
    assert alpha > 0.7 and abs(level - rw[-1]) < 1.0           # rassal yürüyüşte alfa ~ 1, düzey ~ son değer


def test_acf_matches_definition():
    x = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0])
    d = x - x.mean()
    expect = [(d[:-k] @ d[k:]) / (d @ d) for k in (1, 2)]
    np.testing.assert_allclose(M._acf(x, 2), expect)


def test_multiplicative_decompose_recovers_seasonal_figure():
    m = 4
    figure = np.array([0.8, 1.1, 1.3, 0.8]); figure /= figure.mean()
    t = np.arange(40)
    y = (100 + 2 * t) * figure[t % m]
    seas = M._decompose_multiplicative(y, m)
    np.testing.assert_allclose(seas[:m], figure, rtol=5e-3)   # klasik ayrıştırma eğilim varlığında küçük sapma verir
    assert seas.shape == y.shape


def test_theta_matches_hand_formula_on_nonseasonal_trend():
    t = np.arange(50, dtype=float)
    y = 10 + 0.5 * t
    f = M.theta_forecast(y, 4, m=1)
    alpha, level = M.ses_fit(y)
    expect = level + (0.5 / 2) * (np.arange(4) + (1 - (1 - alpha) ** 50) / alpha)
    np.testing.assert_allclose(f, expect)


def test_theta_seasonal_branch_keeps_seasonal_shape():
    rng = np.random.default_rng(2)
    m, t = 7, np.arange(210)
    y = (50 + 0.05 * t) * (1 + 0.4 * np.sin(2 * np.pi * t / m)) + rng.normal(0, 0.3, len(t))
    f = M.theta_forecast(y, 14, m)
    # tahmin haftalık deseni sürdürmeli: 7 adım arayla aynı yönde değişim
    assert np.corrcoef(f[:7], f[7:])[0, 1] > 0.99
    assert np.corrcoef(f[:7], y[-7:])[0, 1] > 0.9


def test_fourier_terms_values_and_shapes():
    X = M.fourier_terms(10, [7.0], 1)
    assert X.shape == (10, 2)
    np.testing.assert_allclose(X[0], [np.sin(2 * np.pi / 7), np.cos(2 * np.pi / 7)])
    Xf = M.fourier_terms(10, [7.0, 52.0], [1, 2], h=3)
    assert Xf.shape == (3, 2 + 4)
    np.testing.assert_allclose(Xf[0, :2], [np.sin(2 * np.pi * 11 / 7), np.cos(2 * np.pi * 11 / 7)])
    assert M.fourier_terms(10, [7.0], 1, h=5)[0, 0] == pytest.approx(np.sin(2 * np.pi * 11 / 7))
    with pytest.raises(ValueError, match="2K"):
        M.fourier_terms(10, [3.0], 2)
    with pytest.raises(ValueError, match="K must"):
        M.fourier_terms(10, [7.0, 52.0], [1, 1, 1])
