"""R `forecast` paketindeki basit yerel modellerin numpy/scipy karşılıkları (üçüncü parti tahmin kütüphanesi gerekmez).

Amaç R ile AYNI algoritmayı uygulamaktır (yalnızca benzer bir model değil):
  * `ses`   : forecast::ses  -> ets(y, "ANN", opt.crit = "mse"); alfa in [1e-4, 0.9999], başlangıç düzeyi serbest.
              Verilen alfa için SSE, başlangıç düzeyinde ikinci dereceden olduğundan düzey KAPALI FORMDA bulunur;
              alfa tek boyutlu aramayla (kaba ızgara + sınırlı Brent) çözülür. R'ın Nelder-Mead'i yaklaşık optimumda
              durabildiği için sonuçlar çok küçük farklarla (~1e-4 bağıl) örtüşür.
  * `theta` : forecast::thetaf -> mevsimsellik sınaması (ACF, %90), çarpımsal klasik ayrıştırma (decompose),
              SES + yarım eğim sürüklenmesi, yeniden mevsimselleştirme.
  * `naive`, `snaive`.
Mevsimsel frekans R'daki `msts` gibi `floor(max(periyot))` alınır (bkz. `r_frequency`).

Kaynaklar: Hyndman & Athanasopoulos (2021) FPP3; Hyndman & Billah (2003) Theta; Assimakopoulos & Nikolopoulos (2000).
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import lfilter
from scipy.stats import norm

ALPHA_LO, ALPHA_HI = 1e-4, 0.9999


def r_frequency(seasonality: Sequence[float] | float) -> int:
    """`forecast::msts` çerçeve frekansı: floor(max(periyotlar)); en az 1."""
    vals = [float(seasonality)] if np.ndim(seasonality) == 0 else [float(s) for s in seasonality]
    return max(1, int(math.floor(max(vals))))


# ------------------------------------------------------------------------------------------------
def naive(y: np.ndarray, h: int) -> np.ndarray:
    return np.full(h, float(y[-1]))


def snaive(y: np.ndarray, h: int, m: int) -> np.ndarray:
    """Mevsimsel naif; seri m'den kısaysa naife düşer (R'da bu durum hata verebilir; burada açık kural)."""
    n = len(y)
    m = m if (m >= 1 and n >= m) else 1
    return y[n - m + (np.arange(h) % m)].astype(np.float64)


# ------------------------------------------------------------------------------------------------
# SES
# ------------------------------------------------------------------------------------------------
def _ses_profile(y: np.ndarray, alpha: float) -> tuple[float, float, float]:
    """Verilen alfa için (SSE, l0*, son düzey l_n). Durum: l_t = l_{t-1} + alfa*(y_t - l_{t-1})."""
    n = len(y)
    b = lfilter([alpha], [1.0, -(1.0 - alpha)], y)           # l0 = 0 ile düzey
    b_prev = np.concatenate(([0.0], b[:-1]))
    u = y - b_prev
    a = (1.0 - alpha) ** np.arange(n)                        # l0'ın e_t üzerindeki katsayısı: a_{t-1}
    denom = float(a @ a)
    l0 = float(a @ u) / denom
    e = u - a * l0
    sse = float(e @ e)
    level_n = (1.0 - alpha) ** n * l0 + float(b[-1])
    return sse, l0, level_n


def ses_fit(y: np.ndarray) -> tuple[float, float]:
    """(alfa, son düzey). Sabit seride alfa = ALPHA_HI kabul edilir (R'da da düz bir amaç işlevi)."""
    y = np.asarray(y, dtype=np.float64)
    if len(y) < 2 or float(np.ptp(y)) == 0.0:
        return ALPHA_HI, float(y[-1])
    grid = np.linspace(ALPHA_LO, ALPHA_HI, 60)
    sses = np.array([_ses_profile(y, a)[0] for a in grid])
    i = int(np.argmin(sses))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(lambda a: _ses_profile(y, a)[0], bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
    alpha = float(res.x)
    if _ses_profile(y, alpha)[0] > sses[i]:                  # sınırlı Brent kötüleşirse ızgara noktasını koru
        alpha = float(grid[i])
    return alpha, _ses_profile(y, alpha)[2]


def ses_forecast(y: np.ndarray, h: int) -> np.ndarray:
    _, level = ses_fit(y)
    return np.full(h, level)


# ------------------------------------------------------------------------------------------------
# Theta (forecast::thetaf)
# ------------------------------------------------------------------------------------------------
def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """R `acf(type = "correlation")`: ortalama çıkarılır, payda n (yanlı), gecikme 1..nlags."""
    n = len(x)
    d = x - x.mean()
    c0 = float(d @ d)
    return np.array([float(d[: n - k] @ d[k:]) / c0 for k in range(1, nlags + 1)])


def _decompose_multiplicative(x: np.ndarray, m: int) -> np.ndarray:
    """R `decompose(x, "multiplicative")$seasonal` (ts başlangıcı 1. mevsim konumu)."""
    n = len(x)
    if m % 2 == 0:
        filt = np.concatenate(([0.5], np.ones(m - 1), [0.5])) / m
    else:
        filt = np.ones(m) / m
    k = len(filt)
    half = k // 2
    trend = np.full(n, np.nan)
    conv = np.convolve(x, filt[::-1], mode="valid")           # R filter(sides=2): ortalanmış, kenarlar NA
    trend[half: half + len(conv)] = conv
    season = x / trend
    figure = np.empty(m)
    for i in range(m):
        vals = season[i::m]
        vals = vals[~np.isnan(vals)]
        figure[i] = vals.mean() if len(vals) else np.nan
    figure = figure / np.nanmean(figure)
    return np.resize(figure, n)


def theta_forecast(y: np.ndarray, h: int, m: int) -> np.ndarray:
    """forecast::thetaf. `m` = çerçeve frekansı (r_frequency)."""
    x = np.asarray(y, dtype=np.float64)
    n = len(x)
    seasonal = False
    if m > 1 and float(np.ptp(x)) != 0.0 and n > 2 * m:
        r = _acf(x, m)
        stat = math.sqrt((1.0 + 2.0 * float(np.sum(r[:-1] ** 2))) / n)
        seasonal = abs(r[-1]) / stat > norm.ppf(0.95)
    seas = None
    if seasonal:
        seas = _decompose_multiplicative(x, m)
        if np.any(np.abs(seas) < 1e-10) or np.any(np.isnan(seas)):
            seas = None                                       # R: uyarı + mevsimsiz Theta
        else:
            x = x / seas
    alpha, level = ses_fit(x)
    t = np.arange(n, dtype=np.float64)
    slope = float(np.polyfit(t, x, 1)[0]) if n > 1 else 0.0  # lsfit(0:(n-1), x) eğimi
    alpha = max(1e-10, alpha)
    f = level + (slope / 2.0) * (np.arange(h) + (1.0 - (1.0 - alpha) ** n) / alpha)
    if seas is not None:
        last = seas[n - m:]                                   # son tam mevsim; faz doğru kalacak biçimde tekrarlanır
        f = f * np.resize(last, h)
    return f


# ------------------------------------------------------------------------------------------------
# Fourier terimleri (forecast::fourier) — DHR-ARIMA için dış regresörler
# ------------------------------------------------------------------------------------------------
def fourier_terms(n: int, periods: Sequence[float], K: int | Sequence[int], h: int | None = None) -> np.ndarray:
    """R `fourier(x, K, h)` ile aynı terimler: her periyot ve k = 1..K için sin(2πk t/m) ve cos(2πk t/m);
    t = 1..n (h verilirse gelecek için n+1..n+h). Sütun sırası regresyon sonucunu etkilemez.
    """
    periods = [float(p) for p in periods]
    Ks = [int(K)] * len(periods) if np.ndim(K) == 0 else [int(k) for k in K]
    if len(Ks) != len(periods):
        raise ValueError("K must have as many entries as there are seasonal periods (or be a scalar).")
    t = np.arange(1, n + 1, dtype=np.float64) if h is None else np.arange(n + 1, n + h + 1, dtype=np.float64)
    cols = []
    for p, k in zip(periods, Ks):
        if 2 * k > p:
            raise ValueError(f"K = {k} is too large: 2K <= period ({p}) is required.")
        for j in range(1, k + 1):
            cols.append(np.sin(2 * np.pi * j * t / p))
            cols.append(np.cos(2 * np.pi * j * t / p))
    return np.column_stack(cols) if cols else np.empty((len(t), 0))
