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
# M3/M4 kıyas yöntemleri (2026-11-22): drift, Naive2, Comb S-H-D
#   Kaynak: M4 yarışmasının resmi kıyas kodu (Makridakis, Spiliotis & Assimakopoulos, 2020, IJF 36(1));
#   `rwf(drift = TRUE)` (Hyndman & Athanasopoulos, FPP3). Holt / sönümlü Holt: forecast::holt ile aynı model sınıfı
#   (ets "AAN"/"AAdN", opt.crit = "mse", R sınırları) ama kendi en iyileyicimiz; R ile birebir sınanmadı.
# ------------------------------------------------------------------------------------------------
def drift(y: np.ndarray, h: int) -> np.ndarray:
    """Rastgele yürüyüş + sürüklenme: y_T + k (y_T - y_1)/(T - 1), k = 1..h (forecast::rwf(drift = TRUE))."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    slope = (y[-1] - y[0]) / (n - 1) if n > 1 else 0.0
    return y[-1] + slope * np.arange(1, h + 1, dtype=np.float64)


def m4_seasonality_test(x: np.ndarray, m: int) -> bool:
    """M4 kıyas kodundaki `SeasonalityTest`: %90 (t = 1.645), n >= 3m, ACF R `acf()` varsayılan lag.max ile.

    R: xacf <- acf(x)$acf[-1]; clim <- 1.645/sqrt(n) * sqrt(cumsum(c(1, 2*xacf^2))); abs(xacf[m]) > clim[m].
    m, R'ın varsayılan lag.max'ından (floor(10 log10 n)) büyükse xacf[m] NA olur -> mevsimsiz kabul edilir."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if m <= 1 or n < 3 * m or float(np.ptp(x)) == 0.0:
        return False
    lag_max = min(int(math.floor(10.0 * math.log10(n))), n - 1)
    if m > lag_max:
        return False
    r = _acf(x, lag_max)
    clim = 1.645 / math.sqrt(n) * math.sqrt(1.0 + 2.0 * float(np.sum(r[: m - 1] ** 2)))
    return bool(abs(r[m - 1]) > clim)


def _m4_deseasonalise(y: np.ndarray, m: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """(mevsimden arındırılmış seri, ufuk için mevsim indeksleri). Mevsimsiz ya da indeks tanımsızsa indeksler 1."""
    x = np.asarray(y, dtype=np.float64)
    if m4_seasonality_test(x, m):
        seas = _decompose_multiplicative(x, m)
        if np.all(np.isfinite(seas)) and not np.any(np.abs(seas) < 1e-10):
            return x / seas, np.resize(seas[len(x) - m:], h)
    return x, np.ones(h)


def naive2(y: np.ndarray, h: int, m: int) -> np.ndarray:
    """M4 Naive2: mevsimsellik sınaması geçerse çarpımsal klasik ayrıştırma, naif, yeniden mevsimselleştirme."""
    x, si = _m4_deseasonalise(y, m, h)
    return naive(x, h) * si


def _holt_sse(params: np.ndarray, y: np.ndarray, damped: bool) -> float:
    alpha, beta, phi, l, b = params if damped else (*params[:2], 1.0, *params[2:])
    if beta > alpha:                                         # ets sınırı: beta <= alpha
        return 1e300
    sse = 0.0
    for yt in y:
        f = l + phi * b
        e = yt - f
        sse += e * e
        l_new = f + alpha * e
        b = phi * b + beta * e
        l = l_new
    return sse


def holt_forecast(y: np.ndarray, h: int, damped: bool = False) -> np.ndarray:
    """Holt doğrusal eğilim (damped=True: sönümlü). SSE en küçüklenir (forecast::holt -> ets(opt.crit = "mse")).

    Sınırlar R ets ile aynı: alfa in [1e-4, 0.9999], beta in [1e-4, alfa], phi in [0.8, 0.98]. Başlangıç düzeyi/eğimi de
    serbest parametredir (ilk değerleri: ilk en fazla 10 gözlem üzerinde doğrusal regresyon, ets gibi)."""
    from scipy.optimize import minimize

    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 3 or float(np.ptp(y)) == 0.0:
        return drift(y, h) if n > 1 else naive(y, h)
    k = min(10, n)
    b0, l0 = np.polyfit(np.arange(1, k + 1, dtype=np.float64), y[:k], 1)
    l0 = float(l0)
    b0 = float(b0)
    scale = max(float(np.std(y)), 1e-8)
    best = None
    starts = [(0.5, 0.1), (0.2, 0.05), (0.9, 0.3)]
    for a0, be0 in starts:
        x0 = [a0, be0] + ([0.95] if damped else []) + [l0, b0]
        bounds = [(ALPHA_LO, ALPHA_HI), (ALPHA_LO, ALPHA_HI)] + ([(0.8, 0.98)] if damped else []) + \
                 [(l0 - 10 * scale, l0 + 10 * scale), (b0 - 10 * scale, b0 + 10 * scale)]
        res = minimize(_holt_sse, np.array(x0), args=(y, damped), method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    alpha, beta, phi, l, b = best.x if damped else (*best.x[:2], 1.0, *best.x[2:])
    for yt in y:                                             # son durum
        f = l + phi * b
        e = yt - f
        l, b = f + alpha * e, phi * b + beta * e
    steps = np.cumsum(phi ** np.arange(1, h + 1)) if damped else np.arange(1, h + 1, dtype=np.float64)
    return l + steps * b


def comb_shd(y: np.ndarray, h: int, m: int) -> np.ndarray:
    """M4 'Comb' kıyası: Naive2 ile aynı mevsim arındırma; SES, Holt ve sönümlü Holt ortalaması; yeniden mevsimselleştirme."""
    x, si = _m4_deseasonalise(y, m, h)
    f = (ses_forecast(x, h) + holt_forecast(x, h, False) + holt_forecast(x, h, True)) / 3.0
    return f * si


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
