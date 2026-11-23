"""tsfeatures (R package by Hyndman, Kang, Montero-Manso, O'Hara-Wild, Talagala, Wang, Yang; GPL-3) as the ranking
study computed it (reference/experiments/feature_functions.R), ported to Python.

The study called, per series x (an msts with the dataset's seasonal periods, ts frequency floor(max(periods))):

* ``tsfeatures(x, c("mean", "var"), scale = FALSE)``;
* for each of max_kl_shift, max_level_shift, max_var_shift, acf_features, arch_stat, crossing_points, entropy,
  flat_spots, holt_parameters, hurst, lumpiness, nonlinearity, pacf_features, stability, unitroot_kpss and
  unitroot_pp: ``tsfeatures(x, f)`` (the series scaled to mean 0, sd 1); if a value is NA, again on
  ``ts(x, frequency = 1)``; if still NA, the shift features with ``width = 1`` and arch_stat with ``lags = 1``;
* ``tsfeatures(x, "stl_features", s.window = "periodic", robust = TRUE)``, falling back to the defaults and then to
  frequency 1 when that fails.

A column that some series lack (a fallback to frequency 1 drops seas_acf1 / seas_pacf) is dropped for the whole
dataset, as the study's intersection did.

The port follows the tsfeatures version the study ran (time_*_shift = which.max + 1, as in tsfeatures 1.0.x; entropy
from spec.ar with Burg's method). Heavy parts run in the native library: forecast::ets "AAN" (holt_parameters),
fracdiff (hurst), robust stl() and supsmu() (stl_features). Checked against data/ranking_study/tsfeatures on all 15
study datasets; the stored m4_daily values behave as if those series had a ts frequency above half their length (no
seasonal columns, shift windows of 10, lumpiness 0), a property of the study's data conversion, not of tsfeatures.
"""
from __future__ import annotations

import math

import numpy as np

NAN = float("nan")

# The study's seasonal periods per .tsf frequency (feature_functions.R, SEASONALITY_VALS).
STUDY_PERIODS = {
    "4_seconds": (21600, 151200, 7889400), "minutely": (1440, 10080, 525960), "10_minutes": (144, 1008, 52596),
    "15_minutes": (96, 672, 35064), "half_hourly": (48, 336, 17532), "hourly": (24, 168, 8766), "daily": (7,),
    "weekly": (365.25 / 7,), "monthly": (12,), "quarterly": (4,), "yearly": (1,),
}

def study_periods(frequency: str | None) -> tuple[float, ...]:
    """The study's seasonal periods for a .tsf frequency (1 when the file states none)."""
    return STUDY_PERIODS.get(frequency or "", (1,))


TSFEATURE_FUNCS = ("max_kl_shift", "max_level_shift", "max_var_shift", "acf_features", "arch_stat", "crossing_points",
                   "entropy", "flat_spots", "holt_parameters", "hurst", "lumpiness", "nonlinearity", "pacf_features",
                   "stability", "unitroot_kpss", "unitroot_pp")


# ---------------------------------------------------------------------------------------------------- R helpers
def r_var(x) -> float:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size < 2:
        return NAN
    m = float(np.mean(x))
    return float(np.sum((x - m) ** 2) / (x.size - 1))


def scalets(x: np.ndarray) -> np.ndarray:
    """tsfeatures:::scalets: (x - mean) / sd unless x is constant."""
    if np.all(x == x[0]):
        return x
    # scale(x): colMeans (long double), then sd = sqrt(sum(v^2) / (n - 1)) with R's long double sum
    import ctypes
    from .engines import cpp_engine
    lib = cpp_engine.load_lib()
    if lib is not None:
        if not getattr(lib, "_scale_ready", False):
            dp = ctypes.POINTER(ctypes.c_double)
            lib.tsf_r_scale.restype = None
            lib.tsf_r_scale.argtypes = [dp, ctypes.c_int32, dp]
            lib._scale_ready = True
        xc = np.ascontiguousarray(x, dtype=np.float64)
        out = np.empty_like(xc)
        lib.tsf_r_scale(cpp_engine._ptr(xc, ctypes.c_double), xc.size, cpp_engine._ptr(out, ctypes.c_double))
        return out
    from .engines.r_boot import _ld, _ld_sum, r_col_mean
    v = x - r_col_mean(x.tolist())
    ss = float(_ld(_ld_sum((v * v).tolist())))
    return v / math.sqrt(ss / (x.size - 1))


def r_acf(x: np.ndarray, lag_max: int) -> np.ndarray:
    """stats::acf(x, lag.max)$acf (lags 0..min(lag.max, n - 1)), demeaned, divided by n."""
    n = x.size
    lag_max = int(min(lag_max, n - 1))
    d = x - x.mean()
    if lag_max > 64:                                     # many lags (high frequencies): one C-level correlation
        acov = np.correlate(d, d, "full")[n - 1:n + lag_max] / n
    else:
        acov = np.array([np.dot(d[: n - k], d[k:]) / n for k in range(lag_max + 1)])
    return acov / acov[0]


def r_pacf(x: np.ndarray, lag_max: int) -> np.ndarray:
    """stats::pacf(x, lag.max)$acf: Durbin-Levinson on the sample autocorrelations (R's uacf2pacf)."""
    rho = r_acf(x, lag_max)
    nlag = rho.size - 1
    phi = np.zeros(nlag + 1)
    work = np.zeros(nlag + 1)
    out = np.zeros(nlag)
    if nlag < 1:
        return out
    phi[1] = rho[1]
    out[0] = rho[1]
    if nlag > 64:                                        # many lags: the same recursion with vector updates
        for ll in range(2, nlag + 1):
            c = (rho[ll] - np.dot(phi[1:ll], rho[ll - 1:0:-1])) / (1.0 - np.dot(phi[1:ll], rho[1:ll]))
            out[ll - 1] = c
            if ll == nlag:
                break
            phi[ll] = c
            phi[1:ll] = phi[1:ll] - c * phi[ll - 1:0:-1]
        return out
    for ll in range(2, nlag + 1):
        a = rho[ll]
        b = 1.0
        for i in range(1, ll):
            a -= phi[i] * rho[ll - i]
            b -= phi[i] * rho[i]
        c = a / b
        out[ll - 1] = c
        if ll == nlag:
            break
        phi[ll] = c
        for i in range(1, ll):
            work[ll - i] = phi[i]
        for i in range(1, ll):
            phi[i] -= c * work[i]
    return out


def _seq_get(a: np.ndarray, k: int) -> float:
    """a[k] (R 1-based); NA beyond the end."""
    return float(a[k - 1]) if 1 <= k <= a.size else NAN


def _sumsq_first(a: np.ndarray, k: int) -> float:
    """sum(a[seq(k)]^2) in R: NA when a has fewer than k elements."""
    return float(np.sum(a[:k] ** 2)) if a.size >= k else NAN


def _dnorm(x: np.ndarray, mean: np.ndarray | float, sd: float) -> np.ndarray:
    """R's dnorm(x, mean, sd) (dnorm4, with its accurate branch for |z| >= 5)."""
    z = np.abs((x - mean) / sd)
    out = np.where(z < 5, np.exp(-0.5 * z * z), 0.0)
    big = (z >= 5) & (z <= math.sqrt(-2 * math.log(2) * (-1021 + 1 - 53)))
    if big.any():
        zb = z[big]
        x1 = np.ldexp(np.rint(np.ldexp(zb, 16)), -16)
        x2 = zb - x1
        out[big] = np.exp(-0.5 * x1 * x1) * np.exp((-0.5 * x2 - x1) * x2)
    return out * (1.0 / math.sqrt(2 * math.pi)) / sd


def _quantile7(x: np.ndarray, p: float) -> float:
    s = np.sort(x)
    h = (s.size - 1) * p
    lo = math.floor(h)
    return float(s[lo] + (h - lo) * (s[min(lo + 1, s.size - 1)] - s[lo]))


def bw_nrd0(x: np.ndarray) -> float:
    hi = math.sqrt(r_var(x))
    lo = min(hi, (_quantile7(x, 0.75) - _quantile7(x, 0.25)) / 1.34)
    if not lo:
        lo = hi or abs(x[0]) or 1.0
    return 0.9 * lo * x.size ** (-0.2)


def _lm_fit(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients and residuals of lm(y ~ X) (X includes the intercept column)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


# ---------------------------------------------------------------------------------------------------- features
def lumpiness(x, width):
    nr = x.size
    if nr < 2 * width:
        return {"lumpiness": 0.0}
    nseg = nr // width
    seg = x[: nseg * width].reshape(nseg, width)
    return {"lumpiness": r_var([r_var(s) for s in seg])}


def stability(x, width):
    nr = x.size
    if nr < 2 * width:
        return {"stability": 0.0}
    nseg = nr // width
    seg = x[: nseg * width].reshape(nseg, width)
    return {"stability": r_var([float(np.mean(s)) for s in seg])}


def _seq_sum(w: np.ndarray, axis: int) -> np.ndarray:
    """Sum along `axis` in sequential order (RcppRoll's loop), not numpy's pairwise order."""
    w = np.moveaxis(w, axis, 0)
    s = w[0].copy()
    for k in range(1, w.shape[0]):
        s += w[k]
    return s


def _roll_mean(w: np.ndarray) -> np.ndarray:
    return _seq_sum(w, 1) / w.shape[1]


def _roll_var(w: np.ndarray) -> np.ndarray:
    n = w.shape[1]
    if n < 2:
        return np.full(w.shape[0], NAN)
    mu = _roll_mean(w)
    return _seq_sum((w - mu[:, None]) ** 2, 1) / (n - 1)


def _roll(x: np.ndarray, width: int, fn) -> np.ndarray:
    if x.size < width:
        raise ValueError("width larger than the series")
    w = np.lib.stride_tricks.sliding_window_view(x, width)
    return fn(w)


def _max_shift(stat: np.ndarray, width: int) -> tuple[float, float]:
    d = np.abs(stat[width:] - stat[:-width]) if stat.size > width else np.zeros(0)
    if d.size == 0:
        return 0.0, NAN
    if np.all(np.isnan(d)):
        return NAN, NAN
    return float(np.nanmax(d)), float(np.nanargmax(d) + 2)              # which.max + 1 (tsfeatures 1.0.x)


def max_level_shift(x, width):
    try:
        means = _roll(x, width, _roll_mean)
    except ValueError:
        return {"max_level_shift": NAN, "time_level_shift": NAN}
    a, b = _max_shift(means, width)
    return {"max_level_shift": a, "time_level_shift": b}


def max_var_shift(x, width):
    try:
        vars_ = _roll(x, width, _roll_var)
    except ValueError:
        return {"max_var_shift": NAN, "time_var_shift": NAN}
    a, b = _max_shift(vars_, width)
    return {"max_var_shift": a, "time_var_shift": b}


def max_kl_shift(x, width):
    gw = 100
    xgrid = np.linspace(x.min(), x.max(), gw)
    grid = xgrid[1] - xgrid[0]
    bw = bw_nrd0(x)
    n = x.size
    if n <= 2 * width:
        return {"max_kl_shift": NAN, "time_kl_shift": NAN}
    dens = _dnorm(xgrid[None, :], x[:, None], bw)
    dens = np.maximum(dens, _dnorm(np.array([38.0]), 0.0, 1.0)[0])
    rmean = np.full((n, gw), NAN)
    cs = _seq_sum(np.lib.stride_tricks.sliding_window_view(dens, width, axis=0), 2) / width
    rmean[width - 1:] = cs
    k = n - width
    lo, hi = rmean[0:k], rmean[width:width + k]
    with np.errstate(invalid="ignore", divide="ignore"):
        terms = lo * (np.log(lo) - np.log(hi)) * grid
    kl = np.nansum(terms, axis=1)
    dkl = np.diff(kl)
    if dkl.size == 0:
        return {"max_kl_shift": 0.0, "time_kl_shift": NAN}
    return {"max_kl_shift": float(np.max(dkl)), "time_kl_shift": float(np.argmax(dkl) + 2)}


def crossing_points(x):
    ab = x <= np.median(x)
    return {"crossing_points": float(np.sum(ab[:-1] != ab[1:]))}


def flat_spots(x):
    lo, hi = float(x.min()), float(x.max())
    dx = hi - lo
    if dx == 0:
        dx = abs(lo) if lo != 0 else 1.0
        breaks = lo - dx / 1000 + np.arange(11) * ((hi + dx / 1000) - (lo - dx / 1000)) / 10
    else:
        breaks = lo + np.arange(11) * (dx / 10)                  # seq.int(lo, hi, length.out = 11)
        breaks[0], breaks[10] = lo - dx / 1000, hi + dx / 1000
    codes = np.searchsorted(breaks, x, side="left")           # right = TRUE: (b[i-1], b[i]]
    run, best = 1, 1
    for i in range(1, codes.size):
        run = run + 1 if codes[i] == codes[i - 1] else 1
        best = max(best, run)
    return {"flat_spots": float(best)}


def acf_features(x, m):
    n = x.size
    acfx = r_acf(x, max(10, m))[1:] if n > 1 else np.array([NAN])
    d1 = r_acf(np.diff(x), 10)[1:] if n > 10 else np.array([NAN])
    d2 = r_acf(np.diff(x, 2), 10)[1:] if n > 11 else np.array([NAN])
    out = {"x_acf1": _seq_get(acfx, 1), "x_acf10": _sumsq_first(acfx, 10), "diff1_acf1": _seq_get(d1, 1),
           "diff1_acf10": _sumsq_first(d1, 10), "diff2_acf1": _seq_get(d2, 1), "diff2_acf10": _sumsq_first(d2, 10)}
    if m > 1:
        out["seas_acf1"] = _seq_get(acfx, m)
    return out


def pacf_features(x, m):
    n = x.size
    p = r_pacf(x, max(5, m)) if n > 1 else np.array([NAN])
    out = {"x_pacf5": _sumsq_first(p, 5) if n > 5 else NAN,
           "diff1x_pacf5": float(np.sum(r_pacf(np.diff(x), 5) ** 2)) if n > 6 else NAN,
           "diff2x_pacf5": float(np.sum(r_pacf(np.diff(x, 2), 5) ** 2)) if n > 7 else NAN}
    if m > 1:
        out["seas_pacf"] = _seq_get(p, m)
    return out


def arch_stat(x, lags=12):
    n = x.size
    if n <= 13:                                          # tsfeatures 1.0.x: fixed at 13, whatever `lags` is
        return {"ARCH.LM": NAN}
    x2 = (x - x.mean()) ** 2
    rows = n - lags
    y = x2[lags:]
    X = np.column_stack([np.ones(rows)] + [x2[lags - j: n - j] for j in range(1, lags + 1)])
    _, res = _lm_fit(y, X)
    fitted = y - res
    mss = float(np.sum((fitted - fitted.mean()) ** 2))
    rss = float(np.sum(res ** 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = mss / (mss + rss) if (mss + rss) != 0 else NAN
    return {"ARCH.LM": 1.0 if math.isnan(r2) else r2}


def unitroot_kpss(x):
    n = x.size
    lmax = math.trunc(4 * (n / 100) ** 0.25)
    res = x - x.mean()
    S = np.cumsum(res)
    nom = float(np.sum(S ** 2)) / n ** 2
    s2 = float(np.sum(res ** 2)) / n
    if lmax == 0:
        den = s2
    else:
        idx = np.arange(1, lmax + 1)
        cov = np.array([np.dot(res[k:], res[: n - k]) for k in idx])
        den = s2 + 2 / n * float(np.dot(1 - idx / (lmax + 1), cov))
    return {"unitroot_kpss": nom / den}


def unitroot_pp(x):
    y, yl = x[1:], x[:-1]
    n = y.size
    lmax = math.trunc(4 * (n / 100) ** 0.25)
    beta, res = _lm_fit(y, np.column_stack([np.ones(n), yl]))
    s = float(np.sum(res ** 2)) / n
    myybar = float(np.sum((y - y.mean()) ** 2)) / n ** 2
    idx = np.arange(1, lmax + 1) if lmax >= 1 else np.array([1, 0])
    cop = np.array([np.dot(res[k:], res[: n - k]) if k > 0 else np.dot(res, res) for k in idx])
    sig = s + 2 / n * float(np.dot(1 - idx / (lmax + 1), cop))
    lam = 0.5 * (sig - s)
    return {"unitroot_pp": n * (beta[1] - 1) - lam / myybar}


def nonlinearity(x):
    """10 * tseries::terasvirta.test(x, type = "Chisq")$stat / length(x) (lag 1, the series scaled first)."""
    n = x.size
    z = (x - x.mean()) / math.sqrt(r_var(x)) if not np.all(x == x[0]) else x
    xs, y = z[1:], z[:-1]
    t = xs.size
    _, u = _lm_fit(xs, np.column_stack([np.ones(t), y]))
    ssr0 = float(np.sum(u ** 2))
    _, v = _lm_fit(u, np.column_stack([np.ones(t), y, y * y, y * y * y]))
    ssr = float(np.sum(v ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = n * math.log(ssr0 / ssr) if ssr > 0 and ssr0 > 0 else NAN       # tseries uses length(x)
    return {"nonlinearity": 10 * stat / n}


def _burg(x: np.ndarray, pmax: int):
    """R's C_Burg (stats/src/burg.c): coefficient matrix (pmax x pmax, row p-1 = AR(p)) and var1."""
    n = x.size
    coefs = np.zeros((pmax, pmax))
    u = x[::-1].copy()
    v = x[::-1].copy()
    var1 = np.zeros(pmax + 1)
    var1[0] = float(np.sum(x * x)) / n
    for p in range(1, pmax + 1):
        sm = float(np.dot(v[p:], u[p - 1:n - 1]))
        d = float(np.dot(v[p:], v[p:]) + np.dot(u[p - 1:n - 1], u[p - 1:n - 1]))
        phii = 2 * sm / d
        coefs[p - 1, p - 1] = phii
        for j in range(1, p):
            coefs[p - 1, j - 1] = coefs[p - 2, j - 1] - phii * coefs[p - 2, p - j - 1]
        u0 = u.copy()
        u[p:] = u0[p - 1:n - 1] - phii * v[p:]
        v[p:] = v[p:] - phii * u0[p - 1:n - 1]
        var1[p] = var1[p - 1] * (1 - phii * phii)
    return coefs, var1


def entropy(x):
    """Spectral entropy from stats::spec.ar(x, method = "burg", n.freq = ceiling(n / 2 + 1)), as tsfeatures 1.1."""
    n = x.size
    xc = x - x.mean()
    pmax = min(n - 1, math.floor(10 * math.log10(n)))
    if pmax < 1:
        return {"entropy": NAN}
    coefs, var1 = _burg(xc, pmax)
    with np.errstate(divide="ignore", invalid="ignore"):
        aic = n * np.log(var1) + 2 * np.arange(pmax + 1) + 2
    if not np.isfinite(aic).any():
        return {"entropy": NAN}
    order = int(np.nanargmin(aic))
    ar = coefs[order - 1, :order] if order else np.zeros(0)
    vp = var1[order]                                         # ar.burg: var.pred is not rescaled
    nf = math.ceil(n / 2 + 1)
    freq = np.linspace(0, 0.5, nf)
    if order:
        k = np.arange(1, order + 1)
        cs = np.cos(2 * np.pi * np.outer(freq, k)) @ ar
        sn = np.sin(2 * np.pi * np.outer(freq, k)) @ ar
        spec = vp / ((1 - cs) ** 2 + sn ** 2)
    else:
        spec = np.full(nf, vp)
    fx = np.concatenate([spec[1:][::-1], spec]) / n
    fx = fx / fx.sum()
    fx = 0.999 * fx + 0.001 / fx.size
    return {"entropy": float(min(1.0, -np.sum(fx * np.log(fx) / math.log(n))))}


# ---------------------------------------------------------------------------------------------------- the study's calls
_SHIFT = {"max_kl_shift": max_kl_shift, "max_level_shift": max_level_shift, "max_var_shift": max_var_shift}
_WIDTH = {"lumpiness": lumpiness, "stability": stability}
_PLAIN = {"crossing_points": crossing_points, "flat_spots": flat_spots, "entropy": entropy,
          "nonlinearity": nonlinearity, "unitroot_kpss": unitroot_kpss, "unitroot_pp": unitroot_pp}
_EXTRA: dict = {}                     # holt_parameters, hurst: filled in below when available


def _one(f: str, z: np.ndarray, m: int, width: int | None = None, lags: int = 12) -> dict:
    w = width if width is not None else (m if m > 1 else 10)
    if f in _SHIFT:
        return _SHIFT[f](z, w)
    if f in _WIDTH:
        return _WIDTH[f](z, w)
    if f == "acf_features":
        return acf_features(z, m)
    if f == "pacf_features":
        return pacf_features(z, m)
    if f == "arch_stat":
        return arch_stat(z, lags)
    if f in _PLAIN:
        return _PLAIN[f](z)
    if f in _EXTRA:
        return _EXTRA[f](z, m)
    raise KeyError(f)


def _has_na(d: dict) -> bool:
    return any(v is None or (isinstance(v, float) and math.isnan(v)) for v in d.values())


def study_features(y, periods, funcs=TSFEATURE_FUNCS, stl: bool = False, startx: int = 0) -> dict:
    """The study's tsfeatures for one series (see the module docstring); periods = the dataset's seasonal periods;
    stl adds stl_features (startx = cycle position of the first observation minus 1, for peak and trough)."""
    y = np.asarray(y, dtype=float)
    m = max(1, int(math.floor(max(periods))))
    out = {"mean": float(np.mean(y)), "var": r_var(y)}
    z = scalets(y)
    for f in funcs:
        try:
            r = _one(f, z, m)
        except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
            r = None
        if r is None or _has_na(r):
            try:
                r = _one(f, z, 1)
            except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
                r = None
            if r is not None and _has_na(r) and (f in _SHIFT or f == "arch_stat"):
                r = _one(f, z, m, width=1) if f in _SHIFT else _one(f, z, m, lags=1)
        out.update(r or {})
    if stl:
        out.update(study_stl_features(z, periods, startx))
    return out


def study_stl_features(z: np.ndarray, periods, startx: int = 0) -> dict:
    """stl_features as the study called it: s.window = "periodic", robust = TRUE; else the defaults; else on
    ts(x, frequency = 1)."""
    if len(periods) > 1:
        kind, per = "msts", tuple(float(p) for p in periods)
    else:
        kind, per = "ts", (max(1, int(math.floor(periods[0]))),)
    for periodic, robust in ((True, True), (False, False)):
        try:
            return stl_features(z, kind, per, startx, periodic, robust)
        except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
            continue
    return stl_features(z, "ts", (1,), 0, True, True)


def holt_parameters(x, m=1):
    """forecast::ets(x, model = "AAN")$par[c("alpha", "beta")] (native ETS port)."""
    import ctypes
    from .engines import cpp_engine
    lib = cpp_engine.load_lib()
    if lib is None:
        return {"alpha": NAN, "beta": NAN}
    if not getattr(lib, "_holt_ready", False):
        lib.tsf_holt_params.restype = ctypes.c_int
        lib.tsf_holt_params.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int32, ctypes.POINTER(ctypes.c_double)]
        lib._holt_ready = True
    xc = np.ascontiguousarray(x, dtype=np.float64)
    out = (ctypes.c_double * 2)()
    if lib.tsf_holt_params(cpp_engine._ptr(xc, ctypes.c_double), xc.size, out) != 0:
        return {"alpha": NAN, "beta": NAN}
    return {"alpha": float(out[0]), "beta": float(out[1])}


_EXTRA["holt_parameters"] = holt_parameters


def hurst(x, m=1):
    """fracdiff::fracdiff(x, 0, 0)$d + 0.5 (native port of fracdiff's C routines)."""
    import ctypes
    from .engines import cpp_engine
    lib = cpp_engine.load_lib()
    if lib is None:
        return {"hurst": NAN}
    if not getattr(lib, "_fd_ready", False):
        lib.tsf_fracdiff_d.restype = ctypes.c_double
        lib.tsf_fracdiff_d.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int32]
        lib._fd_ready = True
    xc = np.ascontiguousarray(x, dtype=np.float64)
    return {"hurst": float(lib.tsf_fracdiff_d(cpp_engine._ptr(xc, ctypes.c_double), xc.size)) + 0.5}


_EXTRA["hurst"] = hurst


# ---------------------------------------------------------------------------------------------------- stl_features
def _native_stl(x: np.ndarray, period: float, periodic: bool, s_window: int, robust: bool):
    import ctypes
    from .engines import cpp_engine
    lib = cpp_engine.load_lib()
    if lib is None:
        raise RuntimeError("native library unavailable")
    if not getattr(lib, "_stl_ready", False):
        dp = ctypes.POINTER(ctypes.c_double)
        lib.tsf_stl.restype = ctypes.c_int
        lib.tsf_stl.argtypes = [dp, ctypes.c_int32, ctypes.c_double, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, dp, dp]
        lib.tsf_supsmu.restype = None
        lib.tsf_supsmu.argtypes = [dp, ctypes.c_int32, dp]
        lib._stl_ready = True
    xc = np.ascontiguousarray(x, dtype=np.float64)
    if period is None:
        out = np.zeros(xc.size)
        lib.tsf_supsmu(cpp_engine._ptr(xc, ctypes.c_double), xc.size, cpp_engine._ptr(out, ctypes.c_double))
        return out
    s, t = np.zeros(xc.size), np.zeros(xc.size)
    if lib.tsf_stl(cpp_engine._ptr(xc, ctypes.c_double), xc.size, float(period), int(periodic), int(s_window),
                   int(robust), cpp_engine._ptr(s, ctypes.c_double), cpp_engine._ptr(t, ctypes.c_double)) != 0:
        raise ValueError("series is not periodic or has less than two periods")
    return s, t


def _mstl(x: np.ndarray, kind: str, periods: tuple, periodic: bool, robust: bool):
    """forecast::mstl(x, s.window, robust): (trend, [seasonal components], seasonal periods used, remainder).
    kind: "msts" (several periods), "ts" (one frequency) or "plain" (no ts attributes)."""
    n = x.size
    if kind == "msts":
        msts = sorted(p for p in periods if p < n / 2)
        if not msts:
            raise ValueError("no seasonal period with two full cycles")
        iterate = 2
    elif kind == "ts":
        msts, iterate = [periods[0]], 1
    else:
        msts, iterate = [1], 1
    if msts[0] > 1:
        swin = [11, 15, 19, 23, 27, 31]
        seas = [np.zeros(n) for _ in msts]
        deseas = x.copy()
        trend = None
        for _ in range(iterate):
            for i, p in enumerate(msts):
                deseas = deseas + seas[i]
                s, trend = _native_stl(deseas, p, periodic, swin[i] if i < 6 else 31, robust)
                seas[i] = s
                deseas = deseas - seas[i]
    else:
        msts, seas, deseas = [], [], x
        trend = _native_stl(x, None, False, 0, False)
    return trend, seas, msts, deseas - trend


def _poly_coefs(y: np.ndarray) -> tuple[float, float]:
    """coef(lm(y ~ poly(seq(n), degree = min(n - 1, 2))))[2:3] (orthonormal polynomial basis)."""
    n = y.size
    t = np.arange(1, n + 1, dtype=float)
    deg = min(n - 1, 2)
    X = np.column_stack([(t - t.mean()) ** k for k in range(deg + 1)])
    Q, R = np.linalg.qr(X)
    Q = Q * np.sign(np.diag(R))
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), Q[:, 1:]]), y, rcond=None)
    return float(beta[1]) if deg >= 1 else NAN, float(beta[2]) if deg >= 2 else NAN


def stl_features(x: np.ndarray, kind: str, periods: tuple, startx: int = 0, periodic: bool = True,
                 robust: bool = True) -> dict:
    """tsfeatures::stl_features(x, s.window = "periodic", robust = TRUE) on the scaled series x.

    kind "msts": several seasonal periods (attributes(x)$msts = periods); "ts": one frequency periods[0]; startx =
    start(x)[2] - 1 (the cycle position of the first observation, for peak and trough)."""
    n = x.size
    # The study's tsfeatures stopped for a multi-period (msts) series not longer than two of its longest periods (the
    # caller then falls back to frequency 1), as the stored m4_hourly values show; a short single-period series kept
    # nperiods and seasonal_period but no seasonal components (stored m4_weekly).
    if kind == "msts" and n <= 2 * max(periods):
        raise ValueError("insufficient data to compute STL decomposition")
    if kind == "msts":
        msts_out, nperiods = list(periods), len(periods)
    else:
        msts_out, nperiods = [periods[0]], int(periods[0] > 1)
        if n <= 2 * periods[0]:
            kind = "plain"                             # x <- c(x): mstl sees no seasonality
    trend0, seas, used, remainder = _mstl(x, kind, periods, periodic, robust)
    deseason = x - np.sum(seas, axis=0) if seas else x
    varx, vare, vardes = r_var(x), r_var(remainder), r_var(deseason)
    if varx < np.finfo(float).eps or vardes / varx < 1e-10:
        trend = 0.0
    else:
        trend = max(0.0, min(1.0, 1 - vare / vardes))
    d = (remainder - remainder.mean()) ** 2
    varloo = (vare * (n - 1) - d) / (n - 2)
    lin, curv = _poly_coefs(trend0)
    m_rem = periods[0] if kind == "ts" else (max(1, int(math.floor(max(periods)))) if kind == "msts" else 1)
    acfr = acf_features(remainder, m_rem)
    out = {"nperiods": float(nperiods)}
    if len(msts_out) == 1:
        out["seasonal_period"] = float(msts_out[0])
    else:
        for i, p in enumerate(msts_out, 1):
            out[f"seasonal_period{i}"] = float(p)
    out.update({"trend": trend, "spike": r_var(varloo), "linearity": lin, "curvature": curv,
                "e_acf1": acfr["x_acf1"], "e_acf10": acfr["x_acf10"]})
    if seas:
        names = [""] if len(seas) == 1 else [str(i) for i in range(1, len(seas) + 1)]
        for nm, s in zip(names, seas):
            out["seasonal_strength" + nm] = max(0.0, min(1.0, 1 - vare / r_var(remainder + s)))
        for nm, s, p in zip(names, seas, used):
            pk = (startx + int(np.argmax(s)) + 1) % p
            th = (startx + int(np.argmin(s)) + 1) % p
            out["peak" + nm] = float(p if pk == 0 else pk)
            out["trough" + nm] = float(p if th == 0 else th)
    return out


def study_startx(timestamp, periods) -> int:
    """start(x)[2] - 1 of the study's msts for a series starting at `timestamp` (None: no index, 0).

    The study built start = c(floor(s), floor((s - floor(s)) * max(periods))) from s = lubridate::decimal_date of the
    first date (via tsibble's as.ts): a start falling on a whole cycle (1 January, or e.g. 1 November of a leap year for
    monthly data) keeps that cycle; any other date lands one cycle earlier than the floor (cycle 0 wraps to the previous
    year's last cycle)."""
    if timestamp is None or (isinstance(timestamp, float) and math.isnan(timestamp)):
        return 0
    import pandas as pd
    t = pd.Timestamp(timestamp)
    y0 = pd.Timestamp(year=t.year, month=1, day=1)
    sofar = (t.normalize() - y0).total_seconds()
    total = (pd.Timestamp(year=t.year + 1, month=1, day=1) - y0).total_seconds()
    s = t.year + sofar / total
    frac = s - math.floor(s)
    maxs = max(periods)
    v = frac * maxs
    if abs(v - round(v)) < 1e-5:                         # start() sees a whole cycle (ts.eps): cycle floor(v + .001)
        return int(math.floor(v + 0.001))
    b = math.floor(v)
    freq = max(1, int(math.floor(maxs)))
    return b - 1 if b >= 1 else freq - 1
