"""Global modeller (Python): havuzlanmış regresyon ve CatBoost — R `global_models.R` ile aynı veri hattı.

Veri hattı (R ile birebir):
  1. Her seri (varsa son `max_obs` gözlemle sınırlı) kendi ortalamasına bölünür (ortalama 0 ise 1).
  2. `embed(x, lag + 1)`: satır = [x_t, x_{t-1}, ..., x_{t-lag}]; ilk sütun y, kalanlar Lag1..Lag_lag.
  3. Tüm serilerin satırları alt alta birleştirilir (kesişimsiz doğrusal model `y ~ Lag1 + ... + 0`).
  4. Özyinelemeli tahmin: Lag'ler [yeni_tahmin, Lag1..Lag_{lag-1}] olarak kayar; sonunda seri ortalamasıyla çarpılır.

Havuzlanmış regresyon R'da `glm` (gauss, kimlik bağı) = sıradan en küçük kareler. Burada tasarım matrisi BELLEĞE
ALINMAZ: satır blokları akış hâlinde QR ile birleştirilir (R-faktörlerinin üst üste QR'ı), böylece M5 L12 gibi
55 milyon satırlık problemler de birkaç yüz MB ile çözülür ve sonuç tam matrisli en küçük kareyle aynıdır.
R, doğrusal bağımlı sütunları `NA` katsayı yapıp tahminden düşürür; aynı kural burada uygulanır (`aliased`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_ALIAS_TOL = 1e-11          # R glm.fit / dqrls tol = min(1e-7, eps/1000) ~ 1e-11


@dataclass
class GlobalFitInfo:
    n_rows: int = 0
    rank: int = 0
    aliased: list[int] = field(default_factory=list)      # 1 tabanlı Lag numaraları
    notes: list[str] = field(default_factory=list)


def normalise(series: Sequence[np.ndarray], lag: int, max_obs: int | None = None) -> tuple[list[np.ndarray], np.ndarray]:
    """Ortalama normalizasyonu (+ isteğe bağlı son `max_obs` pencere). Çok kısa seri için açık hata."""
    out: list[np.ndarray] = []
    means = np.empty(len(series))
    for i, s in enumerate(series):
        x = np.asarray(s, dtype=np.float64)
        if max_obs and len(x) > max_obs:
            x = x[-max_obs:]
        if len(x) < lag + 1:
            raise ValueError(f"Series {i + 1} is too short for lag = {lag} (length {len(x)}; at least {lag + 1} required).")
        mu = float(x.mean())
        if mu == 0.0:
            mu = 1.0
        out.append(x / mu)
        means[i] = mu
    return out, means


def embed(x: np.ndarray, lag: int) -> np.ndarray:
    """R `embed(x, lag + 1)`: (len - lag) x (lag + 1); sütun 0 = y, sütun j = Lag j."""
    return sliding_window_view(x, lag + 1)[:, ::-1]


def final_lags(norm_series: Sequence[np.ndarray], lag: int) -> np.ndarray:
    """Her seri için `rev(tail(x, lag))` (n x lag); Lag1 = son gözlem."""
    return np.vstack([x[-lag:][::-1] for x in norm_series])


def _blocks(norm_series: Sequence[np.ndarray], lag: int, chunk_rows: int) -> Iterator[np.ndarray]:
    """[Lag1..Lag_lag | y] bloklarını (en çok ~chunk_rows satır) üretir; sıra R'daki rbind sırasıyla aynı."""
    parts: list[np.ndarray] = []
    rows = 0
    for x in norm_series:
        e = embed(x, lag)
        parts.append(np.concatenate([e[:, 1:], e[:, :1]], axis=1))
        rows += e.shape[0]
        if rows >= chunk_rows:
            yield np.vstack(parts)
            parts, rows = [], 0
    if parts:
        yield np.vstack(parts)


def _qr_accumulate(blocks: Iterator[np.ndarray], ncols: int) -> tuple[np.ndarray, int]:
    R = np.empty((0, ncols))
    n = 0
    for A in blocks:
        n += A.shape[0]
        R = np.linalg.qr(np.vstack([R, A]), mode="r")
    return R, n


def _solve_with_aliasing(R: np.ndarray, p: int) -> tuple[np.ndarray, list[int]]:
    """R: artırılmış matrisin R-faktörü ([X | y]). Soldan sağa, önceki sütunlara göre yaklaşık bağımlı sütunlar
    (kalan norm < tol * özgün norm) düşürülür (dqrdc2 mantığı); kalanlar için küçük en küçük kareler çözülür."""
    cols = R[:, :p]
    yv = R[:, p]
    keep: list[int] = []
    Q = np.empty((R.shape[0], 0))
    aliased: list[int] = []
    for j in range(p):
        c = cols[:, j]
        nrm = float(np.linalg.norm(c))
        r = c - Q @ (Q.T @ c) if Q.shape[1] else c.copy()
        r = r - Q @ (Q.T @ r) if Q.shape[1] else r              # yeniden dikgenleştirme (kararlılık)
        rn = float(np.linalg.norm(r))
        if nrm == 0.0 or rn < _ALIAS_TOL * nrm:
            aliased.append(j + 1)
            continue
        keep.append(j)
        Q = np.column_stack([Q, r / rn])
    beta = np.zeros(p)
    if keep:
        sol, *_ = np.linalg.lstsq(cols[:, keep], yv, rcond=None)
        beta[keep] = sol
    return beta, aliased


def fit_pooled_regression(norm_series: Sequence[np.ndarray], lag: int, chunk_rows: int = 200_000) -> tuple[np.ndarray, GlobalFitInfo]:
    R, n = _qr_accumulate(_blocks(norm_series, lag, chunk_rows), lag + 1)
    if n < lag + 1:
        raise ValueError(f"Not enough rows for the regression ({n} < {lag + 1}).")
    beta, aliased = _solve_with_aliasing(R, lag)
    info = GlobalFitInfo(n_rows=n, rank=lag - len(aliased), aliased=aliased)
    if aliased:
        info.notes.append(f"{len(aliased)} linearly dependent lag columns (Lag {aliased[:8]}{'…' if len(aliased) > 8 else ''}) "
                          f"were treated as NA coefficients and set to 0, as R `glm` does.")
    return beta, info


def forecast_recursive(predict, lags0: np.ndarray, horizon: int, means: np.ndarray) -> np.ndarray:
    """`predict(lags[n x lag]) -> (n,)`. R `forec_recursive` ile aynı kayma kuralı; sonda ortalamayla ölçeklenir."""
    lags = lags0.copy()
    preds = np.empty((lags.shape[0], horizon))
    for k in range(horizon):
        p = np.asarray(predict(lags), dtype=np.float64).reshape(-1)
        preds[:, k] = p
        if k < horizon - 1:
            lags = np.concatenate([p[:, None], lags[:, :-1]], axis=1)
    return preds * means[:, None]


def pooled_regression_forecast(series: Sequence[np.ndarray], lag: int, horizon: int, max_obs: int | None = None,
                               chunk_rows: int = 200_000) -> tuple[np.ndarray, GlobalFitInfo]:
    norm, means = normalise(series, lag, max_obs)
    beta, info = fit_pooled_regression(norm, lag, chunk_rows)
    return forecast_recursive(lambda L: L @ beta, final_lags(norm, lag), horizon, means), info


# ------------------------------------------------------------------------------------------------
# CatBoost
# ------------------------------------------------------------------------------------------------
def catboost_available() -> bool:
    try:
        import catboost  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def catboost_forecast(series: Sequence[np.ndarray], lag: int, horizon: int, max_obs: int | None = None,
                      params: dict | None = None, workers: int = 1) -> tuple[np.ndarray, GlobalFitInfo]:
    """R `catboost.train` varsayılanlarıyla aynı: yalnızca `params` verilenler değiştirilir (random_seed varsayılanı 0)."""
    from catboost import CatBoostRegressor, Pool

    norm, means = normalise(series, lag, max_obs)
    n_rows = sum(len(x) - lag for x in norm)
    X = np.empty((n_rows, lag), dtype=np.float32)
    y = np.empty(n_rows, dtype=np.float32)
    pos = 0
    for x in norm:                                    # R'daki rbind sırası: seri seri, embed satır sırası
        e = embed(x, lag)
        k = e.shape[0]
        X[pos: pos + k] = e[:, 1:]
        y[pos: pos + k] = e[:, 0]
        pos += k
    kw = {"verbose": False, "allow_writing_files": False, "thread_count": workers if workers > 1 else -1}
    kw.update(params or {})
    model = CatBoostRegressor(**kw)
    model.fit(Pool(X, label=y))
    del X, y
    info = GlobalFitInfo(n_rows=n_rows, rank=lag)
    pred = lambda L: model.predict(np.asarray(L, dtype=np.float32))   # noqa: E731
    return forecast_recursive(pred, final_lags(norm, lag), horizon, means), info


# ------------------------------------------------------------------------------------------------
# LightGBM
# ------------------------------------------------------------------------------------------------
# LightGBM's own defaults, plus a fixed seed and deterministic training so a run is reproducible (and the C++ engine,
# which drives the same lib_lightgbm through its C API with the same parameter string, gives the same numbers).
LIGHTGBM_DEFAULTS = {"objective": "regression", "num_iterations": 100, "learning_rate": 0.1, "num_leaves": 31,
                     "min_data_in_leaf": 20, "seed": 0}
LIGHTGBM_FIXED = {"deterministic": True, "force_col_wise": True, "verbosity": -1}
LIGHTGBM_KEYS = ("num_iterations", "learning_rate", "num_leaves", "min_data_in_leaf", "seed")


def lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def lightgbm_params(params: dict | None = None, workers: int = 1) -> dict:
    """Effective parameters: defaults, the user's values (only the keys in LIGHTGBM_KEYS) and the fixed settings."""
    p = dict(LIGHTGBM_DEFAULTS)
    for k, v in (params or {}).items():
        if k not in LIGHTGBM_KEYS:
            raise ValueError(f"Unknown LightGBM parameter '{k}' (allowed: {', '.join(LIGHTGBM_KEYS)}).")
        if v is not None:
            p[k] = float(v) if k == "learning_rate" else int(v)
    p.update(LIGHTGBM_FIXED)
    p["num_threads"] = int(workers) if workers and int(workers) > 1 else 0
    return p


def lag_matrix(norm_series: Sequence[np.ndarray], lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Row-major float64 design matrix [Lag1..Lag_lag] and target, rows in R's rbind order."""
    n_rows = sum(len(x) - lag for x in norm_series)
    X = np.empty((n_rows, lag), dtype=np.float64)
    y = np.empty(n_rows, dtype=np.float64)
    pos = 0
    for x in norm_series:
        e = embed(x, lag)
        k = e.shape[0]
        X[pos: pos + k] = e[:, 1:]
        y[pos: pos + k] = e[:, 0]
        pos += k
    return X, y


def lgb_feature_spec(norm_series: Sequence[np.ndarray], lag: int, m: int) -> tuple[int, int, int]:
    """(m_eff, w1, w2): seasonal features only when the shortest series holds three full seasons (m_eff = m, else 1);
    rolling windows w1 = max(4, m_eff) and w2 = 2 w1."""
    nmin = min(len(x) for x in norm_series)
    m_eff = m if (m > 1 and nmin >= 3 * m) else 1
    w1 = max(4, m_eff)
    return m_eff, w1, 2 * w1


def lgb_feature_names(lag: int, m_eff: int, w1: int, w2: int) -> list[str]:
    names = [f"lag{j}" for j in range(1, lag + 1)]
    if m_eff > 1:
        names += [f"slag{m_eff}", f"slag{2 * m_eff}"]
    names += [f"rmean{w1}", f"rstd{w1}", f"rmean{w2}", f"rstd{w2}"]
    if m_eff > 1:
        names.append("season_pos")
    return names


def _roll_mean_std(win: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise mean and population sd of windows (oldest -> newest), with sequential sums (np.cumsum), so the C++
    engine reproduces them exactly."""
    w = win.shape[1]
    mean = np.cumsum(win, axis=1)[:, -1] / w
    dev = win - mean[:, None]
    return mean, np.sqrt(np.cumsum(dev * dev, axis=1)[:, -1] / w)


def lgb_features(hist: np.ndarray, t: np.ndarray, lag: int, m_eff: int, w1: int, w2: int) -> np.ndarray:
    """Feature rows for targets at positions t of the (normalised) history hist."""
    cols = [hist[t - j] for j in range(1, lag + 1)]
    if m_eff > 1:
        cols += [hist[t - m_eff], hist[t - 2 * m_eff]]
    for w in (w1, w2):
        win = np.stack([hist[t - w + k] for k in range(w)], axis=1)
        mu, sd = _roll_mean_std(win)
        cols += [mu, sd]
    if m_eff > 1:
        cols.append((t % m_eff).astype(np.float64))
    return np.column_stack(cols)


def lightgbm_feat_forecast(series: Sequence[np.ndarray], lag: int, horizon: int, m: int, max_obs: int | None = None,
                           params: dict | None = None, workers: int = 1) -> tuple[np.ndarray, GlobalFitInfo]:
    """Global LightGBM with M5-style features (lags, seasonal lags, rolling mean / sd over two windows, position in
    the season) on mean-scaled series; recursive forecasts (the history grows by each prediction)."""
    import lightgbm as lgb

    p = lightgbm_params(params, workers)
    norm, means = normalise(series, lag, max_obs)
    m_eff, w1, w2 = lgb_feature_spec(norm, lag, m)
    h0 = max(lag, 2 * m_eff if m_eff > 1 else 0, w2)
    short = [i + 1 for i, x in enumerate(norm) if len(x) <= h0]
    if short:
        raise ValueError(f"Series {short[0]} is too short for the LightGBM features (at least {h0 + 1} observations required).")
    X = np.vstack([lgb_features(x, np.arange(h0, len(x)), lag, m_eff, w1, w2) for x in norm])
    y = np.concatenate([x[h0:] for x in norm])
    rounds = p.pop("num_iterations")
    model = lgb.train(p, lgb.Dataset(X, label=y, params=p), num_boost_round=rounds)
    info = GlobalFitInfo(n_rows=X.shape[0], rank=X.shape[1])
    info.notes.append(f"Features: {', '.join(lgb_feature_names(lag, m_eff, w1, w2))}.")
    del X, y
    hists = [np.concatenate([x, np.zeros(horizon)]) for x in norm]
    out = np.empty((len(norm), horizon))
    for k in range(horizon):
        F = np.vstack([lgb_features(hh, np.array([len(x) + k]), lag, m_eff, w1, w2) for hh, x in zip(hists, norm)])
        pred = model.predict(np.ascontiguousarray(F))
        for i, (hh, x) in enumerate(zip(hists, norm)):
            hh[len(x) + k] = pred[i]
        out[:, k] = pred
    return out * means[:, None], info


def lightgbm_direct_forecast(series: Sequence[np.ndarray], lag: int, horizon: int, m: int, max_obs: int | None = None,
                             params: dict | None = None, workers: int = 1) -> tuple[np.ndarray, GlobalFitInfo]:
    """Direct multi-horizon LightGBM: the features of lightgbm_feat at the forecast origin t, and one model per step k
    whose target is the value at t + k (k = 0 .. horizon - 1). No forecast is fed back, so errors do not compound over
    the horizon; the price is `horizon` models. Model k learns from the origins t = h0 .. n - 1 - k of every series."""
    import lightgbm as lgb

    p = lightgbm_params(params, workers)
    norm, means = normalise(series, lag, max_obs)
    m_eff, w1, w2 = lgb_feature_spec(norm, lag, m)
    h0 = max(lag, 2 * m_eff if m_eff > 1 else 0, w2)
    short = [i + 1 for i, x in enumerate(norm) if len(x) <= h0]
    if short:
        raise ValueError(f"Series {short[0]} is too short for the LightGBM features (at least {h0 + 1} observations required).")
    feats = [lgb_features(x, np.arange(h0, len(x)), lag, m_eff, w1, w2) for x in norm]
    F_last = np.vstack([lgb_features(np.concatenate([x, [0.0]]), np.array([len(x)]), lag, m_eff, w1, w2) for x in norm])
    rounds = p.pop("num_iterations")
    out = np.empty((len(norm), horizon))
    n_rows = 0
    for k in range(horizon):
        rows = [(f[: len(x) - h0 - k], x[h0 + k:]) for f, x in zip(feats, norm) if len(x) - h0 - k > 0]
        if not rows:
            raise ValueError(f"No training rows for forecast step {k + 1}: the series are too short for this horizon.")
        X = np.vstack([r[0] for r in rows])
        y = np.concatenate([r[1] for r in rows])
        n_rows = max(n_rows, X.shape[0])
        model = lgb.train(p, lgb.Dataset(X, label=y, params=p), num_boost_round=rounds)
        out[:, k] = model.predict(np.ascontiguousarray(F_last))
    info = GlobalFitInfo(n_rows=n_rows, rank=F_last.shape[1])
    info.notes.append(f"Direct strategy: {horizon} models, one per forecast step. Features: "
                      f"{', '.join(lgb_feature_names(lag, m_eff, w1, w2))}.")
    return out * means[:, None], info


def lightgbm_forecast(series: Sequence[np.ndarray], lag: int, horizon: int, max_obs: int | None = None,
                      params: dict | None = None, workers: int = 1) -> tuple[np.ndarray, GlobalFitInfo]:
    """Global LightGBM (Ke et al. 2017) on the same lag matrix as pooled regression; recursive forecasts."""
    import lightgbm as lgb

    p = lightgbm_params(params, workers)
    norm, means = normalise(series, lag, max_obs)
    X, y = lag_matrix(norm, lag)
    rounds = p.pop("num_iterations")
    model = lgb.train(p, lgb.Dataset(X, label=y, params=p), num_boost_round=rounds)
    info = GlobalFitInfo(n_rows=X.shape[0], rank=lag)
    del X, y
    pred = lambda L: model.predict(np.ascontiguousarray(L, dtype=np.float64))   # noqa: E731
    return forecast_recursive(pred, final_lags(norm, lag), horizon, means), info
