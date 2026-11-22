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
