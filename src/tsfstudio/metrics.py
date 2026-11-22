"""Hata metrikleri.

Tanımlar kullanıcının R kodundan (utils/error_calculator_original.R ve error_calculator.R) birebir
alınmıştır; yalnızca aşağıdaki NOKTALARDA bilinçli olarak farklıdır (hepsi `reference/` altındaki
orijinallerde hataya ya da hizasızlığa yol açıyordu):

  1. Tanımsız değer (0/0, sıfır ölçek, boş test) seri başına NaN olarak KALIR; MASE'te seriler
     atılıp vektör kısaltılmaz. Böylece tüm metrik sütunları seri adıyla hizalı kalır.
     Özetlerde ortalama/medyan yalnızca sonlu değerler üzerinden alınır ve kaç serinin dışarıda
     kaldığı (`N_excluded`) raporlanır. (Orijinal MASE zaten sonsuz/NA serileri atıyordu.)
  2. WAPE ve nRMSE'de payda 0 ise sonuç NaN'dır (R'da Inf olur ve `mean(na.rm=TRUE)` özeti Inf yapar).
  3. MASE'te tahmin/gerçek çifti, NaN içermeyen çiftler üzerinden hizalanır (orijinal ikisini ayrı ayrı
     süzüp uzunlukları uyuşmaz hale getirebilirdi). NaN içermeyen veride sonuç aynıdır.
  4. MASE ölçeği: `mean(|diff(train, lag = min(mevsimsellik))|)`; gecikme tamsayıya kesilir
     (örn. 365,25/7 -> 52), ölçek tanımsızsa lag 1'e düşülür — R kodundaki gibi.
     `smooth::MASE` eski sürümlerde sonucu 3 haneye yuvarlıyordu; güncel greybox/smooth yuvarlamaz,
     burada da yuvarlama yoktur (`mase_digits` ile isteğe bağlı açılabilir).

Girdi sözleşmesi: `forecasts` ve `actuals` (n_seri, ufuk) biçiminde; farklı uzunlukta test kümeleri NaN ile
sağdan doldurulur. `train` her seri için eğitim dizisi (ragged liste).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .seasonality import primary_period

ORIGINAL_METRICS = ("SMAPE", "mSMAPE", "MASE", "MAE", "RMSE")
DEFAULT_METRICS = ORIGINAL_METRICS + ("WAPE", "nRMSE_Mean", "nRMSE_Range")  # error_calculator.R ile aynı küme
# 2026-11-22: isteğe bağlı ek metrikler (varsayılan kümeyi DEĞİŞTİRMEZ; R çıktısıyla karşılaştırma bozulmasın diye).
# Tanımlar ve kaynaklar: docs/METRICS.md ve docs/RESEARCH_MODELS_METRICS.md.
EXTRA_METRICS = ("RMSSE", "MSE", "MAPE", "MAAPE", "MedAE", "RMSLE", "ME", "PBias")
ALL_METRICS = DEFAULT_METRICS + EXTRA_METRICS
_TRAIN_METRICS = ("MASE", "RMSSE")                     # örnek-içi ölçek gerektirenler
MSMAPE_EPSILON = 0.1


def _as2d(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"A 2-D array of shape (n_series, horizon) was expected; got shape: {a.shape}")
    return a


def _check(forecasts, actuals) -> tuple[np.ndarray, np.ndarray]:
    f, a = _as2d(forecasts), _as2d(actuals)
    if f.shape != a.shape:
        raise ValueError(f"Forecast {f.shape} and actual {a.shape} must have the same shape")
    return f, a


def _row_nanmean(x: np.ndarray) -> np.ndarray:
    """R: rowMeans(x, na.rm = TRUE) — tamamı NaN olan satır NaN verir (uyarısız)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=1)


def smape(forecasts, actuals) -> np.ndarray:
    f, a = _check(forecasts, actuals)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = 2.0 * np.abs(f - a) / (np.abs(f) + np.abs(a))
    return _row_nanmean(e)


def msmape(forecasts, actuals, epsilon: float = MSMAPE_EPSILON) -> np.ndarray:
    f, a = _check(forecasts, actuals)
    denom = np.maximum(0.5 + epsilon, np.abs(f) + np.abs(a) + epsilon)
    return _row_nanmean(2.0 * np.abs(f - a) / denom)


def mae(forecasts, actuals) -> np.ndarray:
    f, a = _check(forecasts, actuals)
    return _row_nanmean(np.abs(f - a))


def rmse(forecasts, actuals) -> np.ndarray:
    f, a = _check(forecasts, actuals)
    return np.sqrt(_row_nanmean((f - a) ** 2))


def wape(forecasts, actuals) -> np.ndarray:
    """Σ|f-a| / Σ|a| × 100 (R: calculate_wape). Payda 0 -> NaN."""
    f, a = _check(forecasts, actuals)
    ok = ~(np.isnan(f) | np.isnan(a))
    num = np.where(ok, np.abs(f - a), 0.0).sum(axis=1)
    den = np.where(ok, np.abs(a), 0.0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den * 100.0
    out[(den == 0) | (ok.sum(axis=1) == 0)] = np.nan
    return out


def nrmse_mean(forecasts, actuals) -> np.ndarray:
    """RMSE / mean(gerçek). Ortalama 0 ise NaN (R: calculate_nrmse_mean)."""
    f, a = _check(forecasts, actuals)
    r = rmse(f, a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(a, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = r / m
    out[m == 0] = np.nan
    return out


def nrmse_range(forecasts, actuals) -> np.ndarray:
    """RMSE / (max-min of gerçek). Aralık 0 ise NaN (R: calculate_nrmse_range)."""
    f, a = _check(forecasts, actuals)
    r = rmse(f, a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rng = np.nanmax(a, axis=1) - np.nanmin(a, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = r / rng
    out[rng == 0] = np.nan
    return out


def _ratio(err: float, scale: float) -> float:
    if np.isnan(scale):
        return np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(err) / np.float64(scale))


def _mase_scale(train: np.ndarray, lag: int) -> float:
    if train.size <= lag:
        return np.nan
    d = np.abs(train[lag:] - train[:-lag])
    return float(d.mean()) if d.size else np.nan


def mase(
    forecasts,
    actuals,
    train: Sequence[np.ndarray],
    seasonality: Sequence[float] | float,
    digits: int | None = None,
) -> np.ndarray:
    """Seri başına MASE (Hyndman & Koehler, 2006), ölçek = mevsimsel naif örnek-içi MAE."""
    f, a = _check(forecasts, actuals)
    if len(train) != f.shape[0]:
        raise ValueError(f"the length of train ({len(train)}) does not match the number of series ({f.shape[0]})")
    lag = primary_period(seasonality)
    out = np.full(f.shape[0], np.nan)
    for k in range(f.shape[0]):
        tr = np.asarray(train[k], dtype=np.float64)
        tr = tr[~np.isnan(tr)]
        ok = ~(np.isnan(f[k]) | np.isnan(a[k]))
        if not ok.any():
            continue
        err = float(np.abs(f[k][ok] - a[k][ok]).mean())
        # R: mase <- MASE(te, f, seasonal scale); if(is.na(mase)) mase <- MASE(te, f, lag-1 scale)
        m = _ratio(err, _mase_scale(tr, lag))
        if np.isnan(m):
            m = _ratio(err, _mase_scale(tr, 1))
        if np.isfinite(m):                    # Inf/NaN -> NaN bırakılır (özette dışarıda kalır; R vektörden atıyordu)
            out[k] = m
    if digits is not None:
        out = np.round(out, digits)
    return out


# ---------------------------------------------------------------------------------------------
# Ek metrikler (2026-11-22). Hepsi seri başına; tanımsız durum NaN (özette N_excluded olarak sayılır).
# ---------------------------------------------------------------------------------------------
def mse(forecasts, actuals) -> np.ndarray:
    f, a = _check(forecasts, actuals)
    return _row_nanmean((f - a) ** 2)


def mape(forecasts, actuals) -> np.ndarray:
    """100 · mean(|f − a| / |a|). Gerçek değeri 0 olan noktalar tanımsızdır ve dışarıda kalır (Hyndman & Koehler 2006)."""
    f, a = _check(forecasts, actuals)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.abs(f - a) / np.abs(a)
    e[a == 0] = np.nan
    return 100.0 * _row_nanmean(e)


def maape(forecasts, actuals) -> np.ndarray:
    """mean(arctan(|f − a| / |a|)) radyan, [0, π/2] (Kim & Kim 2016). a = 0 ve f ≠ 0 -> π/2; a = f = 0 -> 0."""
    f, a = _check(forecasts, actuals)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.arctan(np.abs(f - a) / np.abs(a))          # x/0 -> inf -> arctan = π/2
    e[(a == 0) & (f == 0)] = 0.0
    return _row_nanmean(e)


def medae(forecasts, actuals) -> np.ndarray:
    f, a = _check(forecasts, actuals)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.abs(f - a), axis=1)


def rmsle(forecasts, actuals) -> np.ndarray:
    """sqrt(mean((ln(1+f) − ln(1+a))²)). Negatif tahmin/gerçek içeren seri NaN (kırpma yapılmaz; sessiz düzeltme olmasın)."""
    f, a = _check(forecasts, actuals)
    with np.errstate(invalid="ignore"):
        bad = np.nansum((f < 0) | (a < 0), axis=1) > 0
        out = np.sqrt(_row_nanmean((np.log1p(np.where(f < 0, np.nan, f)) - np.log1p(np.where(a < 0, np.nan, a))) ** 2))
    out[bad] = np.nan
    return out


def me(forecasts, actuals) -> np.ndarray:
    """Ortalama hata, mean(f − a): pozitif = aşırı tahmin (yanlılık)."""
    f, a = _check(forecasts, actuals)
    return _row_nanmean(f - a)


def pbias(forecasts, actuals) -> np.ndarray:
    """100 · Σ(f − a) / Σ a. Σ a = 0 -> NaN."""
    f, a = _check(forecasts, actuals)
    ok = ~(np.isnan(f) | np.isnan(a))
    num = np.where(ok, f - a, 0.0).sum(axis=1)
    den = np.where(ok, a, 0.0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 * num / den
    out[(den == 0) | (ok.sum(axis=1) == 0)] = np.nan
    return out


def rmsse(forecasts, actuals, train: Sequence[np.ndarray]) -> np.ndarray:
    """RMSSE (M5; Makridakis, Spiliotis & Assimakopoulos 2022): sqrt(mean(e²) / mean((y_t − y_{t−1})², örnek içi)).

    Ölçek tüm eğitim kısmı üzerinden alınır. M5 resmi tanımında ölçek ilk sıfır olmayan satıştan başlar; bu fark
    yalnızca başında sıfırlar olan serileri etkiler (docs/METRICS.md). Ölçek 0 -> NaN."""
    f, a = _check(forecasts, actuals)
    if len(train) != f.shape[0]:
        raise ValueError(f"the length of train ({len(train)}) does not match the number of series ({f.shape[0]})")
    num = _row_nanmean((f - a) ** 2)
    out = np.full(f.shape[0], np.nan)
    for k in range(f.shape[0]):
        tr = np.asarray(train[k], dtype=np.float64)
        tr = tr[~np.isnan(tr)]
        if tr.size < 2:
            continue
        s = float(np.mean(np.diff(tr) ** 2))
        if s > 0 and np.isfinite(num[k]):
            out[k] = np.sqrt(num[k] / s)
    return out


# ---------------------------------------------------------------------------------------------
# Toplu hesap + özet + Excel
# ---------------------------------------------------------------------------------------------
_SIMPLE: dict[str, Callable] = {
    "SMAPE": smape, "mSMAPE": msmape, "MAE": mae, "RMSE": rmse,
    "WAPE": wape, "nRMSE_Mean": nrmse_mean, "nRMSE_Range": nrmse_range,
    "MSE": mse, "MAPE": mape, "MAAPE": maape, "MedAE": medae, "RMSLE": rmsle, "ME": me, "PBias": pbias,
}


def compute_error_table(
    forecasts,
    actuals,
    train: Sequence[np.ndarray],
    seasonality: Sequence[float] | float,
    series_names: Sequence[str] | None = None,
    metrics: Sequence[str] = DEFAULT_METRICS,
    mase_digits: int | None = None,
) -> pd.DataFrame:
    """Seri başına metrik tablosu (satır = seri, sütun = metrik)."""
    f, a = _check(forecasts, actuals)
    unknown = [m for m in metrics if m not in _TRAIN_METRICS and m not in _SIMPLE]
    if unknown:
        raise ValueError(f"Unknown metric(s): {unknown}. Valid: {sorted(_SIMPLE) + list(_TRAIN_METRICS)}")
    cols = {}
    for m in metrics:
        if m == "MASE":
            cols[m] = mase(f, a, train, seasonality, mase_digits)
        elif m == "RMSSE":
            cols[m] = rmsse(f, a, train)
        else:
            cols[m] = _SIMPLE[m](f, a)
    idx = pd.Index(series_names if series_names is not None else range(1, f.shape[0] + 1), name="series_name")
    return pd.DataFrame(cols, index=idx)


def summarize(per_series: pd.DataFrame) -> pd.DataFrame:
    """Metrik başına Mean/Median (yalnızca sonlu değerler) ile geçerli/dışarıda kalan seri sayıları."""
    rows = []
    for m in per_series.columns:
        v = per_series[m].to_numpy(dtype=np.float64)
        ok = np.isfinite(v)
        rows.append({
            "Metric": m,
            "Mean": float(v[ok].mean()) if ok.any() else np.nan,
            "Median": float(np.median(v[ok])) if ok.any() else np.nan,
            "N_valid": int(ok.sum()),
            "N_excluded": int((~ok).sum()),
        })
    return pd.DataFrame(rows)


def write_metrics_xlsx(
    path,
    per_series: pd.DataFrame,
    summary: pd.DataFrame,
    run_info: dict | None = None,
) -> None:
    """Orijinal R çıktısıyla aynı sayfa adları (Errors_Per_Series, Summary) + Run_Info."""
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        per_series.reset_index().to_excel(xw, sheet_name="Errors_Per_Series", index=False)
        summary.to_excel(xw, sheet_name="Summary", index=False)
        if run_info:
            pd.DataFrame({"key": list(run_info), "value": [str(v) for v in run_info.values()]}).to_excel(
                xw, sheet_name="Run_Info", index=False
            )
        for name in ("Errors_Per_Series", "Summary", "Run_Info"):
            if name in xw.sheets:
                xw.sheets[name].freeze_panes(1, 0)
                xw.sheets[name].set_column(0, 0, 28)
