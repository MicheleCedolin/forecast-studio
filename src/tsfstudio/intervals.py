"""Prediction intervals for any engine and method (split conformal, scaled), and their accuracy measures.

Intervals: the model is also fitted one horizon earlier (the calibration origin). Its errors on that window are scaled
per series by the in-sample seasonal-naive MAE (the MASE scale) and pooled across series per forecast step h; the
empirical quantiles q_p(h) of the scaled signed errors give, around each final forecast,
    bound_p = forecast + q_p(h) · scale_i,     p = (1 - level) / 2 and (1 + level) / 2.
This needs no model-specific variance formula, so local and global models of all three engines get intervals on the
same footing (scaled split-conformal prediction; Vovk et al. 2005, Stankeviciute et al. 2021).

Measures, per series:
  Coverage<level>  share of test points inside the interval (ideal: level / 100);
  MSIS             mean scaled interval score of the 95% interval (or the widest level), as in the M4 competition:
                   mean_h[(U - L) + 2/α (L - y) 1{y < L} + 2/α (y - U) 1{y > U}] / scale;
  WQL              weighted quantile loss over the interval quantiles and the median (the point forecast):
                   mean_q[2 · Σ_h pinball_q / Σ_h |y|].
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .metrics import _mase_scale
from .seasonality import primary_period

VALID_LEVELS = range(50, 100)


def probs_for(levels: Sequence[int]) -> list[float]:
    """The quantile probabilities of the interval bounds, ascending."""
    ps = set()
    for lv in levels:
        a = 1 - lv / 100
        ps.update((round(a / 2, 6), round(1 - a / 2, 6)))
    return sorted(ps)


def series_scales(train: Sequence[np.ndarray], seasonality) -> np.ndarray:
    """In-sample MAE of the seasonal naive (lag 1 when undefined), as MASE uses it; NaN when zero or undefined."""
    lag = primary_period(seasonality)
    out = np.full(len(train), np.nan)
    for i, y in enumerate(train):
        y = np.asarray(y, dtype=np.float64)
        y = y[~np.isnan(y)]
        s = _mase_scale(y, lag)
        if not (np.isfinite(s) and s > 0):
            s = _mase_scale(y, 1)
        if np.isfinite(s) and s > 0:
            out[i] = s
    return out


def conformal_quantiles(cal_forecasts: np.ndarray, cal_actuals: np.ndarray, cal_scales: np.ndarray,
                        probs: Sequence[float]) -> np.ndarray:
    """q[p, h]: quantiles (type 7) of the scaled signed calibration errors (actual - forecast) / scale per step h."""
    e = (np.asarray(cal_actuals, float) - np.asarray(cal_forecasts, float)) / cal_scales[:, None]
    q = np.full((len(probs), e.shape[1]), np.nan)
    for h in range(e.shape[1]):
        col = e[:, h]
        col = col[np.isfinite(col)]
        if col.size:
            q[:, h] = np.quantile(col, probs)
    return q


def bounds(forecasts: np.ndarray, scales: np.ndarray, q: np.ndarray, probs: Sequence[float]) -> dict[float, np.ndarray]:
    """{p: forecast + q[p, h] · scale_i} for every probability p."""
    f = np.asarray(forecasts, float)
    s = np.where(np.isfinite(scales), scales, np.nan)[:, None]
    return {p: f + q[k][None, :] * s for k, p in enumerate(probs)}


def interval_metrics(forecasts: np.ndarray, actuals: np.ndarray, scales: np.ndarray, levels: Sequence[int],
                     bnds: dict[float, np.ndarray], series_names=None) -> pd.DataFrame:
    y = np.asarray(actuals, float)
    f = np.asarray(forecasts, float)
    cols: dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for lv in sorted(levels):
            a = 1 - lv / 100
            lo, hi = bnds[round(a / 2, 6)], bnds[round(1 - a / 2, 6)]
            ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
            inside = ((y >= lo) & (y <= hi) & ok).sum(axis=1)
            n = ok.sum(axis=1)
            cols[f"Coverage{lv}"] = np.where(n > 0, inside / np.maximum(n, 1), np.nan)
        lv = 95 if 95 in levels else max(levels)
        a = 1 - lv / 100
        lo, hi = bnds[round(a / 2, 6)], bnds[round(1 - a / 2, 6)]
        score = (hi - lo) + 2 / a * (lo - y) * (y < lo) + 2 / a * (y - hi) * (y > hi)
        ok = np.isfinite(score)
        msis = np.where(ok.any(axis=1), np.nansum(np.where(ok, score, 0), axis=1) / np.maximum(ok.sum(axis=1), 1), np.nan)
        cols["MSIS"] = msis / scales
        qs = {0.5: f, **bnds}
        denom = np.nansum(np.abs(y), axis=1)
        losses = []
        for p, b in qs.items():
            d = y - b
            pin = np.where(np.isfinite(d), np.maximum(p * d, (p - 1) * d), np.nan)
            losses.append(2 * np.nansum(pin, axis=1) / denom)
        wql = np.mean(losses, axis=0)
        cols["WQL"] = np.where(denom > 0, wql, np.nan)
    df = pd.DataFrame(cols, index=pd.Index(series_names, name="series_name") if series_names is not None else None)
    return df.replace([np.inf, -np.inf], np.nan)
