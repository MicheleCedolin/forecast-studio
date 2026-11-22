"""Frekans, mevsimsellik ve zaman damgası kuralları.

Değerler kullanıcının R kodundan (`SEASONALITY_VALS`, `FREQUENCIES`, `FREQ_MAP` in
experiments/fixed_horizon_functions.R ve utils/data_loader.R) birebir alınmıştır;
Monash TSForecasting depo çalışmasındaki tanımlarla aynıdır.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

# Sıra önemlidir (R'daki FREQUENCIES ile aynı): düşük frekanslılar -> yüksek frekanslılar
FREQUENCIES: tuple[str, ...] = (
    "4_seconds", "minutely", "10_minutes", "15_minutes", "half_hourly", "hourly",
    "daily", "weekly", "monthly", "quarterly", "yearly",
)
LOW_FREQUENCIES: tuple[str, ...] = FREQUENCIES[:6]   # zaman damgası saat içerir
HIGH_FREQUENCIES: tuple[str, ...] = FREQUENCIES[6:]  # yalnızca tarih

# R: SEASONALITY_VALS (frekans başına mevsimsellik; <günlük için çoklu mevsimsellik)
SEASONALITY_MAP: dict[str, tuple[float, ...]] = {
    "4_seconds": (21600.0, 151200.0, 7889400.0),
    "minutely": (1440.0, 10080.0, 525960.0),
    "10_minutes": (144.0, 1008.0, 52596.0),
    "15_minutes": (96.0, 672.0, 35064.0),
    "half_hourly": (48.0, 336.0, 17532.0),
    "hourly": (24.0, 168.0, 8766.0),
    "daily": (7.0,),
    "weekly": (365.25 / 7,),
    "monthly": (12.0,),
    "quarterly": (4.0,),
    "yearly": (1.0,),
}

# R: FREQ_MAP -> pandas karşılığı. Hafta = tam 7 gün (R'daki "1 week"), pandas "W" Pazar'a
# sabitlediği için kullanılmaz.
_STEP: dict[str, object] = {
    "4_seconds": pd.Timedelta(seconds=4),
    "minutely": pd.Timedelta(minutes=1),
    "10_minutes": pd.Timedelta(minutes=10),
    "15_minutes": pd.Timedelta(minutes=15),
    "half_hourly": pd.Timedelta(minutes=30),
    "hourly": pd.Timedelta(hours=1),
    "daily": pd.Timedelta(days=1),
    "weekly": pd.Timedelta(days=7),
    "monthly": pd.DateOffset(months=1),
    "quarterly": pd.DateOffset(months=3),
    "yearly": pd.DateOffset(years=1),
}


def seasonality_for(frequency: str | None) -> tuple[float, ...]:
    """Frekanstan mevsimsellik listesi; frekans yoksa R kodundaki gibi 1."""
    if frequency is None:
        return (1.0,)
    try:
        return SEASONALITY_MAP[frequency]
    except KeyError as exc:
        raise ValueError(f"Invalid frequency: {frequency!r}. Valid values: {', '.join(FREQUENCIES)}") from exc


def primary_period(seasonality: Sequence[float] | float) -> int:
    """MASE ölçeği için kullanılan gecikme: `min(seasonality)` tamsayıya kesilir (R `diff(lag=)`
    kesirli gecikmeyi `seq_len` ile kısaltır; ör. 365.25/7 = 52.18 -> 52). En az 1."""
    vals = [float(seasonality)] if np.ndim(seasonality) == 0 else [float(s) for s in seasonality]
    return max(1, int(min(vals)))


def timestamps(start: pd.Timestamp, n: int, frequency: str) -> pd.DatetimeIndex:
    """`start`'tan başlayan n zaman damgası (R: seq(start, length.out = n, by = FREQ_MAP[[f]]))."""
    step = _STEP[frequency]
    start = pd.Timestamp(start)
    if isinstance(step, pd.DateOffset):
        # Ay/çeyrek/yıl: başlangıca göre k adım (ay sonu taşmasında pandas gün sayısını kırpar)
        return pd.DatetimeIndex([start + k * step for k in range(n)])
    return pd.date_range(start=start, periods=n, freq=step)


def has_time_of_day(frequency: str) -> bool:
    return frequency in LOW_FREQUENCIES
