"""Model sıralaması ve sıralamanın güvenilirliği (2026-11-22).

Girdi: satır = seri (ya da görev), sütun = model olan bir hata tablosu (küçük = iyi). Bir koşudaki `Errors_Per_Series`
sayfasındaki bir metrik sütunu, modeller yan yana konarak bu biçime getirilir (`errors_from_runs`).

  mean_ranks        seri başına sıra (eşitlikte ortalama sıra), ortalaması
  friedman          Friedman ki-kare testi (eksiksiz bloklar)
  nemenyi_cd        Nemenyi kritik farkı CD = q_α √(k(k+1)/(6N))  (Demšar 2006)
  avg_rel_mae       AvgRelMAE: göreli MAE'lerin (ağırlıklı) geometrik ortalaması (Davydenko & Fildes 2013)
  owa               M4 OWA: ½(sMAPE/sMAPE_Naive2 + MASE/MASE_Naive2), veri kümesi ortalamaları üzerinden (Makridakis vd. 2020)
  rank_stability    serileri yeniden örnekleyen bootstrap: her modelin 1. olma ve ilk-k'da olma olasılığı, tam sıralamayla
                    Kendall τ dağılımı

Not: SMAPE bu pakette oran (0–2) olarak tutulur; OWA'da pay ve payda aynı birimde olduğu için sonuç etkilenmez.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


def _complete(errors: pd.DataFrame) -> pd.DataFrame:
    """Tüm modellerde sonlu değeri olan satırlar (eksiksiz bloklar). Hiç kalmazsa hata."""
    e = errors.astype(float)
    ok = np.isfinite(e.to_numpy()).all(axis=1)
    out = e.loc[ok]
    if out.empty:
        raise ValueError("No series has a finite error for every model (no complete blocks).")
    return out


def mean_ranks(errors: pd.DataFrame) -> pd.Series:
    """Ortalama sıra (1 = en iyi), yalnızca eksiksiz bloklarda. Artan sırada döner."""
    e = _complete(errors)
    r = e.rank(axis=1, method="average")
    return r.mean(axis=0).sort_values()


@dataclass
class FriedmanResult:
    statistic: float
    p_value: float
    n_blocks: int
    n_models: int
    n_dropped: int


def friedman(errors: pd.DataFrame) -> FriedmanResult:
    e = _complete(errors)
    if e.shape[1] < 3:
        raise ValueError("The Friedman test needs at least three models.")
    stat, p = stats.friedmanchisquare(*[e[c].to_numpy() for c in e.columns])
    return FriedmanResult(float(stat), float(p), int(e.shape[0]), int(e.shape[1]), int(len(errors) - len(e)))


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Nemenyi kritik farkı. q_α = studentize aralık dağılımının (k, ∞) üst α noktası / √2 (Demšar 2006, Tablo 5)."""
    if k < 2 or n < 1:
        raise ValueError("k >= 2 and n >= 1 are required")
    q = float(stats.studentized_range.ppf(1.0 - alpha, k, np.inf)) / np.sqrt(2.0)
    return q * np.sqrt(k * (k + 1) / (6.0 * n))


def avg_rel_mae(mae: pd.DataFrame, benchmark: str, weights: Sequence[float] | None = None) -> pd.Series:
    """AvgRelMAE_j = exp(Σ_i w_i ln(MAE_ij / MAE_i,bench) / Σ_i w_i). w_i = serinin ufuk uzunluğu (verilmezse eşit).

    Kıyas MAE'si 0 olan ya da herhangi bir modelde sonlu olmayan seri dışarıda kalır. < 1: kıyastan iyi."""
    if benchmark not in mae.columns:
        raise ValueError(f"benchmark '{benchmark}' is not a column")
    m = mae.astype(float)
    w = np.ones(len(m)) if weights is None else np.asarray(weights, dtype=float)
    b = m[benchmark].to_numpy()
    ok = np.isfinite(m.to_numpy()).all(axis=1) & (b > 0) & (m.to_numpy() > 0).all(axis=1)
    if not ok.any():
        raise ValueError("No usable series for AvgRelMAE.")
    lr = np.log(m.to_numpy()[ok] / b[ok, None])
    return pd.Series(np.exp((w[ok, None] * lr).sum(axis=0) / w[ok].sum()), index=m.columns).sort_values()


def owa(smape: pd.Series | dict, mase: pd.Series | dict, naive2: str = "naive2") -> pd.Series:
    """M4 OWA. `smape`, `mase`: model -> veri kümesi ortalaması. Naive2 = 1."""
    s, q = pd.Series(smape, dtype=float), pd.Series(mase, dtype=float)
    if naive2 not in s or naive2 not in q:
        raise ValueError(f"'{naive2}' must be present in both smape and mase")
    return (0.5 * (s / s[naive2] + q / q[naive2])).sort_values()


@dataclass
class StabilityResult:
    p_first: pd.Series          # P(model 1. sırada)
    p_top_k: pd.Series          # P(model ilk k'da)
    mean_rank_ci: pd.DataFrame  # ortalama sıranın %2,5 / %50 / %97,5 yüzdelikleri
    kendall_tau: np.ndarray     # her bootstrap sıralamasıyla tam sıralama arasındaki τ
    k: int
    n_boot: int


def rank_stability(errors: pd.DataFrame, n_boot: int = 1000, top_k: int = 3, seed: int = 1) -> StabilityResult:
    """Serileri (blokları) yerine koyarak yeniden örnekler; her örnekte ortalama sıraya göre modeller sıralanır."""
    e = _complete(errors)
    r = e.rank(axis=1, method="average").to_numpy()
    models = list(e.columns)
    n, k = r.shape
    full = r.mean(axis=0)
    rng = np.random.default_rng(seed)
    firsts = np.zeros(k)
    topk = np.zeros(k)
    means = np.empty((n_boot, k))
    taus = np.empty(n_boot)
    kk = min(top_k, k)
    for b in range(n_boot):
        mr = r[rng.integers(0, n, n)].mean(axis=0)
        means[b] = mr
        # Ties share the credit instead of going to whichever model comes first in the column order
        # (e.g. naive and naive2 are identical when Naive2 finds no seasonality).
        best = np.isclose(mr, mr.min(), rtol=0, atol=1e-12)
        firsts[best] += 1.0 / best.sum()
        cut = np.sort(mr)[kk - 1]
        above = mr < cut - 1e-12
        at = np.isclose(mr, cut, rtol=0, atol=1e-12)
        topk[above] += 1.0
        topk[at] += (kk - above.sum()) / at.sum()
        taus[b] = stats.kendalltau(full, mr).statistic
    ci = pd.DataFrame(np.percentile(means, [2.5, 50, 97.5], axis=0).T, index=models, columns=["p2.5", "p50", "p97.5"])
    return StabilityResult(pd.Series(firsts / n_boot, index=models).sort_values(ascending=False),
                           pd.Series(topk / n_boot, index=models).sort_values(ascending=False),
                           ci.sort_values("p50"), taus, top_k, n_boot)


def errors_from_runs(run_dirs: dict[str, str], metric: str) -> pd.DataFrame:
    """Koşu klasörlerinden (`metrics.xlsx` -> Errors_Per_Series) seri x model tablosu. Anahtar: sütunda görünecek model adı."""
    from pathlib import Path

    cols = {}
    for name, d in run_dirs.items():
        df = pd.read_excel(Path(d) / "metrics.xlsx", sheet_name="Errors_Per_Series").set_index("series_name")
        if metric not in df.columns:
            raise ValueError(f"{d}: metric '{metric}' not found (available: {list(df.columns)})")
        cols[name] = df[metric]
    return pd.DataFrame(cols)
