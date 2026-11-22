"""statsforecast tabanlı yerel modeller: ets, arima, tbats, dhr_arima (R `forecast` paketinin Python karşılıkları).

DİKKAT — bu modüldeki adaptörler statsforecast API'sine göre yazılmıştır ve geliştirme ortamında (statsforecast
kurulu değil) ÇALIŞTIRILAMADI. Doğrulama kullanıcının makinesinde `python -m tsfstudio doctor` ve
`python -m tsfstudio golden` ile yapılır. Hata durumunda seri başına R'daki gibi SNaive'e düşülür ve sayılır.

R davranışını yakalamak için bilinçli eşlemeler (kaynak: R `forecast` belgeleri):
  * ets   : R `ets()` frekans > 24 ise mevsimselliği yok sayar (uyarı verir) -> season_length = 1.
  * arima : R `auto.arima(lambda = 0)` (log dönüşümü, y > 0 gerekir) -> olmadı `auto.arima()` -> olmadı mevsimsiz.
            R varsayılanları: allowdrift = TRUE, approximation = (n > 150 veya frekans > 12), stepwise = TRUE.
  * dhr_arima: Fourier terimleri (K = 1; orijinal kod) dış regresör, mevsimsiz auto.arima.
statsforecast'ın sürümler arasında değişebilen anahtar kelimeleri `make_model` ile imzaya göre süzülür; atılan
anahtarlar `DROPPED` içinde toplanır (doctor bunları raporlar) — sessizce yok sayılmaz.
"""
from __future__ import annotations

import inspect
import warnings
from typing import Sequence

import numpy as np

from .py_models import fourier_terms, snaive  # noqa: F401  (snaive: yedek)

DROPPED: set[str] = set()
ARIMA_MAX_SEASON_DEFAULT = 200      # bunun üstünde mevsimli ARIMA pratik değil; mevsimsize düşülür (R'da da sınırlar var)


def sf_available() -> bool:
    try:
        import statsforecast  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def sf_version() -> str | None:
    try:
        import statsforecast
        return getattr(statsforecast, "__version__", "?")
    except Exception:  # noqa: BLE001
        return None


def has_model(name: str) -> bool:
    try:
        import statsforecast.models as M
        return hasattr(M, name)
    except Exception:  # noqa: BLE001
        return False


def make_model(cls, _strict: bool = False, **kw):
    """Yalnızca sınıfın kabul ettiği anahtar kelimelerle örnek üretir; atılanlar DROPPED'a yazılır.

    `_strict=True` (kullanıcı parametresi taşıyan yeni modeller): tanınmayan anahtar SESSİZCE atılmaz, hata verilir —
    aksi hâlde kullanıcının seçtiği değer yok sayılıp kütüphane varsayılanıyla çalışılırdı."""
    try:
        params = set(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):
        params = set(kw)
    ok = {k: v for k, v in kw.items() if k in params}
    dropped = [k for k in kw if k not in params]
    for k in dropped:
        DROPPED.add(f"{cls.__name__}.{k}")
    if _strict and dropped:
        raise TypeError(f"{cls.__name__}: this statsforecast version does not recognise the key(s) {', '.join(dropped)}")
    try:
        return cls(**ok)
    except TypeError as exc:
        if dropped:
            raise TypeError(f"{cls.__name__}: this statsforecast version does not recognise the key(s) {', '.join(dropped)}; {exc}") from exc
        raise


def _mean(out) -> np.ndarray:
    m = out["mean"] if isinstance(out, dict) else out
    return np.asarray(m, dtype=np.float64).reshape(-1)


# ------------------------------------------------------------------------------------------------
class AdapterNote(UserWarning):
    """Adaptörün kullanıcıya bildirmesi gereken sessiz uyarlama (ör. mevsimselliğin atılması). Motor bunları yakalayıp
    `EngineResult.notes` içinde sayar; ekrana basılmaz."""


def note(msg: str) -> None:
    warnings.warn(AdapterNote(msg), stacklevel=2)


def forecast_ets(y: np.ndarray, h: int, m: int) -> np.ndarray:
    from statsforecast.models import AutoETS
    sl = m if m <= 24 else 1
    if sl != m:
        note(f"ets: seasonality was not used because the period {m} > 24 (same rule as R ets()).")
    return _mean(make_model(AutoETS, season_length=sl, model="ZZZ").forecast(y=y, h=h))


def _arima(y: np.ndarray, h: int, season_length: int, n_for_approx: int, freq: int) -> np.ndarray:
    from statsforecast.models import AutoARIMA
    mdl = make_model(AutoARIMA, season_length=season_length, allowdrift=True, stepwise=True,
                     approximation=bool(n_for_approx > 150 or freq > 12))
    return _mean(mdl.forecast(y=y, h=h))


def forecast_arima(y: np.ndarray, h: int, m: int, max_season: int = ARIMA_MAX_SEASON_DEFAULT) -> np.ndarray:
    """R zinciri: log (lambda=0) -> ham -> mevsimsiz. Her adım başarısız olursa bir sonrakine geçilir."""
    sl = m if m <= max_season else 1
    errors: list[str] = []
    if float(np.min(y)) > 0:
        try:
            return np.exp(_arima(np.log(y), h, sl, len(y), m))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"log: {exc}")
    try:
        return _arima(y, h, sl, len(y), m)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"raw: {exc}")
    try:
        return _arima(y, h, 1, len(y), m)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"non-seasonal: {exc}")
        raise RuntimeError("; ".join(errors)) from exc


def forecast_dhr_arima(y: np.ndarray, h: int, periods: Sequence[float], K=1) -> np.ndarray:
    """Dinamik harmonik regresyon: Fourier terimleri dış regresör, mevsimsiz auto.arima.

    Orijinal R çağrısı skaler K = 1 kullanır. Çok mevsimli `msts`'te R'ın `fourier()` işlevi sürüme göre hata verip
    seriyi sessizce SNaive'e düşürebilir; burada skaler K her periyoda uygulanır (hata yerine terim üretilir).
    Bu bilinçli bir farktır (docs/CHANGES_R.md).
    """
    from statsforecast.models import AutoARIMA
    n = len(y)
    X = fourier_terms(n, periods, K)
    Xf = fourier_terms(n, periods, K, h=h)
    mdl = make_model(AutoARIMA, season_length=1, allowdrift=True, stepwise=True, approximation=bool(n > 150))
    return _mean(mdl.forecast(y=y, h=h, X=X, X_future=Xf))


def forecast_tbats(y: np.ndarray, h: int, periods: Sequence[float]) -> np.ndarray:
    """statsforecast `AutoTBATS(season_length: int | list[int], use_boxcox=None, use_trend=None, use_damped_trend=None,
    use_arma_errors=True)`. Varsayılanlar R `tbats()` ile aynı (Box-Cox/trend/sönümlü trend denenir, ARMA hataları açık,
    AIC ile seçim). Belgeye göre dönemler TAM SAYIDIR; R'ın gerçek değerli dönemleri (ör. 365.25) en yakın tam sayıya
    yuvarlanır — bilinçli bir fark (docs/CHANGES_R.md); altın karşılaştırma etkisini ölçer.
    (Not: ilk sürümde parametre yanlışlıkla `seasonal_periods` diye gönderilip süzüldü ve her seri yedeğe düştü.)"""
    from statsforecast.models import AutoTBATS
    sl = [max(int(round(float(p))), 1) for p in periods]
    mdl = make_model(AutoTBATS, season_length=sl if len(sl) > 1 else sl[0])
    return _mean(mdl.forecast(y=y, h=h))


# ------------------------------------------------------------------------------------------------
# Aşama C3: ek hazır statsforecast modelleri (kullanıcı kararı 2026-11-04: "A + B").
#   A) aralıklı talep : croston_classic, croston_sba, croston_optimized, tsb, adida, imapa
#   B) ek klasik      : auto_theta, auto_ces, holt_winters, mstl, auto_mfles
# Bunlar R modellerinin çevirisi DEĞİLDİR; statsforecast'ın kendi modelleridir. R karşılığı yoktur, `golden` ile
# sınanamaz; doğrulama `doctor` (çalışır mı, sonlu mu, yedek var mı) ve kullanıcının kendi verisindeki sonuçlarıdır.
# İmzalar resmi belgeden doğrulandı: CrostonClassic/CrostonOptimized/ADIDA(alias, prediction_intervals),
# AutoTheta(season_length, decomposition_type, model), AutoCES(season_length, model), HoltWinters(season_length, error_type),
# AutoMFLES(test_size [ZORUNLU], season_length, n_windows, config, step_size, metric). CrostonSBA, TSB(alpha_d, alpha_p),
# IMAPA ve MSTL(season_length, trend_forecaster) imzaları belge sayfasında bulunamadı (başka kaynaklardan teyit edildi);
# imza uyuşmazlığında `make_model` anahtarı sessizce atmaz, "tanımıyor" hatası verir ve seri yedeğe düşer (nedeni yazılır).
# ------------------------------------------------------------------------------------------------
SPECS: dict[str, dict] = {
    "croston_classic":   {"cls": "CrostonClassic", "group": "intermittent", "params": {}},
    "croston_sba":       {"cls": "CrostonSBA", "group": "intermittent", "params": {}},
    "croston_optimized": {"cls": "CrostonOptimized", "group": "intermittent", "params": {}},
    "tsb":               {"cls": "TSB", "group": "intermittent", "params": {"alpha_d": 0.1, "alpha_p": 0.1}},
    "adida":             {"cls": "ADIDA", "group": "intermittent", "params": {}},
    "imapa":             {"cls": "IMAPA", "group": "intermittent", "params": {}},
    "auto_theta":        {"cls": "AutoTheta", "group": "classical", "params": {"decomposition_type": "multiplicative"}},
    "auto_ces":          {"cls": "AutoCES", "group": "classical", "params": {"model": "Z", "max_season": 24}},
    "holt_winters":      {"cls": "HoltWinters", "group": "classical", "params": {"error_type": "A", "max_season": 24}},
    "mstl":              {"cls": "MSTL", "group": "classical", "params": {}},
    "auto_mfles":        {"cls": "AutoMFLES", "group": "classical", "params": {"n_windows": 2, "metric": "smape"},
                          "requires": ("sklearn",), "pip": "scikit-learn"},
}
INTERMITTENT = tuple(k for k, v in SPECS.items() if v["group"] == "intermittent")


def missing_requirements(method: str) -> list[str]:
    """Yöntemin gerektirdiği ama kurulu olmayan Python paketleri (pip adlarıyla)."""
    import importlib.util
    spec = SPECS.get(method, {})
    return [spec.get("pip", mod) for mod in spec.get("requires", ()) if importlib.util.find_spec(mod) is None]


def usable_season(n: int, m: int, max_season: int | None = None, name: str = "") -> int:
    """Tek dönemli modeller için kullanılacak mevsim uzunluğu; kullanılamıyorsa 1 (ve neden bildirilir)."""
    m = int(m)
    if m <= 1:
        return 1
    if max_season is not None and m > int(max_season):
        note(f"{name}: seasonality was not used because the period {m} > max_season ({int(max_season)}).")
        return 1
    if n < 2 * m:
        note(f"{name}: period {m} needs at least 2 periods (={2 * m} observations); seasonality was not used on shorter series.")
        return 1
    return m


def valid_periods(n: int, periods: Sequence[float]) -> list[int]:
    """Çok mevsimli modeller için geçerli tam sayı dönemler: >= 2 ve en az iki dönem uzunluğunda veri."""
    out = sorted({int(round(float(p))) for p in periods})
    return [p for p in out if p >= 2 and 2 * p <= n]


def _alpha(P: dict, key: str) -> float:
    v = float(P[key])
    if not 0.0 < v <= 1.0:
        raise ValueError(f"{key} must be in (0, 1]; given: {v}")
    return v


def forecast_spec(method: str, y: np.ndarray, h: int, m: int, periods: Sequence[float], params: dict | None) -> np.ndarray:
    """SPECS içindeki bir yöntem için tahmin. `params` yalnızca yöntemin kendi anahtarlarını kullanır (başkaları yok sayılır)."""
    import statsforecast.models as SM

    spec = SPECS[method]
    P = dict(spec["params"])
    P.update({k: v for k, v in (params or {}).items() if k in spec["params"]})
    n = len(y)
    kw: dict = {}

    if method in INTERMITTENT:
        if not np.any(y):                       # hep sıfır: matematiksel olarak tahmin 0; kütüphane 0/0 üretebilir
            return np.zeros(h)
        if method == "tsb":
            kw = {"alpha_d": _alpha(P, "alpha_d"), "alpha_p": _alpha(P, "alpha_p")}
    elif method == "auto_theta":
        dec = str(P["decomposition_type"])
        if dec not in ("multiplicative", "additive"):
            raise ValueError("decomposition_type must be 'multiplicative' or 'additive'")
        if dec == "multiplicative" and float(np.min(y)) <= 0:
            note("auto_theta: for series with zero/negative values, additive decomposition was used instead of multiplicative.")
            dec = "additive"
        kw = {"season_length": usable_season(n, m, None, "auto_theta"), "decomposition_type": dec}
    elif method == "auto_ces":
        kw = {"season_length": usable_season(n, m, P["max_season"], "auto_ces"), "model": str(P["model"])}
    elif method == "holt_winters":
        et = str(P["error_type"]).upper()
        if et not in ("A", "M"):
            raise ValueError("error_type must be 'A' or 'M'")
        if et == "M" and float(np.min(y)) <= 0:
            raise ValueError("multiplicative error (error_type='M') requires all values > 0")
        kw = {"season_length": usable_season(n, m, P["max_season"], "holt_winters"), "error_type": et}
    elif method == "mstl":
        per = valid_periods(n, periods)
        if not per:
            raise ValueError("There is no valid seasonal period for MSTL (a period >= 2 and at least two periods of data are required)")
        kw = {"season_length": per if len(per) > 1 else per[0]}
    elif method == "auto_mfles":
        per = valid_periods(n, periods)
        kw = {"test_size": int(h), "season_length": (per if len(per) > 1 else per[0]) if per else None,
              "n_windows": int(P["n_windows"]), "metric": str(P["metric"])}

    cls = getattr(SM, spec["cls"])
    return _mean(make_model(cls, _strict=True, **kw).forecast(y=y, h=h))
