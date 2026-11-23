"""Time-series features computed per series: catch22 (native library) and a set of basic descriptive features
(level, spread, intermittency with the Syntetos-Boylan class, Box-Cox lambda). Trend and seasonal strength come with
the tsfeatures set (stl_features).

catch22 is Rcatch22 v0.1.13's C code (GPL-3, cpp/src/catch22/), run inside the C++ library tsfcore over all series of
a dataset in one call. Two variants:

* ``study_fft=False`` (default): catch22 as published, with a correct FFT.
* ``study_fft=True``: reproduces the catch22 values stored with the ranking study (data/ranking_study/catch22). The
  Rcatch22 build that produced them computed every FFT twiddle factor as 1, which changes the five features built on
  FFT autocorrelations (CO_f1ecac, CO_FirstMin_ac, SB_TransitionMatrix_3ac_sumdiagcov, CO_Embed2_Dist_tau_d_expfit_
  meandiff, FC_LocalSimple_mean1_tauresrat) and the two Welch-spectrum features (SP_Summaries_*). With this switch the
  stored values are reproduced to ~1e-14 (tests/test_features.py).
"""
from __future__ import annotations

import ctypes

import numpy as np

from .engines import cpp_engine

CATCH22_NAMES: tuple[str, ...] = (
    "DN_HistogramMode_5", "DN_HistogramMode_10", "CO_f1ecac", "CO_FirstMin_ac", "CO_HistogramAMI_even_2_5",
    "CO_trev_1_num", "MD_hrv_classic_pnn40", "SB_BinaryStats_mean_longstretch1", "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01", "CO_Embed2_Dist_tau_d_expfit_meandiff", "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "FC_LocalSimple_mean1_tauresrat", "DN_OutlierInclude_p_001_mdrmd", "DN_OutlierInclude_n_001_mdrmd",
    "SP_Summaries_welch_rect_area_5_1", "SB_BinaryStats_diff_longstretch0", "SB_MotifThree_quantile_hh",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1", "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "SP_Summaries_welch_rect_centroid", "FC_LocalSimple_mean3_stderr",
)


class FeatureError(RuntimeError):
    pass


def _lib():
    lib = cpp_engine.load_lib()
    if lib is None:
        raise FeatureError(cpp_engine._LIB_ERR or "native library unavailable")
    if not getattr(lib, "_c22_ready", False):
        dp = ctypes.POINTER(ctypes.c_double)
        lib.tsf_catch22.restype = ctypes.c_int32
        lib.tsf_catch22.argtypes = [dp, ctypes.POINTER(ctypes.c_int64), ctypes.c_int32, ctypes.c_int32,
                                    ctypes.c_int32, dp]
        lib.tsf_catch22_count.restype = ctypes.c_int32
        lib.tsf_catch22_name.restype = ctypes.c_char_p
        lib.tsf_catch22_name.argtypes = [ctypes.c_int32]
        names = tuple(lib.tsf_catch22_name(i).decode() for i in range(lib.tsf_catch22_count()))
        if names != CATCH22_NAMES:
            raise FeatureError("the native library's catch22 feature list differs from this module's")
        lib._c22_ready = True
    return lib


def catch22(series: list[np.ndarray], study_fft: bool = False, threads: int = 0) -> np.ndarray:
    """catch22 features, one row per series (columns in CATCH22_NAMES order).

    Rows are NaN for series shorter than 10 points, constant series and series with missing values.
    """
    lib = _lib()
    arrs = [np.asarray(s, dtype=np.float64).ravel() for s in series]
    offsets = np.zeros(len(arrs) + 1, dtype=np.int64)
    np.cumsum([a.size for a in arrs], out=offsets[1:])
    values = np.ascontiguousarray(np.concatenate(arrs) if arrs else np.zeros(0))
    out = np.empty((len(arrs), len(CATCH22_NAMES)), dtype=np.float64)
    if arrs:
        lib.tsf_catch22(cpp_engine._ptr(values, ctypes.c_double), cpp_engine._ptr(offsets, ctypes.c_int64),
                        len(arrs), int(bool(study_fft)), int(threads), cpp_engine._ptr(out, ctypes.c_double))
    return out


# ---------------------------------------------------------------- basic features
# Syntetos, Boylan & Croston (2005) cut-offs for the demand classes.
ADI_CUT, CV2_CUT = 1.32, 0.49
DEMAND_CLASSES = ("smooth", "intermittent", "erratic", "lumpy")

BASIC_NAMES: tuple[str, ...] = (
    "length", "mean", "sd", "cv", "min", "max", "zero_share", "adi", "cv2", "demand_class", "boxcox_lambda",
)


def demand_class(adi: float, cv2: float) -> str | None:
    """Syntetos-Boylan class from ADI and CV^2 (None when either is undefined)."""
    if not (np.isfinite(adi) and np.isfinite(cv2)):
        return None
    return DEMAND_CLASSES[(adi >= ADI_CUT) + 2 * (cv2 >= CV2_CUT)]


def basic_features(y, m: float = 1) -> dict:
    """Descriptive features of one series (missing values dropped).

    length, mean, sd (n-1), cv = sd/|mean|, min, max, zero_share (share of exact zeros);
    adi = length / number of non-zero values (average demand interval), cv2 = (sd/mean)^2 of the non-zero values
    and demand_class, the Syntetos-Boylan class from them (ADI 1.32, CV^2 0.49);
    boxcox_lambda = forecast::BoxCox.lambda(y, "guerrero", -1, 2) for seasonal period m (lower bound 0 when y has
    values <= 0; 1 when the series is shorter than two periods), as R computes it.
    """
    from .engines.r_boot import box_cox_lambda

    x = np.asarray(y, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    nan = float("nan")
    out = dict.fromkeys(BASIC_NAMES, nan)
    out["length"] = n
    out["demand_class"] = None
    if n == 0:
        return out
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else nan
    nz = x[x != 0]
    out.update(mean=mu, sd=sd, cv=sd / abs(mu) if mu != 0 else nan, min=float(x.min()), max=float(x.max()),
               zero_share=float(np.mean(x == 0)))
    if nz.size:
        out["adi"] = n / nz.size
        nz_mu = float(np.mean(nz))
        out["cv2"] = (float(np.std(nz, ddof=1)) / nz_mu) ** 2 if nz.size > 1 and nz_mu != 0 else nan
    out["demand_class"] = demand_class(out["adi"], out["cv2"])
    try:
        lam = box_cox_lambda(x, -1.0, 2.0, float(m))
        out["boxcox_lambda"] = lam if np.isfinite(lam) else nan
    except (ValueError, ZeroDivisionError, OverflowError):
        pass
    return out


def compute(series: list[np.ndarray], m: float = 1, sets: tuple[str, ...] = ("basic", "catch22"),
            study_fft: bool = False, threads: int = 0, names: list[str] | None = None):
    """A DataFrame of features (one row per series, index = series names) for the requested feature sets."""
    import pandas as pd

    idx = pd.Index(names if names is not None else [f"T{i + 1}" for i in range(len(series))], name="series")
    parts = []
    for s in sets:
        if s == "basic":
            parts.append(pd.DataFrame([basic_features(y, m) for y in series], index=idx, columns=list(BASIC_NAMES)))
        elif s == "catch22":
            parts.append(pd.DataFrame(catch22(series, study_fft, threads), index=idx, columns=list(CATCH22_NAMES)))
        else:
            raise ValueError(f"unknown feature set: {s}")
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=idx)
