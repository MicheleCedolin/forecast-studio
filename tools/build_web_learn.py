"""Writes web/learn.json: the texts of the browser page's method list and Learn tab.

Usage:  python tools/build_web_learn.py

Everything comes from the interface's own registry (src/tsfstudio/app/models_info.py), so the browser page and the desktop
application describe a method, a measure and a reference in the same words. Only the local methods of the C++ engine are
listed, because those are what the browser build can run. Run this again after models_info.py changes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfstudio.app import models_info as MI  # noqa: E402
from tsfstudio.engines.cpp_engine import CppEngine  # noqa: E402
from tsfstudio.metrics import ALL_METRICS, DEFAULT_METRICS  # noqa: E402


# The series characteristics the page computes (tsfstudio.features: basic_features and catch22) and the map drawn from
# them. The basic texts follow basic_features' definitions; the catch22 texts are the short descriptions of the catch22
# documentation (feature overview table), matched to the names of the Rcatch22 0.1.13 code that tsfcore runs.
BASIC_TEXT = {
    "length": "Number of observations (missing values not counted).",
    "mean": "Mean of the series.",
    "sd": "Standard deviation of the series (divisor n − 1).",
    "cv": "Coefficient of variation: standard deviation divided by the absolute mean. Undefined when the mean is 0.",
    "min": "Smallest value.",
    "max": "Largest value.",
    "zero_share": "Share of observations that are exactly 0.",
    "adi": "Average demand interval: number of observations divided by the number of non-zero observations. 1 means demand in "
           "every period; larger values mean rarer demand.",
    "cv2": "Squared coefficient of variation of the non-zero values: how much the size of the demand varies when there is demand.",
    "boxcox_lambda": "Box-Cox transformation parameter chosen by Guerrero's method between −1 and 2 (as forecast::BoxCox.lambda in R): "
                     "near 1 the series needs no transformation, near 0 a logarithm stabilises its variance.",
}
CATCH22_TEXT = {
    "DN_HistogramMode_5": "5-bin histogram mode.",
    "DN_HistogramMode_10": "10-bin histogram mode.",
    "CO_f1ecac": "First 1/e crossing of the autocorrelation function.",
    "CO_FirstMin_ac": "First minimum of the autocorrelation function.",
    "CO_HistogramAMI_even_2_5": "Histogram-based automutual information (lag 2, 5 bins).",
    "CO_trev_1_num": "Time reversibility.",
    "MD_hrv_classic_pnn40": "Proportion of high incremental changes in the series.",
    "SB_BinaryStats_mean_longstretch1": "Longest stretch of above-mean values.",
    "SB_TransitionMatrix_3ac_sumdiagcov": "Transition matrix column variance (series coarse-grained to three symbols).",
    "PD_PeriodicityWang_th0_01": "Wang's periodicity metric.",
    "CO_Embed2_Dist_tau_d_expfit_meandiff": "Goodness of exponential fit to the embedding distance distribution.",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi": "First minimum of the automutual information function.",
    "FC_LocalSimple_mean1_tauresrat": "Change in autocorrelation timescale after incremental differencing.",
    "DN_OutlierInclude_p_001_mdrmd": "Positive outlier timing.",
    "DN_OutlierInclude_n_001_mdrmd": "Negative outlier timing.",
    "SP_Summaries_welch_rect_area_5_1": "Power in the lowest 20% of frequencies.",
    "SB_BinaryStats_diff_longstretch0": "Longest stretch of non-increasing values.",
    "SB_MotifThree_quantile_hh": "Entropy of successive pairs in the symbolized series.",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1": "Rescaled range fluctuation analysis (low-scale scaling).",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1": "Detrended fluctuation analysis (low-scale scaling).",
    "SP_Summaries_welch_rect_centroid": "Centroid frequency of the power spectrum (the frequency that splits its power in half).",
    "FC_LocalSimple_mean3_stderr": "Error of a 3-point rolling mean forecast.",
}
CLASS_TEXT = [
    ("smooth", "Demand in most periods, sizes that vary little (ADI below 1.32, CV² below 0.49)."),
    ("intermittent", "Demand in few periods, sizes that vary little (ADI 1.32 or more, CV² below 0.49)."),
    ("erratic", "Demand in most periods, sizes that vary a lot (ADI below 1.32, CV² 0.49 or more)."),
    ("lumpy", "Demand in few periods, sizes that vary a lot (ADI 1.32 or more, CV² 0.49 or more)."),
]
MAP_TEXT = [
    "The map is a principal component analysis (PCA) of the characteristics. Each characteristic is first standardised (mean 0, "
    "standard deviation 1), so that none dominates because of its unit; this is a PCA on the correlation matrix. A characteristic "
    "that is constant, or missing for more than half of the series, is left out; other missing values are replaced by the median.",
    "The first component (PC1) is the direction along which the series differ most, PC2 the next one that is unrelated to it, and "
    "so on. The percentage next to each axis is the share of the total variation it shows; when PC1 and PC2 together show a small "
    "share, the flat picture hides much of the difference between series.",
    "Each point is one series. Two points close together are series with similar characteristics; the distance to the centre says "
    "how unusual a series is. The arrows (loadings) show which characteristics pull the points in which direction: a series far "
    "along an arrow has a high value of that characteristic. The sign of a component has no meaning of its own.",
    "Colouring the map by the error of a run shows where on the map a method does well or badly, that is, for which kind of "
    "series. This is the idea of the instance space of Kang, Hyndman and Smith-Miles (2017).",
]


def main() -> int:
    used_refs: list[str] = []

    def refs(keys) -> list[str]:
        for k in keys:
            if k not in used_refs:
                used_refs.append(k)
        return list(keys)

    groups = [{"id": g, "name": name, "text": text} for g, name, text in MI.GROUPS]
    models = []
    for mid, m in MI.MODELS.items():
        if m.get("task") != "local" or mid not in CppEngine.LOCAL_METHODS:
            continue
        models.append({"id": mid, "name": m["name"], "group": m["group"], "summary": m["summary"], "how": m["how"],
                       "use": m["use"], "caution": m["caution"], "refs": refs(m.get("refs", []))})
    metrics = []
    for key in ALL_METRICS:
        info = MI.METRICS_INFO[key]
        metrics.append({"id": key, "name": info["name"], "unit": info["unit"], "text": info["text"], "caution": info["caution"],
                        "default": key in DEFAULT_METRICS})
    metric_refs = refs(MI.METRIC_REFS)
    from tsfstudio import features as F
    missing = [c for c in F.CATCH22_NAMES if c not in CATCH22_TEXT] + [c for c in F.BASIC_NAMES if c != "demand_class" and c not in BASIC_TEXT]
    if missing:
        raise SystemExit(f"No description for: {missing}")
    characteristics = {
        "basic": [{"id": c, "text": BASIC_TEXT[c]} for c in F.BASIC_NAMES if c != "demand_class"],
        "basic_refs": refs(["guerrero1993", "hyndman2008"]),
        "classes": [{"id": c, "text": t} for c, t in CLASS_TEXT],
        "class_refs": refs(["syntetos2005cat"]),
        "catch22": [{"id": c, "text": CATCH22_TEXT[c]} for c in F.CATCH22_NAMES],
        "catch22_refs": refs(["lubba2019"]),
        "map": MAP_TEXT,
        "map_refs": refs(["jolliffe2016", "kang2017"]),
    }
    refs(["godahewa2021"])
    references = {k: MI.REFS[k] for k in used_refs}
    out = {"groups": [g for g in groups if any(m["group"] == g["id"] for m in models)], "models": models, "metrics": metrics,
           "metric_refs": metric_refs, "characteristics": characteristics, "references": references}
    path = ROOT / "web" / "learn.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{path}: {len(models)} methods, {len(metrics)} measures, {len(references)} references "
          f"({sum(1 for r in references.values() if r['verified'])} checked against a bibliographic record)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
