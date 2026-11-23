"""The interface's model information registry: each method's explanation, parameter schema (with defaults), engine notes and references.

Single-source principle: parameter KEYS and DEFAULTS are identical to the values the engines actually use
(`engines/py_sf.py: SPECS`, `py_engine.forecast_one`, `r/worker.R`). `tests/test_app_models_info.py` checks this; if an engine
parameter changes the test breaks, so the interface never silently shows a wrong default.

References: `verified=True` only for citations matched in this session against a bibliographic record/page (RePEc, publisher, arXiv, Lancaster eprints);
the others are known citations and MUST BE VERIFIED before use in a paper (they are also flagged in the interface).
Timing information: only times measured on the user's own machine and times from their own study are written; none are invented.
"""
from __future__ import annotations

GROUPS = (
    ("yerel", "Local statistical", "A separate model is fitted to each series."),
    ("aralikli", "Intermittent demand", "Methods for series with many zeros (spare parts, retail)."),
    ("kuresel", "Global", "A single model is learned from all series (through lagged values)."),
)


def _ref(text: str, url: str | None = None, verified: bool = False) -> dict:
    return {"text": text, "url": url, "verified": verified}


REFS = {
    "fpp3": _ref("Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice (3rd ed.). OTexts.", "https://otexts.com/fpp3/"),
    "hyndman2002": _ref("Hyndman, R. J., Koehler, A. B., Snyder, R. D., & Grose, S. (2002). A state space framework for automatic forecasting "
                        "using exponential smoothing methods. International Journal of Forecasting, 18(3), 439–454.",
                        "https://econpapers.repec.org/RePEc:eee:intfor:v:18:y:2002:i:3:p:439-454", True),
    "hyndman2008": _ref("Hyndman, R. J., & Khandakar, Y. (2008). Automatic time series forecasting: the forecast package for R. "
                        "Journal of Statistical Software, 27(3), 1–22.", "https://www.jstatsoft.org/v27/i03", True),
    "delivera2011": _ref("De Livera, A. M., Hyndman, R. J., & Snyder, R. D. (2011). Forecasting time series with complex seasonal patterns "
                         "using exponential smoothing. Journal of the American Statistical Association, 106(496), 1513–1527."),
    "assimakopoulos2000": _ref("Assimakopoulos, V., & Nikolopoulos, K. (2000). The theta model: a decomposition approach to forecasting. "
                               "International Journal of Forecasting, 16(4), 521–530."),
    "hyndmanbillah2003": _ref("Hyndman, R. J., & Billah, B. (2003). Unmasking the Theta method. International Journal of Forecasting, 19(2), 287–290."),
    "croston1972": _ref("Croston, J. D. (1972). Forecasting and stock control for intermittent demands. Operational Research Quarterly, 23(3), 289–303."),
    "syntetos2005": _ref("Syntetos, A. A., & Boylan, J. E. (2005). The accuracy of intermittent demand estimates. "
                         "International Journal of Forecasting, 21(2), 303–314."),
    "teunter2011": _ref("Teunter, R. H., Syntetos, A. A., & Babai, M. Z. (2011). Intermittent demand: linking forecasting to inventory obsolescence. "
                        "European Journal of Operational Research, 214(3), 606–615."),
    "nikolopoulos2011": _ref("Nikolopoulos, K., Syntetos, A. A., Boylan, J. E., Petropoulos, F., & Assimakopoulos, V. (2011). An aggregate–disaggregate "
                             "intermittent demand approach (ADIDA) to forecasting. Journal of the Operational Research Society, 62(3), 544–554."),
    "petropoulos2015": _ref("Petropoulos, F., & Kourentzes, N. (2015). Forecast combinations for intermittent demand. "
                            "Journal of the Operational Research Society, 66(6), 914–924."),
    "svetunkov2022": _ref("Svetunkov, I., Kourentzes, N., & Ord, J. K. (2022). Complex exponential smoothing. Naval Research Logistics, 69(8), 1108–1123."),
    "winters1960": _ref("Winters, P. R. (1960). Forecasting sales by exponentially weighted moving averages. Management Science, 6(3), 324–342."),
    "bandara2021": _ref("Bandara, K., Hyndman, R. J., & Bergmeir, C. (2021). MSTL: a seasonal-trend decomposition algorithm for time series with "
                        "multiple seasonal patterns. arXiv:2107.13462.", "https://arxiv.org/abs/2107.13462"),
    "cleveland1990": _ref("Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: a seasonal-trend decomposition procedure "
                          "based on loess. Journal of Official Statistics, 6(1), 3–73."),
    "prokhorenkova2018": _ref("Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: unbiased boosting with "
                              "categorical features. Advances in Neural Information Processing Systems 31.", "https://arxiv.org/abs/1706.09516"),
    "athanasopoulos2017": _ref("Athanasopoulos, G., Hyndman, R. J., Kourentzes, N., & Petropoulos, F. (2017). Forecasting with temporal "
                               "hierarchies. European Journal of Operational Research, 262(1), 60–74.",
                               "https://doi.org/10.1016/j.ejor.2017.02.046"),
    "kourentzes2014": _ref("Kourentzes, N., Petropoulos, F., & Trapero, J. R. (2014). Improving forecasting by estimating time series "
                           "structural components across multiple frequencies. International Journal of Forecasting, 30(2), 291–302.",
                           "https://doi.org/10.1016/j.ijforecast.2013.09.006"),
    "bergmeir2016": _ref("Bergmeir, C., Hyndman, R. J., & Benítez, J. M. (2016). Bagging exponential smoothing methods using STL "
                         "decomposition and Box–Cox transformation. International Journal of Forecasting, 32(2), 303–312.",
                         "https://doi.org/10.1016/j.ijforecast.2015.07.002"),
    "ke2017": _ref("Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: a highly efficient "
                   "gradient boosting decision tree. Advances in Neural Information Processing Systems 30.",
                   "https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree"),
    "taieb2012": _ref("Ben Taieb, S., Bontempi, G., Atiya, A. F., & Sorjamaa, A. (2012). A review and comparison of strategies for multi-step "
                      "ahead time series forecasting based on the NN5 forecasting competition. Expert Systems with Applications, 39(8), "
                      "7067–7083.", "https://doi.org/10.1016/j.eswa.2012.01.039"),
    "makridakis2000": _ref("Makridakis, S., & Hibon, M. (2000). The M3-Competition: results, conclusions and implications. International "
                           "Journal of Forecasting, 16(4), 451–476.", "https://doi.org/10.1016/S0169-2070(00)00057-1"),
    "oreshkin2020": _ref("Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2020). N-BEATS: Neural basis expansion analysis for "
                         "interpretable time series forecasting. International Conference on Learning Representations.",
                         "https://openreview.net/forum?id=r1ecqn4YwB"),
    "challu2023": _ref("Challu, C., Olivares, K. G., Oreshkin, B. N., Garza Ramirez, F., Mergenthaler Canseco, M., & Dubrawski, A. "
                       "(2023). NHITS: Neural hierarchical interpolation for time series forecasting. Proceedings of the AAAI "
                       "Conference on Artificial Intelligence, 37(6), 6989–6997.", "https://doi.org/10.1609/aaai.v37i6.25854"),
    "ansari2024": _ref("Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., et al. (2024). Chronos: Learning the "
                       "language of time series. Transactions on Machine Learning Research.", "https://openreview.net/forum?id=gerNCVqqtR"),
    "montero2021": _ref("Montero-Manso, P., & Hyndman, R. J. (2021). Principles and algorithms for forecasting groups of time series: locality "
                        "and globality. International Journal of Forecasting, 37(4), 1632–1653.",
                        "https://econpapers.repec.org/RePEc:eee:intfor:v:37:y:2021:i:4:p:1632-1653", True),
    "godahewa2021": _ref("Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J., & Montero-Manso, P. (2021). Monash Time Series Forecasting "
                         "Archive. NeurIPS Datasets and Benchmarks Track.", "https://arxiv.org/abs/2105.06643", True),
    "syntetos2005cat": _ref("Syntetos, A. A., Boylan, J. E., & Croston, J. D. (2005). On the categorization of demand patterns. "
                            "Journal of the Operational Research Society, 56(5), 495–503.", "https://eprints.lancs.ac.uk/id/eprint/73095", True),
    "hyndman2006": _ref("Hyndman, R. J. (2006). Another look at forecast-accuracy metrics for intermittent demand. Foresight, 4(4), 43–46.",
                        "https://econpapers.repec.org/RePEc:for:ijafaa:y:2006:i:4:p:43-46", True),
    "hyndmankoehler2006": _ref("Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. "
                               "International Journal of Forecasting, 22(4), 679–688."),
    "makridakis2020": _ref("Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). The M4 Competition: 100,000 time series and 61 "
                           "forecasting methods. International Journal of Forecasting, 36(1), 54–74."),
    "makridakis2022": _ref("Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and "
                           "conclusions. International Journal of Forecasting, 38(4), 1346–1364."),
    "petropoulos2020": _ref("Petropoulos, F., & Svetunkov, I. (2020). A simple combination of univariate models. "
                            "International Journal of Forecasting, 36(1), 110–115."),
    "fiorucci2016": _ref("Fiorucci, J. A., Pellegrini, T. R., Louzada, F., Petropoulos, F., & Koehler, A. B. (2016). Models for optimising "
                         "the theta method and their relationship to state space models. International Journal of Forecasting, 32(4), 1151–1161."),
    "holt2004": _ref("Holt, C. C. (2004). Forecasting seasonals and trends by exponentially weighted moving averages. International Journal of Forecasting, 20(1), 5-10 (reprint of the 1957 ONR memorandum).", "https://doi.org/10.1016/j.ijforecast.2003.09.015"),
    "gardner1985": _ref("Gardner, E. S., Jr., & McKenzie, E. (1985). Forecasting trends in time series. Management Science, 31(10), 1237–1246."),
    "kim2016": _ref("Kim, S., & Kim, H. (2016). A new metric of absolute percentage error for intermittent demand forecasts. "
                    "International Journal of Forecasting, 32(3), 669–679."),
    "davydenko2013": _ref("Davydenko, A., & Fildes, R. (2013). Measuring forecasting accuracy: The case of judgmental adjustments to SKU-level "
                          "demand forecasts. International Journal of Forecasting, 29(3), 510–522."),
    "nixtla": _ref("Nixtla StatsForecast (open source, Apache-2.0) documentation: in the Python engine this model is the library's own implementation.",
                   "https://nixtlaverse.nixtla.io/statsforecast/"),
}


def P(key, label, kind, default, help, *, min=None, max=None, step=None, choices=None, engines=("python", "r"),
      placeholder=None, advanced=False):
    """Parameter schema. `default=None` + `placeholder`: if not sent, the library default is used (no value is invented)."""
    return {"key": key, "label": label, "kind": kind, "default": default, "help": help, "min": min, "max": max, "step": step,
            "choices": [{"value": v, "label": l} for v, l in (choices or ())], "engines": list(engines), "placeholder": placeholder,
            "advanced": advanced}


MAX_OBS = P("max_obs", "Last N observations", "int", None,
            "Uses only the last N observations of each series (reduces memory and time). If left empty the whole series is used.",
            min=2, placeholder="all")

CATBOOST_PARAMS = [
    MAX_OBS,
    P("catboost.iterations", "Number of trees (iterations)", "int", None, "Number of boosting rounds. Empty: CatBoost default (1000).",
      min=1, max=20000, placeholder="1000"),
    P("catboost.depth", "Depth", "int", None, "Tree depth. Empty: CatBoost default (6).", min=1, max=16, placeholder="6"),
    P("catboost.learning_rate", "Learning rate", "float", None, "Empty: CatBoost chooses it itself according to the data size.", min=0.001, max=1,
      step=0.01, placeholder="automatic"),
    P("catboost.l2_leaf_reg", "L2 regularisation", "float", None, "Empty: CatBoost default (3).", min=0, max=100, step=0.5, placeholder="3"),
    P("catboost.random_seed", "Random seed", "int", None, "Empty: 0 (same as the original R call).", min=0, placeholder="0", advanced=True),
]

LIGHTGBM_PARAMS = [
    MAX_OBS,
    P("lightgbm.num_iterations", "Number of trees (iterations)", "int", None, "Number of boosting rounds. Empty: LightGBM default (100).",
      min=1, max=20000, placeholder="100"),
    P("lightgbm.learning_rate", "Learning rate", "float", None, "Empty: LightGBM default (0.1).", min=0.001, max=1, step=0.01,
      placeholder="0.1"),
    P("lightgbm.num_leaves", "Leaves per tree", "int", None, "Maximum leaves in one tree. Empty: LightGBM default (31).", min=2,
      max=4096, placeholder="31"),
    P("lightgbm.min_data_in_leaf", "Minimum rows per leaf", "int", None, "Empty: LightGBM default (20).", min=1, max=100000,
      placeholder="20"),
    P("lightgbm.seed", "Random seed", "int", None, "Empty: 0. Training is deterministic for a given seed.", min=0, placeholder="0",
      advanced=True),
]

PY = "python"
R = "r"

MODELS: dict[str, dict] = {
    "naive": {
        "name": "Naive", "group": "yerel", "task": "local",
        "summary": "Repeats the last observation unchanged over the horizon.",
        "how": "The forecast equals the last value of the series and is flat over the horizon. It has no parameters.",
        "use": "The simplest benchmark. Hard to beat on series that behave like a random walk.",
        "caution": "Ignores seasonality and trend.",
        "engines": {PY: "Built in (numpy). This method does not exist in the R engine."},
        "params": [], "refs": ["fpp3"], "cost": "Instant.",
    },
    "snaive": {
        "name": "Seasonal naive", "group": "yerel", "task": "local",
        "summary": "Carries the values of the previous season forward.",
        "how": "Forecast(T+h) = y(T+h−m); m is the integer part of the largest seasonal period (same as the msts frequency in R). "
               "Series for which other models fail also receive this method as a fallback forecast, and these are counted and reported.",
        "use": "A baseline for seasonal data; also the fallback forecast.",
        "caution": "Not meaningful if the series is shorter than one period. It also carries the noise forward unchanged.",
        "engines": {PY: "Built in (numpy), same rule as R.", R: "forecast::snaive (msts)."},
        "params": [], "refs": ["fpp3"], "cost": "Instant.",
    },
    "ses": {
        "name": "Simple exponential smoothing (SES)", "group": "yerel", "task": "local",
        "summary": "Smooths the level component with exponential weights; gives a flat forecast.",
        "how": "The single smoothing parameter (alpha) is chosen by maximum likelihood (alpha in the range 0.0001–0.9999). "
               "There is no trend or seasonality; the forecast is constant over the horizon.",
        "use": "Series without trend or seasonality, or whose seasonality is handled separately; a quick baseline.",
        "caution": "Makes systematic errors if there is trend or seasonality.",
        "engines": {PY: "Built in (numpy/scipy), same algorithm as R; median difference on NN5 about 4×10⁻⁶ (docs/VALIDATION.md).",
                    R: "forecast::ses. The original call from your study."},
        "params": [], "refs": ["fpp3"],
        "cost": "NN5 weekly, 111 series: 0.3 s in R; daily: 0.6 s (measured on your machine).",
    },
    "ets": {
        "name": "ETS (exponential smoothing state space)", "group": "yerel", "task": "local",
        "summary": "An exponential smoothing family that selects the error, trend and seasonality form automatically with an information criterion.",
        "how": "All combinations of additive/multiplicative error, trend (possibly damped) and seasonality are fitted; the information criterion picks the best.",
        "use": "A strong general-purpose classical method; common for monthly, quarterly and weekly data.",
        "caution": "Seasonality is not used if the period exceeds 24 (same rule as R ets); it cannot capture seasonality in long daily/hourly periods. "
                   "In that case consider dhr_arima, mstl or tbats.",
        "engines": {PY: "statsforecast AutoETS. Similar to R, not identical: on NN5 weekly the mean SMAPE difference is −1.55% (docs/VALIDATION.md).",
                    R: "forecast::ets. The original call from your study."},
        "params": [], "refs": ["hyndman2002", "fpp3"],
        "cost": "NN5 weekly, 111 series: 1.8 s in R; daily: 55 s (measured on your machine).",
    },
    "theta": {
        "name": "Theta", "group": "yerel", "task": "local",
        "summary": "Combines simple exponential smoothing with a drift equal to half the slope of the series.",
        "how": "If seasonality is detected by an ACF test, a classical multiplicative decomposition is applied; then the SES forecast and a linear drift with half the slope are added.",
        "use": "The method that won the M3 competition; a fast and robust baseline for short series.",
        "caution": "Does not handle complex multi-seasonal structures.",
        "engines": {PY: "Built in (numpy/scipy), same algorithm as R; median difference on NN5 about 4–8×10⁻⁶.",
                    R: "forecast::thetaf. The original call from your study."},
        "params": [], "refs": ["assimakopoulos2000", "hyndmanbillah2003"],
        "cost": "NN5 weekly, 111 series: 0.5 s in R; daily: 1.2 s (measured on your machine).",
    },
    "drift": {
        "name": "Random walk with drift", "group": "yerel", "task": "local",
        "summary": "Extends the line from the first to the last observation into the future.",
        "how": "Forecast(T+h) = y(T) + h·(y(T) − y(1))/(T − 1); the same rule as R rwf(drift = TRUE). It has no parameters.",
        "use": "A standard benchmark for trended series.",
        "caution": "Only the first and last observations determine the slope, so an outlier at either end changes the whole forecast.",
        "engines": {PY: "Built in (numpy), closed form."},
        "params": [], "refs": ["fpp3"], "cost": "Instant.",
    },
    "naive2": {
        "name": "Naive2 (M4 benchmark)", "group": "yerel", "task": "local",
        "summary": "Seasonally adjusted naive forecast; the reference method of the M4 OWA measure.",
        "how": "If a 90% autocorrelation test (M4 benchmark code, at least 3 periods of data) finds seasonality, a classical multiplicative "
               "decomposition removes it; the last adjusted value is repeated and the seasonal indices are put back.",
        "use": "Needed to compute OWA; a seasonal benchmark that is less noisy than seasonal naive.",
        "caution": "The seasonality test uses R's default number of autocorrelation lags, so long periods (e.g. 365 in daily data) are treated as non-seasonal.",
        "engines": {PY: "Built in (numpy), following the official M4 benchmark code; identical to R on NN5 weekly and daily (docs/VALIDATION.md 1d)."},
        "params": [], "refs": ["makridakis2020"], "cost": "Instant.",
    },
    "comb_shd": {
        "name": "Comb S-H-D (M4 benchmark)", "group": "yerel", "task": "local",
        "summary": "Average of simple, Holt and damped Holt exponential smoothing on the seasonally adjusted series.",
        "how": "Seasonality is removed as in Naive2; SES, Holt (linear trend) and damped Holt forecasts are averaged and the seasonal indices are put back. "
               "Holt parameters are fitted by least squares within the R ets bounds.",
        "use": "The combination benchmark of the M3/M4 competitions; strong and hard to beat on yearly, quarterly and monthly data.",
        "caution": "Same model as R holt() (ets AAN / AAdN, least squares). On NN5 the mean SMAPE and MASE are within 0.4% of R; "
                   "single series can differ slightly because R's optimiser sometimes stops at a worse fit (docs/VALIDATION.md 1d).",
        "engines": {PY: "Built in (numpy/scipy); checked against R forecast 9.0.2 on NN5 weekly and daily."},
        "params": [], "refs": ["makridakis2020", "gardner1985"], "cost": "Fast.",
    },
    "dotm": {
        "name": "Dynamic optimised Theta (DOTM)", "group": "yerel", "task": "local",
        "summary": "A Theta model whose theta coefficient and parameters are optimised and updated over time.",
        "how": "State-space form of the Theta method with an optimised theta line; seasonality is removed first by classical decomposition when detected.",
        "use": "A strong benchmark in M4 for yearly/quarterly/monthly data; also a component of SCUM.",
        "caution": "For series containing zero/negative values, additive decomposition is used instead of multiplicative (a note is recorded).",
        "engines": {PY: "statsforecast DynamicOptimizedTheta. No R counterpart in this app."},
        "params": [P("decomposition_type", "Decomposition", "choice", "multiplicative", "Type of seasonal decomposition.",
                     choices=(("multiplicative", "Multiplicative"), ("additive", "Additive")))],
        "refs": ["fiorucci2016", "nixtla"], "cost": "Fast.",
    },
    "scum": {
        "name": "SCUM (median of ETS, CES, ARIMA, DOTM)", "group": "yerel", "task": "local",
        "summary": "Point-wise median of four automatic statistical models.",
        "how": "ETS, CES, ARIMA and DOTM are fitted with this engine's settings and the median of their forecasts is taken at each horizon step. "
               "If a component fails the median of the remaining ones is used (a note is recorded); with fewer than two the series falls back.",
        "use": "A simple combination that ranked among the best statistical benchmarks in M4; useful to test whether combinations change rankings.",
        "caution": "As slow as its slowest component (ARIMA); measure with a short trial first.",
        "engines": {PY: "Built from statsforecast AutoETS, AutoCES, AutoARIMA and DynamicOptimizedTheta."},
        "params": [], "refs": ["petropoulos2020"], "cost": "Slow (sum of the four components).",
    },
    "comb_eat": {
        "name": "Combination: mean of ETS, ARIMA, Theta", "group": "yerel", "task": "local",
        "summary": "Equal-weight average of the automatic ETS, ARIMA and Theta forecasts.",
        "how": "ETS, ARIMA and Theta are fitted with this engine's settings and their forecasts are averaged at each horizon step. "
               "If a component fails the mean of the other two is used (a note is recorded); with fewer than two the series falls back.",
        "use": "The classic simple combination: it is often more accurate than each of its parts (Makridakis & Hibon 2000; "
               "Hyndman's ETS-ARIMA-Theta average was a strong M4 benchmark). Good for testing whether combining changes rankings.",
        "caution": "As slow as its slowest component (ARIMA). One badly wrong component pulls the mean; the median variant is more robust.",
        "engines": {PY: "statsforecast AutoETS and AutoARIMA with the built-in Theta (R thetaf)."},
        "params": [], "refs": ["makridakis2000", "petropoulos2020"], "cost": "Slow (sum of the three components).",
    },
    "comb_eat_median": {
        "name": "Combination: median of ETS, ARIMA, Theta", "group": "yerel", "task": "local",
        "summary": "Pointwise median of the automatic ETS, ARIMA and Theta forecasts.",
        "how": "The same three components as the mean combination; at each horizon step the middle value is taken, so a single "
               "badly wrong component has no effect. With one component failed it is the mean of the other two.",
        "use": "A robust alternative to the mean combination (the idea behind SCUM, with three components).",
        "caution": "As slow as its slowest component (ARIMA).",
        "engines": {PY: "statsforecast AutoETS and AutoARIMA with the built-in Theta (R thetaf)."},
        "params": [], "refs": ["petropoulos2020"], "cost": "Slow (sum of the three components).",
    },
    "tbats": {
        "name": "TBATS", "group": "yerel", "task": "local",
        "summary": "Exponential smoothing with trigonometric seasonality for series with multiple and fractional seasonal periods.",
        "how": "Box-Cox transformation, ARMA errors, trend and trigonometric (Fourier) seasonality; the components to use are selected by AIC. "
               "Periods may be fractional (e.g. 365.25).",
        "use": "When hourly/daily data have more than one seasonality.",
        "caution": "Very slow. NOT available in the Python engine: statsforecast AutoTBATS did not find the same model as R tbats() (disabled by user decision).",
        "engines": {R: "forecast::tbats (R engine only)."},
        "params": [], "refs": ["delivera2011"],
        "cost": "In your own study, NN5, 111 series: about 12 min weekly, about 13 min daily (R). A long run needs confirmation.",
    },
    "arima": {
        "name": "Auto ARIMA", "group": "yerel", "task": "local",
        "summary": "ARIMA with automatic selection of the differencing order and the (p,d,q)(P,D,Q) structure.",
        "how": "The differencing order is chosen by unit-root tests, and the orders by stepwise search and an information criterion; drift is allowed. "
               "Fallback chain: first with a log transformation (lambda=0), otherwise raw, otherwise non-seasonal.",
        "use": "A flexible classical method for series with strong autocorrelation.",
        "caution": "Slow; on long daily series (R) a single run can take about half an hour. The log transformation is tried only if all values are positive.",
        "engines": {PY: "statsforecast AutoARIMA; imitates the R chain, not identical (NN5 weekly: ΔSMAPE −0.07%).",
                    R: "forecast::auto.arima. The original call from your study."},
        "params": [P("arima_max_season", "Largest seasonal period", "int", 200,
                     "If the period exceeds this value it falls back to non-seasonal ARIMA (seasonal ARIMA with a long period is impractical).",
                     min=2, max=2000, engines=(PY,), advanced=True)],
        "refs": ["hyndman2008", "fpp3"],
        "cost": "NN5 weekly, 111 series: 14 s in R; daily: about 30 min in your study (R).",
    },
    "dhr_arima": {
        "name": "Dynamic harmonic regression (DHR-ARIMA)", "group": "yerel", "task": "local",
        "summary": "Models seasonality with Fourier terms and the remaining structure with a non-seasonal ARIMA.",
        "how": "For each seasonal period, K pairs of sin/cos terms are added as external regressors; the errors are modelled with auto.arima. "
               "K=1 in the original study.",
        "use": "Long or multi-seasonal periods (daily, weekly, hourly).",
        "caution": "As K grows the model is more flexible but slower and prone to overfitting.",
        "engines": {PY: "statsforecast AutoARIMA + Fourier regressors. With multi-seasonal data K is applied to every period (in some versions R may fail and fall back to SNaive).",
                    R: "forecast::auto.arima(xreg=fourier). The original call from your study."},
        "params": [
            P("fourier_k", "Number of Fourier pairs (K)", "int", 1, "Number of sin/cos pairs per period. 1 in the original study.", min=1, max=100),
            P("fourier_k_per_period", "Apply K to each period separately", "bool", False,
              "With multi-seasonal data, repeats K for every period.", engines=(R,), advanced=True),
        ],
        "refs": ["hyndman2008", "fpp3"],
        "cost": "NN5 weekly, 111 series: 10 s in R.",
    },
    # ---------------------------------------------------------------- intermittent demand (Python only)
    "croston_classic": {
        "name": "Croston (classic)", "group": "aralikli", "task": "local",
        "summary": "Smooths the demand size and the interval between demands separately; gives a flat forecast as a demand rate.",
        "how": "Only non-zero demands are exponentially smoothed; the interval between demands is smoothed separately. Forecast = size / interval, constant over the horizon.",
        "use": "Demand series with many zeros, such as spare parts.",
        "caution": "Produces a flat RATE (often below 1). If rounding to integers is on, it falls to 0: all forecasts can become 0. "
                   "Turn rounding off. Classic Croston is known to be biased upwards.",
        "engines": {PY: "statsforecast CrostonClassic. Not in the R engine."},
        "params": [], "refs": ["croston1972", "syntetos2005"], "cost": "Very fast.",
    },
    "croston_sba": {
        "name": "Croston–SBA", "group": "aralikli", "task": "local",
        "summary": "Applies to Croston the Syntetos–Boylan correction, which reduces its bias.",
        "how": "Shrinks the Croston forecast by a correction factor (approximately 1 − alpha/2).",
        "use": "To prevent Croston from consistently over-forecasting intermittent demand.",
        "caution": "Rounding to integers can turn forecasts into 0; turn rounding off.",
        "engines": {PY: "statsforecast CrostonSBA."},
        "params": [], "refs": ["syntetos2005"], "cost": "Very fast.",
    },
    "croston_optimized": {
        "name": "Croston (optimised)", "group": "aralikli", "task": "local",
        "summary": "Classic Croston; the smoothing parameter is optimised per series.",
        "how": "Instead of a fixed smoothing value, the parameter is chosen by the in-sample error.",
        "use": "When you do not want to choose the smoothing value by hand.",
        "caution": "On short and very sparse series the optimisation can be unstable. Turn rounding off.",
        "engines": {PY: "statsforecast CrostonOptimized."},
        "params": [], "refs": ["croston1972"], "cost": "Fast.",
    },
    "tsb": {
        "name": "TSB (Teunter–Syntetos–Babai)", "group": "aralikli", "task": "local",
        "summary": "Updates the demand probability every period; for products whose demand stops, the forecast decays towards zero over time.",
        "how": "The demand probability (alpha_p) is smoothed every period, the demand size (alpha_d) only when demand occurs. Forecast = probability × size.",
        "use": "Products at risk of obsolescence: the Croston family does not update its forecast when demand stops.",
        "caution": "The default 0.1/0.1 is our own choice, not an evidence-based recommendation; tune it on your data. Turn rounding off.",
        "engines": {PY: "statsforecast TSB."},
        "params": [
            P("alpha_d", "Size smoothing (alpha_d)", "float", 0.1, "Smoothing for demand size. Range (0, 1]. Our own choice.", min=0.01, max=1, step=0.05),
            P("alpha_p", "Probability smoothing (alpha_p)", "float", 0.1, "Smoothing for demand probability. Range (0, 1]. Our own choice.", min=0.01, max=1, step=0.05),
        ],
        "refs": ["teunter2011"], "cost": "Very fast.",
    },
    "adida": {
        "name": "ADIDA", "group": "aralikli", "task": "local",
        "summary": "Reduces intermittency by temporal aggregation, forecasts, and then disaggregates the result.",
        "how": "The series is aggregated at one aggregation level (demand becomes less sparse), forecast with a simple method, and the forecast is divided back to the original scale.",
        "use": "An alternative to Croston on very sparse series.",
        "caution": "The aggregation level is chosen by the library. Turn rounding off.",
        "engines": {PY: "statsforecast ADIDA."},
        "params": [], "refs": ["nikolopoulos2011"], "cost": "Very fast.",
    },
    "imapa": {
        "name": "IMAPA", "group": "aralikli", "task": "local",
        "summary": "Combines the forecasts of several aggregation levels (MAPA for intermittent demand).",
        "how": "Forecasts are made at different aggregation levels and averaged; this reduces the risk of choosing a single level.",
        "use": "To avoid ADIDA's sensitivity to the choice of level.",
        "caution": "Turn rounding off.", "engines": {PY: "statsforecast IMAPA."},
        "params": [], "refs": ["petropoulos2015"], "cost": "Fast.",
    },
    # ---------------------------------------------------------------- classical methods specific to Python
    "auto_theta": {
        "name": "Auto Theta", "group": "yerel", "task": "local",
        "summary": "Selects the best of the four Theta variants by in-sample error.",
        "how": "Standard, optimised, dynamic standard and dynamic optimised Theta models are tried; the library selects by its criterion.",
        "use": "An automated, more flexible version of Theta.",
        "caution": "For series containing zero/negative values, additive decomposition is used instead of multiplicative (a note is recorded). Without at least 2 periods of data the model is non-seasonal.",
        "engines": {PY: "statsforecast AutoTheta. No R counterpart; validation is the doctor command and your own data."},
        "params": [P("decomposition_type", "Decomposition", "choice", "multiplicative", "Type of seasonal decomposition.",
                     choices=(("multiplicative", "Multiplicative"), ("additive", "Additive")))],
        "refs": ["assimakopoulos2000", "nixtla"], "cost": "Fast.",
    },
    "auto_ces": {
        "name": "Complex exponential smoothing (CES)", "group": "yerel", "task": "local",
        "summary": "Smooths the series in the complex plane (real + imaginary component); selects the seasonality type automatically.",
        "how": "Captures trend and seasonality with complex-valued smoothing instead of separate components as in ETS.",
        "use": "Series with intertwined trend/seasonality where ETS falls short.",
        "caution": "If the period exceeds the 'largest seasonal period' setting the model is non-seasonal; this threshold (24) is our own choice.",
        "engines": {PY: "statsforecast AutoCES."},
        "params": [
            P("model", "Seasonality type", "choice", "Z", "Z: select automatically; N: none; S: simple; P: partial; F: full.",
              choices=(("Z", "Z (automatic)"), ("N", "N (none)"), ("S", "S (simple)"), ("P", "P (partial)"), ("F", "F (full)"))),
            P("max_season", "Largest seasonal period", "int", 24, "If the period exceeds this value the model is used non-seasonally. Our own choice.", min=2, max=2000, advanced=True),
        ],
        "refs": ["svetunkov2022", "nixtla"], "cost": "Medium (measure with a short trial).",
    },
    "holt_winters": {
        "name": "Holt–Winters", "group": "yerel", "task": "local",
        "summary": "Tracks level, trend and seasonality with three separate exponential smoothings.",
        "how": "The classical three-component method; if there is no seasonality it reduces to Holt (a note is recorded).",
        "use": "Relatively short series with clear trend and seasonality.",
        "caution": "For multiplicative error all values must be greater than 0. The period threshold (24) is our own choice.",
        "engines": {PY: "statsforecast HoltWinters."},
        "params": [
            P("error_type", "Error type", "choice", "A", "A: additive; M: multiplicative (all values must be positive).",
              choices=(("A", "Additive"), ("M", "Multiplicative"))),
            P("max_season", "Largest seasonal period", "int", 24, "If the period exceeds this value the model is used non-seasonally. Our own choice.", min=2, max=2000, advanced=True),
        ],
        "refs": ["winters1960", "nixtla"], "cost": "Fast.",
    },
    "mstl": {
        "name": "MSTL (multiple seasonal-trend decomposition)", "group": "yerel", "task": "local",
        "summary": "Decomposes the series into trend, several seasonal components and a remainder; trend/remainder are forecast with ETS.",
        "how": "STL decomposition is applied successively for each seasonal period. Periods are rounded to integers; only those with at least two periods of data are used.",
        "use": "Multiple seasonalities in hourly/daily data.",
        "caution": "If no valid period is found (e.g. yearly data) the series falls back to the fallback forecast; this is expected.",
        "engines": {PY: "statsforecast MSTL; trend forecaster is the library default (AutoETS, non-seasonal)."},
        "params": [], "refs": ["bandara2021", "cleveland1990"], "cost": "Medium (measure with a short trial).",
    },
    "stl_ets": {
        "name": "STL + ETS (stlf)", "group": "yerel", "task": "local",
        "summary": "R's stlf: STL removes the seasonality, ETS forecasts the seasonally adjusted series, the last seasonal cycle is added back.",
        "how": "Same as R forecast::stlf(y): mstl() with R's stl() settings (s.window 11, 15, … per period; one pass for one period, "
               "two passes for several; periods shorter than half the series), ets(model = \"ZZN\") on trend + remainder, the last "
               "cycle of each seasonal component repeated.",
        "use": "Seasonal daily/weekly data where a slowly changing seasonal pattern and a separate trend model are wanted; a strong, fast alternative to seasonal ETS (also for periods > 24, where ets() ignores seasonality).",
        "caution": "Needs a seasonal period and more than two full periods of data; otherwise the series falls back. Close to MSTL, which uses statsmodels' STL settings instead of R's.",
        "engines": {PY: "R's stl/mstl ported to Python (identical decomposition to the C++ engine); statsforecast AutoETS(ZZN) for the adjusted series, so forecasts are close to, not equal to, R."},
        "params": [], "refs": ["cleveland1990", "hyndman2008", "fpp3"], "cost": "Low (NN5 daily, 111 series: 0.1 s in C++).",
    },
    "stl_arima": {
        "name": "STL + ARIMA", "group": "yerel", "task": "local",
        "summary": "As STL + ETS, but the seasonally adjusted series is forecast with a non-seasonal auto.arima.",
        "how": "R forecast(mstl(y), method = \"arima\"): the same R STL decomposition, auto.arima(seasonal = FALSE) on trend + remainder, the last seasonal cycle added back.",
        "use": "Seasonal series whose adjusted part has autocorrelation that ETS does not capture.",
        "caution": "Needs a seasonal period and more than two full periods of data; otherwise the series falls back.",
        "engines": {PY: "R's stl/mstl ported to Python; statsforecast AutoARIMA (non-seasonal) for the adjusted series, so forecasts are close to, not equal to, R."},
        "params": [], "refs": ["cleveland1990", "hyndman2008", "fpp3"], "cost": "Medium (NN5 daily, 111 series: 2 s in C++).",
    },
    "bagged_ets": {
        "name": "Bagged ETS", "group": "yerel", "task": "local",
        "summary": "ETS fitted to the series and to bootstrapped copies of it; the forecasts are averaged.",
        "how": "R forecast::baggedETS after set.seed(seed): Box-Cox (Guerrero lambda in [0, 1] for positive data), periodic STL, a moving-block "
               "bootstrap of the remainder (blocks of two seasons) gives n_boot − 1 new series; ets() is fitted to each and to the "
               "original, and the point forecasts are averaged. R's random number generator is reproduced, so a seed gives R's numbers.",
        "use": "When ETS is a good model but its selection is unstable; bagging usually improves accuracy (strong on M3 monthly).",
        "caution": "About n_boot times the cost of ETS. Non-seasonal series (frequency 1, or fewer than two full periods) use R's loess "
                   "trend instead of STL, as R does.",
        "engines": {PY: "The bootstrap equals R's (and the C++ engine's); each copy is forecast with statsforecast AutoETS, so the result is close to, "
                        "not equal to, R. Slow: about 100 AutoETS fits per series."},
        "params": [P("n_boot", "Number of bootstrap series", "int", 100, "Including the original series (R's default 100). C++ engine: 100 only.",
                     min=1, max=1000),
                   P("seed", "Random seed", "int", 0, "set.seed() value used before each series (C++ engine: 0 only).", min=0, advanced=True)],
        "refs": ["bergmeir2016", "hyndman2008"], "cost": "C++: NN5 weekly 111 series 4 s (R: 200 s). Python: minutes.",
    },
    "thief": {
        "name": "THieF (temporal hierarchies, ETS)", "group": "yerel", "task": "local",
        "summary": "Forecasts the series at every temporal aggregation level that divides the season (e.g. monthly, 2-monthly, quarterly, …, annual) and reconciles them.",
        "how": "R thief::thief(y, usemodel = \"ets\", comb = \"struc\"): non-overlapping sums at the levels that divide the seasonal period m, ets() at each "
               "level, then the forecasts are made coherent by structural-scaling weighted least squares; the reconciled bottom level is returned.",
        "use": "Monthly, quarterly or weekly data where the annual picture is more stable than the fine detail; typically improves on plain ETS.",
        "caution": "Needs a seasonal period m > 1 and at least two full periods of data; otherwise the series falls back. Levels with very short series use R's small-sample ETS branch.",
        "engines": {PY: "The R algorithm (aggregation, reconciliation) with statsforecast AutoETS at each level; close to, not equal to, R."},
        "params": [], "refs": ["athanasopoulos2017", "hyndman2008"], "cost": "A few ETS fits per series.",
    },
    "mapa": {
        "name": "MAPA (multiple aggregation prediction)", "group": "yerel", "task": "local",
        "summary": "ETS at temporal aggregation levels 1..m; the level, trend and seasonal components are combined across levels.",
        "how": "R MAPA::mapa(y, comb = \"w.mean\", hybrid = TRUE, model = \"ZZZ\", type = \"ets\"): the series is averaged over 1..m periods, ets() is fitted at "
               "each level, the final states are translated into level / trend / season forecasts, level and trend are averaged, the seasonal parts of "
               "the levels that divide m are combined with 1/AL weights, and the result is averaged with the level-1 ETS forecast.",
        "use": "Noisy series where the model chosen at the original frequency is unstable; strong on intermittent and monthly data.",
        "caution": "About m ETS fits per series. Aggregation levels with fewer than four observations are skipped, as in R.",
        "engines": {PY: "The R algorithm with statsforecast AutoETS fits (final states taken directly); close to, not equal to, R."},
        "params": [], "refs": ["kourentzes2014", "hyndman2008"], "cost": "About m ETS fits per series.",
    },
    "auto_mfles": {
        "name": "Auto MFLES", "group": "yerel", "task": "local",
        "summary": "Gradient-boosted time-series decomposition: trend, seasonal and level steps are learned in sequence.",
        "how": "The library selects its settings by taking the last 'horizon' observations as test windows (n_windows of them); the criterion can be chosen.",
        "use": "To try a quick modern alternative to classical methods.",
        "caution": "Requires scikit-learn. A peer-reviewed reference has not been verified; cite the library documentation before using it in a paper.",
        "engines": {PY: "statsforecast AutoMFLES (not listed if scikit-learn is not installed)."},
        "params": [
            P("n_windows", "Number of validation windows", "int", 2, "Number of windows used for selecting the settings.", min=1, max=10),
            P("metric", "Measure", "choice", "smape", "The measure used when selecting settings on the windows.",
              choices=(("smape", "sMAPE"), ("mape", "MAPE"), ("mse", "MSE"), ("mae", "MAE"))),
        ],
        "refs": ["nixtla"], "cost": "Medium (measure with a short trial).",
    },
    # ---------------------------------------------------------------- global
    "holt_ets": {
        "name": "Holt (ETS with additive trend)", "group": "yerel", "task": "local",
        "summary": "Exponential smoothing of level and trend, with the trend damped or not as the information criterion decides.",
        "how": "ETS(A,A,N) and ETS(A,Ad,N) (or the multiplicative-error forms) are fitted and the one with the lower AICc is kept.",
        "use": "Trended, non-seasonal series; a classical benchmark between SES and the full automatic ETS.",
        "caution": "For multiplicative error all values must be greater than 0. Seasonality is ignored; use ETS or Holt-Winters for seasonal data.",
        "engines": {PY: "statsforecast Holt."},
        "params": [P("error_type", "Error type", "choice", "A", "A: additive; M: multiplicative (all values must be positive).",
                     choices=(("A", "Additive"), ("M", "Multiplicative")), engines=(PY,))],
        "refs": ["holt2004", "gardner1985", "nixtla"], "cost": "Fast.",
    },
    "historic_average": {
        "name": "Historic average", "group": "yerel", "task": "local",
        "summary": "Forecasts every future value with the mean of the whole training series.",
        "how": "The arithmetic mean of all observations is repeated over the horizon (R: meanf).",
        "use": "The simplest benchmark for stationary series without trend or seasonality.",
        "caution": "Ignores trend, seasonality and recent level shifts; only useful as a lower bound.",
        "engines": {PY: "statsforecast HistoricAverage."},
        "params": [], "refs": ["fpp3", "nixtla"], "cost": "Very fast.",
    },
    "window_average": {
        "name": "Window average", "group": "yerel", "task": "local",
        "summary": "Forecasts with the mean of the last few observations.",
        "how": "The mean of the last window_size values is repeated over the horizon.",
        "use": "A benchmark that follows level shifts more quickly than the historic average.",
        "caution": "A series shorter than the window falls back. The default window (4) is our own choice.",
        "engines": {PY: "statsforecast WindowAverage."},
        "params": [P("window_size", "Window size", "int", 4, "Number of last observations averaged. Our own choice.", min=1, max=10000, engines=(PY,))],
        "refs": ["nixtla"], "cost": "Very fast.",
    },
    "seasonal_window_average": {
        "name": "Seasonal window average", "group": "yerel", "task": "local",
        "summary": "Forecasts each season position with the mean of that position over the last few seasons.",
        "how": "The last window_size full seasons are averaged position by position and the seasonal profile is repeated.",
        "use": "A seasonal benchmark that is less noisy than the seasonal naive method (daily data with weekly cycles, NN5).",
        "caution": "Needs window_size full seasons, otherwise the series falls back; without a usable season it is the window average.",
        "engines": {PY: "statsforecast SeasonalWindowAverage."},
        "params": [P("window_size", "Number of seasons", "int", 2, "How many past seasons are averaged. Our own choice.", min=1, max=1000, engines=(PY,))],
        "refs": ["nixtla"], "cost": "Very fast.",
    },
    "seasonal_es": {
        "name": "Seasonal exponential smoothing (optimised)", "group": "yerel", "task": "local",
        "summary": "Simple exponential smoothing run separately on each season position, with its own optimised alpha.",
        "how": "The series is split into its m season positions; SES with alpha chosen in [0.01, 0.99] by golden-section search forecasts each one.",
        "use": "Seasonal series whose seasonal pattern changes slowly; a smoother alternative to the seasonal naive method.",
        "caution": "There is no trend component. A series shorter than one season falls back.",
        "engines": {PY: "statsforecast SeasonalExponentialSmoothingOptimized."},
        "params": [], "refs": ["fpp3", "nixtla"], "cost": "Very fast.",
    },
    "zero_model": {
        "name": "Zero forecast", "group": "aralikli", "task": "local",
        "summary": "Forecasts zero for every period.",
        "how": "Returns 0 over the whole horizon.",
        "use": "Sanity baseline for intermittent demand (M5): a model that cannot beat it on a given metric adds nothing.",
        "caution": "Only meaningful for intermittent series and for metrics that do not divide by the forecast; rounding has no effect on it.",
        "engines": {PY: "statsforecast ZeroModel."},
        "params": [], "refs": ["makridakis2022", "nixtla"], "cost": "Instant.",
    },    "pooled_regression": {
        "name": "Pooled regression", "group": "kuresel", "task": "global",
        "summary": "Learns a single linear model on the lagged values of all series.",
        "how": "Each series is divided by its own mean; rows made of (lag) lagged values are stacked from all series and a linear regression "
               "without intercept is fitted. Forecasting is recursive; the result is rescaled with the series mean.",
        "use": "A fast, strong global baseline when there are many similar series.",
        "caution": "Cannot be used if a series is shorter than lag+1 observations. 'lag' is chosen in the Settings step (the catalogue suggests the value from your study).",
        "engines": {PY: "Built in; streaming QR (the design matrix is not held in memory). Same result as R glm (difference about 5×10⁻¹⁵ on NN5).",
                    R: "glm (gaussian). The original call from your study."},
        "params": [MAX_OBS], "refs": ["montero2021"],
        "cost": "NN5 weekly, 111 series: 0.2 s; daily: 0.3 s (measured on your machine).",
    },
    "catboost": {
        "name": "CatBoost", "group": "kuresel", "task": "global",
        "summary": "Gradient-boosted decision trees on the same lag matrix.",
        "how": "The same data pipeline as pooled regression; a CatBoost regressor instead of the linear model. Forecasting is recursive.",
        "use": "Non-linear relationships; many series.",
        "caution": "Heavy in memory and time (on 8 GB the pre-run check warns for large datasets). The R catboost package may not be installed on Windows.",
        "engines": {PY: "Requires the catboost Python package; the 'doctor' command checks whether it is installed.",
                    R: "Requires the R catboost package; not installed on your machine (doctor warning)."},
        "params": CATBOOST_PARAMS, "refs": ["prokhorenkova2018", "montero2021"], "cost": "Depends on the data; try a small subset.",
    },
    "lightgbm": {
        "name": "LightGBM", "group": "kuresel", "task": "global",
        "summary": "Histogram-based gradient-boosted trees on the same lag matrix.",
        "how": "The same data pipeline as pooled regression (mean-scaled series, lag matrix of all series stacked); a LightGBM "
               "regressor (L2 loss, leaf-wise trees) instead of the linear model. Forecasting is recursive. Training is "
               "deterministic with a fixed seed.",
        "use": "Non-linear relationships across many series; usually much faster than CatBoost on large data.",
        "caution": "Like every recursive tree model it cannot forecast beyond the range of values seen in training (no trend "
                   "extrapolation after mean scaling).",
        "engines": {PY: "Requires the lightgbm Python package; the 'doctor' command checks whether it is installed. This method "
                        "does not exist in the R engine."},
        "params": LIGHTGBM_PARAMS, "refs": ["ke2017", "montero2021"], "cost": "Depends on the data; typically seconds for NN5.",
    },
    "lightgbm_feat": {
        "name": "LightGBM (M5-style features)", "group": "kuresel", "task": "global",
        "summary": "Global LightGBM on lags plus seasonal lags, rolling means / standard deviations and the position in the season.",
        "how": "Mean-scaled series; features per row: lags 1..lag, seasonal lags m and 2m, rolling mean and standard deviation over the last "
               "max(4, m) and 2·max(4, m) values, and t mod m. Seasonal features are used only when the shortest series holds three full "
               "seasons (otherwise windows of 4 and 8). Recursive forecasting: each prediction extends the history used for the next features.",
        "use": "Many related series with weekly or daily patterns, as in the M5 competition; usually a little more accurate than LightGBM on lags alone.",
        "caution": "Every series needs more than max(lag, 2m, 2·max(4, m)) observations. Like all tree models it does not extrapolate trends beyond the training range.",
        "engines": {PY: "Requires the lightgbm Python package. The C++ engine builds the same features and gives the same forecasts bit for bit. "
                        "This method does not exist in the R engine."},
        "params": LIGHTGBM_PARAMS, "refs": ["ke2017", "makridakis2022", "montero2021"], "cost": "About the cost of LightGBM; NN5 daily about 1 s in C++.",
    },
    "lightgbm_direct": {
        "name": "LightGBM, direct multi-horizon (M5-style features)", "group": "kuresel", "task": "global",
        "summary": "One global LightGBM model per forecast step, each predicting its own step directly from the same features.",
        "how": "The features of the M5-style LightGBM (lags, seasonal lags, rolling mean and standard deviation, position in the "
               "season) are taken at the forecast origin. For step k a separate model learns the value k steps after the origin, "
               "from every origin in the training data. No forecast is fed back as an input, so errors do not compound along the "
               "horizon (direct strategy; Taieb et al. 2012).",
        "use": "Long horizons, where recursive forecasts drift; compare it with the recursive M5-style LightGBM on the same data.",
        "caution": "Trains as many models as the horizon is long (56 for NN5 daily), so it is that many times slower than the "
                   "recursive version. Later steps learn from fewer rows. Every series needs more than max(lag, 2m, 2·max(4, m)) observations.",
        "engines": {PY: "Requires the lightgbm Python package. The C++ engine builds the same features and gives the same forecasts. "
                        "This method does not exist in the R engine."},
        "params": LIGHTGBM_PARAMS, "refs": ["taieb2012", "ke2017", "makridakis2022"],
        "cost": "About horizon × the cost of the recursive LightGBM.",
    },
    "nhits": {
        "name": "N-HiTS (deep learning)", "group": "kuresel", "task": "global",
        "summary": "A global neural network (MLP stacks with multi-rate pooling) trained on all series at once.",
        "how": "Windows of the last 'lag' observations of every series are robust-scaled and fed to N-HiTS, which forecasts the whole "
               "horizon in one step (no recursion), with hierarchical interpolation across scales. Trained with Adam on MAE for a "
               "fixed number of steps; the seed fixes the result on one machine.",
        "use": "Many series with long horizons; a modern deep-learning baseline.",
        "caution": "Takes tens of seconds to minutes on the CPU. Results can vary slightly between machines (floating point in PyTorch). "
                   "Series shorter than the window are zero-padded at the start.",
        "engines": {PY: "Requires torch and neuralforecast (optional extras; installed on this machine). Not in the R engine."},
        "params": [P("nhits.max_steps", "Training steps", "int", 500, "Number of training steps (mini-batches).", min=10, max=20000,
                     engines=("python",)),
                   P("nhits.learning_rate", "Learning rate", "float", 0.001, "Adam learning rate.", min=1e-5, max=0.1, step=0.0001,
                     engines=("python",), advanced=True)],
        "refs": ["challu2023"], "cost": "NN5 weekly, 111 series, 100 steps: about 25 s on the CPU.",
    },
    "nbeats": {
        "name": "N-BEATS (deep learning)", "group": "kuresel", "task": "global",
        "summary": "A global neural network of fully connected blocks with backward and forward residual links, trained on all series.",
        "how": "Windows of the last 'lag' observations of every series are robust-scaled; stacks of blocks (trend, seasonality and "
               "generic) each remove what they explain from the input and add their part to the forecast of the whole horizon. "
               "Trained with Adam on MAE for a fixed number of steps; the seed fixes the result on one machine.",
        "use": "The deep-learning model that beat the statistical benchmarks in M4; compare it with N-HiTS (its successor).",
        "caution": "Takes tens of seconds to minutes on the CPU. Results can vary slightly between machines (floating point in PyTorch). "
                   "Series shorter than the window are zero-padded at the start.",
        "engines": {PY: "Requires torch and neuralforecast (optional extras; installed on this machine). Not in the R engine."},
        "params": [P("nbeats.max_steps", "Training steps", "int", 500, "Number of training steps (mini-batches).", min=10, max=20000,
                     engines=("python",)),
                   P("nbeats.learning_rate", "Learning rate", "float", 0.001, "Adam learning rate.", min=1e-5, max=0.1, step=0.0001,
                     engines=("python",), advanced=True)],
        "refs": ["oreshkin2020"], "cost": "Similar to N-HiTS; NN5 weekly, 111 series, 200 steps: tens of seconds on the CPU.",
    },
    "chronos": {
        "name": "Chronos-Bolt (foundation model)", "group": "kuresel", "task": "global",
        "summary": "A pretrained time-series foundation model used zero-shot: no training on your data.",
        "how": "Each series (its last 2048 values at most) is given as context to Chronos-Bolt, a T5-based model pretrained on a large "
               "collection of real and synthetic series; it returns quantile forecasts for the whole horizon and the median is used.",
        "use": "A strong reference without any model fitting; useful to check how far fitted models are from a foundation model.",
        "caution": "Its pretraining corpus includes public benchmark data (M4 and Monash datasets among them), so scores on those "
                   "datasets may be optimistic (leakage) and must be reported as such. The weights (about 200 MB for 'small') are "
                   "downloaded once from Hugging Face. 'lag' is not used.",
        "engines": {PY: "Requires torch and chronos-forecasting (optional extras; installed on this machine). Not in the R engine."},
        "params": [P("chronos.model", "Model size", "choice", "amazon/chronos-bolt-small", "Larger models are slower and somewhat "
                     "more accurate.", choices=[("amazon/chronos-bolt-tiny", "tiny (9M)"), ("amazon/chronos-bolt-mini", "mini (21M)"),
                                                ("amazon/chronos-bolt-small", "small (48M)"), ("amazon/chronos-bolt-base", "base (205M)")],
                     engines=("python",))],
        "refs": ["ansari2024"], "cost": "NN5 weekly, 111 series: a few seconds after the first download.",
    },
}

# ----------------------------------------------------------------------------------------------------
METRICS_INFO = {
    # Prediction intervals (shown when a run asks for them; tsfstudio.intervals)
    "Coverage80": {"name": "Coverage (80%)", "unit": "share", "text": "Share of the test points inside the 80% prediction interval. Ideal: 0.80.",
                   "caution": "Coverage alone rewards wide intervals; read it together with MSIS. With few series the intervals rest on few calibration errors."},
    "Coverage95": {"name": "Coverage (95%)", "unit": "share", "text": "Share of the test points inside the 95% prediction interval. Ideal: 0.95.",
                   "caution": "Coverage alone rewards wide intervals; read it together with MSIS."},
    "MSIS": {"name": "MSIS", "unit": "scale-free", "text": "Mean scaled interval score of the 95% interval (the widest level when 95 is not requested), as in the M4 competition: interval width plus 2/α times every miss, divided by the in-sample seasonal-naive MAE (the MASE scale). Lower is better.",
             "caution": "Intervals are split-conformal: built from the model's own errors one horizon earlier, pooled across series, so every engine and method is treated alike (not the model's analytic interval)."},
    "WQL": {"name": "Weighted quantile loss", "unit": "ratio", "text": "Pinball loss of the interval bounds and of the point forecast (as the median), times 2, summed over the horizon and divided by the sum of |actual|; averaged over the quantiles (GluonTS mean wQL). Lower is better.",
            "caution": "Undefined when all actuals are 0. Uses the point forecast as the median."},
    "SMAPE": {"name": "SMAPE", "unit": "ratio (0–2)",
              "text": "Mean of 2·|forecast−actual| / (|forecast|+|actual|). Given as a ratio (0–2), NOT a percentage. It is undefined when forecast and actual are both 0, and that point is excluded.",
              "caution": "Misleading for intermittent demand: every point with forecast 0 and actual > 0 gives exactly 2; a model that forecasts all zeros scores exactly 2.0."},
    "mSMAPE": {"name": "mSMAPE", "unit": "ratio",
               "text": "A form of SMAPE corrected (epsilon = 0.1) so that its denominator does not blow up for small values.",
               "caution": "Reduces SMAPE's zero problem but does not remove it."},
    "MASE": {"name": "MASE", "unit": "scale-free",
             "text": "Mean absolute error / mean absolute error of the seasonal naive on the training part (lag = seasonal period). Below 1 means better than seasonal naive.",
             "caution": "If the whole training part is zero the denominator is 0 (the series is excluded). With intermittent demand it can reward zero forecasts."},
    "MAE": {"name": "MAE", "unit": "data unit", "text": "Mean absolute error.",
            "caution": "With intermittent demand it rewards the median (often 0) forecast; it can be meaningless for inventory."},
    "RMSE": {"name": "RMSE", "unit": "data unit", "text": "Square root of the mean of squared errors; penalises large errors more.",
             "caution": "Computed per series and then averaged across series; watch the scale effect when comparing series of different scales."},
    "WAPE": {"name": "WAPE", "unit": "percent", "text": "Total |error| / total |actual| × 100 (per series).",
             "caution": "Undefined if the total of actuals is 0. A model that forecasts all zeros scores exactly 100%."},
    "nRMSE_Mean": {"name": "nRMSE (mean)", "unit": "ratio", "text": "RMSE / mean of the actuals.", "caution": "Undefined if the mean is 0."},
    "nRMSE_Range": {"name": "nRMSE (range)", "unit": "ratio", "text": "RMSE / (maximum of actuals − minimum of actuals).", "caution": "Undefined if the range is 0."},
    # 2026-11-22: optional measures (not selected by default)
    "RMSSE": {"name": "RMSSE", "unit": "scale-free",
              "text": "Root of: mean squared error / mean squared one-step naive error on the training part (the M5 measure). Below 1 means better than the in-sample naive.",
              "caution": "The scale uses the whole training part; the official M5 definition starts at the first non-zero sale, which differs only for series with leading zeros."},
    "MSE": {"name": "MSE", "unit": "data unit²", "text": "Mean of squared errors.", "caution": "Scale-dependent; do not average across series of different scales."},
    "MAPE": {"name": "MAPE", "unit": "percent", "text": "Mean of |error| / |actual| × 100.",
             "caution": "Points with actual = 0 are excluded; asymmetric (penalises over-forecasts more). Not suitable for intermittent demand."},
    "MAAPE": {"name": "MAAPE", "unit": "radians (0–π/2)", "text": "Mean of arctan(|error| / |actual|); bounded, defined when the actual is 0.",
              "caution": "A zero forecast for a positive actual scores about 0.785 (π/4) per point, so it can still favour zero forecasts on intermittent data."},
    "MedAE": {"name": "MedAE", "unit": "data unit", "text": "Median absolute error over the horizon.", "caution": "Robust to a few large errors; scale-dependent."},
    "RMSLE": {"name": "RMSLE", "unit": "log ratio", "text": "Root mean squared difference of log(1 + forecast) and log(1 + actual).",
              "caution": "Undefined (series excluded) if any forecast or actual is negative; no clipping is applied."},
    "ME": {"name": "ME (bias)", "unit": "data unit", "text": "Mean of forecast − actual. Positive: over-forecasting.",
           "caution": "Positive and negative errors cancel; read together with an absolute measure."},
    "PBias": {"name": "Percent bias", "unit": "percent", "text": "100 · Σ(forecast − actual) / Σ actual.",
              "caution": "Undefined if the actuals sum to 0. Shows when a 'winning' model is simply biased toward zero."},
}

METRIC_REFS = ("hyndmankoehler2006", "hyndman2006", "makridakis2022", "kim2016", "davydenko2013")


# ----------------------------------------------------------------------------------------------------
def model_param_keys(model_id: str, engine: str | None = None) -> list[str]:
    return [p["key"] for p in MODELS[model_id]["params"] if engine is None or engine in p["engines"]]


# ------------------------------------------------------------------------------------------------ C++ engine
# The C++ engine (engines/cpp_engine.py, native library tsfcore) offers the methods in CppEngine.LOCAL_METHODS /
# GLOBAL_METHODS. Its notes say what it reproduces (cpp/README.md has the full validation); parameters are offered in
# C++ only where the C++ engine accepts every value (elsewhere it runs the method's defaults and the setting is hidden).
CPP = "cpp"
_CPP_R = "Native C++ port of the R code; forecasts equal R's to rounding (cpp/README.md)."
_CPP_PY = "Native C++ port of the Python engine's statsforecast model; forecasts equal the Python engine's to rounding."
CPP_NOTES = {
    "naive": "Native C++; same rule as the Python engine.", "snaive": _CPP_R, "drift": "Native C++; same rule as the Python engine.",
    "ses": _CPP_R, "theta": _CPP_R, "naive2": _CPP_R, "comb_shd": _CPP_R,
    "ets": "Native C++ port of R forecast::ets (incl. its small-sample branch and class-3 mean forecasts); equals R to rounding.",
    "holt_winters": _CPP_R + " Additive errors only (the default).", "holt_ets": _CPP_R,
    "arima": "Native C++ port of R auto.arima with the study's call chain; equals R to rounding.",
    "dhr_arima": _CPP_R + " K = 1 Fourier terms (the study's setting).",
    "tbats": "Native C++ port of R forecast::tbats; equals the study's R forecasts to rounding.",
    "stl_ets": _CPP_R, "stl_arima": _CPP_R,
    "bagged_ets": _CPP_R + " Runs n_boot = 100 and seed 0 (the defaults); other values need the Python engine.",
    "thief": _CPP_R, "mapa": _CPP_R,
    "comb_eat": "Native C++: mean of the C++ ets and arima (both reproduce R, so the result differs slightly from the Python engine, which uses statsforecast) and theta.",
    "comb_eat_median": "Native C++: median of the C++ ets, arima and theta (R-based ETS / ARIMA, so it differs slightly from the Python engine).",
    "scum": "Native C++: median of the C++ ets, auto_ces, arima and dotm (R-based ETS / ARIMA, so it differs slightly from the Python SCUM).",
    "mstl": _CPP_PY + " The trend forecaster is the C++ (R) ETS.",
    "pooled_regression": "Native C++ least squares (streamed QR); equals R glm to rounding.",
    "catboost": "C++ drives the official CatBoost 1.2.10 programs (CatBoost's defaults); the same forecasts as the Python engine.",
    "lightgbm": "C++ drives LightGBM's C library with the Python engine's parameters; the same forecasts bit for bit.",
    "lightgbm_feat": "C++ builds the same features and drives LightGBM's C library; the same forecasts bit for bit.",
    "lightgbm_direct": "C++ builds the same features and trains one booster per step through LightGBM's C library; the same forecasts as Python.",
}
CPP_PARAMS = {"pooled_regression": {"max_obs"}, "catboost": {"max_obs"}, "lightgbm": None, "lightgbm_feat": None, "lightgbm_direct": None}  # None: all


def _wire_cpp() -> None:
    from ..engines.cpp_engine import CppEngine
    for mid in tuple(CppEngine.LOCAL_METHODS) + tuple(CppEngine.GLOBAL_METHODS):
        if mid not in MODELS:
            continue
        MODELS[mid]["engines"].setdefault(CPP, CPP_NOTES.get(mid, _CPP_PY + " Runs the method's default settings."))
        allowed = CPP_PARAMS.get(mid, set())
        for p in MODELS[mid]["params"]:
            if (allowed is None or p["key"] in allowed) and CPP not in p["engines"]:
                p["engines"].append(CPP)


_wire_cpp()


def build_catalog(engine_methods: dict[str, dict | None]) -> list[dict]:
    """Model list for the interface. `engine_methods`: engine -> {"local": (...), "global": (...)} (unavailable engine: None).

    A method that the engine does not actually offer is marked "available: false"; models that the engine does not know but that are in the registry are listed too."""
    out = []
    for mid, info in MODELS.items():
        engines = {}
        for eng in (PY, R, CPP):
            ms = engine_methods.get(eng)
            avail = bool(ms) and mid in tuple(ms.get(info["task"], ()))
            note = info["engines"].get(eng)
            if eng not in info["engines"]:
                note = "Not available in this engine."
            engines[eng] = {"available": avail, "note": note, "supported": eng in info["engines"]}
        item = {k: v for k, v in info.items() if k not in ("engines", "refs")}
        item["id"] = mid
        item["engines"] = engines
        item["refs"] = [dict(REFS[r], key=r) for r in info["refs"]]
        out.append(item)
    return out
