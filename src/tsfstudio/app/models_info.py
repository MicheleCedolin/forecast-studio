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
    "pooled_regression": {
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
}

# ----------------------------------------------------------------------------------------------------
METRICS_INFO = {
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
}

METRIC_REFS = ("hyndmankoehler2006", "hyndman2006")


# ----------------------------------------------------------------------------------------------------
def model_param_keys(model_id: str, engine: str | None = None) -> list[str]:
    return [p["key"] for p in MODELS[model_id]["params"] if engine is None or engine in p["engines"]]


def build_catalog(engine_methods: dict[str, dict | None]) -> list[dict]:
    """Model list for the interface. `engine_methods`: engine -> {"local": (...), "global": (...)} (unavailable engine: None).

    A method that the engine does not actually offer is marked "available: false"; models that the engine does not know but that are in the registry are listed too."""
    out = []
    for mid, info in MODELS.items():
        engines = {}
        for eng in (PY, R):
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
