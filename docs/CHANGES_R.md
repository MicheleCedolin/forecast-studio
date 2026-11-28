# Changes to the R code

The sources under `reference/` are kept **unchanged**. The copies under `src/tsfstudio/r/` keep the modelling decisions
(model calls, parameters, normalisation, recursion); only the points below differ.

| # | Original | Change | Effect on results |
|---|---|---|---|
| 1 | `tryCatch(..., error = function(e) SNaive)` was silent | Every fallback is counted and its message recorded (`fallbacks.csv`, manifest) | Forecasts unchanged; only reported. Warning if the fallback rate is >50% |
| 2 | NA forecasts silently set to 0 | Same behaviour kept, but counted (`na_replaced`) | None |
| 3 | `warning(e)` filled the log | Removed; the message is in the fallback record | None |
| 4 | `create_input_matrix`: `rbind` inside the loop (quadratic), `print(i)` | Embedded matrices are collected in a list and joined with a single `rbind` | Identical matrix, much faster |
| 5 | Unclear error on a very short series | An error that names the problem series | None |
| 6 | Memory blow-up on large data (M5) | Optional `max_obs` (default off = original): the last `max_obs` observations of each series; mean normalisation over this window | None by default |
| 7 | CatBoost ran with the `catboost.train` defaults | Can be overridden with `params.catboost` (empty = the same defaults) | None by default |
| 8 | `set.seed(1)` at the top of the file | The seed is set explicitly for every run with `job.seed` (default 1) | None by default |
| 9 | Overwriting `forecast_horizon <- 1`, example runs when `rolling_origin.R` is sourced, `os.system("rm")`, fixed `BASE_DIR`, `distutils.strtobool` | Not carried over; replaced by the worker protocol and the Python side | None |

## Open point on DHR-ARIMA

The original call `fourier(ts, K = 1)` uses a scalar K. For a multi-seasonal `msts` (e.g. hourly: 24, 168, 8766), the `forecast`
package requires K to have one entry per seasonal period; otherwise it fails → **silently SNaive**. The worker keeps the original
behaviour and allows changing it with `params.fourier_k` / `params.fourier_k_per_period`. Whether this happens with your package
versions is shown in `fallbacks.csv`; if it does, part of the DHR-ARIMA results on multi-seasonal data are in fact SNaive.

## Difference from the Python side

`PythonBaselineEngine.snaive` (engine `python-baseline`) uses the first seasonal period; R's `snaive(msts)` uses the frequency `floor(max(period))`.
Results differ on multi-seasonal data; for comparisons the R engine is the reference.

## Counterparts in the Python engine
Details: `docs/PYTHON_ENGINE.md`. Summary: SES, Theta and pooled regression are implemented with exactly the R algorithm (NN5 golden tests);
ETS/ARIMA/TBATS/DHR-ARIMA are similar-but-not-identical counterparts through statsforecast. In the Python engine NaN/Inf output is not set to 0;
the series gets the fallback forecast instead. In DHR-ARIMA a scalar K is applied to every period on multi-seasonal data.
The C++ engine (`cpp/README.md`) ports the R calls of ets, arima, dhr_arima and tbats and reproduces the R forecasts to rounding
(`src/tsfstudio/app/models_info.py`, `CPP_NOTES`).

### TBATS periods (Python engine)
R `tbats()` accepts real-valued periods (e.g. 52.18; 365.25). According to its documentation, statsforecast `AutoTBATS(season_length: int | list[int])`
requires integers; the adapter rounds the periods to the nearest integer (52.18 → 52; 365.25 → 365). This is a deliberate difference; its effect is measured with `golden`.

## Methods added to the R engine (2026-11-25)
`src/tsfstudio/r/lib/extra_models.R` adds 30 local methods to the nine of the original study (`local_models.R`, `global_models.R`), which
gives the R engine 39 methods. They are not copies of the original code: each is the R counterpart of a method that the C++ and Python
engines offer, either the R call that the C++ method ports ("REFERENCE") or an R library function for the same method ("LIBRARY",
forecasts close to but not equal to the other engines). Unlike `local_models.R`, these functions do not catch errors; `worker.R` records
the failure and gives the series the seasonal naive forecast, the same rule as the C++ and Python engines. Methods that need R packages
beyond `forecast` (`extra_packages`: thief, MAPA, smooth, tsintermittent, forecTheta) are listed only when those packages are installed.
