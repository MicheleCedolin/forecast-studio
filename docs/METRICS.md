# Metrics: the original R definitions and the differences in this application

Source: `reference/utils/error_calculator_original.R` (SMAPE, mSMAPE, MASE, MAE, RMSE) and
`reference/utils/error_calculator.R` (WAPE, nRMSE). Implementation: `src/tsfstudio/metrics.py`.
Each formula is tested against hand-computed values in `tests/test_metrics.py`.

## Definitions (per series; ε = 0.1)

| Metric | Definition | Unit |
|---|---|---|
| SMAPE | mean( 2·\|f−a\| / (\|f\|+\|a\|) ) | ratio, range 0–2 (**not** ×100) |
| mSMAPE | mean( 2·\|f−a\| / max(0.5+ε, \|f\|+\|a\|+ε) ) | ratio |
| MASE | mean(\|f−a\|) / mean(\|Δ_m y_train\|), m = ⌊min(seasonality)⌋; if the scale is undefined, m = 1 | – |
| MAE, RMSE | standard | data unit |
| WAPE | Σ\|f−a\| / Σ\|a\| × 100 | **percent** |
| nRMSE_Mean / nRMSE_Range | RMSE / mean(a)  ;  RMSE / (max(a)−min(a)) | ratio |

Note: SMAPE/mSMAPE are ratios (0–2) and WAPE is a percentage, as in the original code; mind the units when comparing.

## Deliberate differences from the original

1. **Alignment.** In the original, `calculate_mase` dropped NaN/Inf series from the vector while the other metrics did not;
   inside `data.frame(...)` the columns shifted (`error_calculator.R`) or were recycled. Here an undefined series value
   **stays in place as NaN**; the summary (`N_valid`, `N_excluded`) shows how many series were left out.
   Mean/median are taken over finite values only — the same result as the original MASE summary.
2. **Zero denominator.** For WAPE/nRMSE, R produces `Inf` and `mean(na.rm=TRUE)` turns the summary into `Inf`; here the value is `NaN` and is left out.
3. **Pairs containing NaN.** In MASE, filtering forecasts and actuals separately could give vectors of different lengths; the pairs are filtered together.
   On data without NaN the result is identical.
4. **MASE lag.** In R, `diff(lag = min(seasonality))` truncates a fractional lag (365.25/7 → 52); it is truncated the same way here.
   If the seasonal scale is `NaN` (series shorter than the lag, or 0/0), lag 1 is tried — exactly the `is.na(mase)` logic of the original.
5. **Rounding.** The old `smooth::MASE` rounded the result to 3 digits; current greybox does not round. There is no rounding here
   (it can be switched on with `mase_digits=3`).

## Problems found in the original code and fixed here

| File | Problem | Fix |
|---|---|---|
| `fixed_horizon_functions.R` (local model) | Forecasts are written as `.xlsx`, but the error calculation still reads `<data>_<method>.txt` → error when the loop ends | R engine with a corrected driver (Phase B) |
| `error_calculator.R` | `calculate_smape/msmape/mase/mae/rmse` undefined (only in `_original.R`) | A single Python module |
| `error_calculator_full.R` | `calculate_mape` undefined; `Theils_U` row count mismatch; naive baseline broken for fractional/vector seasonality | Extended measures added on 2026-11-22 without these bugs (MAPE among them; see "Optional measures" below). Theil's U is not implemented |
| `fixed_horizon_functions.R` | The assignment `forecast_horizon <- 1` affects the following series; silent SNaive fallback and NA→0 | Fallback use is counted and reported per series (`fallbacks.csv`) |
| `rolling_origin.R` | Sourcing the file runs the example runs; fixed Colab `BASE_DIR` | Functions kept separate, path is a parameter |
| `data_loader.py` | `distutils.util.strtobool` is missing in Python ≥ 3.12 | Local `strtobool` in `tsfstudio.tsf.reader` |

## Optional measures (added 2026-11-22)
The default set is unchanged (it reproduces the R error files). These can be selected in the UI or passed via `RunConfig.metrics`:
`RMSSE` (M5, scale = in-sample one-step squared naive error over the whole training part), `MSE`, `MAPE` (points with actual 0 excluded),
`MAAPE` (Kim & Kim 2016, radians), `MedAE`, `RMSLE` (series with negative values -> NaN, no clipping), `ME` (bias), `PBias`.
Dataset-level comparisons across models (OWA, AvgRelMAE, Friedman / Nemenyi, bootstrap rank stability) are in `tsfstudio.ranking`.
Formulas and references: `docs/RESEARCH_MODELS_METRICS.md`.

## Further measures as implemented

Per series, f = forecast, a = actual, over the test horizon; undefined values are NaN and are counted in `N_excluded`.
The point measures are in `src/tsfstudio/metrics.py` (`EXTRA_METRICS`), the interval measures in `src/tsfstudio/intervals.py`
and WRMSSE in `src/tsfstudio/wrmsse.py`.

### Point measures (`EXTRA_METRICS`)

| Metric | Definition as implemented | Unit / undefined case |
|---|---|---|
| RMSSE | sqrt( mean((f−a)²) / mean((y_t − y_{t−1})²) ), denominator over the whole training part | – ; NaN if the scale is 0 or the training part has < 2 values. The official M5 scale starts at the first non-zero sale; this differs only for series with leading zeros |
| MSE | mean((f−a)²) | data unit² |
| MAPE | 100 · mean(\|f−a\| / \|a\|) | percent; points with a = 0 are excluded |
| MAAPE | mean(arctan(\|f−a\| / \|a\|)) | radians, 0–π/2; a = 0 and f ≠ 0 gives π/2, a = f = 0 gives 0 |
| MedAE | median(\|f−a\|) | data unit |
| RMSLE | sqrt( mean((ln(1+f) − ln(1+a))²) ) | log ratio; NaN for a series with any negative forecast or actual (no clipping) |
| ME | mean(f−a) | data unit; positive = over-forecasting |
| PBias | 100 · Σ(f−a) / Σa | percent; NaN if Σa = 0 |

### Interval measures (`intervals.py`)

Computed when a run asks for prediction intervals (`RunConfig.intervals`, whole percentages between 50 and 99). The intervals are
split conformal: the model is also fitted one horizon earlier; its errors on that window, scaled per series by the MASE scale
(in-sample seasonal-naive MAE, lag 1 if undefined), are pooled across series per step h, and their empirical quantiles q_p(h)
give bound_p = f + q_p(h) · scale.

| Metric | Definition as implemented |
|---|---|
| Coverage\<level\> (e.g. Coverage80, Coverage95) | share of the test points with L ≤ a ≤ U; ideal = level / 100 |
| MSIS | mean_h[(U − L) + 2/α (L − a) 1{a < L} + 2/α (a − U) 1{a > U}] / scale, for the 95% interval (the widest level if 95 is not requested) |
| WQL | mean over the quantiles q (the interval bounds and the point forecast as the median) of 2 · Σ_h pinball_q / Σ_h \|a\|; NaN if Σ\|a\| = 0 |

### WRMSSE (`wrmsse.py`)

The M5 measure, computed for data with the M5 hierarchy attributes (`item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`). The bottom
series are summed into the 12 levels of the M5 hierarchy. For each aggregated series, RMSSE = sqrt( mean_h (a − f)² / mean_{t≥2} (y_t − y_{t−1})² ),
with the denominator over its training part from its first non-zero value on. Within a level, series are weighted by their share of
the level's sales over the last 28 training days; WRMSSE is the mean of the 12 level values. With M5's `sell_prices.csv` and
`calendar.csv` the weights are dollar sales (the official definition); otherwise they are unit sales (the manifest records the weighting as
"unit sales"), which is not the official figure. A run writes the per-level values to `wrmsse.csv`.
