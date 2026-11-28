# Python engine (`--engine python`)

Needs neither R nor Docker. Runs with `pip` on a local Windows/Linux/macOS machine.

## Methods of the original R study

This table covers only the methods that have a counterpart in the original R study (`src/tsfstudio/r/lib/local_models.R`,
`global_models.R`). The full list of Python-engine methods is under [All methods](#all-methods) below; the source of truth is the
`engines` field of each model in `src/tsfstudio/app/models_info.py`. `python -m tsfstudio engines` shows which methods are unavailable on a
machine because a package is missing, and `python -m tsfstudio doctor` runs a short live test of every available method.

| Method | Type | Needs | R counterpart | Relation to R | Validation |
|---|---|---|---|---|---|
| `naive`, `snaive` | local | – | `snaive(msts)` | same (frequency = `floor(max(period))`) | unit test |
| `ses` | local | – | `forecast::ses` | **same algorithm** (ML, α ∈ [1e-4, 0.9999], initial level in closed form) | NN5 golden: median difference ~4×10⁻⁶ |
| `theta` | local | – | `forecast::thetaf` | **same algorithm** (ACF seasonality test, multiplicative decomposition, SES + half slope) | NN5 golden: median difference ~4–8×10⁻⁶ |
| `pooled_regression` | global | – | `glm` | **same result** (streaming QR; no memory problem even for M5 L12) | NN5 golden: 5×10⁻¹⁵ |
| `ets` | local | statsforecast | `forecast::ets` | similar, **not identical** (AutoETS); frequency > 24 → non-seasonal (as in R) | `golden` on the user's machine (VALIDATION 1c) |
| `arima` | local | statsforecast | `auto.arima` | similar (AutoARIMA); R chain: log → raw → non-seasonal; `allowdrift`, `approximation` as in R | `golden` (VALIDATION 1c) |
| `dhr_arima` | local | statsforecast | `auto.arima(xreg=fourier)` | similar; Fourier K=1 scalar, applied to every period on multi-seasonal data (R may fail) | `golden` (VALIDATION 1c) |
| `tbats` | local | statsforecast | `forecast::tbats` | **not the same model** (AutoTBATS); periods rounded to integers. Disabled on 2026-11-03 (user decision), offered again since 2026-11-25 so that every C++ method has its Python library counterpart; the interface states the difference | NN5 weekly, 8 series: median forecast difference 3.1%. For R's TBATS use the R or the C++ engine |
| `catboost` | global | catboost | `catboost.train` | same core and defaults; `random_seed` default 0 (as in R) | no `golden` (there is no R CatBoost output for NN5); the C++ engine gives bit-identical forecasts (`cpp/README.md`, row 6b) |

Methods that are "not installed" do not appear in the list and give a clear error if called.

## statsforecast models of phase C3 (intermittent demand and further classical models)
The goal is not to translate R into Python (user decision, 2026-11-03); these are statsforecast's own models and cannot be tested with `golden`
(there are no R outputs of the study for them). Since 2026-11-25 the R engine has R library counterparts for `croston_classic`, `croston_sba`,
`croston_optimized`, `tsb`, `imapa`, `auto_ces` and `holt_winters` (`src/tsfstudio/r/lib/extra_models.R`); these are other implementations,
and their differences are stated in `models_info.py` (`R_NOTES`). `adida`, `auto_theta`, `mstl` and `auto_mfles` have no R counterpart.
Validation: `doctor` (does it run, is it finite, is there a fallback), the C++ ports listed in `cpp/README.md`, and inspection of the results on
your own data. **Defaults marked "our choice" are not evidence-based recommendations; tune them on your data.** Parameters are passed with
`--param key=value`.

| Method | Class | Parameters (default) | Note |
|---|---|---|---|
| `croston_classic` | `CrostonClassic` | – | Flat (constant) forecast: demand size / demand interval |
| `croston_sba` | `CrostonSBA` | – | Croston with the Syntetos–Boylan bias correction |
| `croston_optimized` | `CrostonOptimized` | – | The smoothing parameter is optimised by the library |
| `tsb` | `TSB` | `alpha_d`=0.1; `alpha_p`=0.1 (our choice) | Updates the demand probability every period (sensitive to obsolescence risk) |
| `adida` | `ADIDA` | – | Temporal aggregation + disaggregation |
| `imapa` | `IMAPA` | – | Multiple aggregation (intermittent MAPA) |
| `auto_theta` | `AutoTheta` | `decomposition_type`="multiplicative" | Automatic choice among Theta variants. Switches to additive if there are zero/negative values (a note is written). Non-seasonal if there are fewer than 2 seasons of data |
| `auto_ces` | `AutoCES` | `model`="Z", `max_season`=24 (our choice) | Complex exponential smoothing. Non-seasonal if the period > `max_season` (similar to the R `ets` rule; can be changed) |
| `holt_winters` | `HoltWinters` | `error_type`="A", `max_season`=24 (our choice) | Falls back to Holt if non-seasonal (a note is written). `error_type="M"` needs all values > 0 |
| `mstl` | `MSTL` | – | Multi-seasonal decomposition; periods are rounded to integers, those ≥ 2 with at least two seasons of data are used; if none remains the series gets the fallback (expected for yearly data). The trend forecaster is the library default (AutoETS, non-seasonal) |
| `auto_mfles` | `AutoMFLES` | `n_windows`=2, `metric`="smape" | **Needs scikit-learn** (statsforecast does not declare it; if it is not installed the method is not listed). `test_size` = horizon (required by the library); no peer-reviewed reference / not validated |

* **No silent dropping:** for these models, a key that your version does not recognise raises an error (the series gets the fallback and the reason is written).
* **Zeros:** on an all-zero series the intermittent models forecast 0 directly (the library can produce 0/0).
* **Measure warning:** on intermittent series sMAPE/MAPE are undefined or misleading; use scale-based measures (MASE, RMSSE and, for M5, WRMSSE).
  If the whole training part is zero, the MASE denominator is zero.
* **Run time:** there are no run-time priors for the new methods; try `estimate`/a run on a small sample of series first (especially `auto_mfles`, `auto_ces`, `mstl`).
* **Sources (checked against bibliographic records; full citations in `models_info.py`, `REFS`):** Croston (1972), *ORQ* 23(3); Syntetos & Boylan (2005), *IJF* 21(2);
  Teunter, Syntetos & Babai (2011), *EJOR* 214(3) [TSB]; Nikolopoulos et al. (2011), *JORS* 62(3) [ADIDA]; Petropoulos & Kourentzes (2015),
  *JORS* 66(6) [IMAPA]; Assimakopoulos & Nikolopoulos (2000), *IJF* 16(4) [Theta]; Svetunkov, Kourentzes & Ord (2022), *NRL* 69(8) [CES];
  Winters (1960), *Management Science* 6(3); Bandara, Hyndman & Bergmeir (2025), *International Journal of Operational Research* 52(1), preprint arXiv:2107.13462 (2021) [MSTL].
  Library: Nixtla StatsForecast (open source, Apache-2.0).

## Rounding to integers and intermittent demand (important)
In the catalogue, datasets with count data (car_parts, M5, etc.) have `integer_conversion: true` (as in your own study): forecasts are rounded.
Intermittent-demand methods, however, forecast a flat demand **rate** (e.g. 0.3 units per period); rounding turns this into 0. This is what
the first `car_parts` trial showed (`croston_sba`, 100 series): all forecasts 0 → sMAPE = 2.0 and WAPE = 100% (all exact). Use
`--integer-conversion off` with intermittent-demand methods; if you leave it on, the runner writes a warning. `auto` (catalogue) remains for comparison with earlier studies.

**Measure note:** measures based on the mean absolute error (MAE, MASE, WAPE) reward the median forecast (often 0) on an intermittent series; a zero
forecast can look good on these measures but is meaningless for inventory (Hyndman, 2006, *Foresight* 4(4), 43–46).
On intermittent series, bias and inventory outcomes (service level, stock cost) should be reported in addition to RMSE/RMSSE; this is exactly the
analysis the project theme (decision impact rather than accuracy) calls for, and the first candidate for the inventory extension left out of v1.

## Deliberate differences
* **Error semantics:** if the model fails, the series gets `snaive` and this is **counted** (`fallbacks.csv`). If the model produces NaN/Inf, R sets it to 0;
  here the series also gets the fallback forecast (instead of inventing 0). `na_replaced` is always 0 in the Python engine.
* **SES α bound:** on some series (where the α optimum is at the lower bound) R's Nelder-Mead stops at a point with a higher error; Python finds the true optimum.
  On NN5, 4–6 of the 111 series differ by > 10⁻³ (at most 17%), **but the effect on mean SMAPE/MASE is < 0.4%** (see VALIDATION.md).
* **DHR-ARIMA:** on a multi-seasonal `msts`, R `fourier(K=1)` may fail depending on the version and fall back to SNaive; Python produces the terms.
* **ARIMA, season > 200:** falls back to non-seasonal (changed with `arima_max_season`). R's behaviour here has not been verified.
* **Parallelism:** process pool with `--workers N` (`spawn` on Windows; R's `mclapply` does not run in parallel on Windows).

## Validation commands
```powershell
.\scripts\windows\tsfs.ps1 doctor     # short live test of every method
.\scripts\windows\tsfs.ps1 golden     # comparison with your own R outputs (NN5 weekly + daily), saved with --csv
```

## Added 2026-11-22: M3/M4 benchmarks and SCUM

| Method | Type | Needs | Source | Relation to R | Validation |
|---|---|---|---|---|---|
| `drift` | local | - | `rwf(drift = TRUE)` | same closed form | equals statsforecast `RandomWalkWithDrift` to 1e-10 (test) |
| `naive2` | local | - | M4 benchmark code (Makridakis et al. 2020) | seasonality test and `decompose()` follow the M4 R code | **identical** to R on NN5 weekly and daily (golden test, VALIDATION 1d) |
| `comb_shd` | local | - | M4 benchmark "Comb" (SES + Holt + damped Holt) | same model as `holt()` (ets AAN/AAdN, least squares, ets bounds); initial state solved in closed form | NN5: mean SMAPE/MASE within 0.4% of R; fit at least as good as R's on every series (golden test, VALIDATION 1d) |
| `dotm` | local | statsforecast | Fiorucci et al. 2016 | `DynamicOptimizedTheta` (signature checked on 2.1.1) | runs; no R reference |
| `scum` | local | statsforecast | Petropoulos & Svetunkov 2020 | median of ETS, CES, ARIMA, DOTM from this engine | median identity test; a failed component is reported in notes |

Ranking tools for comparing several runs: `python -m tsfstudio rank --run ets=runs/<dir> --run theta=runs/<dir> ... --metric MASE`
(mean ranks, Friedman, Nemenyi critical difference, bootstrap P(first) / P(top k) and Kendall tau). Module: `src/tsfstudio/ranking.py`.

## All methods

The 55 methods that `models_info.py` lists for the Python engine (`PythonEngine.LOCAL_METHODS` / `GLOBAL_METHODS` in
`src/tsfstudio/engines/py_engine.py`). "Needs" names the Python packages beyond numpy/scipy; the extras are defined in `pyproject.toml`
(`pyengine`: statsforecast, catboost, lightgbm, scikit-learn; `deep`: torch, neuralforecast, gluonts[torch], chronos-forecasting).
"R engine" says whether the R engine also offers the method. "Validation" gives a recorded check or says "not recorded here"; rows of
`cpp/README.md` compare the C++ port with this engine's forecasts.

| Method | Type | Needs | R engine | Validation |
|---|---|---|---|---|
| `naive`, `snaive` | local | – | yes | unit test |
| `ses`, `theta` | local | – | yes | NN5 golden (VALIDATION 1b, 1c) |
| `drift`, `naive2`, `comb_shd` | local | – | yes | see the table "M3/M4 benchmarks" above (VALIDATION 1d) |
| `pooled_regression` | global | – | yes | NN5 golden (VALIDATION 1b) |
| `ets`, `arima`, `dhr_arima` | local | statsforecast | yes | VALIDATION 1c |
| `tbats` | local | statsforecast | yes | NN5 weekly, 8 series: median forecast difference 3.1% from R |
| `croston_classic`, `croston_sba`, `croston_optimized`, `tsb`, `imapa` | local (intermittent) | statsforecast | yes | C++ port bit-identical to this engine (`cpp/README.md`, row 4) |
| `adida` | local (intermittent) | statsforecast | no | C++ port bit-identical to this engine (`cpp/README.md`, row 4) |
| `zero_model` | local (intermittent) | statsforecast | no | C++ port bit-identical to this engine (`cpp/README.md`, row 7) |
| `dotm`, `stm`, `otm`, `dstm` | local | statsforecast | yes | C++ port equal to this engine to machine precision (`cpp/README.md`, rows 4b, 8b) |
| `auto_theta` | local | statsforecast | no | C++ port equal to this engine to machine precision (`cpp/README.md`, row 4b') |
| `auto_ces` | local | statsforecast | yes | C++ port equal to this engine (`cpp/README.md`, row 4c) |
| `holt_winters` | local | statsforecast | yes | differs from R by ~1% on every NN5 daily series (`cpp/README.md`, row 4a) |
| `holt_ets` | local | statsforecast | yes | statsforecast's Holt differs from R by ~0.3% (`cpp/README.md`, row 7) |
| `historic_average` | local | statsforecast | yes | C++ port bit-identical to this engine (`cpp/README.md`, row 7) |
| `window_average`, `seasonal_window_average`, `seasonal_es` | local | statsforecast | no | C++ port bit-identical to this engine (`cpp/README.md`, row 7) |
| `mstl` | local | statsforecast | no | decomposition equal to statsforecast in the C++ port (`cpp/README.md`, row 4e) |
| `auto_mfles` | local | statsforecast, scikit-learn | no | C++ port equal to this engine to machine precision (`cpp/README.md`, row 4f) |
| `scum` | local | statsforecast | yes | median identity test; against the C++ (R-based) variant: `cpp/README.md`, row 4d |
| `comb_eat`, `comb_eat_median` | local | statsforecast | yes | not recorded here |
| `stl_ets`, `stl_arima` | local | statsforecast | yes | decomposition bit-identical to R's `stl()` and to the C++ engine (`cpp/README.md`, row 7c); forecasts close to, not equal to, R |
| `bagged_ets` | local | statsforecast | yes | bootstrap series equal to R's (`cpp/README.md`, row 7d); forecasts close to, not equal to, R |
| `thief`, `mapa` | local | statsforecast | yes | not recorded here (R algorithm with statsforecast AutoETS; close to, not equal to, R) |
| `catboost` | global | catboost | yes | C++ engine bit-identical (`cpp/README.md`, row 6b) |
| `lightgbm`, `lightgbm_feat` | global | lightgbm | no | C++ engine bit-identical (`cpp/README.md`, rows 7b, 7g) |
| `lightgbm_direct` | global | lightgbm | no | not recorded here |
| `nhits`, `nbeats`, `lstm`, `rnn`, `gru`, `deepar` | global | torch, neuralforecast | no | not recorded here |
| `deepar_gluonts`, `feed_forward`, `wavenet` | global | gluonts[torch] | no | not recorded here |
| `chronos` | global | torch, chronos-forecasting | no | not recorded here |
