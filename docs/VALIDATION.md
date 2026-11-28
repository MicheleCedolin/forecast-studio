# Validation

## 1. Metrics — against the user's real R outputs (done)
For NN5 weekly (111 series, training 105, horizon 8) and NN5 daily (111 series, training 735, horizon 56), the user's R error
files (`smape, msmape, mase, mae, rmse`; per series) against `tsfstudio.metrics`:

* 11 models × 5 metrics × 111 series: largest relative difference **≈5×10⁻¹⁵** (machine precision).
* `MAE, RMSE, WAPE, NRMSE_mean, NRMSE_range` and `sMAPE` (×100) in the extended summary file, over 23 model/data pairs: **≈5×10⁻¹⁶**.
* MASE lag: weekly 52 (from 365.25/7), daily 7.

Automated test: `tests/test_golden_nn5.py` (`tests/golden/nn5_weekly/`, 412 KB at the time; later R reference fixtures have been added to the same folder). Regeneration: `tools/make_nn5_fixture.py`.

## 1b. Python engine — against the user's R forecasts (done, in this environment)
`python -m tsfstudio golden` / `tests/test_golden_python.py` (NN5 weekly h=8 and daily h=56, 111 series):

| Data | Method | median rel. diff. | series differing by >1e-3 | ΔSMAPE (rel.) | ΔMASE (rel.) |
|---|---|---|---|---|---|
| weekly | pooled_regression (lag 65) | 1×10⁻¹⁵ | 0/111 | 0 | 0 |
| weekly | ses | 4×10⁻⁶ | 6/111 | +0.27% | +0.36% |
| weekly | theta | 4×10⁻⁶ | 5/111 | +0.06% | +0.08% |
| daily | pooled_regression (lag 9) | 0 | 0/111 | 0 | 0 |
| daily | ses | 1.5×10⁻⁵ | 6/111 | +0.18% | +0.18% |
| daily | theta | 8×10⁻⁶ | 4/111 | −0.32% | −0.39% |

Cause of the deviation: in **all** deviating SES series, Python's best α is at the lower bound (1×10⁻⁴), while R's stopped at a higher SSE
(0.2–41% higher on 6 weekly series); that is, R's Nelder-Mead does not find the boundary optimum. This is tested by
`test_ses_deviations_occur_only_at_alpha_lower_bound`. Run time: < 0.5 s each for 111 series (7–11 s in your R log).

At the time of these runs the statsforecast-based methods **could not be run** in the development environment; the adapter logic (R chain,
frequency rules, Fourier regressors, fallback counting) is tested with a fake statsforecast module. The real validation is done with `golden` on the user's machine (1c).

### 1c. statsforecast — on the user's Windows machine, NN5 weekly (h=8, 111 series, Python 3.12, single core)
Source: the user's `golden` output (2026-11-03). The fallback (SNaive) count is 0 in every row, **except TBATS**.

| Method | median rel. diff. | series differing by >1e-3 | ΔSMAPE (rel.) | ΔMASE (rel.) | time (Python / R log) |
|---|---|---|---|---|---|
| ses | 4.4×10⁻⁶ | 6/111 | +0.27% | +0.36% | 0.25 s / 8 s |
| theta | 4.0×10⁻⁶ | 5/111 | +0.06% | +0.08% | 0.29 s / 9 s |
| pooled_regression | 1.4×10⁻¹⁵ | 0/111 | ≈0 | ≈0 | 0.03 s |
| ets (`AutoETS`) | 4.7×10⁻⁵ | 21/111 | −1.55% | −1.38% | 0.9 s / 11 s |
| arima (`AutoARIMA`) | 2.4×10⁻³ | 70/111 | −0.07% | −0.33% | 25 s / 41 s |
| dhr_arima | 4.2×10⁻³ | 96/111 | −0.11% | −0.40% | 50 s / 42 s |
| tbats | **INVALID** | – | – | – | 111/111 series fell back |

Reading: (i) the SES/Theta/pooled regression results on Windows + Python 3.12 are **the same** as the numbers in the development environment (platform-independent).
(ii) The ETS/ARIMA/DHR-ARIMA forecasts differ from R *at the series level* (differences in model selection/optimisation: the `statsforecast` ARIMA gave many
"possible convergence problem: minimize gave code 2" warnings), but the *aggregate accuracy measures* move by less than 1.6%, and the sign always favours
Python. This means "equivalent in accuracy", not "exact reproduction"; a paper should state the engine name and version.
For series-level equivalence a paired test (e.g. Wilcoxon on the per-series SMAPE difference) is recommended; it has not been done yet.
(iii) The TBATS row was a **bug**: the `AutoTBATS` signature expects `season_length`, the adapter sent `seasonal_periods`, the key was filtered out and every series
fell back (SMAPE 0.164 = SNaive). Fixed (`season_length`, integer periods); golden now writes the reason when the fallback count is > 0.
A check after the fix, on 8 NN5 weekly series, found a median forecast difference of 3.1% from R's `tbats()`. TBATS was therefore removed from the Python
engine on 2026-11-03 and offered again on 2026-11-25 as statsforecast's model, with this difference stated in the interface
(`src/tsfstudio/engines/py_engine.py`, `src/tsfstudio/app/models_info.py`). For R's TBATS use the R engine or the C++ engine, whose port of
`forecast::tbats` equals the study's R forecasts to rounding (`cpp/README.md`). The daily-set results (`golden_result.csv`) have not been examined yet.

### 1d. Naive2 and Comb S-H-D against R (done 2026-11-22, Windows, R 4.6.2, forecast 9.0.2)
Reference: `tools/r_reference/m4_benchmarks.R`, which follows the official M4 benchmark code (SeasonalityTest, `decompose`,
`naive`, `ses`, `holt`, `holt(damped = TRUE)`). Comparison: `tools/r_reference/compare_m4_benchmarks.py`. Golden test:
`tests/test_golden_m4_benchmarks.py` (fixtures `r_m4_naive2`, `r_m4_comb` in `tests/golden/nn5_*`).

| Data | Method | median rel. diff. | >1e-3 series | ΔSMAPE (rel.) | ΔMASE (rel.) |
|---|---|---|---|---|---|
| weekly (m 52, h 8; no series seasonal) | naive2 | 0 | 0/111 | 0 | 0 |
| weekly | comb_shd | 1.3×10⁻³ | 70/111 | −0.03% | −0.05% |
| daily (m 7, h 56; all 111 seasonal) | naive2 | 8×10⁻¹⁶ | 0/111 | 0 | 0 |
| daily | comb_shd | 5.5×10⁻⁴ | 55/111 | −0.31% | −0.35% |

- The seasonality test agrees with R on 222/222 series.
- **A first version of the Holt fit was wrong.** It optimised the initial level and trend numerically next to the smoothing
  parameters, with a hard penalty for beta > alpha. It stopped early (on one series exactly at its starting point). Daily Holt
  was 15% worse in SMAPE than R and Comb 5% worse. It was rewritten: for fixed (alpha, beta, phi) the initial state is solved
  in closed form, and beta is searched as alpha·u with u in [0, 1].
- After the fix, Python's in-sample MSE is lower than or equal to R's on every one of the 444 fits (111 series × 2 datasets ×
  Holt and damped Holt). It is up to 7.8% lower, because R's Nelder-Mead stops before the optimum. This is the same
  situation as SES in 1b. The remaining forecast differences are therefore R optimiser artefacts, not a different model.

## 2. R engine — against the user's R forecasts (done: Windows, local R, weekly set)
Environment: R 4.6.2 (ucrt, installed in the user folder), `forecast` 9.0.2, `jsonlite` 2.0.0; `python -m tsfstudio golden --engine r`
(NN5 weekly, h=8, 111 series, 0 fallbacks). Reference: the user's own R outputs (probably from an older `forecast` version).

| Method | median rel. diff. | max. rel. diff. | series differing by >1e-3 | ΔSMAPE | time |
|---|---|---|---|---|---|
| ses | 1×10⁻¹⁵ | 5×10⁻¹⁵ | 0/111 | ≈0 | 0.3 s |
| ets | 1×10⁻¹⁵ | 4×10⁻¹⁵ | 0/111 | ≈0 | 1.8 s |
| theta | 1×10⁻¹⁵ | 5×10⁻¹⁵ | 0/111 | ≈0 | 0.5 s |
| arima | 3×10⁻¹⁵ | 1.6×10⁻⁶ | 0/111 | −5×10⁻⁶ % | 14 s |
| dhr_arima | 4×10⁻¹⁴ | 5×10⁻⁸ | 0/111 | −6×10⁻⁸ % | 10 s |
| pooled_regression (lag 65) | 3×10⁻¹⁵ | 8×10⁻¹⁵ | 0/111 | ≈0 | 0.2 s |

Reading: despite the package-version difference, the original R path reproduces the user's R results at machine precision (10⁻⁶–10⁻⁸ for arima/dhr_arima,
i.e. at the level of the optimiser tolerance). This is the main evidence for the claim of exact reproducibility.
Daily set (NN5 daily, h=56, 111 series, 0 fallbacks): ses/ets/theta max. rel. diff. ≤ 5×10⁻¹⁵, pooled_regression (lag 9) 9×10⁻¹⁴, ΔSMAPE/ΔMASE ≈ 0
(ets 55 s; the same command was run twice and the outputs were identical to the last digit: deterministic). R-engine runs of tbats (weekly/daily) and of daily arima have not been done yet. The CatBoost package is not installed in R (optional).
`tests/test_r_engine_live.py` makes the same comparison automatically. Tolerances: ses/ets/theta/pooled regression `1e-6`, arima `1e-4`,
tbats and dhr_arima `1e-3`.

Windows note: `parallel::mclapply` does not support multiple cores on Windows; the R worker is reduced to a single core on Windows (multi-core with PSOCK
does not exist yet).

## 2b. C3 models (intermittent demand + further classical models) — validation status
They have no R counterpart in the original study, so a golden comparison with the study's R outputs is not possible. Originally they were checked only by:
(i) unit tests with a fake statsforecast whose signatures follow the documentation (key passing, seasonality rules, error/fallback behaviour, note collection);
(ii) signatures: CrostonClassic/Optimized, ADIDA, AutoTheta, AutoCES, HoltWinters, AutoMFLES from the official documentation; CrostonSBA, TSB, IMAPA, MSTL from
other sources (the documentation page was truncated). They have since been run with the real library (statsforecast 2.1.1): the C++ ports were
compared with the Python engine's forecasts on NN5 weekly/daily and synthetic series (`cpp/README.md`, rows 4, 4a–4f). Since 2026-11-25 the R engine
also offers R library counterparts of several of them (`src/tsfstudio/r/lib/extra_models.R`); how far their forecasts differ is stated in
`src/tsfstudio/app/models_info.py` (`R_NOTES`).

## 3. Verified here (in an environment without R)
The job protocol (column-major reading, fallback/NA counters, progress, error and crash messages, cancellation) with an `Rscript` mock; the runner, missing-value
policies, splitting, manifest, memory and run-time estimator are tested with the reference engine. R itself, PyTorch and Docker cannot be run in this environment
(network and permission limits) — so section 2 and `doctor` are required steps on your machine.

## 4. Not yet known / open
* The `rocker/r-ver:4.5.1` tag exists on Docker Hub (amd64 + arm64); the date of the CRAN snapshot and the `forecast` version appear in the `doctor` output once the image is built.
* The CatBoost R package `catboost-R-linux-x86_64-1.2.10.tgz` exists in the GitHub release (verified); whether the installation succeeds is shown in the build log.
* The run-time priors come from a single run on a single machine (rough guide).
* Of the further metrics in the user's files (`MSE, MAPE, RMSLE, MedAE, MdAPE, NRMSE_IQR, MASE summary`), `MSE`, `MAPE`, `RMSLE` and `MedAE` were added to the module on
  2026-11-22 (`EXTRA_METRICS` in `src/tsfstudio/metrics.py`; see `docs/METRICS.md`); `MdAPE` and `NRMSE_IQR` are not. Whether the "MASE summary" of those files corresponds to the Mean/Median of MASE in `metrics.summarize` has not been checked.
