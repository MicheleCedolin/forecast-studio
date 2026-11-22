# tsfcore: C++ forecasting core

C++17 counterparts of the Python engine's methods, built as one DLL with a plain C interface. Python uses it through
`ctypes` as the `cpp` engine (`--engine cpp`, `src/tsfstudio/engines/cpp_engine.py`). All series of a job go to the library
in one call and are spread over CPU threads inside C++.

## Build
```powershell
winget install BrechtSanders.WinLibs.POSIX.UCRT     # portable GCC + CMake, no admin rights (once)
powershell -ExecutionPolicy Bypass -File cpp\build.ps1
```
The DLL is copied to `src/tsfstudio/native/tsfcore.dll` (not in git). It links the GCC runtime statically and depends only on
Windows system DLLs. GitHub Actions builds it with MSVC (`-Generator ""`) for the release package.

## Status
| Phase | Methods | Check against Python (tests/test_cpp_engine.py) |
|---|---|---|
| 1 (done) | naive, snaive, drift, naive2, ses, theta, comb_shd (with holt / damped holt) | naive/snaive/drift/naive2 to 1e-10; ses/theta to ~3e-8 (same Brent minimiser, different summation order); Holt fits equal to or better than Python on all NN5 fits; Comb S-H-D within 1% of R in mean SMAPE/MASE |
| 2 (done) | auto ETS: a port of R `forecast::ets()` 9.x (initstate, initparam, bounds "both" with the characteristic-root test, etscalc likelihood, R's Nelder-Mead `nmmin`, AICc selection, m > 24 non-seasonal) | against the study's R forecasts (`tests/golden/*/forecasts_ets`): same model on 111/111 weekly and 111/111 daily series; forecasts equal to machine precision (median relative difference ~1e-14); 1 daily series differs by > 0.1% (the two Nelder-Mead runs drift apart) |
| 3 (done) | auto ARIMA: a port of R `stats::arima` (CSS / CSS-ML / ML, Gardner Q0, Kalman likelihood, R's BFGS `vmmin` and `optimhess`) and `forecast::auto.arima()` 9.0.2 (KPSS `ndiffs`, STL-based `nsdiffs` with a port of `stl.f`, stepwise search, approximation offset and refit, root and variance checks), run as the study's chain `auto.arima(lambda = 0)` → `auto.arima()` → `auto.arima(seasonal = FALSE)` | fixed orders: 40 NN5 fits against R `stats::arima`, log-likelihood within 2e-9 (`tools/r_reference/compare_arima_fixed.py`); auto: same model, constant and fallback step as R on 111/111 weekly and 111/111 daily series, forecasts median relative difference ~1e-13, worst 1.5e-6 weekly / 9e-5 daily (`compare_auto_arima.py`); R 9.0.2 itself reproduces the study's golden forecasts to 4e-5 |
| 3c (done) | dhr_arima: the study's dynamic harmonic regression, `auto.arima(y, xreg = fourier(y, K = 1), seasonal = FALSE)`; adds general regressors to the `stats::arima` port (intercept / drift / xreg columns, SVD rotation, `lm` starting values) and to `auto.arima` (residualised differencing tests, rank check), and `fourier()` with R's `sinpi`/`cospi`. Several seasonal periods fail as in R (scalar K), so those series fall back to snaive; K other than 1 is refused by the engine | same model on 111/111 weekly (vs the study's forecasts, worst 1.2e-8) and 111/111 daily series (vs R 9.0.2, `tools/r_reference/compare_dhr.py`, worst 4.3e-6) |
| 4a (done) | holt_winters: R `ets(y, model = "AAA")` (undamped and damped trend compared by AICc) through the C++ ets port; season length as the Python engine's `usable_season(n, m, 24)`, otherwise "Nonseasonal data" and the snaive fallback (as statsforecast); error_type "A" only | against R 9.0.2 on NN5 daily (`tests/golden/nn5_daily/r_ets_aaa`): median relative difference 1e-13, 1/111 series 4.7% (Nelder-Mead drift, as with ets). statsforecast's HoltWinters (Python engine) differs from R by ~1% on every series; C++ follows R. Daily: R 28.7 s / Python 16.1 s / C++ 5.6 s (1 thread) |
| 4b (done) | dotm: statsforecast 2.1.1 `DynamicOptimizedTheta` as the Python engine runs it — theta state recursion and objective (`src/theta.cpp`), its bounded adaptive Nelder-Mead (`nelder_mead.h`), statsmodels' acf seasonality test and `seasonal_decompose`, seasonal re-application; season length by `usable_season(n, m, None)`, multiplicative → additive for data ≤ 0; decomposition_type other than multiplicative is refused. The forecast step reads only the previous state row (statsforecast copies the whole history each step), which makes a fit O(n) instead of O(n²) with the same arithmetic | equal to the Python engine to machine precision on NN5 weekly/daily and 200 synthetic series (worst 1.3e-14; differences come from statsmodels' FFT convolution and Eigen's vectorised sums). Daily: Python 9.0 s / C++ 2.3 s (1 thread) |
| 4c (done) | auto_ces: statsforecast 2.1.1 `AutoCES(model = "Z")` as the Python engine runs it — initial states (`ces.py`, with statsmodels' additive decomposition), the forward / backward / forward fit with objective n·log(SSE), the inline and buffered multi-step forecasts, AICc selection over N / S / P / F (`src/ces.cpp`), the shared Nelder-Mead; `usable_season(n, m, 24)`; other model / max_season settings are refused | equal to the Python engine on NN5 weekly (bit for bit), daily (worst 1.7e-10) and 200 synthetic series (worst 3.7e-7, Nelder-Mead paths after 1e-16 differences in the decomposition). Daily: Python 3.5 s / C++ 2.1 s |
| 4d (done) | scum (Petropoulos & Svetunkov 2020): pointwise median of the C++ ets, auto_ces, arima and dotm forecasts, with the Python engine's rules (a failing or non-finite component is left out, fewer than two → fallback; ARIMA non-seasonal above arima_max_season = 200, other values refused). Because the C++ ets and arima reproduce R while the Python engine's are statsforecast's, the C++ scum is the R-based variant | exactly the median of the C++ components (weekly and daily). Against the Python engine's scum: relative difference median 2e-5 weekly / 4e-3 daily, mean MASE 0.852 vs 0.849 weekly, 0.843 vs 0.842 daily. Weekly 0.9 s (C++, 6 threads) vs 28.5 s (Python, 1 process); daily 105 s vs 160 s, dominated by the exact R auto.arima |
| 4 (done) | intermittent demand: croston_classic, croston_sba, croston_optimized, tsb, adida, imapa — ports of the Python engine's statsforecast 2.1.1 models (`src/ses.cpp` SES kernel and golden-section search, `models.py` logic, numpy's pairwise mean and round-half-to-even), with the engine's all-zero → zeros rule; tsb takes only the default alphas (0.1, 0.1) in C++ | bit-identical to the Python engine on NN5 weekly/daily and 307 synthetic intermittent series, error cases included (`test_intermittent_methods_match_python_bit_for_bit`). statsforecast's core is already C++, so the gain is the Python overhead per series |

Speed on this machine (i5, 6 cores; 111 NN5 series; Python in-process, one core):

| | SES | Theta | Comb S-H-D |
|---|---|---|---|
| weekly: Python / C++ 1 thread / C++ 6 threads | 246 / 11 / 2.7 ms | 297 / 5.2 / 3.0 ms | 4,973 / 429 / 80 ms |
| daily: Python / C++ 1 thread / C++ 6 threads | 445 / 43 / 9.3 ms | 448 / 42 / 9.2 ms | 7,603 / 2,824 / 743 ms |

ETS (111 series): weekly statsforecast 2.2 s / R 1.9 s / C++ 0.23 s (1 thread) / 0.04 s (6 threads); daily statsforecast 28.3 s /
R 54.8 s / C++ 13.0 s / 2.4 s. Note that the Python engine's ets is statsforecast AutoETS, which is close to but not the
same as R; the C++ ets reproduces R.

ARIMA (111 series, the study's three-step chain): weekly R 8–15 s / C++ 2.7 s (1 thread) / 0.7 s (6 threads); daily R 943 s /
C++ 100 s (6 threads). Daily is dominated by R's refit step: candidates are refitted by exact likelihood in AICc order until
one passes the root and variance checks (up to ~30 fits, ~25,000 Kalman passes for one series), and the port keeps that
behaviour. Two deliberate departures from R's arithmetic, both ~1e-12: the stationary covariance Q0 is computed from the
ARMA autocovariances when the state dimension exceeds 8 (Gardner's method costs O(r^6)), and loops skip the zero
coefficients of the seasonal differencing polynomial. Weekly data use frequency floor(52.18) = 52, as the study's R run did
(a single-period `msts` is a plain `ts` with frequency floor(period), so R's Fourier terms also use 52).

DHR-ARIMA (111 series, 1 thread): weekly R 10.5 s / C++ 1.3 s; daily R 34.3 s / C++ 13.3 s.

## Interface
`tsf_forecast_batch(method, values, offsets, n_series, h, horizons, periods, n_periods, n_threads, out, flags, msg, msg_len)`:
series are concatenated in `values` (series i is `values[offsets[i]:offsets[i+1]]`); `out` is `n_series x h`. A series whose
model fails or returns non-finite values gets the seasonal naive forecast and `flags[i] = 1`, as in the Python engine.
