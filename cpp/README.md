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
coefficients of the seasonal differencing polynomial. Weekly data use frequency floor(52.18) = 52, as the study's R run did.

## Interface
`tsf_forecast_batch(method, values, offsets, n_series, h, horizons, periods, n_periods, n_threads, out, flags, msg, msg_len)`:
series are concatenated in `values` (series i is `values[offsets[i]:offsets[i+1]]`); `out` is `n_series x h`. A series whose
model fails or returns non-finite values gets the seasonal naive forecast and `flags[i] = 1`, as in the Python engine.
