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
| 2 | auto ETS (R `ets()`: 30 models, likelihood, AICc selection) | planned |
| 3 | auto ARIMA (R `auto.arima()`: stepwise search, Kalman-filter likelihood) | planned |

Speed on this machine (i5, 6 cores; 111 NN5 series; Python in-process, one core):

| | SES | Theta | Comb S-H-D |
|---|---|---|---|
| weekly: Python / C++ 1 thread / C++ 6 threads | 246 / 11 / 2.7 ms | 297 / 5.2 / 3.0 ms | 4,973 / 429 / 80 ms |
| daily: Python / C++ 1 thread / C++ 6 threads | 445 / 43 / 9.3 ms | 448 / 42 / 9.2 ms | 7,603 / 2,824 / 743 ms |

## Interface
`tsf_forecast_batch(method, values, offsets, n_series, h, horizons, periods, n_periods, n_threads, out, flags, msg, msg_len)`:
series are concatenated in `values` (series i is `values[offsets[i]:offsets[i+1]]`); `out` is `n_series x h`. A series whose
model fails or returns non-finite values gets the seasonal naive forecast and `flags[i] = 1`, as in the Python engine.
