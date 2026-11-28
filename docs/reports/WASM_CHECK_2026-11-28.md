> Scripts and the final job list (jobs.json, with horizons and the extra dshw jobs) are copied to `tools/wasm_check/`. The paths below show where they were run: `wasm_work/` is a scratch folder under the repository root (emsdk, venv, outputs; not in git).

# tsfcore: WebAssembly build vs native build

- Repo: /home/user/forecast-studio, branch claude/forecast-studio-hpc-speed-p4qixh, commit 49778993d362959726de5df615df0c98c9e20c32 (clean tree)
- emsdk 7.0.1 (git clone https://github.com/emscripten-core/emsdk; ./emsdk install 7.0.1; ./emsdk activate 7.0.1)
- em++: `emcc (Emscripten gcc/clang-like replacement + linker emulating GNU ld) 7.0.1 (ac0578743420bcbe9a4f77d0b6599c25f8d43702)`
- wasm build: wasm_work/wasm_check/build_wasm.sh (the pages.yml recipe verbatim: em++ -std=c++17 -O3 -fexceptions per file, emcc -O2 -std=gnu99 -w for catch22, same link flags/exports; only compile parallelised). No warnings. web/tsfcore.wasm sha256 472f7bbc40bdd1afc808f8aaadc7ef6fc0d5a864f27f49f1c1e89191a4c9fba2
- native: `bash cpp/build.sh` (g++ 13.3.0, cmake 3.28.3, -O3) -> src/tsfstudio/native/tsfcore.so, rebuilt at the same commit
- Node v22.23.5 (/opt/node22/bin/node)

## Commands
```
python wasm_work/wasm_check/make_jobs.py                                   # writes jobs.json (104 jobs; then horizon from web/data/index.json and the extra dshw jobs were added)
wasm_work/wasm_check/../venv/bin/python wasm_work/wasm_check/run_native.py wasm_work/wasm_check/jobs.json wasm_work/wasm_check/out_native     # tsf_run via ctypes on tsfcore.so
/opt/node22/bin/node wasm_work/wasm_check/run_wasm.js wasm_work/wasm_check/jobs.json wasm_work/wasm_check/out_wasm           # tsf_run on web/tsfcore.js in Node, as web/worker.js runOnce
wasm_work/wasm_check/../venv/bin/python wasm_work/wasm_check/compare.py > wasm_work/wasm_check/table.md
```
Both sides call the same entry point the browser uses (tsf_run) with identical config lines:
`method=<m>, dataset_id=<ds>, horizon=<index.json horizon>, max_series=<N>, threads=1, eol=lf` (dshw: plus `seasonality=a,b`).
N = 100 for tslm/stm/otm/dstm/theta/dshw (111 for dshw on nn5_daily), 30 for ets/arima/bats/tbats/struct_ts/nnetar (first N series).
forecasts.csv (full repr precision) and fallbacks.csv are compared; rel diff = |wasm-native|/max(|native|,1e-12), per series = max over horizon.

## Method list
tsf_version() and tsf_methods() strings are identical in both builds. Probing every name of the METHODS registry in capi.cpp
with tsf_forecast_batch: all 47 (incl. tslm, stm, otm, dstm, bats, struct_ts, dshw, nnetar) return 0 on both; catboost/lightgbm/unknown return -1 on both.
Note (both builds, not a wasm issue): tsf_methods() is stale - it lists 32 names and omits stl_ets, stl_arima, bagged_ets, thief, mapa,
comb_eat, comb_eat_median and all eight new models.

## Results
| model | dataset | n | max rel diff | median rel diff | n > 1e-6 | n identical | fallbacks (both) | fallback one side only | native s | wasm s |
|---|---|---|---|---|---|---|---|---|---|---|
| tslm | m1_yearly | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.0 | 0.1 |
| stm | m1_yearly | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.0 | 0.1 |
| otm | m1_yearly | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.1 | 0.1 |
| dstm | m1_yearly | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.0 | 0.1 |
| theta | m1_yearly | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.0 | 0.0 |
| ets | m1_yearly | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 0.0 | 0.1 |
| arima | m1_yearly | 30 | 2.03e-07 | 0.00e+00 | 0 | 24 | 0 | - | 0.1 | 0.4 |
| bats | m1_yearly | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 0.4 | 0.7 |
| tbats | m1_yearly | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 0.5 | 0.9 |
| struct_ts | m1_yearly | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 0.0 | 0.0 |
| nnetar | m1_yearly | 30 | 4.53e-04 | 3.50e-14 | 1 | 7 | 0 | - | 0.1 | 0.1 |
| tslm | m1_quarterly | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.0 | 0.0 |
| stm | m1_quarterly | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.0 | 0.1 |
| otm | m1_quarterly | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.1 | 0.2 |
| dstm | m1_quarterly | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.0 | 0.1 |
| theta | m1_quarterly | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.0 | 0.0 |
| ets | m1_quarterly | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 1.3 | 2.3 |
| arima | m1_quarterly | 30 | 7.06e-06 | 0.00e+00 | 1 | 17 | 0 | - | 1.5 | 3.7 |
| bats | m1_quarterly | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 1.4 | 2.6 |
| tbats | m1_quarterly | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 2.8 | 6.3 |
| struct_ts | m1_quarterly | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 0.1 | 0.4 |
| nnetar | m1_quarterly | 30 | 4.43e-04 | 2.91e-07 | 11 | 0 | 0 | - | 0.3 | 0.3 |
| tslm | m1_monthly | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.0 | 0.0 |
| stm | m1_monthly | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.1 | 0.1 |
| otm | m1_monthly | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.3 | 0.3 |
| dstm | m1_monthly | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.1 | 0.2 |
| theta | m1_monthly | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.0 | 0.0 |
| ets | m1_monthly | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 12.5 | 29.8 |
| arima | m1_monthly | 30 | 3.87e-07 | 6.17e-14 | 0 | 13 | 0 | - | 35.1 | 88.1 |
| bats | m1_monthly | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 1.2 | 3.5 |
| tbats | m1_monthly | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 6.1 | 17.6 |
| struct_ts | m1_monthly | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 0.9 | 3.2 |
| nnetar | m1_monthly | 30 | 1.81e-04 | 3.03e-08 | 6 | 0 | 0 | - | 1.1 | 2.1 |
| tslm | tourism_quarterly | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.0 | 0.0 |
| stm | tourism_quarterly | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.1 | 0.3 |
| otm | tourism_quarterly | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.3 | 0.8 |
| dstm | tourism_quarterly | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.1 | 0.3 |
| theta | tourism_quarterly | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.0 | 0.0 |
| ets | tourism_quarterly | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 1.5 | 4.4 |
| arima | tourism_quarterly | 30 | 1.48e-07 | 8.57e-14 | 0 | 15 | 0 | - | 5.4 | 13.0 |
| bats | tourism_quarterly | 30 | 4.32e-15 | 0.00e+00 | 0 | 29 | 0 | - | 3.3 | 10.2 |
| tbats | tourism_quarterly | 30 | 4.32e-15 | 0.00e+00 | 0 | 28 | 0 | - | 9.5 | 24.9 |
| struct_ts | tourism_quarterly | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 0.2 | 0.4 |
| nnetar | tourism_quarterly | 30 | 2.96e-03 | 3.18e-06 | 17 | 0 | 0 | - | 0.5 | 0.9 |
| tslm | nn5_weekly | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.0 | 0.1 |
| stm | nn5_weekly | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.1 | 0.2 |
| otm | nn5_weekly | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.3 | 1.0 |
| dstm | nn5_weekly | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.1 | 0.4 |
| theta | nn5_weekly | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.0 | 0.0 |
| ets | nn5_weekly | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 0.1 | 0.3 |
| arima | nn5_weekly | 30 | 1.40e-06 | 0.00e+00 | 2 | 22 | 0 | - | 1.4 | 4.3 |
| bats | nn5_weekly | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 1.5 | 2.7 |
| tbats | nn5_weekly | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 16.3 | 24.4 |
| struct_ts | nn5_weekly | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 0.0 | 0.1 |
| nnetar | nn5_weekly | 30 | 8.24e-04 | 1.44e-08 | 5 | 0 | 0 | - | 0.9 | 1.6 |
| tslm | hospital | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.0 | 0.1 |
| stm | hospital | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.1 | 0.1 |
| otm | hospital | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.2 | 0.2 |
| dstm | hospital | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.1 | 0.1 |
| theta | hospital | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.0 | 0.0 |
| ets | hospital | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 13.5 | 17.5 |
| arima | hospital | 30 | 6.05e-11 | 0.00e+00 | 0 | 23 | 0 | - | 16.0 | 27.2 |
| bats | hospital | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 0.9 | 1.4 |
| tbats | hospital | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 3.5 | 4.7 |
| struct_ts | hospital | 30 | 1.40e-11 | 0.00e+00 | 0 | 29 | 0 | - | 0.8 | 1.3 |
| nnetar | hospital | 30 | 2.33e-04 | 1.23e-08 | 7 | 0 | 0 | - | 0.8 | 1.0 |
| tslm | car_parts | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.0 | 0.0 |
| stm | car_parts | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.0 | 0.0 |
| otm | car_parts | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.0 | 0.1 |
| dstm | car_parts | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.0 | 0.1 |
| theta | car_parts | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.0 | 0.0 |
| ets | car_parts | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 4.8 | 5.9 |
| arima | car_parts | 30 | 1.66e-02 | 0.00e+00 | 1 | 24 | 0 | - | 3.7 | 6.7 |
| bats | car_parts | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 0.4 | 0.6 |
| tbats | car_parts | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 2.0 | 2.5 |
| struct_ts | car_parts | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 0.5 | 1.1 |
| nnetar | car_parts | 30 | 6.44e-09 | 0.00e+00 | 0 | 22 | 0 | - | 0.1 | 0.2 |
| tslm | covid_deaths | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.0 | 0.0 |
| stm | covid_deaths | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.1 | 0.1 |
| otm | covid_deaths | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.2 | 0.3 |
| dstm | covid_deaths | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.1 | 0.1 |
| theta | covid_deaths | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.0 | 0.0 |
| ets | covid_deaths | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 2 | - | 2.0 | 4.3 |
| arima | covid_deaths | 30 | 7.03e-10 | 0.00e+00 | 0 | 27 | 0 | - | 0.7 | 1.1 |
| bats | covid_deaths | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 0.6 | 1.2 |
| tbats | covid_deaths | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 3.3 | 3.9 |
| struct_ts | covid_deaths | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 2 | - | 1.5 | 3.2 |
| nnetar | covid_deaths | 30 | 2.55e-03 | 4.15e-07 | 13 | 3 | 0 | - | 1.2 | 1.7 |
| tslm | nn5_daily | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.0 | 0.1 |
| stm | nn5_daily | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.4 | 1.0 |
| otm | nn5_daily | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 1.0 | 2.6 |
| dstm | nn5_daily | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.6 | 1.3 |
| theta | nn5_daily | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.1 | 0.2 |
| ets | nn5_daily | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 5.7 | 18.1 |
| arima | nn5_daily | 30 | 2.53e-10 | 6.19e-13 | 0 | 7 | 0 | - | 170.3 | 370.7 |
| bats | nn5_daily | 30 | 1.52e-11 | 0.00e+00 | 0 | 28 | 0 | - | 15.6 | 33.7 |
| tbats | nn5_daily | 30 | 1.91e-16 | 0.00e+00 | 0 | 29 | 0 | - | 37.1 | 85.7 |
| struct_ts | nn5_daily | 30 | 0.00e+00 | 0.00e+00 | 0 | 30 | 0 | - | 4.8 | 16.7 |
| nnetar | nn5_daily | 30 | 6.08e-09 | 1.48e-10 | 0 | 0 | 0 | - | 120.1 | 236.3 |
| dshw | nn5_daily_7_28 | 111 | 0.00e+00 | 0.00e+00 | 0 | 111 | 93 | - | 0.1 | 1.0 |
| dshw | nn5_daily_7_364 | 111 | 0.00e+00 | 0.00e+00 | 0 | 111 | 93 | - | 0.1 | 0.5 |
| dshw | m1_monthly_3_12 | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.1 | 0.3 |
| dshw | hospital_3_12 | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 0 | - | 0.0 | 0.3 |
| dshw | tourism_quarterly_2_4 | 100 | 0.00e+00 | 0.00e+00 | 0 | 100 | 1 | - | 0.1 | 0.5 |

(native s / wasm s = single-threaded tsf_run wall time; wasm total 1109 s vs native 521 s, ~2.1x.)

## Summary per model (all datasets)
| model | series | max rel diff | series > 1e-6 | bit-identical | one-sided fallback/failure |
|---|---|---|---|---|---|
| tslm | 900 | 0 | 0 | 900 | 0 |
| stm | 900 | 0 | 0 | 900 | 0 |
| otm | 900 | 0 | 0 | 900 | 0 |
| dstm | 900 | 0 | 0 | 900 | 0 |
| dshw | 522 | 0 | 0 | 522 | 0 |
| bats | 270 | 1.5e-11 | 0 | 267 | 0 |
| struct_ts | 270 | 1.4e-11 | 0 | 269 | 0 |
| nnetar | 270 | 3.0e-3 | 60 | 32 | 0 |
| theta (ctrl) | 900 | 0 | 0 | 900 | 0 |
| ets (ctrl) | 270 | 0 | 0 | 270 | 0 |
| tbats (ctrl) | 270 | 4.3e-15 | 0 | 267 | 0 |
| arima (ctrl) | 270 | 1.7e-2 | 4 | 172 | 0 |

## Notes
- No run failed on either side; fallback sets and messages are identical everywhere (ets covid_deaths 2, struct_ts covid_deaths 2, dshw nn5_daily 93/111 "data contain zeros or negative numbers", dshw tourism 1).
- dshw: no web/data file has two seasonal periods (no sub-daily data). Tested nn5_daily with 7,28 and 7,364 (only 18/111 series are strictly positive, so 93 fall back identically on both sides), plus synthetic nested periods m1_monthly 3,12, hospital 3,12, tourism_quarterly 2,4 (all bit-identical).
- Cause of the differences: libm. Emscripten (musl) and glibc differ in the last bit for some inputs: of 200k random arguments exp differs in 130, pow 152, expm1 323, tanh 15329, sin/cos ~6200, atan 8624 (max 1 ulp, ~2.2e-16); log, sqrt, lgamma identical (wasm_work/wasm_check/libm). nnetar (logistic via exp, 20 nets fitted by BFGS) and auto.arima (optimiser + model selection) amplify these 1-ulp differences into optimiser path differences.
- nnetar: median per-dataset rel diff 1e-10..3e-6, worst 3.0e-3 (tourism_quarterly T18: 12547.9 vs 12576.3 at h1). Same order as expected from a different libm; on nn5_daily and car_parts all within 1e-8.
- arima: 4/270 series > 1e-6; the worst, car_parts T18 (an almost all-zero intermittent series, forecasts ~0.37), differs by 1.7e-2 - the likelihood is flat and the optimiser ends at a slightly different point. Others <= 7e-6.
- The Windows x87 paths (r_exp/r_log long double, X87Extended) are compiled out on both Linux native and wasm, so this compares SSE2/glibc against wasm/musl; the R bit-for-bit reference (Windows) was not tested here.
