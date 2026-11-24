# Forecast Studio, web edition

A static page that runs the statistical methods of Forecast Studio **in the visitor's browser**. The forecasting core is
the C++ library of the desktop edition (`cpp/`) compiled to WebAssembly; the page starts one worker per CPU core of the
visitor's device. There is no server part: the web host only delivers these files (about 1.2 MB), and the data a visitor
loads never leave their device.

## Publishing

Copy this folder as it is to any web host, at the site root or in a sub-folder (for example `https://example.org/forecast/`).
All addresses inside it are relative. It must be served over http(s); opening `index.html` from the disk does not work,
because browsers do not start module workers from `file://`. To try it locally:

```bash
python -m http.server 8000 --directory web      # then open http://127.0.0.1:8000/
```

Browsers: current Chrome, Edge, Firefox and Safari (WebAssembly exception handling and module workers are required).

## What it contains

| File | Role |
|---|---|
| `index.html`, `app.css`, `app.js` | the page: data, method choice, run, result table, series chart, CSV downloads |
| `core.js` | reading `.tsf` and `.csv`, the train/test split, the error measures (the definitions of `src/tsfstudio`) |
| `worker.js`, `engine.js` | the worker and its calls into the WebAssembly core |
| `tsfcore.mjs`, `tsfcore.wasm` | the core, **built** from `cpp/src` by `cpp/build_wasm.sh` (Emscripten 7.0.1) |
| `methods.js` | the method list, **generated** from the desktop edition's descriptions by `tools/web/make_methods.py` |
| `samples/nn5_weekly.tsf` | the sample (NN5 Weekly, 111 series), **generated** by `tools/web/make_sample.py` |

Methods: the 35 local methods of the C++ engine and pooled regression (see `cpp/README.md` for what each one ports and how
it was checked). Measures: MASE, sMAPE, msMAPE, MAE, RMSE, RMSSE, WAPE, with mean, median and the mean rank by MASE.
The evaluation is the study's fixed origin: the last *h* observations of each series are the test part.

Not in the web edition: the R engine, CatBoost, LightGBM, N-HiTS and Chronos-Bolt, method parameters, prediction intervals,
rolling origins, series features, Excel output, run history, and the `rank` / `reliability` / `instability` analyses.

## Rebuilding the core

```bash
source /path/to/emsdk/emsdk_env.sh      # Emscripten SDK
bash cpp/build_wasm.sh                  # writes web/tsfcore.mjs and web/tsfcore.wasm
python -m pytest tests/test_web_edition.py
```

The built files are kept in git so that the folder can be published without a compiler. After a change under `cpp/src`
they must be rebuilt.

## Checks (`tests/test_web_edition.py`, 20 tests; needs Node)

On NN5 Weekly (111 series, h = 8), through the same modules the page uses:

| Check | Result |
|---|---|
| WebAssembly core against the study's R forecasts | ETS ≤ 1e-10, TBATS ≤ 1e-8, DHR-ARIMA ≤ 1e-6, ARIMA ≤ 1e-4, pooled regression ≤ 1e-10 (largest relative difference; the bounds of the native library) |
| WebAssembly core against the native library | 11 methods equal to 1e-9 |
| `core.js` measures against `tsfstudio.metrics` | all seven measures equal to 1e-12 on every series |
| `core.js` reader against `tsfstudio.tsf` | same series, names, frequency, horizon and split |

## Speed

Measured in Chromium on 4 cores, NN5 Weekly: the seven "Monash statistical set" methods take 22 s in total, of which
TBATS 18 s and ARIMA 2.4 s; the seven quick benchmarks take 3.3 s. In a single thread, Comb S-H-D, THieF and MAPA are
much slower than the native library (12 s, 53 s and 54 s for the 111 series); the cause has not been investigated.
Large collections with ARIMA or TBATS will take long in a browser; the desktop edition is the tool for those.
