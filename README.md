# Forecast Studio

Forecast Studio runs statistical, machine-learning and deep forecasting methods on collections of time series in the
Monash `.tsf` format and reports the same error measures for every method. The same run can be started from a desktop
interface, a page in the browser, the command line or a batch job on a cluster, and each run writes its forecasts,
errors per series and a `manifest.json` that records everything needed to repeat it.

Methods run on three engines:

| Engine | What it is | Methods |
|:--|:--|--:|
| `cpp` | **tsfcore**, a C++17 library with a plain C interface (`cpp/`). Ports of R `forecast` (ETS, auto ARIMA, TBATS, BATS, STL, bagged ETS, thief, MAPA, …) and of statsforecast models, checked series by series against their reference (`cpp/README.md`). Multi-threaded. Also compiled to WebAssembly for the browser page. | 50 |
| `python` | statsforecast, LightGBM, CatBoost, neuralforecast / GluonTS (N-HiTS, N-BEATS, DeepAR, LSTM, GRU, RNN, FFNN, WaveNet), Chronos, and built-in ports of R's ses / theta / pooled regression | 55 |
| `r` | Rscript worker calling `forecast`, `thief`, `MAPA`, `smooth`, `tsintermittent` and `forecTheta` (`src/tsfstudio/r/`) | 39 |

Together these are 60 methods (45 local, 15 global). `python -m tsfstudio engines` lists what is available on a
machine; `src/tsfstudio/app/models_info.py` holds every method's engines, parameters and references.

## Editions

- **Desktop interface:** a local web interface served by Python, reachable from this computer only (`docs/APP.md`).
  You choose data and methods, see an estimate and a pre-run check, follow progress, and compare results, charts and
  history. Windows users can use the self-contained package from the
  [releases page](https://github.com/MicheleCedolin/forecast-studio/releases) and need no Python installation.
- **Browser page (`web/`):** tsfcore compiled to WebAssembly. The page runs the C++ methods (without CatBoost and
  LightGBM) on the example datasets or on a `.tsf` file of the user's, and the data never leaves the device.
  `.github/workflows/pages.yml` builds it and publishes it with GitHub Pages.
- **Command line:** `python -m tsfstudio run | estimate | sweep | rank | reliability | instability | …`.
- **Cluster:** batch runs over datasets × methods with SGE or SLURM scripts, and an engine speed study (`docs/HPC.md`,
  `docs/HPC_SPEED.md`).

## What a run produces

- **Point measures:** SMAPE, mSMAPE, MASE, MAE, RMSE, WAPE, nRMSE, as defined in the original R code. Optional:
  RMSSE, MSE, MAPE, MAAPE, MedAE, RMSLE, ME, PBias, and WRMSSE for M5. See `docs/METRICS.md`.
- **Prediction intervals:** split-conformal intervals for every engine and method, with Coverage, MSIS and WQL.
- **Rolling-origin evaluation:** several forecast origins, each with its own fit, so a global model never sees its
  test window.
- **Series features:** catch22, R `tsfeatures` (as called in the study), the demand class (ADI / CV²) and the Box-Cox λ,
  plus a 2-D / 3-D PCA map of the series coloured by a feature, by the error, or by the best method.
- **Ranking analyses** (command line): Friedman and Nemenyi tests, AvgRelMAE, OWA and bootstrap rank stability within a
  task (`rank`); ranking reliability across tasks with incomplete method coverage (`reliability`); disagreement between
  measures inside single series (`instability`).
- **Files:** `runs/<data>__<engine>-<method>__<time>/` holds `forecasts.csv`, `test_actuals.csv`,
  `errors_per_series.csv`, `summary.csv`, `metrics.xlsx`, `fallbacks.csv` and `manifest.json`. The manifest records the
  configuration, seed, software versions, the SHA-256 of the data file, skipped series and fallback counts.

## Installation

Python ≥ 3.10 (3.11 or 3.12 recommended).

```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[pyengine,dev]"                      # core + statsforecast, LightGBM, CatBoost
pip install -e ".[deep]"                              # optional: deep and foundation models (PyTorch)
bash cpp/build.sh                                     # optional: the C++ engine (Windows: cpp\build.ps1)
python -m tsfstudio engines                           # which engines and methods are available here
make test
```

- **R engine:** needs R with `forecast` and `jsonlite`; the other R packages are optional, and each enables the
  methods that use it. On Windows the Docker image provides R (`docs/DOCKER_WINDOWS.md`). On a cluster without a usable
  R, `scripts/hpc/setup_r.sh` installs it from conda-forge.
- **Old Linux systems** (CentOS 7, g++ 4.8): `scripts/hpc/setup_conda.sh` takes the compiler, CMake and Python from
  conda-forge and builds tsfcore (`docs/HPC.md`).

## Quick start

```bash
python -m tsfstudio app                                              # the interface: http://127.0.0.1:8765/
python -m tsfstudio run --dataset nn5_weekly --engine cpp --method ets
python -m tsfstudio run --dataset nn5_daily --engine python --method auto_theta --workers 4
python -m tsfstudio estimate --dataset m4_monthly --method arima      # time and memory estimate before a long run
```

Windows: double-click `ForecastStudio.bat`, or use `scripts\windows\tsfs.ps1 <command>`.

## Data

Put the `.tsf` files in one folder, `data/tsf/`, or point `TSFSTUDIO_DATA_DIR` to another one. The catalogue
`src/tsfstudio/data/catalog.yaml` gives each dataset's file name, byte size, lag, horizon and the methods used in the
study; `make data-check` compares the folder with it. The files are those of the Monash archive (Godahewa et al.,
2021) plus two M5 files. `web/data/` holds nine small examples for the browser page.

Excel input: `python -m tsfstudio template template.xlsx` writes a template, and
`python -m tsfstudio convert mydata.xlsx mydata.tsf --fill-gaps` converts a filled one. Long format
(`series_name | date | value`) and wide format (`date | series1 | series2 …`) are accepted. Errors are reported with
their cell address.

## Validation

Every port is compared with its reference implementation on real data, and the comparisons are tests:

- **Measures:** against the study's R error files, to machine precision (`docs/VALIDATION.md`).
- **C++ engine:** each method series by series against R `forecast` 9.0.2 or statsforecast 2.1.1. For example, auto
  ETS, auto ARIMA and TBATS select the same model as R on all NN5 series. Methods, data and differences are listed in
  `cpp/README.md`; the checks are in `tests/test_cpp_engine.py` and `tools/r_reference/`.
- **Engines against each other:** `tools/compare_engines.py` runs one method on the C++, Python and R engines and
  reports the forecast differences, fallbacks and mean MASE.
- **Linux and cluster builds:** `qsub scripts/hpc/check.sge` repeats the agreement checks on a compute node.

## Documentation

| File | Content |
|:--|:--|
| `docs/APP.md` | the desktop interface: use, security, limits |
| `docs/METRICS.md` | definitions of the measures, differences from the original R code |
| `docs/PYTHON_ENGINE.md` | Python engine: methods and their relation to R |
| `docs/CHANGES_R.md` | changes to the original R code, with reasons |
| `docs/VALIDATION.md` | golden tests: what is compared, how, results |
| `cpp/README.md` | tsfcore: build, every ported method and its check |
| `docs/HPC.md`, `docs/HPC_SPEED.md` | cluster set-up, batch runs, speed study |
| `docs/DOCKER_WINDOWS.md` | Docker on Windows (R engine) |
| `docs/RESEARCH_MODELS_METRICS.md`, `docs/UI_REVIEW*.md` | research notes on methods and measures; reviews of the interface |

## Repository layout

```
src/tsfstudio/
  tsf/                 Monash .tsf reader and writer
  catalog.py, data/    dataset catalogue
  metrics.py, intervals.py, wrmsse.py      error measures, conformal intervals, WRMSSE
  features.py, tsfeat.py                   series features (catch22, tsfeatures, demand class)
  engines/             cpp_engine.py (tsfcore via ctypes), py_*.py (Python), r_engine.py (Rscript), registry.py
  r/                   R worker and model library
  runner.py            split, missing values, engine call, measures, outputs, manifest
  sweep.py             batch runs for clusters
  ranking.py, reliability.py, instability.py   ranking analyses
  app/                 desktop interface: server, service, models_info.py, static/
  cli.py               command line
cpp/                   tsfcore sources, build scripts (native and WebAssembly), README
web/                   browser page (WebAssembly)
tools/                 validation, comparison, benchmark and packaging scripts
scripts/hpc/, scripts/windows/      cluster and Windows launchers
tests/                 unit, golden and engine-agreement tests
reference/             the original R / Python code of the study (unchanged)
manuscript/            papers that use the software
```

Some source comments and the Windows set-up scripts are still in Turkish.

## Licence and citation

GNU General Public License v3.0 (`LICENSE`); third-party notices in `NOTICE`. A software paper describing Forecast
Studio is in preparation; until it is published, please cite the repository and the release you used.

## References

- Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J., Montero-Manso, P. (2021). Monash Time Series Forecasting Archive. NeurIPS Datasets and Benchmarks.
- Hyndman, R. J., Khandakar, Y. (2008). Automatic time series forecasting: the forecast package for R. Journal of Statistical Software, 27(3).
- Hyndman, R. J., Koehler, A. B. (2006). Another look at measures of forecast accuracy. International Journal of Forecasting, 22(4).
- Makridakis, S., Spiliotis, E., Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. International Journal of Forecasting, 38(4).
