# Forecast Studio

A Python application, distributed with Docker, that lets you choose and run local (R), global (R) and deep (PyTorch GluonTS)
forecasting models on time series in TSF (Monash) format, and that produces error measures and the selected analyses automatically.
This repository contains **Phase A** (core), **Phase B** (R engine, runner, Docker), the first part of **Phase C** (Python engine; local work without Docker) and the local interface (Phase F). PyTorch and WRMSSE belong to later phases.

## Status

| Phase | Content | Status |
|---|---|---|
| A | TSF reader/writer, data catalogue, Excel template + converter, Python metrics module, CLI, tests | **Done** |
| B | R engine (corrected driver, fallback counter), runner + manifest, memory/time estimate, Dockerfile/compose, Windows scripts, golden tests | **Coded**; real R and Docker verification is on the user's machine (`doctor`, `golden-test`) |
| C | Python engine: built-in ses/theta/naive/snaive/pooled regression (verified against R) + statsforecast (ets/arima/dhr_arima; TBATS disabled; C3: intermittent demand Croston/SBA/TSB/ADIDA/IMAPA and AutoTheta/AutoCES/Holt-Winters/MSTL/AutoMFLES) + CatBoost | **Coded**; statsforecast/catboost verification is on the user's machine (`golden`) |
| C4 | M3/M4 benchmarks drift, Naive2, Comb S-H-D (numpy), DOTM and SCUM (statsforecast); optional measures RMSSE, MSE, MAPE, MAAPE, MedAE, RMSLE, ME, PBias; `ranking.py` + `rank` command (Friedman, Nemenyi CD, AvgRelMAE, OWA, bootstrap rank stability) | **Coded and unit-tested** (2026-11-22); see docs/RESEARCH_MODELS_METRICS.md for what to add next |
| C5 | `reliability.py` + `reliability` command: ranking reliability across tasks with incomplete model coverage (common-support ranks, family-balanced scores, task-balanced Bradley-Terry with task bootstrap, permutation Skillings-Mack, Kendall's W, cross-metric selection regret, leave-one-family-out) | **Coded and tested** (2026-11-22); reproduces the manuscript's Bradley-Terry ordering and Skillings-Mack statistic (`tests/test_reliability.py`). Command line only; the P/S/X evidence screen is not ported |
| C6 | `instability.py` + `instability` command: metric disagreement and selection regret per series (measures that are equivalent inside a series count once), the aggregation effect (task-level concordance of such measures), six built-in series features, within-task feature association pooled over tasks | **Coded and tested** (2026-11-22); produces the per-series values of manuscript Section 5.5 (`tests/test_instability.py`). Built-in features approximate R's tsfeatures (acf1, CV, level identical; entropy and trend approximate); R tables can be passed with `--features`. Command line only |
| D | Deep models: the research report recommends `neuralforecast` (N-HiTS, N-BEATS) and one foundation model (Chronos-Bolt / Chronos-2) over GluonTS | – |
| E | WRMSSE (M5), data features (tsfeatures/catch22/lambda), rolling origin | – |
| F | Local interface (in the browser; standard library only): data/model selection, margin-note explanations, editable parameters, pre-run check, progress/cancel, result charts, history and comparison, Excel upload | **Coded** (docs/APP.md) |

## Claude Code agents
`.claude/agents/` defines three project agents for Claude Code: `test-manager` (runs and triages the test suite),
`ui-design-reviewer` (evaluates the web interface; latest report: `docs/UI_REVIEW.md`) and `forecasting-researcher`
(alternative models, error measures and ranking methods; latest report: `docs/RESEARCH_MODELS_METRICS.md`).

## Installation and tests

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
make test            # needs neither R nor torch
```

## Interface (running it as an application)

```powershell
.\ForecastStudio.bat                                  # or: .venv\Scripts\python.exe -m tsfstudio app
```
`http://127.0.0.1:8765/` opens in the browser (this computer only; closed to the outside network). The interface is in English. Details: `docs/APP.md`.

## Quick start A: no Docker, local Python (Windows; no administrator rights needed)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\setup_native.ps1   # .venv + packages + doctor + golden
.\scripts\windows\tsfs.ps1 run --dataset nn5_weekly --method theta           # default engine: python
.\scripts\windows\tsfs.ps1 golden                                            # comparison with your own R results
```
The Python engine does not need R (docs/PYTHON_ENGINE.md). **Python 3.12 is recommended** (`py install 3.12`); very new versions such as 3.15 lack ready-made wheels for some packages and the installation then requires a C++ compiler. Put the project in a folder outside OneDrive (so that `.venv` and `data\tsf`, ~2 GB, are not synchronised). The R engine works only in Docker (below) or locally if R is installed.

## Quick start B: Docker (R engine + Python engine)

Details: `docs/DOCKER_WINDOWS.md`.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1      # check + image + doctor
.\scripts\windows\run.ps1 estimate --dataset nn5_weekly --method tbats  # memory/time estimate without running
.\scripts\windows\run.ps1 run --dataset nn5_weekly --engine r --method ets
.\scripts\windows\golden-test.ps1                                       # does the R engine reproduce your R output?
```

Outputs are written to `runs/<data>__<engine>-<method>__<time>/`: `forecasts.csv/.xlsx`, `test_actuals.csv`, `metrics.xlsx`
(Errors_Per_Series, Summary, Run_Info), `fallbacks.csv`, `manifest.json`.

## Data package

Put all .tsf files into **a single folder** (`data/tsf/`): the 24 files in the Drive folder `tsf_data` + the 17 files in `not used`
+ the two M5 files (`m5_daily_dataset.tsf`, `m5_item_state_daily_dataset.tsf`) = 43 files, about 1.98 GB.

```bash
make data-check      # compares file names and byte sizes with the catalogue
make index           # produces the summary of number of series, length, share of missing/zero (_index.json)
make hash            # SHA256SUMS.txt (data fingerprint for experiment manifests)
```

The catalogue (`src/tsfstudio/data/catalog.yaml`) extracts the lag, horizon, integer conversion and the lists of methods used in the study
from `reference/experiments/fixed_horizon.R` and `deep_learning_experiments.py` (`tools/build_catalog.py`).
Entries marked `lag_origin: suggested` (M5, sample, london_smart_meters) are not in the study; they are suggestions.
Datasets mentioned in your code but not in Drive: `kaggle_web_traffic_daily`, `saugeen_river_flow`, `sunspot`, `us_births`.

## Loading data from Excel

```bash
python -m tsfstudio template template.xlsx               # Instructions + Meta + Veri_Uzun + Veri_Genis
python -m tsfstudio convert mydata.xlsx mydata.tsf --fill-gaps
```

The long format is `series_name | date | value | [constant attributes]`, the wide format is `date | series1 | series2 ...`.
Errors are reported with the Excel cell address (`Veri_Uzun!C7`): first cell errors, and once those are fixed, structural errors (duplicate dates,
gaps, irregular frequency). Monthly/quarterly/yearly dates are moved to the 1st day of the period. Non-ASCII characters are converted to ASCII.
(The data sheet names `Veri_Uzun` = long data and `Veri_Genis` = wide data are kept for compatibility with files made earlier.)

## Ranking reliability across tasks

```powershell
# one run folder per task and model; a model may be missing on some tasks
.\scripts\windows\tsfs.ps1 reliability --run nn5_weekly:ets=runs\... --run nn5_weekly:theta=runs\... --run m3_monthly:ets=runs\... --out reliability_out
# or from a table of mean losses with the columns task, metric, model, loss
.\scripts\windows\tsfs.ps1 reliability --losses losses.csv
```
Prints the task-balanced Bradley-Terry ordering with each model's coverage, P(first) and P(top k); the Skillings-Mack test;
Kendall's W and selection regret per task; and the ordering after dropping each metric family. `rank` is the within-task
counterpart (blocks = series). The manuscript draft is in `manuscript/Manuscript_v2.md`.

## Where do the measures disagree?

```powershell
# at least three models per task; the data file is taken from the run manifest
.\scripts\windows\tsfs.ps1 instability --run nn5_weekly:ets=runs\... --run nn5_weekly:theta=runs\... --run nn5_weekly:ses=runs\... --out instability_out
# with features computed elsewhere (first column = series name), e.g. R tsfeatures
.\scripts\windows\tsfs.ps1 instability --run ... --features nn5_weekly=my_features.csv
```
For each task it prints how much the measures disagree inside single series, whether measures that are equivalent
inside a series (MAE, MASE, WAPE; or RMSE, MSE, RMSSE) still rank the models differently after averaging over series,
and how series features relate to the disagreement. With five or more tasks the correlations are pooled with tasks as
the units. The analysis behind manuscript Section 5.5 is documented in `manuscript/analysis/README.md`.

## Structure

```
src/tsfstudio/
  tsf/reader.py, writer.py   Monash format (compatible with Python ≥ 3.12; byte-level round-trip test on the M5 files)
  seasonality.py             frequency, seasonality and timestamp rules (identical to the R code)
  catalog.py, data/catalog.yaml
  metrics.py                 SMAPE, mSMAPE, MASE, MAE, RMSE, WAPE, nRMSE  (docs/METRICS.md)
  excel/template.py, converter.py
  engines/                   Engine interface; py_engine.py (Python), py_models.py/py_global.py/py_sf.py, r_engine.py (Rscript), baseline.py, registry.py
  golden.py                  compares engine forecasts with your own R outputs
  r/worker.R, r/lib/*.R      R worker + corrected original models (docs/CHANGES_R.md)
  runner.py                  split, missing-value policy, engine, measures, outputs, manifest
  estimate.py                memory (cgroup) and time (NN5 priors / trial run) estimate
  cli.py                     data-check, index, hash, template, convert, engines, estimate, run, doctor, golden, app
  app/                       local interface: server.py (http.server), service.py (business logic), models_info.py (model/parameter/reference schema), static/
reference/                   the user's original R/Python code (NOT MODIFIED)
tools/                       catalogue builder, M5 TSF production/verification scripts
docs/METRICS.md              definitions, differences from the original, problems fixed
docs/CHANGES_R.md            changes in the R code (line-by-line rationale)
docs/PYTHON_ENGINE.md        Python engine: method table, relation to R, deliberate differences
docs/APP.md                  interface: use, security, limits, references
docs/DOCKER_WINDOWS.md       installation, .wslconfig, troubleshooting
docs/VALIDATION.md           golden tests: what, how, results
Dockerfile, docker-compose.yml, docker/, scripts/windows/
```

Language note: the interface, the run messages and the CLI output are in English. Source comments, the older documents (`docs/METRICS.md`, `docs/CHANGES_R.md`, `docs/PYTHON_ENGINE.md`, `docs/VALIDATION.md`, `docs/DOCKER_WINDOWS.md`) and the Windows set-up scripts are still in Turkish.

## Reproducibility

The data fingerprint (`make hash`), the source of the catalogue defaults (`tools/build_catalog.py`), fixed version limits
(`pyproject.toml`) and hand-computed tests of the measures are in the repository. Every run is written to `manifest.json` with its configuration, seed, R/package versions, data SHA-256, skipped series, fallback counts and warnings.
On the R side, the fixed tag `rocker/r-ver:4.5.1` ties CRAN to the date on which that version was released.

## References

- Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J., Montero-Manso, P. (2021). Monash Time Series Forecasting Archive. NeurIPS Datasets and Benchmarks.
- Hyndman, R. J., Koehler, A. B. (2006). Another look at measures of forecast accuracy. International Journal of Forecasting, 22(4).
- Makridakis, S., Spiliotis, E., Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. International Journal of Forecasting, 38(4).
