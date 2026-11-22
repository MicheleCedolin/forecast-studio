# Forecast Studio interface (Phase F)

A workbench that runs in the browser and is open to this computer only. It calls the same core as the command line (`runner.run`);
every run made in the interface produces the same `manifest.json` and output files as a command-line run, so the results are reproducible.
The interface language is English.

## Starting

```powershell
.\ForecastStudio.bat                                   # double-click it in the project folder, or run it like this
.venv\Scripts\python.exe -m tsfstudio app              # the same thing; extra options: --port 8800 --no-browser
.venv\Scripts\python.exe -m tsfstudio app --data-dir D:\tsf --runs-dir D:\runs --uploads-dir D:\uploads
```

The default address is `http://127.0.0.1:8765/` (if it is busy, the next 10 ports are tried). To stop, press `Ctrl+C` in the command window or use
**Quit application** in the left column of the interface (it asks for two clicks). Folder defaults: data `data/tsf` (`$TSFSTUDIO_DATA_DIR`),
runs `runs/` (`$TSFSTUDIO_RUNS_DIR`), uploads `uploads/` (`$TSFSTUDIO_UPLOADS_DIR`).

## Flow

| Step | What you do | In the background |
|---|---|---|
| 1 Data | Choose one of the 43 catalogue datasets or an uploaded `.tsf`/`.xlsx` file | When chosen, frequency, horizon, number of series, length, share of missing and zero values, demand pattern (ADI/CV²) and sample series are extracted automatically |
| 2 Model | 21 methods in three groups (local, intermittent demand, global); Python or R engine | If the engine is not installed the method is marked and cannot be started |
| 3 Settings | Horizon, lag, number of series, rounding to integers, missing values, measures, model-specific parameters | Defaults are the values from your study; a changed field is highlighted and one click returns it to the default |
| 4 Run | Pre-run check (number of series, horizon, memory, time), optional 8-series time measurement, start, cancel | A run longer than 15 minutes asks for a confirmation tick; it does not start if memory will not suffice |
| 5 Results | Summary table, forecast diagnostics, sample-series chart, error distribution, per-series error table, file download | Everything is read from the files under `runs/<run>/` |

**Margin note.** When you hover over (or focus with the keyboard) a model, a setting, a measure or a dataset, its explanation appears in the right column:
how it works, when to use it, what to watch out for, what each engine offers, run time, references. On narrow screens it opens as a bottom panel
that can be collapsed. In the references, the **citation verified** tag appears only on citations that were matched against a bibliographic record page; the others
are marked **citation to be verified** (check them before publication).

**Past runs.** The link in the left column lists all runs in the `runs/` folder. Two or more ticked runs are shown side by side per measure
(mean or median; the lowest value in each measure is highlighted). A warning is given for different datasets or horizons.

## Design decisions and their rationale

* **Standard library only, instead of Streamlit.** No extra package installation is needed (one dependency fewer on an 8 GB machine and in a OneDrive folder); the interface could be
  tested automatically in a real browser. The price: more code. If wished, the same `service.py` can later be put behind a
  Streamlit/FastAPI layer.
* **A single concurrent run.** Because the machine is limited in memory and cores, a second run is rejected (HTTP 409).
* **Reproducibility.** Every parameter that has a default is written explicitly to the manifest; parameters whose default is left to the library (e.g. CatBoost
  `iterations`) are not sent and do not appear in the manifest. Two runs that finish in the same second do not overwrite each other (the second gets `-2`).
* **Typeface.** Latin Modern (GUST Font License; `static/fonts/LICENSE-LatinModern.txt`) is served from files; no internet is needed.

## Security

The application binds to `127.0.0.1` only. Because another page in the browser can send requests to a local application too: (1) the `Host` header
may only be `127.0.0.1`/`localhost`/`[::1]` (against DNS rebinding); (2) every `/api` request must carry the random token generated at start-up; (3) the client never sends a file path: datasets are chosen by catalogue name or by `upload:<name>`, and downloads are
limited to the outputs recorded in the run's manifest; (4) uploads are limited to 400 MB, the uploaded file name is sanitised and an existing file is not overwritten.
These properties are tested in `tests/test_app_server.py`.

## Warnings for interpreting results

* **SMAPE** is a ratio in the range 0–2 (not a percentage); 0/0 cells are left out and counted in the "Excluded" column.
* **MAE, MASE and WAPE can reward zero forecasts on intermittent demand** (Hyndman, 2006). This is why *Share of zero forecasts* is
  shown in the forecast diagnostics; if more than half of the series received only 0 forecasts, a red warning is shown in addition. Do not trust such a table. For `car_parts` the cause is rounding
  to integers: the Croston family gives a flat demand rate (often below 0.5) and rounding turns it into 0. When an intermittent-demand
  method is chosen on a dataset whose rounding is on, the interface turns rounding off for you and says so explicitly; you can turn it back on.
* **Total bias** `(Σ forecast − Σ actual) / Σ actual` is a preliminary indicator. Inventory outcomes such as service level and stock cost are **not
  computed** in this version (see below).

## What this version does not include

Inventory simulation (service level, holding/shortage cost, bullwhip), additional measures (MSE, MAPE, RMSLE, MedAE, MdAPE; RMSSE and bias for intermittent demand),
WRMSSE, rolling origin, probabilistic forecast measures, deep models (PyTorch GluonTS) and optional mlforecast/Chronos-Bolt
methods. Which of them to add, and in what order, will be settled with the user.

## For further research (candidate reading list)

The aim of the project is to show the effect of forecasting models on inventory decisions beyond accuracy measures. The topics below are sources to be
read in that direction; **their citations have not been matched against record pages in this session, so verify them before use**:

1. The relation between forecast error and stock performance for intermittent demand: Syntetos & Boylan's work on SBA and inventory control; Teunter, Syntetos & Babai (2011, TSB; the citation is also marked "to be verified" in the interface).
2. Inconsistency of error measures on intermittent demand: Hyndman (2006, verified); Kolassa (2016) and Prestwich et al. (2014) on mean-based error measures.
3. Tail- and service-level-oriented evaluation: probabilistic measures such as pinball/CRPS, safety stock from quantile forecasts, and base-stock (order-up-to) policies.
4. WRMSSE and hierarchical reconciliation for M5 (Makridakis, Spiliotis, Assimakopoulos, 2022; the citation is in the README).

Open data and code: the Monash Forecasting Archive (Godahewa et al., 2021; verified) is the source of all datasets in this tool; Nixtla StatsForecast
(Apache-2.0) is the open-source implementation of the statistical methods in the Python engine; on the R side, the `forecast` package (Hyndman & Khandakar, 2008;
verified) and (not verified in this session) the `tsintermittent` package can serve as a separate reference implementation for intermittent-demand methods.

## File layout

```
src/tsfstudio/app/
  server.py        HTTP layer based on http.server (the security checks are here)
  service.py       business logic: catalogue, pre-run check, job management, reading results, upload
  models_info.py   model/parameter/measure/reference schema (checked by tests against the engines' real keys)
  static/          index.html, app.css, app.js (dependency-free), fonts/
tests/test_app_models_info.py, tests/test_app_server.py
```

If something goes wrong, details are written to `runs/_app_errors.log`; setting the environment variable `TSFSTUDIO_APP_LOG=1` prints every HTTP request to the command
window.
