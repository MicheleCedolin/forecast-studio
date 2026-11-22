# Standalone Windows package

`ForecastStudio.exe` runs the app without installing Python. It is a PyInstaller "one folder" build:
`dist\ForecastStudio\` holds the exe and an `_internal` folder (about 640 MB, mostly statsforecast/numba,
catboost, scipy and scikit-learn).

## For users
1. Unzip the `ForecastStudio` folder anywhere outside OneDrive (for example `C:\ForecastStudio`).
2. Double-click `ForecastStudio.exe`. A console window opens and the interface opens in the browser.
   The first start takes 10–20 seconds; later starts are faster.
3. Put `.tsf` files in `data\tsf` next to the exe. Runs are written to `runs\`, uploads to `uploads\`.
4. Command line works too: `ForecastStudio.exe run --file my.tsf --method theta`, `ForecastStudio.exe doctor`,
   `ForecastStudio.exe rank ...` (same commands as `python -m tsfstudio`).

The R engine is used only if R is installed on that computer (`Rscript` on PATH or `TSFSTUDIO_RSCRIPT`).
Windows SmartScreen may warn about an unsigned program the first time ("More info" → "Run anyway").

## Building
```powershell
py -3.12 -m venv .venv-build
.venv-build\Scripts\python.exe -m pip install -r packaging\requirements-build.txt
cd packaging
..\.venv-build\Scripts\pyinstaller.exe forecast_studio.spec --noconfirm --distpath ..\dist --workpath ..\build
```
Checked on 2026-11-22 (Windows 11, Python 3.12, PyInstaller 6.22.3):
- `ForecastStudio.exe engines`: Python, R and baseline engines are found.
- `ForecastStudio.exe run --file nn5_weekly.tsf --method scum --workers 4`: 30 s, the same MASE (0.8485) and SMAPE (0.1153)
  as the Python run, so worker processes work inside the package.
- `ForecastStudio.exe app --no-browser`: the interface answers on localhost.

Files: `launcher.py` (entry point; sets data/runs/uploads next to the exe, `freeze_support()` for worker processes),
`forecast_studio.spec` (bundles the package data: catalogue, web UI, R worker), `requirements-build.txt` (pinned versions).
