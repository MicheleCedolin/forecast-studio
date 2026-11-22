# Forecast Studio 0.1.0: Windows package

Runs without installing Python. Download `ForecastStudio-0.1.0-win64.zip`, unzip it outside OneDrive, and double-click
`ForecastStudio.exe`. The interface opens in your browser. See `README.txt` inside the zip.

- **File:** `ForecastStudio-0.1.0-win64.zip`, 224 MB (640 MB unzipped), Windows 10/11, 64-bit
- **SHA-256:** `6310E5CD246F7C5642A924A01497A3AABA90A41E409478B1323EF377F12A1859`
- **Built from:** commit `0cf8549` with Python 3.12, PyInstaller 6.22.3, statsforecast 2.1.1, catboost 1.2.10

## What it does
- Local web interface: choose a dataset (TSF or Excel), models and parameters, check the estimated time, run it,
  then view and download forecasts and error measures.
- Python engine:
  - built-in: naive, seasonal naive, SES, Theta, drift, Naive2, Comb S-H-D, pooled regression
  - statsforecast: ETS, ARIMA, DHR-ARIMA, CES, Auto Theta, DOTM, SCUM, Holt-Winters, MSTL, MFLES, and the
    intermittent-demand methods (Croston, SBA, TSB, ADIDA, IMAPA)
  - CatBoost (global)
- The R engine is used if R is installed on the computer.
- Error measures:
  - default: SMAPE, mSMAPE, MASE, MAE, RMSE, WAPE, nRMSE
  - optional: RMSSE, MSE, MAPE, MAAPE, MedAE, RMSLE, ME, PBias
- `rank` command: mean ranks, Friedman test, Nemenyi critical difference and bootstrap ranking stability across runs.
- Example data: `examples\nn5_weekly.tsf` (NN5, 111 weekly series).

## Checked before release
Unzipped into a fresh folder on Windows 11:
- `run --file examples\nn5_weekly.tsf --method theta` finished with mean MASE 0.8854, the same as the Python engine.
- The interface answered on localhost.
- In the build folder, a SCUM run with 4 worker processes gave the same metrics as Python.

## Known limits
- The program is not signed, so Windows SmartScreen may warn the first time ("More info" → "Run anyway").
- The first start can take 10–20 seconds.
- The arima and scum methods are slow on long daily series. Use the estimate step or `--max-series` first.
