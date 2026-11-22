# Forecast Studio 0.1.1: Windows package

Runs without installing Python. Download `ForecastStudio-0.1.1-win64.zip`, unzip it outside OneDrive and double-click
`ForecastStudio.exe`. See `README.txt` inside the zip.

- **File:** `ForecastStudio-0.1.1-win64.zip`, 224 MB (234,820,278 bytes), Windows 10/11 64-bit
- **SHA-256:** `73F86588E3AA546FC1046F08DAF636621AF037AEDC0B040218C88F852BE72333`
- **Built from:** commit `fa78802` (Python 3.12, PyInstaller 6.22.3). Tested in a fresh folder: the Comb S-H-D example run gives a
  mean MASE of 0.8929, the same as the Python engine, and the interface answers.

## Changes since 0.1.0
**Results**
- **Holt fit corrected (affects Comb S-H-D).** The Holt and damped Holt parts could stop at a poor fit. On NN5 daily, Comb S-H-D
  was 5% worse in SMAPE than R's version. Now its mean SMAPE and MASE are within 0.4% of R (forecast 9.0.2), and Naive2 is
  identical to R. Rerun any Comb S-H-D results made with 0.1.0.
- Bootstrap ranking stability: tied methods share the "first" and "top k" probabilities.

**Interface**
- Dark mode, following the system setting.
- Reloading the page during a run shows its progress and the Cancel button again.
- Results warn when they belong to an earlier run than the current selection.
- New "Run configuration" block: parameters, seed, versions and data SHA-256. New "Load these settings" button.
- Fallback forecasts: you can see which series fell back, and why.
- Numbers use fixed decimals per measure, with units. Short run times are shown in seconds.
- The comparison of past runs no longer marks a "best" value across different datasets or horizons.
- Keyboard: the dataset and model lists are one Tab stop each, navigated with the arrow keys. Screen readers no longer re-read
  the margin note.
- On phones, past runs can be opened again.

## Known limits
- The program is not signed, so Windows SmartScreen may warn the first time ("More info" → "Run anyway").
- The first start can take 10–20 seconds.
