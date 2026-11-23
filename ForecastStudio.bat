@echo off
rem Starts the Forecast Studio interface; it opens in the browser. To stop, press Ctrl+C in this window
rem or use the "Quit application" button in the interface. Extra option: ForecastStudio.bat --port 8800
cd /d "%~dp0"
rem Prefer .venv (setup_native.ps1); otherwise fall back to the build environment and run from the source tree.
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" if exist ".venv-build\Scripts\python.exe" (
  set "PY=.venv-build\Scripts\python.exe"
  set "PYTHONPATH=%~dp0src"
)
if not exist "%PY%" (
  echo ERROR: .venv not found. Set it up first: powershell -ExecutionPolicy Bypass -File scripts\windows\setup_native.ps1
  pause
  exit /b 1
)
"%PY%" -m tsfstudio app %*
if errorlevel 1 pause
