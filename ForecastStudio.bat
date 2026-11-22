@echo off
rem Starts the Forecast Studio interface; it opens in the browser. To stop, press Ctrl+C in this window
rem or use the "Quit application" button in the interface. Extra option: ForecastStudio.bat --port 8800
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv not found. Set it up first: powershell -ExecutionPolicy Bypass -File scripts\windows\setup_native.ps1
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m tsfstudio app %*
if errorlevel 1 pause
