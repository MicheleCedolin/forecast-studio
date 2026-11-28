# Windows 11 + Docker Desktop setup

Machine: i5-8400 (6 cores), 8 GB RAM. Docker Desktop runs with the **WSL2 backend**.

## 1. One-time steps
1. Install Docker Desktop, restart the computer, open Docker Desktop and wait until it shows "Engine running".
2. WSL2 memory: the default is 50% of RAM (~4 GB). Save the contents of `scripts\windows\wslconfig.example` as
   `C:\Users\<user>\.wslconfig` (`memory=6GB`, `swap=8GB`, `processors=5`), then run
   `wsl --shutdown` in PowerShell and restart Docker Desktop. (`setup.ps1` does not write it without asking.)
3. Open the project folder and put the 43 `.tsf` files into `data\tsf\`.
4. `powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1`
   (If the script is blocked: `Get-ChildItem -Recurse | Unblock-File`.)

The first image build can take 15–40 min (R packages + a data layer of about 2 GB). Later code changes do not rebuild the data
layer.

The image installs the R packages `forecast` and `jsonlite` and, if it can, CatBoost for R (`Dockerfile`). Methods of the R engine that need
further R packages (thief, MAPA, smooth, tsintermittent, forecTheta; `extra_packages` in `src/tsfstudio/r/lib/extra_models.R`) are not
listed in the image unless those packages are added.

## 2. Daily use
```powershell
.\scripts\windows\run.ps1 doctor                                   # environment + a quick live test of each available method
.\scripts\windows\run.ps1 estimate --dataset m5_item_store --task global --method pooled_regression --lag 12
.\scripts\windows\run.ps1 run --dataset nn5_weekly --engine r --method ets
.\scripts\windows\golden-test.ps1
```
`estimate` predicts memory and run time without running; `run` asks for confirmation on long runs (>15 min estimated) (skipped with `--yes`).

## 3. Troubleshooting
| Symptom | Cause / fix |
|---|---|
| `docker: command not found` | Docker Desktop is not installed or the computer was not restarted |
| "Docker calismiyor" (Docker is not running; message of `setup.ps1`) | Open Docker Desktop and wait for "Engine running" |
| `The R worker terminated unexpectedly (code 137)` | Out of memory (OOM). Increase the memory in `.wslconfig`, use `max_obs` or `--max-series` |
| Very slow | `.wslconfig` `processors`, `--workers 4` (local models; on Windows R itself does not run in parallel, in Docker/Linux it does) |
| `doctor`: catboost WARNING | The CatBoost R package could not be installed; rebuild the image with `--build-arg CATBOOST_REQUIRED=1` and read the error message |
| Image too large | Mount the data with the bind mount in `docker-compose.yml` and delete the `COPY data/tsf` line in the `Dockerfile` |
