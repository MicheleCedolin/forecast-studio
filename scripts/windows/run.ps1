# Docker icindeki tsfstudio komutunu calistirir. Ornekler:
#   .\scripts\windows\run.ps1 doctor
#   .\scripts\windows\run.ps1 estimate --dataset nn5_weekly --method tbats
#   .\scripts\windows\run.ps1 run --dataset nn5_weekly --engine r --method ets
Set-Location (Join-Path $PSScriptRoot "..\..")
docker compose run --rm studio @args
exit $LASTEXITCODE
