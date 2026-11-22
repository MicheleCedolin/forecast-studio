# Yerel .venv icindeki tsfstudio komutunu calistirir. Ornekler:
#   .\scripts\windows\tsfs.ps1 doctor
#   .\scripts\windows\tsfs.ps1 estimate --dataset nn5_weekly --method theta
#   .\scripts\windows\tsfs.ps1 run --dataset nn5_weekly --method theta
#   .\scripts\windows\tsfs.ps1 golden --case nn5_weekly
# Cikti boruya (Tee-Object) giderken Turkce karakterler icin UTF-8
$env:PYTHONUTF8 = "1"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
Set-Location (Join-Path $PSScriptRoot "..\..")
& ".\.venv\Scripts\python.exe" -m tsfstudio @args
exit $LASTEXITCODE
