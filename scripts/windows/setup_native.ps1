# Forecast Studio - Docker'siz (yerel Python) kurulum. Yonetici yetkisi GEREKMEZ.
# Calistirma (PowerShell, proje klasorunde):
#   powershell -ExecutionPolicy Bypass -File scripts\windows\setup_native.ps1
# Zip'ten acilan dosyalar engelliyse once:  Get-ChildItem -Recurse | Unblock-File
param(
    [switch]$SkipOptional,   # statsforecast/catboost kurulumunu atla (yalnizca yerlesik yontemler)
    [switch]$NoGolden
)
$ErrorActionPreference = "Stop"
# Cikti boruya (Tee-Object) giderken Turkce karakterler icin UTF-8
$env:PYTHONUTF8 = "1"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
Set-Location (Join-Path $PSScriptRoot "..\..")

function Step($m) { Write-Host ""; Write-Host "==> $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "HATA: $m" -ForegroundColor Red; exit 1 }

Step "1/5 Python bulunuyor (3.10-3.13 tercih edilir)"
$candidates = @("py -3.12", "py -3.13", "py -3.11", "py -3.10", "python", "python3")
$chosen = $null; $fallback = $null
foreach ($c in $candidates) {
    $parts = $c.Split(" ")
    if (-not (Get-Command $parts[0] -ErrorAction SilentlyContinue)) { continue }
    $cargs = @(); if ($parts.Length -gt 1) { $cargs += $parts[1..($parts.Length - 1)] }
    $cargs += @("-c", "import sys;print('{}.{}'.format(*sys.version_info[:2]))")
    try { $ver = (& $parts[0] @cargs 2>$null) } catch { continue }
    if ($LASTEXITCODE -ne 0 -or -not $ver) { continue }
    $minor = [int]($ver.Split(".")[1]); $major = [int]($ver.Split(".")[0])
    if ($major -ne 3 -or $minor -lt 10) { continue }
    if ($minor -le 13) { $chosen = @{ cmd = $c; ver = $ver }; break }
    if (-not $fallback) { $fallback = @{ cmd = $c; ver = $ver } }
}
if (-not $chosen -and $fallback) {
    $chosen = $fallback
    Write-Host "UYARI: Python $($chosen.ver) bulundu. Cok yeni surumlerde numba/statsforecast/torch tekerlekleri (wheel) henuz olmayabilir." -ForegroundColor Yellow
    Write-Host "       ONERI: PowerShell'de  py install 3.12  yazin (yonetici yetkisi gerekmez), sonra bu betigi tekrar calistirin." -ForegroundColor Yellow
    Write-Host "       3.15'te PyYAML icin C++ derleyicisi gerekir ve kurulum basarisiz olabilir." -ForegroundColor Yellow
}
if (-not $chosen) { Fail "Python 3.10+ bulunamadi. 'python --version' ve 'py --list' ciktisini bana gonderin." }
Write-Host ("Kullanilacak Python: " + $chosen.cmd + "  (surum " + $chosen.ver + ")")

Step "2/5 Sanal ortam (.venv)"
$p0 = $chosen.cmd.Split(" ")
$margs = @(); if ($p0.Length -gt 1) { $margs += $p0[1..($p0.Length - 1)] }
# Onceki denemeden kalan .venv baska bir Python surumuyle olusturulmus olabilir (ornegin 3.15); o zaman yeniden olustur.
if (Test-Path ".venv\Scripts\python.exe") {
    $vv = $null
    try { $vv = (& ".venv\Scripts\python.exe" -c "import sys;print('{}.{}'.format(*sys.version_info[:2]))" 2>$null) } catch { }
    if ($vv -ne $chosen.ver) {
        Write-Host ("Mevcut .venv Python " + $vv + " ile olusturulmus; " + $chosen.ver + " ile yeniden olusturuluyor.") -ForegroundColor Yellow
        Remove-Item -Recurse -Force .venv
    }
}
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $p0[0] @margs -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail "Sanal ortam olusturulamadi." }
}
$py = (Resolve-Path ".venv\Scripts\python.exe").Path

Step "3/5 Cekirdek paketler (numpy, scipy, pandas, openpyxl, xlsxwriter, pyyaml, pytest)"
& $py -m pip install --upgrade pip
& $py -m pip install -e ".[dev]"
if ($LASTEXITCODE -ne 0) { Fail "Cekirdek kurulum basarisiz. Kurum agi PyPI'ye proxy gerektiriyor olabilir; hata satirlarini bana gonderin." }

if (-not $SkipOptional) {
    Step "4/5 Istege bagli: statsforecast (ets/arima/tbats/dhr_arima) ve catboost (global CatBoost)"
    & $py -m pip install -e ".[pyengine]"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "UYARI: Istege bagli paketler kurulamadi. Yerlesik yontemler (naive, snaive, ses, theta, pooled_regression) calisir." -ForegroundColor Yellow
        Write-Host "       Hatanin son satirlarini bana gonderin; surume uygun bir cozum bulalim." -ForegroundColor Yellow
    }
} else { Step "4/5 Istege bagli paketler atlandi" }

Step "5/5 Ortam testi (doctor) ve R sonuclarinizla HIZLI karsilastirma (golden)"
& $py -m tsfstudio doctor --data-dir data\tsf
if (-not $NoGolden) {
    # Yalnizca hizli yontemler (birkac dakika). Yavas olanlar (arima, dhr_arima, tbats) bilerek AYRI: once sure olcumu.
    & $py -m tsfstudio golden --methods ses theta pooled_regression ets --csv golden_hizli.csv
}
Write-Host ""
Write-Host "Tamam. Yavas yontemleri once KUCUK denemeyle (8 seri) olcun, sonra tam kosu icin bana yazin:" -ForegroundColor Green
Write-Host "  .\scripts\windows\tsfs.ps1 golden --case nn5_weekly --methods tbats --limit 8" -ForegroundColor Green
Write-Host "Cikti: doctor ve golden ekranini bana gonderin; golden_hizli.csv de ise yarar." -ForegroundColor Green
