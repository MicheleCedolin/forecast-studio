# Forecast Studio - Windows kurulum ve dogrulama (Docker Desktop gerekir).
# Calistirma (PowerShell, proje klasorunde):
#   powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1
# Zip'ten acilan dosyalar "internetten indirildi" isaretli olabilir; calismazsa once:  Get-ChildItem -Recurse | Unblock-File
param(
    [switch]$SkipBuild,
    [switch]$NoDoctor
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")

function Step($m) { Write-Host ""; Write-Host "==> $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "HATA: $m" -ForegroundColor Red; exit 1 }

Step "1/5 Docker kontrolu"
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Fail "docker bulunamadi. Docker Desktop kurulu ve oturum acilmis olmali; kurulumdan sonra bilgisayari yeniden baslatin." }
docker version --format "{{.Server.Version}}" 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { Fail "Docker calismiyor. Docker Desktop'i acin ve 'Engine running' gorunene kadar bekleyin." }
Write-Host ("Docker sunucu surumu: " + (docker version --format "{{.Server.Version}}"))

Step "2/5 WSL2 bellek ayari"
$wslcfg = Join-Path $env:USERPROFILE ".wslconfig"
$suggest = Get-Content "scripts\windows\wslconfig.example" -Raw
if (Test-Path $wslcfg) {
    Write-Host "Mevcut .wslconfig bulundu (DEGISTIRILMEDI):"; Get-Content $wslcfg | ForEach-Object { Write-Host "   $_" }
    Write-Host "Onerilen: memory=6GB, swap=8GB, processors=5 (scripts\windows\wslconfig.example)."
} else {
    Write-Host "Onerilen .wslconfig icerigi:"; Write-Host $suggest
    $ans = Read-Host "Bu dosya $wslcfg konumuna yazilsin mi? [e/H]"
    if ($ans -match '^(e|evet|y|yes)$') {
        Set-Content -Path $wslcfg -Value $suggest -Encoding ASCII
        Write-Host "Yazildi. Etkili olmasi icin:  wsl --shutdown  sonra Docker Desktop'i yeniden baslatin, ardindan bu betigi tekrar calistirin." -ForegroundColor Yellow
    }
}

Step "3/5 Veri kontrolu (data\tsf)"
$tsf = @(Get-ChildItem "data\tsf" -Filter *.tsf -ErrorAction SilentlyContinue)
Write-Host ("data\tsf icinde " + $tsf.Count + " adet .tsf dosyasi var (beklenen: 43 = 41 Monash + 2 M5).")
if ($tsf.Count -eq 0) { Write-Host "UYARI: Veri yok. Dosyalari data\tsf klasorune koyun (README.md -> 'Veri')." -ForegroundColor Yellow }

if (-not $SkipBuild) {
    Step "4/5 Imaj olusturma (ilk seferde 15-40 dk surebilir; R paketleri ve ~2 GB veri katmani)"
    docker compose build
    if ($LASTEXITCODE -ne 0) { Fail "Imaj olusturulamadi. Yukaridaki son hata satirlarini bana gonderin." }
}

if (-not $NoDoctor) {
    Step "5/5 Ortam testi (doctor)"
    docker compose run --rm studio doctor
    if ($LASTEXITCODE -ne 0) { Write-Host "Doctor sorun bildirdi; ciktiyi bana gonderin." -ForegroundColor Yellow; exit 2 }
    docker compose run --rm studio data-check
}
Write-Host ""
Write-Host "Tamam. Ornek: .\scripts\windows\run.ps1 run --dataset nn5_weekly --engine r --method ets --max-series 10" -ForegroundColor Green
