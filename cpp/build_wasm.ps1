# Builds tsfcore for the browser (WebAssembly) with Emscripten and writes web\tsfcore.js + web\tsfcore.wasm.
# Usage (from the repository root):  powershell -ExecutionPolicy Bypass -File cpp\build_wasm.ps1 [-Emsdk C:\path\to\emsdk]
# Needs the Emscripten SDK (https://emscripten.org/docs/getting_started/downloads.html):
#   git clone https://github.com/emscripten-core/emsdk ; cd emsdk ; .\emsdk.bat install latest ; .\emsdk.bat activate latest
# The browser build is single-threaded and has no CatBoost / LightGBM (they need native libraries). Files live in the
# module's in-memory file system: the page writes the .tsf file there, calls tsf_run and reads the result files back.
param([string]$Emsdk = (Join-Path $env:USERPROFILE "emsdk"))
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$emcc = Join-Path $Emsdk "upstream\emscripten"
if (-not (Test-Path (Join-Path $emcc "em++.bat")) -and -not (Test-Path (Join-Path $emcc "em++.exe"))) {
    throw "Emscripten not found in $Emsdk (pass -Emsdk <folder>)"
}
$env:EMSDK = $Emsdk
$env:EM_CONFIG = Join-Path $Emsdk ".emscripten"
$node = Get-ChildItem (Join-Path $Emsdk "node") -Recurse -Filter node.exe | Select-Object -First 1
$env:PATH = "$emcc;$($node.DirectoryName);$env:PATH"

$obj = Join-Path $root "cpp\build_wasm"
$web = Join-Path $root "web"
New-Item -ItemType Directory -Force $obj, $web | Out-Null
$objects = @()
foreach ($f in Get-ChildItem (Join-Path $root "cpp\src") -Filter *.cpp) {
    $o = Join-Path $obj "$($f.BaseName).o"
    if (-not (Test-Path $o) -or (Get-Item $o).LastWriteTime -lt $f.LastWriteTime) {
        Write-Host "em++ $($f.Name)"
        # -O3 without -ffast-math, as the native build: IEEE semantics
        cmd /c "em++ -std=c++17 -O3 -fexceptions -I `"$($f.DirectoryName)`" -c `"$($f.FullName)`" -o `"$o`""
        if ($LASTEXITCODE -ne 0) { throw "compiling $($f.Name) failed" }
    }
    $objects += "`"$o`""
}
foreach ($f in Get-ChildItem (Join-Path $root "cpp\src\catch22") -Filter *.c) {
    $o = Join-Path $obj "catch22_$($f.BaseName).o"
    if (-not (Test-Path $o) -or (Get-Item $o).LastWriteTime -lt $f.LastWriteTime) {
        cmd /c "emcc -O2 -std=gnu99 -w -c `"$($f.FullName)`" -o `"$o`""
        if ($LASTEXITCODE -ne 0) { throw "compiling $($f.Name) failed" }
    }
    $objects += "`"$o`""
}
$exports = "_malloc,_free,_tsf_version,_tsf_methods,_tsf_run,_tsf_forecast_batch,_tsf_io_open,_tsf_io_open_buffer,_tsf_io_close,_tsf_io_n_series," +
           "_tsf_io_n_attributes,_tsf_io_series,_tsf_io_attr,_tsf_io_header_json,_tsf_io_stats_json,_tsf_metrics_names," +
           "_tsf_metrics_table,_tsf_metrics_summary,_tsf_primary_period,_tsf_catch22,_tsf_catch22_count,_tsf_catch22_name,_tsf_boxcox_lambda"
$runtime = "FS,UTF8ToString,stringToUTF8,lengthBytesUTF8,HEAPU8,HEAPF64,HEAP64,HEAP32"
$out = Join-Path $web "tsfcore.js"
Write-Host "linking $out"
cmd /c "em++ -O3 -fexceptions $($objects -join ' ') -o `"$out`" -sMODULARIZE=1 -sEXPORT_NAME=createTsfCore -sENVIRONMENT=web,worker,node -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sSTACK_SIZE=8MB -sEXPORTED_FUNCTIONS=$exports -sEXPORTED_RUNTIME_METHODS=$runtime"
if ($LASTEXITCODE -ne 0) { throw "linking failed" }
Get-ChildItem $web -Filter "tsfcore.*" | ForEach-Object { Write-Host ("Built: {0} ({1:N0} KB)" -f $_.FullName, ($_.Length / 1KB)) }
