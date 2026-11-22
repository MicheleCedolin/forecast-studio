# Builds tsfcore.dll with CMake and copies it to src/tsfstudio/native/ (where the Python "cpp" engine loads it).
# Usage (from the repository root):  powershell -ExecutionPolicy Bypass -File cpp\build.ps1
# Needs a C++17 compiler and CMake on PATH (for example WinLibs MinGW-w64: winget install BrechtSanders.WinLibs.POSIX.UCRT).
param([string]$Generator = "MinGW Makefiles")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$build = Join-Path $root "cpp\build"
# -Generator "" uses CMake's default (Visual Studio on GitHub's Windows runners).
$gen = @(); if ($Generator) { $gen = @("-G", $Generator) }
cmake -S (Join-Path $root "cpp") -B $build @gen -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
cmake --build $build --config Release
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }
$dll = Get-ChildItem $build -Recurse -Filter "tsfcore.dll" | Select-Object -First 1
if (-not $dll) { throw "tsfcore.dll not found in $build" }
$dest = Join-Path $root "src\tsfstudio\native"
New-Item -ItemType Directory -Force $dest | Out-Null
Copy-Item $dll.FullName (Join-Path $dest "tsfcore.dll") -Force
Write-Host "Built: $(Join-Path $dest 'tsfcore.dll')"
