#!/usr/bin/env bash
# Builds tsfcore.so with CMake and copies it to src/tsfstudio/native/ (where the Python "cpp" engine loads it).
# The Linux counterpart of cpp/build.ps1. Usage (from anywhere):  bash cpp/build.sh
# Needs a C++17 compiler (GCC >= 8) and CMake >= 3.16 on PATH (on a cluster: module load gcc cmake, or a conda
# environment with gxx_linux-64 and cmake; scripts/hpc/setup_conda.sh does that for systems with an old compiler).
#   TSF_STATIC_RUNTIME=ON bash cpp/build.sh    libstdc++ inside tsfcore.so (compiler newer than the system's)
#
# Use GCC (or Clang) with the flags of CMakeLists.txt as they are. The agreement with R and Python documented in
# cpp/README.md rests on IEEE arithmetic without reassociation and without FMA contraction: do not add -march=native,
# -ffast-math or -Ofast, and do not build with a compiler whose default floating-point model is "fast" (Intel icpx).
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$root/cpp/build"
cmake -S "$root/cpp" -B "$build" -DCMAKE_BUILD_TYPE=Release -DTSF_STATIC_RUNTIME="${TSF_STATIC_RUNTIME:-OFF}"
cmake --build "$build" --config Release -j "${TSF_BUILD_JOBS:-4}"
dest="$root/src/tsfstudio/native"
mkdir -p "$dest"
cp "$build/tsfcore.so" "$dest/tsfcore.so"
echo "Built: $dest/tsfcore.so"
