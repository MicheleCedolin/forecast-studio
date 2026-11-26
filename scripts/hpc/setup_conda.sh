#!/usr/bin/env bash
# One-time set-up on a cluster whose own tools are too old (for example CentOS 7: g++ 4.8, CMake 2.8, Python 3.6).
# Python, CMake and a C++17 compiler come from conda-forge; nothing is installed system-wide. Run on the login node
# (needs the internet; the build takes about a minute of CPU):
#   bash scripts/hpc/setup_conda.sh
# Then check the build on a compute node:  qsub scripts/hpc/check.sge   (see docs/HPC.md)
#
# Needs conda (or mamba) on PATH or in ~/miniforge3. If there is none, install Miniforge first (about 0.5 GB):
#   curl -fLO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
#   bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
# Options (environment variables):
#   TSF_CONDA_ENV=tsfstudio   name of the environment (about 1.5 GB; check the free space of the home file system first)
#   WITH_TESTS=0              leave out statsforecast / scikit-learn / pytest (needed only by the agreement checks)
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
env_name="${TSF_CONDA_ENV:-tsfstudio}"

if ! command -v conda >/dev/null 2>&1; then
  for d in "${TSF_CONDA_ROOT:-}" "$HOME/miniforge3" "$HOME/miniforge3_research" "$HOME/mambaforge" "$HOME/miniconda3" "$HOME/anaconda3"; do
    if [ -n "$d" ] && [ -f "$d/etc/profile.d/conda.sh" ]; then source "$d/etc/profile.d/conda.sh"; break; fi
  done
fi
command -v conda >/dev/null 2>&1 || { echo "conda not found; see the Miniforge lines at the top of this script." >&2; exit 1; }

echo "Free space:"; df -h "$HOME" | tail -1

# Versions: lightgbm 4.6.0 and statsforecast 2.1.1 are the ones the C++ engine was checked against (cpp/README.md).
pkgs=(python=3.11 numpy scipy pandas pyyaml openpyxl xlsxwriter pip cmake make gxx_linux-64 lightgbm=4.6.0)
[ "${WITH_TESTS:-1}" = "1" ] && pkgs+=(pytest statsforecast=2.1.1 scikit-learn)
if conda env list | awk '{print $1}' | grep -qx "$env_name"; then
  conda install -y -n "$env_name" -c conda-forge --override-channels "${pkgs[@]}"
else
  conda create -y -n "$env_name" -c conda-forge --override-channels "${pkgs[@]}"
fi
conda clean -a -y >/dev/null 2>&1 || true        # the package cache is not needed afterwards

set +u; conda activate "$env_name"; set -u
pip install --no-deps -e .

# conda's compiler ($CXX) is newer than the system's: keep its runtime inside tsfcore.so.
rm -rf cpp/build
TSF_STATIC_RUNTIME=ON bash cpp/build.sh
if ldd src/tsfstudio/native/tsfcore.so | grep -q "not found"; then
  echo "WARNING: tsfcore.so needs libraries the system cannot find:"; ldd src/tsfstudio/native/tsfcore.so | grep "not found"
  echo "Install them into the environment (for libquadmath: conda install -n $env_name -c conda-forge libgfortran) and run this script again."
fi

echo
source scripts/hpc/activate.sh
python -m tsfstudio engines
python -m tsfstudio data-check || true
echo
echo "Next: qsub scripts/hpc/check.sge   (agreement checks on a compute node), then docs/HPC.md, 'SGE'."
