#!/usr/bin/env bash
# One-time set-up on the LOGIN node of a cluster (it needs the internet; compute nodes usually have none).
#   bash scripts/hpc/setup.sh
# Makes a Python environment in .venv, builds the C++ core (tsfcore.so) and checks the result. See docs/HPC.md.
#
# Load a compiler, CMake and Python first if the cluster uses modules, for example:
#   module load gcc/13 cmake python/3.11
# Options (environment variables):
#   PYTHON=python3.11     the interpreter to build the environment from (default python3; 3.10 or newer)
#   WITH_CATBOOST=1       also download the CatBoost 1.2.10 Linux binaries for the global catboost model (about 300 MB)
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

"${PYTHON:-python3}" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# lightgbm: its package carries lib_lightgbm.so, which the C++ engine loads for the global lightgbm models.
pip install -e ".[dev]" "lightgbm>=4.0"

bash cpp/build.sh

if [ "${WITH_CATBOOST:-0}" = "1" ]; then
  # Names the C++ engine looks for: third_party/catboost/catboost and libcatboostmodel.so (cpp/README.md, phase 6b).
  base="https://github.com/catboost/catboost/releases/download/v1.2.10"
  mkdir -p third_party/catboost
  curl -fL -o third_party/catboost/catboost "$base/catboost-linux-x86_64-1.2.10"
  curl -fL -o third_party/catboost/libcatboostmodel.so "$base/libcatboostmodel-linux-x86_64-1.2.10.so"
  chmod +x third_party/catboost/catboost
  sha256sum third_party/catboost/catboost third_party/catboost/libcatboostmodel.so | tee third_party/catboost/SHA256SUMS.txt
fi

echo
python -m tsfstudio engines
echo
echo "Agreement of this build with the reference forecasts (a few minutes):"
python -m pytest tests/test_cpp_engine.py tests/test_golden_nn5.py tests/test_sweep.py -q -p no:cacheprovider || \
  echo "Some checks differ on this machine: read the failing lines above and docs/HPC.md, 'Agreement on Linux'."
echo
python -m tsfstudio data-check || true
