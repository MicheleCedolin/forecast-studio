# Source this from the repository root: activates the Python environment made by scripts/hpc/setup.sh (.venv) or by
# scripts/hpc/setup_conda.sh (a conda environment, name in TSF_CONDA_ENV, default "tsfstudio"). Used by the launchers.
# Add the cluster's `module load ...` lines here if the environment needs them.
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
else
  # `conda activate` needs conda's shell function. A job submitted with `qsub -V` from a shell where conda is active
  # inherits PATH (so the conda program is found) but not the function: load it unless it is already defined.
  if [ "$(type -t conda)" != "function" ]; then
    for d in "${TSF_CONDA_ROOT:-}" "${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}" "$HOME/miniforge3" "$HOME/miniforge3_research" "$HOME/mambaforge" "$HOME/miniconda3" "$HOME/anaconda3"; do
      if [ -n "$d" ] && [ -f "$d/etc/profile.d/conda.sh" ]; then source "$d/etc/profile.d/conda.sh"; break; fi
    done
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "No .venv and no conda found: run scripts/hpc/setup.sh or scripts/hpc/setup_conda.sh first (docs/HPC.md)." >&2
    return 1 2>/dev/null || exit 1
  fi
  set +u                                     # conda's activation scripts read unset variables
  conda activate "${TSF_CONDA_ENV:-tsfstudio}" || { echo "conda environment '${TSF_CONDA_ENV:-tsfstudio}' not found" >&2; return 1 2>/dev/null || exit 1; }
  set -u
  # LightGBM's C library for the global lightgbm models, when conda keeps it outside the Python package.
  if [ -z "${TSFSTUDIO_LIGHTGBM_LIB:-}" ] && [ -f "$CONDA_PREFIX/lib/lib_lightgbm.so" ]; then
    export TSFSTUDIO_LIGHTGBM_LIB="$CONDA_PREFIX/lib/lib_lightgbm.so"
  fi
fi
