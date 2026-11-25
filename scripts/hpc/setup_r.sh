#!/usr/bin/env bash
# Optional: R with the forecast package, so that the speed study can also time the R engine (the reference
# implementation). Run on the login node after scripts/hpc/setup_conda.sh or scripts/hpc/setup.sh:
#   bash scripts/hpc/setup_r.sh
# Written for the two set-ups of docs/HPC.md and not yet run on a cluster: read its output.
#   conda environment: R, forecast and jsonlite come from conda-forge (about 1 GB more in the home file system).
#   .venv:             R must already be on PATH (module load R); forecast and jsonlite are installed into the
#                      user's R library from CRAN, compiled on the spot (several minutes, needs gcc and gfortran).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ -f .venv/bin/activate ]; then
  command -v Rscript >/dev/null 2>&1 || { echo "Rscript not on PATH: load the cluster's R module first (module avail R)." >&2; exit 1; }
  mkdir -p "${R_LIBS_USER:=$HOME/R/library}"
  export R_LIBS_USER
  Rscript -e 'for (p in c("forecast", "jsonlite")) if (!requireNamespace(p, quietly = TRUE)) install.packages(p, lib = Sys.getenv("R_LIBS_USER"), repos = "https://cloud.r-project.org")'
  echo "Add to scripts/hpc/activate.sh:  module load <R module>;  export R_LIBS_USER=$R_LIBS_USER"
else
  source scripts/hpc/activate.sh
  conda install -y -n "${TSF_CONDA_ENV:-tsfstudio}" -c conda-forge --override-channels r-base r-forecast r-jsonlite
  conda clean -a -y >/dev/null 2>&1 || true
  source scripts/hpc/activate.sh
fi
# Optional packages of three methods (thief, mapa, auto_ces). Without one of them that method is not offered on the R
# engine; the rest works. They are compiled from CRAN, which can fail on an old system: that is not an error here.
Rscript -e 'need <- setdiff(c("thief", "MAPA", "smooth"), rownames(installed.packages())); if (length(need)) try(install.packages(need, repos = "https://cloud.r-project.org"))' || true
echo
Rscript -e 'cat(R.version.string, "| forecast", as.character(packageVersion("forecast")), "\n"); for (p in c("thief", "MAPA", "smooth")) cat(p, if (requireNamespace(p, quietly = TRUE)) as.character(packageVersion(p)) else "not installed", "\n")'
python -m tsfstudio engines
