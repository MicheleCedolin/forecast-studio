#!/usr/bin/env bash
# Pre-flight check for the speed study (docs/HPC_SPEED.md), on the login node, from the repository root:
#   bash scripts/hpc/preflight.sh 2>&1 | tee preflight.txt
# Changes nothing. Prints system tools, cores, free space, network reach (conda-forge, Zenodo, CRAN), the conda
# installations and environments it finds, the Python and R that the study would use, and the data files present.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
sec() { printf '\n== %s\n' "$*"; }
ver() { if command -v "$1" >/dev/null 2>&1; then printf '%-10s %s  (%s)\n' "$1" "$("$1" --version 2>&1 | head -1 | cut -c1-80)" "$(command -v "$1")"; else printf '%-10s not found\n' "$1"; fi; }

sec "host"
echo "$(hostname) | $(date) | $(uname -sr) | glibc $(ldd --version 2>/dev/null | head -1 | awk '{print $NF}')"
[ -f /etc/os-release ] && grep -m1 PRETTY_NAME /etc/os-release

sec "system tools"
for t in g++ gcc cmake make python3 python R Rscript conda mamba git curl wget qsub qhost; do ver "$t"; done
command -v module >/dev/null 2>&1 && { echo "modules (gcc / cmake / python / R):"; module -t avail 2>&1 | grep -iE '^(gcc|cmake|python|r)[/-]' | head -20; }

sec "cores (login node)"
echo "nproc $(nproc)"
command -v lscpu >/dev/null 2>&1 && lscpu | grep -E '^(Model name|Socket|Core|Thread|CPU\(s\)|NUMA node\(s\))'

sec "SGE: queues and nodes"
command -v qconf >/dev/null 2>&1 && { echo "queues: $(qconf -sql 2>/dev/null | tr '\n' ' ')"; echo "parallel environments: $(qconf -spl 2>/dev/null | tr '\n' ' ')"; }
command -v qhost >/dev/null 2>&1 && qhost 2>/dev/null | head -30
command -v qquota >/dev/null 2>&1 && qquota 2>/dev/null | head -10

sec "free space"
df -h "$HOME" | tail -1
du -sh "$HOME" 2>/dev/null | tail -1

sec "network (HTTP status of a HEAD request; 000 = not reachable, any other code = reachable)"
for u in https://conda.anaconda.org/conda-forge/noarch/repodata.json.bz2 https://zenodo.org/records/4656125 https://cloud.r-project.org/ https://github.com; do
  code=$(curl -s -o /dev/null -m 20 -I -w '%{http_code}' "$u" 2>/dev/null); printf '%-70s %s\n' "$u" "${code:-000}"
done

sec "conda installations and environments"
for d in "${TSF_CONDA_ROOT:-}" "$HOME/miniforge3" "$HOME/miniforge3_research" "$HOME/mambaforge" "$HOME/miniconda3" "$HOME/anaconda3"; do
  [ -n "$d" ] && [ -x "$d/bin/conda" ] && { echo "$d"; "$d/bin/conda" env list 2>/dev/null | grep -v '^#' | sed 's/^/    /'; }
done

sec "Python of earlier work: ~/miniforge3_research/envs/clrp_env"
P="$HOME/miniforge3_research/envs/clrp_env/bin/python3"
if [ -x "$P" ]; then
  "$P" - <<'PY'
import sys
print(sys.version.split()[0], sys.executable)
for m in ("numpy", "scipy", "pandas", "yaml", "statsforecast", "sklearn", "numba"):
    try:
        print(f"  {m:14s} {getattr(__import__(m), '__version__', '?')}")
    except Exception:
        print(f"  {m:14s} MISSING")
PY
else
  echo "not found: $P"
fi

sec "environment of the study (tsfstudio), if already set up"
if ( source scripts/hpc/activate.sh >/dev/null 2>&1 ); then
  ( source scripts/hpc/activate.sh >/dev/null 2>&1; echo "python $(python --version 2>&1) ($(command -v python))"; ver Rscript; ver g++
    PYTHONPATH=src python -m tsfstudio engines 2>&1 | cut -c1-160 )
else
  echo "not set up yet (scripts/hpc/setup_conda.sh creates it)"
fi

sec "data files of the speed study (data/tsf)"
for f in nn5_weekly_dataset.tsf nn5_daily_dataset_without_missing_values.tsf m3_monthly_dataset.tsf m4_hourly_dataset.tsf m4_quarterly_dataset.tsf m4_monthly_dataset.tsf; do
  if [ -f "data/tsf/$f" ]; then printf '  ok       %-48s %s bytes\n' "$f" "$(wc -c < "data/tsf/$f")"; else printf '  MISSING  %s\n' "$f"; fi
done
echo
echo "preflight done"
