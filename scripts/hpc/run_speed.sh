#!/usr/bin/env bash
# Engine speed study (tools/engine_speed_study.py): the same method on the C++, Python and R engines, and the C++
# engine on 1, 2, 4, ... threads. See docs/HPC_SPEED.md. This script does the work; scripts/hpc/speed.sge and
# scripts/hpc/speed.sbatch submit it to a scheduler.
#
# Without a scheduler (a server reached by ssh), from the repository root:
#   nohup bash scripts/hpc/run_speed.sh > speed.log 2>&1 &      # 32 cores, in the background
#   tail -f speed.log
#   SPEED_CORES=16 SPEED_ENGINES=cpp bash scripts/hpc/run_speed.sh
#
# Running the same command again continues: repetitions already in <SPEED_OUT>/runs.csv are skipped.
# Variables:
#   SPEED_CORES          cores to use; default: the slots the scheduler granted, otherwise 32
#   SPEED_THREADS        thread counts for the C++ engine; default 1,2,4,... up to SPEED_CORES
#   SPEED_OUT            output folder (default results/speed)
#   SPEED_DATASETS       catalogue ids, comma separated (default: the plan in tools/engine_speed_study.py)
#   SPEED_METHODS        these methods for every dataset, comma separated (default: the plan's; any method of the application)
#   SPEED_ENGINES        default cpp,python,r; an engine that is not installed is left out
#   SPEED_REF_DATASETS   datasets on which Python and R are timed too (default: the four smaller ones), or "all"
#   SPEED_REPEATS       repetitions of every C++ cell (default 1), SPEED_REF_REPEATS of every Python / R cell (default 1)
#   SPEED_TIMEOUT_MIN    limit for a single run in minutes (default 120); slower levels are skipped, not started
#   SPEED_EXTRA          further options of the tool, e.g. "--discard-runs" or "--dry-run"
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${SPEED_CORES:=${NSLOTS:-${SLURM_CPUS_PER_TASK:-32}}}"
if [ -z "${SPEED_THREADS:-}" ]; then
  t=1; SPEED_THREADS=1
  while [ $(( t * 2 )) -le "$SPEED_CORES" ]; do t=$(( t * 2 )); SPEED_THREADS="$SPEED_THREADS,$t"; done
  [ "$t" -ne "$SPEED_CORES" ] && SPEED_THREADS="$SPEED_THREADS,$SPEED_CORES"
fi
: "${SPEED_OUT:=results/speed}"
: "${SPEED_DATASETS:=}"
: "${SPEED_METHODS:=}"
: "${SPEED_ENGINES:=cpp,python,r}"
: "${SPEED_REPEATS:=1}"
: "${SPEED_REF_REPEATS:=1}"
: "${SPEED_TIMEOUT_MIN:=120}"
: "${SPEED_EXTRA:=}"
: "${TSFSTUDIO_DATA_DIR:=$PWD/data/tsf}"
export TSFSTUDIO_DATA_DIR

source scripts/hpc/activate.sh || exit 1

args=(--out "$SPEED_OUT" --threads "$SPEED_THREADS" --engines "$SPEED_ENGINES" --repeats "$SPEED_REPEATS"
      --ref-repeats "$SPEED_REF_REPEATS" --timeout-min "$SPEED_TIMEOUT_MIN")
[ -n "$SPEED_DATASETS" ] && args+=(--datasets "$SPEED_DATASETS")
[ -n "$SPEED_METHODS" ] && args+=(--methods "$SPEED_METHODS")
[ -n "${SPEED_REF_DATASETS:-}" ] && args+=(--ref-datasets "$SPEED_REF_DATASETS")

echo "host $(hostname) | $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2) | cores $SPEED_CORES | $(date)"
echo "python tools/engine_speed_study.py ${args[*]} $SPEED_EXTRA"
# shellcheck disable=SC2086
exec python tools/engine_speed_study.py "${args[@]}" $SPEED_EXTRA
