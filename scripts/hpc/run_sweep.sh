#!/usr/bin/env bash
# Forecast Studio sweep on a machine WITHOUT a job scheduler (a shared server reached by ssh): runs in the background,
# survives the end of the ssh session, and uses a fixed number of cores. With SLURM use scripts/hpc/sweep.sbatch.
#
#   bash scripts/hpc/run_sweep.sh                         # 32 cores: 4 tasks at a time x 8 threads
#   SWEEP_JOBS=32 SWEEP_THREADS=1 bash scripts/hpc/run_sweep.sh     # 32 tasks at a time, single-thread timings
#   PROBE=20 bash scripts/hpc/run_sweep.sh                # time estimate on 20 series per dataset
#   tail -f results/sweep/sweep.log                       # follow it
#   kill "$(cat results/sweep/sweep.pid)"                 # stop it (the running tasks are recorded as interrupted)
#
# Running the same command again continues: finished tasks are skipped. Run from the repository root. See docs/HPC.md.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${SWEEP_CORES:=32}"                       # the cores you may use
: "${SWEEP_JOBS:=4}"                         # (dataset, method) tasks running at the same time
: "${SWEEP_THREADS:=$(( SWEEP_CORES / SWEEP_JOBS ))}"   # threads inside each task; JOBS x THREADS = cores
: "${SWEEP_OUT:=results/sweep}"
: "${SWEEP_DATASETS:=all}"                   # all | main | extra | m5 | ids, comma separated
: "${SWEEP_METHODS:=study}"                  # study | benchmarks | local | intermittent | global | all | method names
: "${SWEEP_EXCLUDE_DATASETS:=}"
: "${SWEEP_EXCLUDE_METHODS:=}"
: "${SWEEP_TASK_TIMEOUT_MIN:=1440}"          # one (dataset, method) may take this long
: "${SWEEP_EXTRA:=}"                         # further options, e.g. "--missing-policy locf --metrics all"
: "${TSFSTUDIO_DATA_DIR:=$PWD/data/tsf}"
export TSFSTUDIO_DATA_DIR
[ "$SWEEP_THREADS" -lt 1 ] && SWEEP_THREADS=1

source .venv/bin/activate
export OMP_NUM_THREADS="$SWEEP_THREADS" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

args=(--out "$SWEEP_OUT" --datasets "$SWEEP_DATASETS" --methods "$SWEEP_METHODS" --jobs "$SWEEP_JOBS"
      --threads "$SWEEP_THREADS" --task-timeout "$SWEEP_TASK_TIMEOUT_MIN")
[ -n "$SWEEP_EXCLUDE_DATASETS" ] && args+=(--exclude-datasets "$SWEEP_EXCLUDE_DATASETS")
[ -n "$SWEEP_EXCLUDE_METHODS" ] && args+=(--exclude-methods "$SWEEP_EXCLUDE_METHODS")
[ -n "${PROBE:-}" ] && args+=(--probe "$PROBE")

mkdir -p "$SWEEP_OUT"
if [ -f "$SWEEP_OUT/sweep.pid" ] && kill -0 "$(cat "$SWEEP_OUT/sweep.pid")" 2>/dev/null; then
  echo "A sweep is already running in $SWEEP_OUT (pid $(cat "$SWEEP_OUT/sweep.pid")); stop it first or choose another SWEEP_OUT."
  exit 1
fi
echo "=== $(date -Is) $(hostname): python -m tsfstudio sweep ${args[*]} $SWEEP_EXTRA" >> "$SWEEP_OUT/sweep.log"
# shellcheck disable=SC2086
nohup python -m tsfstudio sweep "${args[@]}" $SWEEP_EXTRA >> "$SWEEP_OUT/sweep.log" 2>&1 &
echo $! > "$SWEEP_OUT/sweep.pid"
echo "Started (pid $!). Log: $SWEEP_OUT/sweep.log"
