#!/bin/bash
# Forecast Studio sweep on SLURM: every dataset x every method in turn, on one node. See docs/HPC.md.
#
#   sbatch scripts/hpc/sweep.sbatch                               # the whole plan in one job: 4 tasks at a time x 8 threads
#   SWEEP_JOBS=32 SWEEP_THREADS=1 sbatch scripts/hpc/sweep.sbatch # 32 tasks at a time, single-thread timings
#   sbatch --array=0-41%8 scripts/hpc/sweep.sbatch                # one dataset per array task (at most 8 at a time)
#   PROBE=20 sbatch --time=02:00:00 scripts/hpc/sweep.sbatch      # time estimate on 20 series per dataset
#
# No SLURM on the machine? Use scripts/hpc/run_sweep.sh, which takes the same variables.
# Submitting the same command again continues: finished tasks are skipped. Submit from the repository root.
# Adapt the #SBATCH lines to the cluster (partition and account names are site specific).
#SBATCH --job-name=tsf-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%x-%A_%a.out
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME

set -uo pipefail

# ---- what to run (override at submission: SWEEP_METHODS=all sbatch --export=ALL scripts/hpc/sweep.sbatch)
: "${SWEEP_OUT:=results/sweep}"              # output folder; keep it to continue or to add methods later
: "${SWEEP_DATASETS:=all}"                   # all | main | extra | m5 | ids, comma separated
: "${SWEEP_METHODS:=study}"                  # study | benchmarks | local | intermittent | global | all | method names
: "${SWEEP_EXCLUDE_DATASETS:=}"
: "${SWEEP_EXCLUDE_METHODS:=}"
: "${SWEEP_TASK_TIMEOUT_MIN:=1440}"          # one (dataset, method) may take this long; keep it below --time
: "${SWEEP_JOBS:=4}"                         # (dataset, method) tasks running at the same time
: "${SWEEP_THREADS:=$(( ${SLURM_CPUS_PER_TASK:-1} / SWEEP_JOBS ))}"   # threads inside each task; JOBS x THREADS = cores
[ "$SWEEP_THREADS" -lt 1 ] && SWEEP_THREADS=1
: "${SWEEP_EXTRA:=}"                         # further options, e.g. "--missing-policy locf --metrics all"
: "${TSFSTUDIO_DATA_DIR:=$PWD/data/tsf}"
export TSFSTUDIO_DATA_DIR

cd "${SLURM_SUBMIT_DIR:-$PWD}"
# module load gcc/13 python/3.11             # the modules used in scripts/hpc/setup.sh, if the cluster needs them
source .venv/bin/activate
export OMP_NUM_THREADS="$SWEEP_THREADS" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

args=(--out "$SWEEP_OUT" --datasets "$SWEEP_DATASETS" --methods "$SWEEP_METHODS" --jobs "$SWEEP_JOBS" --threads "$SWEEP_THREADS"
      --task-timeout "$SWEEP_TASK_TIMEOUT_MIN")
[ -n "$SWEEP_EXCLUDE_DATASETS" ] && args+=(--exclude-datasets "$SWEEP_EXCLUDE_DATASETS")
[ -n "$SWEEP_EXCLUDE_METHODS" ] && args+=(--exclude-methods "$SWEEP_EXCLUDE_METHODS")
[ -n "${PROBE:-}" ] && args+=(--probe "$PROBE")
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] && args+=(--array-index "$SLURM_ARRAY_TASK_ID")

echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | $(date -Is)"
echo "python -m tsfstudio sweep ${args[*]} $SWEEP_EXTRA"
# exec: SLURM's end-of-time signal reaches the sweep directly, which records the running task and writes the report.
# shellcheck disable=SC2086
exec python -m tsfstudio sweep "${args[@]}" $SWEEP_EXTRA
