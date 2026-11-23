#!/usr/bin/env bash
# SGE: one job per dataset. Each job runs the chosen methods on its dataset in turn and has its own log
# (sge-tsf-<dataset>.<job id>.log); a last job, held until the others end, builds results/sweep/report.
# Run on the login node from the repository root:
#
#   bash scripts/hpc/submit_datasets.sh                           # all datasets, 32 slots each
#   DRY=1 bash scripts/hpc/submit_datasets.sh                     # only print the qsub commands
#   SWEEP_DATASETS=m4_monthly,nn5_daily bash scripts/hpc/submit_datasets.sh
#   SLOTS=8 bash scripts/hpc/submit_datasets.sh                   # 8 slots per job: four datasets fit in 32 slots
#   PROBE=20 bash scripts/hpc/submit_datasets.sh                  # time estimate per dataset (into <out>/probe)
#
# With a limit of 32 slots per user, jobs of 32 slots run one after another (smallest dataset first), jobs of 8 slots
# four at a time. Running it again resubmits every dataset; finished (dataset, method) pairs are skipped in seconds.
# Variables: those of scripts/hpc/sweep.sge (SWEEP_OUT, SWEEP_METHODS, SWEEP_EXCLUDE_DATASETS, SWEEP_EXCLUDE_METHODS,
# SWEEP_TASK_TIMEOUT_MIN, SWEEP_EXTRA, ...), plus SLOTS (default 32), SWEEP_JOBS (default 1: the methods of a dataset
# one at a time, each on all slots of its job) and QUEUE (default all.q).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${SLOTS:=32}" "${QUEUE:=all.q}" "${SWEEP_JOBS:=1}" "${SWEEP_OUT:=results/sweep}"
: "${SWEEP_DATASETS:=all}" "${SWEEP_METHODS:=study}" "${SWEEP_EXCLUDE_DATASETS:=}" "${SWEEP_EXCLUDE_METHODS:=}"
: "${TSFSTUDIO_DATA_DIR:=$PWD/data/tsf}"
export SWEEP_JOBS SWEEP_OUT SWEEP_METHODS SWEEP_EXCLUDE_METHODS TSFSTUDIO_DATA_DIR
unset SWEEP_THREADS                               # each job: threads = its slots / SWEEP_JOBS

source scripts/hpc/activate.sh || exit 1
list=(--list-datasets --out "$SWEEP_OUT" --datasets "$SWEEP_DATASETS" --methods "$SWEEP_METHODS")
[ -n "$SWEEP_EXCLUDE_DATASETS" ] && list+=(--exclude-datasets "$SWEEP_EXCLUDE_DATASETS")
[ -n "$SWEEP_EXCLUDE_METHODS" ] && list+=(--exclude-methods "$SWEEP_EXCLUDE_METHODS")
ids=$(python -m tsfstudio sweep "${list[@]}") || { echo "$ids"; exit 1; }
[ -n "$ids" ] || { echo "No dataset to submit."; exit 1; }

run() { if [ -n "${DRY:-}" ]; then echo "$*"; else "$@"; fi; }
n=0
unset SWEEP_REPORT_ONLY
export SWEEP_NO_REPORT=1 SWEEP_EXCLUDE_DATASETS=""
for id in $ids; do
  export SWEEP_DATASETS="$id"
  name=$(basename "$id" .tsf)
  run qsub -V -N "tsf-$name" -q "$QUEUE" -pe smp "$SLOTS" scripts/hpc/sweep.sge
  n=$((n + 1))
done
unset SWEEP_NO_REPORT
export SWEEP_REPORT_ONLY=1
[ -n "${PROBE:-}" ] && export SWEEP_OUT="$SWEEP_OUT/probe"
run qsub -V -N tsf-report -q "$QUEUE" -pe smp 1 -hold_jid "tsf-*" scripts/hpc/sweep.sge
[ -n "${DRY:-}" ] && { echo "(dry run: nothing submitted; $n dataset jobs and the report job)"; exit 0; }
echo "$n dataset jobs and the report job submitted (qstat to follow; logs: sge-tsf-<dataset>.<job id>.log)."
