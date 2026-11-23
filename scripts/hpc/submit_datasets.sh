#!/usr/bin/env bash
# SGE: one job per dataset, strictly one after another: the job of a dataset is held (-hold_jid) until the job of
# the dataset before it has ended, so no dataset starts before the previous one is finished. Smallest dataset first.
# Each job runs the chosen methods on its dataset in turn and has its own log (sge-tsf-<dataset>.<job id>.log); a last
# job, held until all others end, builds results/sweep/report. Run on the login node from the repository root:
#
#   DRY=1 bash scripts/hpc/submit_datasets.sh                     # only print the qsub commands
#   PROBE=20 bash scripts/hpc/submit_datasets.sh                  # time estimate per dataset (into <out>/probe)
#   bash scripts/hpc/submit_datasets.sh                           # all datasets, 32 slots each, in turn
#   SWEEP_DATASETS=m4_monthly,nn5_daily bash scripts/hpc/submit_datasets.sh
#   CHAIN=0 SLOTS=8 bash scripts/hpc/submit_datasets.sh           # not chained: datasets side by side, 8 slots each
#
# A dataset whose job fails or is deleted (qdel) does not block the ones after it. Running the script again resubmits
# every dataset; finished (dataset, method) pairs are skipped in seconds.
# Variables: those of scripts/hpc/sweep.sge (SWEEP_OUT, SWEEP_METHODS, SWEEP_EXCLUDE_DATASETS, SWEEP_EXCLUDE_METHODS,
# SWEEP_TASK_TIMEOUT_MIN, SWEEP_EXTRA, ...), plus SLOTS (default 32), SWEEP_JOBS (default 1: the methods of a dataset
# one at a time, each on all slots of its job), CHAIN (default 1) and QUEUE (default all.q).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${SLOTS:=32}" "${QUEUE:=all.q}" "${CHAIN:=1}" "${SWEEP_JOBS:=1}" "${SWEEP_OUT:=results/sweep}"
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

# qsub -terse prints only the job id; in a dry run the command is printed and the job name stands in for the id.
submit() {
  if [ -n "${DRY:-}" ]; then echo "qsub $*" >&2; local a; while [ $# -gt 0 ]; do [ "$1" = "-N" ] && a=$2; shift; done; echo "$a"
  else qsub -terse "$@" | cut -d. -f1; fi
}
n=0
prev=""
all=""
unset SWEEP_REPORT_ONLY
export SWEEP_NO_REPORT=1 SWEEP_EXCLUDE_DATASETS=""
for id in $ids; do
  export SWEEP_DATASETS="$id"
  name=$(basename "$id" .tsf)
  hold=()
  # CHAIN=1 (default): this dataset's job waits until the previous dataset's job has ended, whatever its outcome.
  [ "$CHAIN" = "1" ] && [ -n "$prev" ] && hold=(-hold_jid "$prev")
  jid=$(submit -V -N "tsf-$name" -q "$QUEUE" -pe smp "$SLOTS" ${hold[@]+"${hold[@]}"} scripts/hpc/sweep.sge) || exit 1
  [ -n "$jid" ] || { echo "qsub gave no job id for $name; stopping." >&2; exit 1; }
  after=""; [ "$CHAIN" = "1" ] && [ -n "$prev" ] && after=" (after job $prev)"
  echo "tsf-$name -> job $jid$after"
  prev="$jid"; all="${all:+$all,}$jid"
  n=$((n + 1))
done
unset SWEEP_NO_REPORT
export SWEEP_REPORT_ONLY=1
[ -n "${PROBE:-}" ] && export SWEEP_OUT="$SWEEP_OUT/probe"
jid=$(submit -V -N tsf-report -q "$QUEUE" -pe smp 1 -hold_jid "$all" scripts/hpc/sweep.sge)
echo "tsf-report -> job $jid (after all dataset jobs)"
[ -n "${DRY:-}" ] && { echo "(dry run: nothing submitted; $n dataset jobs and the report job)"; exit 0; }
echo "$n dataset jobs and the report job submitted (qstat to follow; logs: sge-tsf-<dataset>.<job id>.log)."
