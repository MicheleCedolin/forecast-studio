#!/bin/bash
# Engine speed study on SLURM, one node, 32 cores. See docs/HPC_SPEED.md.
#
#   sbatch scripts/hpc/speed.sbatch                                   # C++ on 1,2,4,8,16,32 threads; Python and R on 1 and 32
#   SPEED_ENGINES=cpp sbatch scripts/hpc/speed.sbatch                 # the C++ engine only
#   SPEED_EXTRA=--dry-run sbatch --time=00:05:00 scripts/hpc/speed.sbatch
#
# Submit from the repository root. Submitting again continues where the last job stopped.
# --exclusive: the node does nothing else while the times are taken. Adapt partition, account, memory and time.
#SBATCH --job-name=tsf-speed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
cd "${SLURM_SUBMIT_DIR:-$PWD}"
exec bash scripts/hpc/run_speed.sh
