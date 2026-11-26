# Engine speed study on a cluster (32 cores)

Measures how long the same forecasting method takes on the C++ engine (tsfcore) with 1, 2, 4, 8, 16 and 32 threads, and
on the Python and R engines with 1 and 32 worker processes. The result is the speed table of the software paper
(`manuscript/software_paper`). General cluster notes (set-up details, SGE and SLURM, agreement checks) are in
`docs/HPC.md`; this page is the short path for the speed study.

## 0. Short path: SGE cluster, quick subset (Tables 5 and 6 of the paper)

Four datasets (nn5_weekly, nn5_daily, m3_monthly, m4_hourly), five methods (ses, theta, ets, arima, tbats), C++ on 1, 8
and 32 threads (three repetitions), Python and R on 1 and 32 workers (one repetition), one node, the three engines one
after the other. Steps, from the repository root on the login node:

1. `bash scripts/hpc/preflight.sh 2>&1 | tee preflight.txt` — tools, cores, free space, reach of conda-forge / Zenodo /
   CRAN, conda environments, data files. Changes nothing.
2. If needed: `bash scripts/hpc/setup_conda.sh` (new environment `tsfstudio`; an existing environment such as
   `clrp_env` is not touched) and `bash scripts/hpc/setup_r.sh` (R into the same environment). About 2.5 GB together.
3. Smoke test: `SPEED_DATASETS=nn5_weekly SPEED_METHODS=ses,theta SPEED_THREADS=1,8,32 SPEED_OUT=results/speed_smoke
   qsub -V scripts/hpc/speed.sge` (a few minutes).
4. The study: `SPEED_DATASETS=nn5_weekly,nn5_daily,m3_monthly,m4_hourly SPEED_METHODS=ses,theta,ets,arima,tbats
   SPEED_THREADS=1,8,32 SPEED_REPEATS=3 SPEED_REF_REPEATS=1 SPEED_OUT=results/speed_quick qsub -V scripts/hpc/speed.sge`.
   The single-worker R runs take longest (about 80 minutes together); 1 to 2 hours in all, more on old hardware.

The data files needed in `data/tsf/`: `nn5_weekly_dataset.tsf`, `nn5_daily_dataset_without_missing_values.tsf`,
`m3_monthly_dataset.tsf`, `m4_hourly_dataset.tsf` (byte counts as in `src/tsfstudio/data/catalog.yaml`; the study stops
at the start if one is missing or has another size).

## 1. Upload and unpack

On your computer the archive is built with `python tools/make_hpc_package.py` (`dist/forecast-studio-hpc-speed-<date>.tar.gz`,
code and the six datasets of the plan). Copy it to the cluster and unpack it:

```bash
scp forecast-studio-hpc-speed-<date>.tar.gz <user>@<cluster>:~/
ssh <user>@<cluster>
tar -xzf forecast-studio-hpc-speed-<date>.tar.gz
cd forecast-studio
```

## 2. Set-up (once, on the login node; needs the internet)

Old system tools (CentOS 7 and similar: g++ 4.8, Python 3.6), everything from conda-forge:

```bash
curl -fLO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh   # only if there is no conda
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
bash scripts/hpc/setup_conda.sh        # Python 3.11, compiler, statsforecast; builds tsfcore.so  (about 2 GB in $HOME)
```

Or, where the cluster offers GCC >= 9, CMake >= 3.16 and Python >= 3.10 as modules:

```bash
module load gcc cmake python
bash scripts/hpc/setup.sh
pip install "statsforecast==2.1.1" scikit-learn     # for the Python engine's ETS, ARIMA and TBATS
```

Optional, for the R engine (the reference implementation; without it the study times C++ and Python only):

```bash
bash scripts/hpc/setup_r.sh            # R, forecast, jsonlite; not yet tried on a cluster, read its output
```

Check:

```bash
source scripts/hpc/activate.sh
python -m tsfstudio engines            # cpp must be AVAILABLE; python should name statsforecast; r if installed
python -m tsfstudio data-check         # the six datasets of the plan must be present
python tools/engine_speed_study.py --dry-run
```

## 3. Run

SGE:

```bash
qsub scripts/hpc/check.sge             # optional: agreement of the build with the reference forecasts (a few minutes)
qsub scripts/hpc/speed.sge             # the study; log: sge-tsf-speed.<job id>.log
qstat
```

SLURM: `sbatch scripts/hpc/speed.sbatch` (edit partition and account first).
No scheduler: `nohup bash scripts/hpc/run_speed.sh > speed.log 2>&1 &`.

Edit the `#$` lines of `speed.sge` (queue `all.q`, parallel environment `smp`, 32 slots) if the site uses other names.
The times are only meaningful if the 32 cores are not shared: ask for a whole node, or run when the node is idle, and
say in the paper which it was. Where hyper-threading is on, 32 slots may be 16 physical cores.

Choices (`qsub -v NAME=value,...`; a value with commas: `NAME=a,b qsub -V scripts/hpc/speed.sge`):

| Variable | Default | Meaning |
|---|---|---|
| `SPEED_ENGINES` | `cpp,python,r` | engines to time; one that is not installed is left out |
| `SPEED_THREADS` | `1,2,4,8,16,32` | thread counts of the C++ engine (Python and R use the smallest and the largest) |
| `SPEED_DATASETS` | the plan below | catalogue ids |
| `SPEED_METHODS` | the plan's | these methods for every dataset, for example `ets,arima,stl_ets,croston_sba`; any method of the application (`python -m tsfstudio engines`) |
| `SPEED_REF_DATASETS` | the four smaller datasets | where Python and R are timed too; `all` adds the two large M4 sets (hours) |
| `SPEED_REPEATS` / `SPEED_REF_REPEATS` | 1 / 1 | repetitions per C++ cell / per Python or R cell; with more than one the median is reported |
| `SPEED_TIMEOUT_MIN` | 120 | limit for one run; a level projected to exceed it is recorded as `skipped` |
| `SPEED_OUT` | `results/speed` | output folder |

Plan:

| Dataset | Series | Methods | Engines |
|---|---|---|---|
| nn5_weekly | 111 | ses, theta, ets, arima, tbats | C++, Python, R |
| nn5_daily | 111 | ses, theta, ets, arima, tbats | C++, Python, R |
| m3_monthly | 1,428 | ses, theta, ets, arima, tbats | C++, Python, R |
| m4_hourly | 414 | ses, theta, ets, arima, tbats | C++, Python, R |
| m4_quarterly | 24,000 | ses, theta, ets, arima | C++ |
| m4_monthly | 48,000 | ses, theta, ets, arima | C++ |

How long the whole study takes is not known in advance. Every single run is limited to `SPEED_TIMEOUT_MIN`, and for
each method the largest thread count runs first, so the slow single-thread runs are skipped instead of started when
they would not fit. To get the C++ part quickly, submit `SPEED_ENGINES=cpp` first and the reference engines afterwards
into the same output folder. Submitting the same command again continues where the previous job stopped.

## 4. Results to bring back

```
results/speed/
  speed_table.md     the table: one row per dataset and method, one column per engine and thread count (seconds);
                     below it A. ms per series, median over the datasets (the layout of Table 5), and B. C++ speed-up
  summary.csv        median, minimum and maximum time per cell, ms per series, speed-up against one C++ thread
  runs.csv           every repetition, with its state (ok, skipped, timeout, failed), the 1-minute load average when it
                     started (load1_start) and message
  ratios.csv         every ratio from the medians: C++ speed-up and parallel efficiency per thread count, Python / C++
                     and R / C++ on 1 and on the largest level, Python and R speed-up on the largest level, R / Python
  node.json          host, CPU model, logical and physical cores (sockets x cores x threads per core), CPUs the job
                     may use, memory, load average at the start and the end, granted slots, engine versions
  node_lscpu.txt     lscpu of the compute node;  node_qhost.txt: qhost of the cluster and of this host (SGE only)
  runs/<dataset>/<method>/<engine>_x<level>/<run id>/
                     forecasts.csv (one row per series), test_actuals.csv, errors_per_series.csv, summary.csv,
                     fallbacks.csv, manifest.json of every cell
```

```bash
tar -czf speed_results.tar.gz results/speed sge-tsf-speed.*.log
scp <user>@<cluster>:~/forecast-studio/speed_results.tar.gz .
```

What is timed: `timing.engine_wall_sec` of each run's manifest, the engine call that fits and forecasts all series of
the dataset. Reading the file, the accuracy measures and writing results are not included; for R the time includes
starting Rscript. C++ threads are fixed exactly (`TSFSTUDIO_CPP_THREADS`); Python and R get that many worker
processes (`--workers`) with one BLAS / numba thread each. `mase_mean` in `summary.csv` lets you confirm that a method
gives the same accuracy at every thread count.
