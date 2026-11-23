# Running the C++ models on a cluster (HPC)

`python -m tsfstudio sweep` runs every chosen dataset with every chosen method, one task after another, and records for
each task the error measures, the computation time and the memory, and for every series its forecasts and errors. It is
made for unattended jobs: a task that fails, runs out of memory or exceeds its time limit is recorded and the sweep goes
on; a job that the scheduler stops is continued by submitting the same command again.

Three launchers take the same variables: `scripts/hpc/sweep.sge` for SGE (Grid Engine), `scripts/hpc/sweep.sbatch` for
SLURM, and `scripts/hpc/run_sweep.sh` for a machine without a scheduler (a server reached by ssh; it runs in the
background with `nohup`). The `sweep` command is the same under all of them.

## 1. Set-up (once, on the login node)

```bash
git clone <repository> forecast-studio && cd forecast-studio
module load gcc cmake python          # whatever the cluster calls them; GCC >= 9, CMake >= 3.16, Python >= 3.10
bash scripts/hpc/setup.sh             # .venv, tsfcore.so, engine list, agreement tests, data check
```

On a system whose own tools are too old for the build (C++17 needs GCC >= 8; for example CentOS 7 has g++ 4.8,
CMake 2.8 and Python 3.6), take everything from conda-forge instead:

```bash
bash scripts/hpc/setup_conda.sh       # conda environment "tsfstudio": Python 3.11, CMake, GCC, the packages; builds tsfcore.so
qsub scripts/hpc/check.sge            # agreement tests on a compute node (SGE); read sge-tsf-check.<job>.log
```

The script needs conda on PATH or in `~/miniforge3` (the two lines that install Miniforge are at its top) and about
2 GB in the home file system for Miniforge and the environment. The library is linked with the compiler's runtime
inside (`TSF_STATIC_RUNTIME=ON`), so it does not depend on the old system libstdc++. `scripts/hpc/activate.sh`, which
all launchers source, activates `.venv` if it exists and the conda environment otherwise. This path was written for
CentOS 7 with SGE and has not been run on such a system: read the output of the set-up script and of `check.sge`
before the first sweep.

Copy the `.tsf` files into `data/tsf/` (or set `TSFSTUDIO_DATA_DIR`), then `python -m tsfstudio data-check`. Only files
whose size matches the catalogue are used; a `.tsf` file outside the catalogue can be named by its path.

Build rules (`cpp/build.sh`): GCC or Clang with the flags of `cpp/CMakeLists.txt` as they are. The agreement with R and
Python in `cpp/README.md` rests on IEEE arithmetic without reassociation or FMA contraction, so no `-march=native`,
`-ffast-math`, `-Ofast`, and no compiler whose default floating-point model is "fast" (Intel `icpx`).

Global models: `pooled_regression` needs nothing; `lightgbm` / `lightgbm_feat` use the `lightgbm` package installed by
the set-up script; `catboost` needs the CatBoost 1.2.10 binaries (`WITH_CATBOOST=1 bash scripts/hpc/setup.sh`).
`python -m tsfstudio engines` and `sweep --dry-run` show what the build offers. The CatBoost path has been checked on
Windows only (bit-identical to the Python engine there); on Linux, compare one dataset with the Python engine before
relying on it.

## 2. Plan

```bash
source .venv/bin/activate
python -m tsfstudio sweep --out results/sweep --datasets all --methods study --dry-run
```

`--datasets`: `all`, a bundle (`main`, `extra`, `m5`), catalogue ids, or paths to `.tsf` files, comma separated. Datasets
run smallest file first. `--methods`: method names and presets, comma separated:

| Preset | Methods |
|---|---|
| `study` (default) | per dataset, the methods of the study in the catalogue: ses, ets, theta, tbats, arima or dhr_arima, pooled_regression, catboost |
| `benchmarks` | naive, snaive, drift, naive2, ses, theta, comb_shd, historic_average, window_average, seasonal_window_average, seasonal_es, holt_ets, zero_model |
| `local` | every local C++ method except the intermittent-demand ones |
| `intermittent` | croston_classic, croston_sba, croston_optimized, tsb, adida, imapa |
| `global` | pooled_regression, lightgbm, lightgbm_feat, catboost (those available) |
| `all` | everything the engine offers |

`--exclude-datasets` and `--exclude-methods` remove entries. Horizon, lag, seasonality and integer rounding come from the
catalogue, as in a single `run`. Missing values in a training part stop that task (`--missing-policy error`, the
default) unless a policy is chosen (`zero`, `locf`, `interpolate`); the choice changes the results, so it is not made
silently.

## 3. Time estimate before the real job

```bash
PROBE=20 sbatch --time=02:00:00 scripts/hpc/sweep.sbatch
# or directly:  python -m tsfstudio sweep --out results/sweep --methods study --probe 20 --threads 16 --task-timeout 30
```

The same plan runs on 20 randomly chosen series per dataset into `results/sweep/probe`. `probe/report/estimate.csv` gives
for every local task the projected engine time of the full dataset (sample time x total series / sample series) and the
job log prints the total and the ten most expensive tasks. Use it to set `--time`, `SWEEP_TASK_TIMEOUT_MIN` and to
decide which (dataset, method) pairs to exclude. Limits of the projection: a random sample, so a factor of two either
way; global models are fitted once on all series and are not projected; a task that already exceeds the limit on the
sample is listed as not projected.

## 4. Submit

```bash
sbatch scripts/hpc/sweep.sbatch                                   # everything in one job, in turn
SWEEP_METHODS=study,benchmarks SWEEP_EXCLUDE_DATASETS=london_smart_meters sbatch scripts/hpc/sweep.sbatch
sbatch --array=0-41%8 scripts/hpc/sweep.sbatch                    # one dataset per array task; indices as --dry-run prints them
python -m tsfstudio sweep --out results/sweep --report-only       # after an array: build the report (login node, seconds)
```

Edit the `#SBATCH` lines (partition, account, cores, memory, time) for the cluster. The variables at the top of the
script (`SWEEP_OUT`, `SWEEP_DATASETS`, `SWEEP_METHODS`, `SWEEP_THREADS`, `SWEEP_TASK_TIMEOUT_MIN`, `SWEEP_EXTRA`, ...)
choose the work.

Continuing: submit the same command again. Finished tasks are skipped; a task that was running when the job ended
(`interrupted`) runs again; tasks that `failed`, hit the task limit (`timeout`) or `crashed` are left as they are unless
`--retry-failed` is given (`SWEEP_EXTRA="--retry-failed"`). Keep `SWEEP_TASK_TIMEOUT_MIN` below the job's `--time`, so
that one task cannot use up the whole job. New methods or datasets can be added later to the same `--out` folder.

SGE (`scripts/hpc/sweep.sge`: queue `all.q`, parallel environment `smp`, 32 slots on one node; edit the `#$` lines for
another site):

```bash
qsub scripts/hpc/sweep.sge                                          # 4 tasks at a time x 8 threads
qsub -v PROBE=20 scripts/hpc/sweep.sge                              # time estimate
qsub -v SWEEP_JOBS=32,SWEEP_THREADS=1 scripts/hpc/sweep.sge         # single-thread timings
SWEEP_METHODS=study,benchmarks qsub -V scripts/hpc/sweep.sge        # values with commas: export, then -V
qstat; tail -f sge-tsf-sweep.<job id>.log; qdel <job id>
```

One job per dataset (`scripts/hpc/submit_datasets.sh`, run on the login node): every dataset becomes its own job
`tsf-<dataset>` with its own log, the methods run in turn inside it on all slots of the job, and a last job
`tsf-report`, held until the others end, builds the report. A dataset can then be followed, stopped (`qdel tsf-m4_monthly`)
or resubmitted on its own.

```bash
DRY=1 bash scripts/hpc/submit_datasets.sh                              # print the qsub commands only
PROBE=20 bash scripts/hpc/submit_datasets.sh                           # time estimate per dataset
bash scripts/hpc/submit_datasets.sh                                    # all datasets, 32 slots each
SWEEP_DATASETS=m4_monthly,nn5_daily bash scripts/hpc/submit_datasets.sh
SLOTS=8 bash scripts/hpc/submit_datasets.sh                            # 8 slots per job
```

With a limit of 32 slots per user, jobs of 32 slots run one after another, smallest dataset first; with `SLOTS=8` four
datasets run side by side, each with 8 threads. All jobs write into the same `--out` folder (one sub-folder per
dataset), so the report covers them together.

`qdel` first sends a notice (`-notify`), so the running tasks are recorded as `interrupted` and the report is written.
Where a slot is a hardware thread (hyper-threading on), 32 slots can be the two threads of 16 physical cores when the
rest of the node is busy; computation times measured then are longer and noisier than on 32 free cores. For times that
go into a paper, ask for the whole node (all its slots) and still run with 32 threads in total, or state the setting.

Without a scheduler:

```bash
bash scripts/hpc/run_sweep.sh                      # SWEEP_CORES=32: 4 tasks at a time x 8 threads, in the background
tail -f results/sweep/sweep.log                    # follow it
kill "$(cat results/sweep/sweep.pid)"              # stop it; running tasks are recorded as interrupted
```

### Using the cores

Two levels, `--jobs J` x `--threads T` = the cores (`SWEEP_JOBS`, `SWEEP_THREADS`; with 32 cores the launchers default to
4 x 8):

- `--jobs J`: J (dataset, method) tasks run at the same time, each in its own process.
- `--threads T`: inside a task the C++ library spreads the series over T threads. Global models and datasets with fewer
  series than T cannot use all of them.

| Setting (32 cores) | Use |
|---|---|
| `SWEEP_JOBS=4 SWEEP_THREADS=8` (default) | a balance: small tasks overlap, long ones still get 8 threads |
| `SWEEP_JOBS=32 SWEEP_THREADS=1` | single-thread times for every task while all cores work; the longest task (for example ARIMA or TBATS on tens of thousands of series) then runs on one core and decides when the sweep ends |
| `SWEEP_JOBS=1 SWEEP_THREADS=32` | one task at a time: the least disturbed timings, idle cores during small or global tasks |

Forecasts and errors do not depend on the setting. Times do, slightly: tasks that run side by side share memory
bandwidth and the processor's frequency headroom. On 4 cores, NN5 weekly + daily, 38 tasks: one at a time 80 s, four at
a time 22 s, with the same summed `engine_sec` (38.2 s and 38.3 s). `results.csv` records `threads` and
`concurrent_tasks` for every task. Memory adds up over the J tasks (each holds its dataset), so lower J for the large
files if the job is killed for memory (state `crashed`, "signal 9").

## 5. Outputs

```
results/sweep/
  sweep_runs.jsonl                         one line per invocation: arguments, node, CPU, SLURM job, versions, git commit
  tasks/<dataset>/<method>/
    forecasts.csv                          one row per series, columns h1..hH
    test_actuals.csv                       the held-out observations, same layout
    errors_per_series.csv                  one row per series, one column per measure
    summary.csv                            mean, median, valid and excluded series per measure
    fallbacks.csv                          series that received the seasonal naive forecast instead of the model, with the reason
    manifest.json                          configuration, data SHA-256, split, engine version, warnings
    status.json                            state, times, memory
    log.txt                                output of the task process
  report/
    results.csv                            one row per task: state, series, horizon, mean_/median_<measure>, times, memory, message
    summary_long.csv                       one row per task and measure
    errors_per_series.csv.gz               all tasks: dataset, task, method, series_name, measures
    mean_<measure>.csv, engine_sec.csv, cpu_sec.csv, peak_rss_mb.csv, state.csv      dataset x method tables
    report.xlsx                            results, summary_long and the dataset x method tables as sheets
```

Measures: SMAPE, mSMAPE, MASE, MAE, RMSE, WAPE, nRMSE (`docs/METRICS.md`); `--metrics all` adds RMSSE, MSE, MAPE, MAAPE,
MedAE, RMSLE, ME and PBias. Data with the M5 hierarchy also get `WRMSSE` in `results.csv` and `wrmsse.csv` in the task
folder.

Times, all in seconds:

| Column | Meaning |
|---|---|
| `engine_sec` | time inside the engine: fitting and forecasting all series of the task (for C++ one library call) |
| `engine_sec_per_series` | `engine_sec` / number of series (a mean; the library does not time single series) |
| `cpu_sec` | user + system CPU time of the run, summed over threads and child processes |
| `run_sec` | split, engine, measures and writing the files |
| `load_sec` | reading the `.tsf` file |
| `wall_sec` | the whole task process, Python start-up included |
| `peak_rss_mb` | peak resident memory of the task process (megabytes) |

For a comparison of computation times, report `engine_sec` together with `threads`, the CPU model and the node
(`sweep_runs.jsonl`). `--threads 1` gives single-thread times (see "Using the cores"); with more threads `cpu_sec` stays close to the
single-thread cost while `engine_sec` falls. The thread count is fixed exactly (`TSFSTUDIO_CPP_THREADS`): without it
the library would start one thread per core of the node, not per core granted to the job. Timings taken while other
array tasks share the node are noisier than those of an exclusive node (`#SBATCH --exclusive`).

## Agreement on Linux

The reference checks in `cpp/README.md` were made on Windows (MinGW GCC). On Linux (GCC 13.3, glibc, x86-64), 2026-11-23,
`tests/test_cpp_engine.py` and `tests/test_golden_nn5.py` gave 84 passed, 1 skipped and 2 differing checks: `thief` and
`mapa` on one of the 29 quarterly reference series (18 observations) differ from R by a relative 1.4e-6, where the test
asks for 1e-12; the other 28 quarterly and all 38 monthly series meet it. The likely cause is a last-bit
difference between the C libraries' elementary functions that sends one optimiser run down another path; it has not been
traced. Everything else, including ETS, ARIMA, TBATS and the bagged models on NN5, agrees within the documented
tolerances. `scripts/hpc/setup.sh` repeats these checks on the cluster; run them on the node type that will do the work.
