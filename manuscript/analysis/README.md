# Feature analysis of ranking instability (manuscript Section 5.5)

Written 2026-11-22. Everything here can be regenerated with the two commands at the end.

## Inputs (not in git: `data/ranking_study/`, about 67 MB)

Copied unchanged from the author's Google Drive on 2026-11-22. The folder is listed in `.gitignore`; it must be added to
the reproducibility package separately.

| Local path | Content | Drive source |
|---|---|---|
| `tsfeatures/<task>.csv` | R `tsfeatures` output, one row per series, no series identifier (row i = series T(i)) | "Experimental Analysis…" project folder `1LO43rG6hq2igcu4GyjSLb21Vofn0zMZ0` → `features/tsfeatures` |
| `catch22/<task>.csv` | R `Rcatch22` output, same layout | same folder → `features/catch22_features` |
| `errors/<task>.csv` | per-series, per-model values of the 15 measures with validity flags (`*_all_models_errors_per_series.csv`) | project folder `16tG293dZeue811BYlHbUhcLx3VdumqCO` → `02_Source_Data_and_Forecasts/Forecast_Error_Outputs/<TASK>` |
| `errors/m4_daily/*.csv` | the same for M4 Daily, one file per model (the combined file is 23 MB and could not be downloaded through the connector; ten model files, TBATS absent) | same |
| `ALL_common_support_metric_rankings.csv` | Primary measures and common-support models per task | `01_Final_Submission_Package/03_Data` |

The features were produced by `reference/experiments/feature_experiments.R` on the complete series (test period
included). They were not recomputed here.

## Files in this folder

| File | Produced by | Content |
|---|---|---|
| `task_agreement.csv` | typed in from `04_Results/competition_metric_agreement_regret.csv` plus Table 1 | task-level Kendall's W and regret, number of series, horizon |
| `task_feature_summary.csv` | task-level script | task medians of every feature |
| `feature_instability_correlations.csv` | task-level script | Table 9 and the exploratory screen |
| `series_level_outcomes.csv` | series-level script | disagreement and regret for each of the 10,537 series, with the six pre-specified features |
| `series_level_task_info.csv` | series-level script | judges used per task, alignment check, medians |
| `series_level_within_task_correlations.csv` | series-level script | Spearman correlation of every feature with both outcomes, per task |
| `series_level_pooled_correlations.csv` | series-level script | Table 10 and the exploratory screen |

### Model-compatibility rule (manuscript Section 5.4, Table 8)

Produced by `tools/coverage_sensitivity.py` from the same error files and ranking file.

| File | Content |
|---|---|
| `coverage_exclusion_audit.csv` | every model-task pair removed by the common-support rule: measure, number of series on which it is undefined, series with a negative forecast |
| `coverage_sensitivity_design.csv` | per task: measures removed under "drop the measure", series removed under "drop the series" |
| `coverage_sensitivity_orderings.csv` | Bradley-Terry ordering, bootstrap interval, P(first), P(top 3) under the three rules |
| `coverage_sensitivity_scores.csv` | family-balanced task scores under the three rules |
| `coverage_sensitivity_restored_tasks.csv` | each model's score on the tasks it kept and on the tasks it is restored to |
| `coverage_sensitivity_agreement.csv` | Kendall's W and selection regret per task under the three rules |
| `coverage_sensitivity_leave_one_competition_out.csv` | Table 7 repeated under each rule |

The script first recomputes the study's own analysis from the per-series files and stops if it differs: mean losses agree
with the stored ones to 1e-14 and the 135 family-balanced scores to 1e-16. Bootstrap frequencies for the common-support
rule differ from Table 4 by up to 0.02 (different random stream; Table 4 was not re-estimated).

```powershell
python tools\coverage_sensitivity.py --errors data\ranking_study\errors --rankings data\ranking_study\ALL_common_support_metric_rankings.csv --out manuscript\analysis
```

### Features against forecast error and model standing (`feature_performance/`, written 2026-11-23)

Not part of the manuscript. `tools/feature_model_performance.py` relates the same features to series difficulty
(median error over models), to each model's error, and to each model's rank on the series; it then tests with LightGBM
how much of this holds out of sample (difficulty, and per-series model selection). It uses all models in the error
files, not the common-support sets. `tools/feature_model_performance_excel.py` turns the CSV output into three
workbooks (Turkish sheet names). `series_panel.csv` in that folder is 15 MB.

```powershell
$b = "data\ranking_study"
python tools\feature_model_performance.py --errors "$b\errors" --tsfeatures "$b\tsfeatures" --catch22 "$b\catch22" --agreement manuscript\analysis\task_agreement.csv --out manuscript\analysis\feature_performance
python tools\feature_model_performance_excel.py --dir manuscript\analysis\feature_performance
```

The first command takes about 12 minutes. The features include the test period, so the selection gain is an upper bound.

**Training-period features** (`feature_performance_train/`, `feature_performance_full_validated/`).
`tools/recompute_features.py` recomputes the features in Python (pycatch22 1.0.2, Nixtla tsfeatures 0.4.5, statsmodels;
installed in a separate environment, not the application's) on the whole series (`py_full`, only to check the port
against the R files) and on the training period (`py_train`). `port_agreement.csv` holds the Spearman correlation between
port and R per feature; `validated_features.csv` lists the 39 features with 0.90 or more in every task. Both analyses
below use these 39. Result: mean MASE of per-series selection 1.589 (single best model), 1.375 with whole-series
features, 1.487 with training-period tsfeatures.

```powershell
$b = "data\ranking_study"; $o = "manuscript\analysis"; $c = @("--tsf","data\tsf","--errors","$b\errors","--agreement","$o\task_agreement.csv")
python tools\recompute_features.py @c --part full --out "$b\py_full"
python tools\recompute_features.py @c --part train --out "$b\py_train"
python tools\recompute_features.py --compare --py "$b\py_full" --r $b --out "$o\feature_performance_train"
# validated_features.csv: features of port_agreement.csv with min_rho >= 0.90, prefixed ts_ / c22_, plus ts_cv
$k = "$o\feature_performance_train\validated_features.csv"
python tools\feature_model_performance.py --errors "$b\errors" --tsfeatures "$b\py_train\tsfeatures" --catch22 "$b\py_train\catch22" --agreement "$o\task_agreement.csv" --keep $k --out "$o\feature_performance_train"
python tools\feature_model_performance.py --errors "$b\errors" --tsfeatures "$b\tsfeatures" --catch22 "$b\catch22" --agreement "$o\task_agreement.csv" --keep $k --out "$o\feature_performance_full_validated"
python tools\feature_model_performance_excel.py --dir "$o\feature_performance_train" --baseline "$o\feature_performance_full_validated"
```

The port also settles the alignment question below: features recomputed from the series in archive order agree with the
R files at 0.90 or more in every task (most at 1.00), M3 Yearly, M3 Quarterly and M4 Weekly included.

## Checks worth knowing about

- **Alignment of features and series.** Feature files carry no series identifier. The series-level script correlates the
  feature `mean` with the test-period mean from the error file: 0.85 to 0.99 in most tasks and about 0 after shuffling.
  It is lower for M3 Yearly (0.58), M3 Quarterly (0.72) and M4 Weekly (0.73), where series levels are close together.
  This was not verified by another route.
- **Equivalent measures.** MAE, nMAE, MASE rank models identically inside a series, as do MSE, RMSE, nRMSE, RMSSE (checked:
  at least 99.4% of series in every task; the remainder are numerical ties). Each group is one judge. The squared-error
  group is represented on the RMSE scale, because MSE gives a different normalized regret.
- **Seasonal strength** exists for 8 of the 15 tasks only.

## Regenerate

```powershell
$b = "data\ranking_study"
python tools\feature_instability_analysis.py --tsfeatures "$b\tsfeatures" --catch22 "$b\catch22" --agreement manuscript\analysis\task_agreement.csv --out manuscript\analysis
python tools\feature_instability_series.py --errors "$b\errors" --tsfeatures "$b\tsfeatures" --catch22 "$b\catch22" --rankings "$b\ALL_common_support_metric_rankings.csv" --agreement manuscript\analysis\task_agreement.csv --out manuscript\analysis
```

The series-level computation is `tsfstudio.instability.series_instability`, the same code the application's
`instability` command runs (`tests/test_instability.py` pins the NN5 Weekly values).
