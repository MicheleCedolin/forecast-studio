---
title: "How Reliable Are Forecasting-Model Rankings? A Validity- and Coverage-Aware Benchmark Evaluation with an Open Workbench"
author: "Michele Cedolin [TODO: affiliation and e-mail]"
date: "Revised draft v2, 22 November 2026"
---

<!--
Source: Manuscript.docx and Supplementary_Material.docx (Drive, 01_Final_Submission_Package, 22 Sep 2026).
All numerical results in Sections 3-5 are carried over unchanged from that version; nothing was re-estimated.
Equations (1)-(7) were re-typeset from the prose and from run_reanalysis.py because the Word equation objects
did not survive text export: check them against the original before submission.
Items marked [TODO] need an author decision or a fact that is not available in the repository.

EXCEPTION, added 22 Nov 2026: the comparison of model-compatibility rules in Section 5.4 (Table 8) is a new estimate from
tools/coverage_sensitivity.py (outputs: manuscript/analysis/coverage_*.csv). The abstract, contributions, Sections 3.3,
4.5, 5.1, 5.2, 5.4, 7 and 8 were rewritten on the same day so that the three rules are presented side by side; the
common-support results themselves (Tables 4 to 7) are unchanged. Tables 8 to 14 of the earlier draft are now 9 to 15.
The abstract is about 290 words: check the journal's limit.
-->

# Abstract

Forecasting benchmarks compress large collections of forecast errors into model rankings, yet a ranking depends on the accuracy measure, on which models are observed for each task, and on which tasks enter the benchmark. We study the reliability of such rankings rather than propose a new accuracy measure. The analysis covers 15 dataset-frequency tasks from the M1, M3, M4, Tourism, and NN5 competitions (10,537 series), 11 forecasting methods represented by fixed point forecasts, and 15 accuracy measures. A validity screen decides which evidence enters each task, a task-balanced Bradley-Terry model and a permutation-calibrated Skillings-Mack test handle incomplete model coverage, and Kendall's W is paired with cross-metric selection regret. Systematic rank differences are clearly detected, and the same five methods (TBATS, ETS, N-BEATS, DeepAR, Theta) lead under every specification, but their order is not determined. When a model is dropped from a task on which a measure is undefined for it, TBATS, ETS, and N-BEATS come first, and this set survives alternative metric weightings, 48 screening-threshold configurations, and leave-one-family-out re-estimation. All 29 such exclusions, however, stem from one measure (MSLE), most of them from at most three series with a negative forecast, and they remove the leading methods from tasks on which they rank poorly. Dropping the measure or the series instead of the model moves TBATS to fifth place and puts N-BEATS or Theta first. Benchmark composition changes the leader under every rule. Within series the measures mostly agree, less so for series that are hard to forecast; part of the task-level disagreement arises because differently scaled measures weight series differently in the average. We release Forecast Studio, an open, locally run workbench that regenerates forecasts and computes these diagnostics without programming; its R engine reproduces the archived statistical forecasts to machine precision. Rankings should be reported as evidence-qualified, coverage-aware, benchmark-conditional summaries, together with the rule applied to undefined evidence.

**Keywords:** forecast evaluation; model ranking; incomplete blocks; Bradley-Terry; Skillings-Mack; metric agreement; selection regret; reproducible research software

# 1 Introduction

Forecasting competitions and benchmark archives turn heterogeneous forecast errors into compact model rankings. Those rankings are useful, but they are not intrinsic properties of the forecasting methods. They are conditional on the losses used to evaluate forecasts, on which models are observed for each benchmark task, and on which tasks enter the benchmark. A leaderboard can therefore look precise even when the underlying evidence is incomplete or locally contradictory.

Three sources of conditionality are well documented in isolation. Accuracy measures differ in domain restrictions, scale behavior, and the functional they elicit (Hyndman & Koehler, 2006; Kolassa, 2020). Benchmark composition can change which method appears best (Roque et al., 2025). And enforcing validity restrictions leaves a model-by-task matrix with missing cells, for which the usual complete-block rank tests are not designed. What is missing is an analysis that carries all three through to the final ranking and states how much of that ranking survives.

Computing several accuracy measures and applying a non-parametric comparison test is by now routine in forecasting studies, and we claim no novelty for either. The open practical problem is a different one: what an analyst should conclude when the measures send different messages. Two questions have to be kept apart. Whether a measure is suitable for the data at hand comes first and is settled before any ranking is computed; a measure that fails this check does not vote. Only then does it make sense to ask how far the remaining, admissible measures agree, and what their disagreement costs.

This paper asks a narrower and more operational question than metric selection: *how much confidence should be placed in a forecasting-model ranking once validity, unequal model coverage, admissible metric disagreement, and benchmark composition are made explicit?* Three research questions organize the analysis.

- **RQ1.** Can a global model ordering be estimated without discarding benchmark tasks that have incomplete model support?
- **RQ2.** Where do admissible accuracy measures produce materially different rankings, and when does that disagreement change the selected model enough to create meaningful regret?
- **RQ3.** Does the leading model set survive reasonable changes in metric-family composition, screening thresholds, the treatment of undefined measure values, and benchmark scope?

The analysis uses a fixed set of point forecasts. Because different losses elicit different point functionals, the results are not interpreted as the loss-optimal performance of the underlying model classes (Kolassa, 2020). Holding forecasts fixed instead isolates the sensitivity of the evaluation and ranking procedure, conditional on one forecast-generation protocol.

**Contributions.** The paper makes four contributions.

1. *An incomplete-block ranking framework.* A task-balanced Bradley-Terry ordering and a permutation-calibrated Skillings-Mack test retain every benchmark task whatever the pattern of missing model-task cells. Under the common-support rule only five of the 15 tasks are complete, and under any rule one model has no archived forecasts for one task.
2. *A separation of ranking disagreement from its decision consequence.* Kendall's W measures concordance among admissible measures on a common model support, and cross-metric selection regret measures what that disagreement costs when it changes the selected model. An analysis of 10,537 series with two standard feature sets locates the disagreement: within series it is small and grows as forecastability falls, and part of the task-level disagreement comes from how differently scaled measures weight series in the average.
3. *A robustness audit along three axes.* Evaluation-specification choices (metric families, weighting, screening thresholds), the model-compatibility rule (whether an undefined measure value removes the model, the measure, or the series), and benchmark-composition choices (competition omission, cluster resampling, scope expansion) are perturbed separately. The first leave the leading set intact. The second changes it: every exclusion in this benchmark comes from one measure and a few series, and the exclusions favor the models they remove. The third changes the leader under every rule.
4. *An open workbench.* Forecast Studio is a locally run application whose browser interface regenerates forecasts on Monash-format data, computes 16 accuracy measures, and records full run provenance. Its analysis commands implement the framework of this paper, so the same reliability analysis can be applied to any other collection of tasks, models, and measures. Its R engine reproduces the archived statistical forecasts used in this study to machine precision (Section 6), so the forecast-generation protocol on which the rankings are conditional can be inspected and rerun by readers who do not program.

The interpretation is deliberately set-valued rather than winner-centric. The paper does not propose another rule for choosing an error metric. It treats metric validity as a gatekeeping step and asks whether the remaining evidence supports a stable ranking, a stable leading group, or neither.

Section 2 reviews related work. Section 3 describes the benchmark and the forecast archive, Section 4 the ranking-reliability framework, and Section 5 the results. Section 6 presents Forecast Studio. Section 7 discusses implications, reporting recommendations, and limitations, and Section 8 concludes.

# 2 Related work

## 2.1 Accuracy measures and ranking sensitivity

Forecast losses differ in domain restrictions, scale behavior, robustness, and the statistical functional they target (Hyndman & Koehler, 2006; Kolassa, 2020; Hewamalage et al., 2023). Koutsandreas et al. (2022) study directly whether method rankings change across commonly used measures and report only small discrepancies on average, particularly within measure categories. Spiliotis et al. (2023) show that model-selection criteria can nevertheless disagree substantially in particular settings. This literature makes metric validity a prerequisite for credible evaluation. It leaves open how stable rankings are once valid evidence, model coverage, and benchmark composition are treated together.

The competitions themselves have recognized the problem. The M4 competition ranked methods by a combination of sMAPE and MASE relative to a benchmark (Makridakis et al., 2020), and reporting several measures together with a non-parametric rank comparison has become common practice in applied studies (Koning et al., 2005; Kourentzes, 2013). A parallel line of work argues that accuracy measures should be judged by the decisions they lead to. In inventory settings, the method that is most accurate under a standard measure is often not the one that gives the best stock-control outcome (Syntetos et al., 2010), and measures such as MAE can reward forecasts that are useless for intermittent demand (Kolassa, 2016). Our selection-regret measure is a step in this direction that stays within the accuracy measures: it asks what is lost, under one admissible measure, by having selected with another. It does not replace evaluation on an application-specific cost.

## 2.2 Benchmark heterogeneity and composition

Forecasting performance varies with data characteristics and problem conditions (Petropoulos et al., 2014; Kang et al., 2017). ModelRadar shows how aggregate performance can conceal aspect-specific differences across series and forecast conditions (Cerqueira et al., 2025), and Roque et al. (2025) demonstrate that dataset cherry-picking can materially distort apparent performance. The broader benchmarking literature has begun to treat preservation of model rankings under task selection as an explicit design objective (Gusev & Zaytsev, 2026). Benchmark composition is therefore part of the estimand, not a neutral backdrop.

## 2.3 Statistical comparison across multiple data sets

Forecasting-specific work has long moved beyond raw average ranks toward formal comparison (Koning et al., 2005). Friedman-type procedures remain the standard reference for complete multi-dataset designs (Demšar, 2006), and the Model Confidence Set returns a superior set rather than forcing a single winner (Hansen et al., 2011). Bradley-Terry models offer another route by converting pairwise wins across data sets into latent abilities: Wainer (2023) develops a Bayesian version for multi-algorithm comparison, and Grishina et al. (2026) use Bradley-Terry models to handle incomplete benchmark matrices in recommender-system evaluation. Forecasting comparisons become incomplete precisely when validity and output-compatibility restrictions are enforced, and the usual combination of complete-block tests and fixed-loss leaderboards does not characterize that setting.

## 2.4 Model selection, combination, and automated forecasting

A large literature seeks to choose or combine forecasting methods automatically. Meta-learning approaches predict from time-series features which method, or which combination weights, will perform well (Montero-Manso et al., 2020), and automated machine-learning systems build ensembles by ranking candidate models on a validation loss (Shchur et al., 2023); Petropoulos et al. (2022) review selection and combination more broadly. Our study is complementary to this work rather than a competitor to it. Those methods take a loss and a set of data sets as given and optimize under them, and they are themselves evaluated by leaderboards of the kind studied here. The framework below does not select a model. It quantifies how far a ranking, including a ranking of selection or ensemble procedures, depends on the admissible measures, on coverage, and on benchmark composition.

## 2.5 Reproducible evaluation tooling

The Monash Time Series Forecasting Archive distributes benchmark data in a common format together with the R and Python scripts used to produce its baseline results (Godahewa et al., 2021). Those scripts, like the forecasting libraries they call (Hyndman & Khandakar, 2008; Alexandrov et al., 2020), assume a user who writes and debugs code, manages two language environments, and assembles error tables by hand. A ranking conditional on a forecast-generation protocol is only as inspectable as that protocol is executable, which motivates the software contribution in Section 6. [TODO: add one or two verified citations to comparable open evaluation tools if the target journal expects a software comparison.]

# 3 Benchmark design and forecast archive

## 3.1 Competition benchmark scope

The principal benchmark contains 15 dataset-frequency tasks from five forecasting competitions: M1 (Yearly, Quarterly, Monthly), M3 (Yearly, Quarterly, Monthly, Other), M4 (Weekly, Daily, Hourly), Tourism (Yearly, Quarterly, Monthly), and NN5 (Daily, Weekly) (Makridakis et al., 1982; Makridakis & Hibon, 2000; Athanasopoulos et al., 2011; Makridakis et al., 2018; Andrawis et al., 2011). All are distributed through the Monash archive (Godahewa et al., 2021). Each dataset-frequency combination is one benchmark task, so a collection with thousands of series does not count as thousands of independent replications.

The 15 tasks contain 10,537 series. For every series the final $h$ observations form the test segment and the preceding observations the training segment, with $h$ the repository horizon of the task. Cross-task inference treats tasks, not series, as the benchmark units. Series-level observations are used only within a task, to construct losses.

**Table 1.** Principal competition benchmark. Common support is the number of models jointly eligible across the task's Primary measures.

| Competition | Frequency | Series | Horizon | Common-support models | Primary measures |
|:--|:--|--:|--:|--:|--:|
| M1 | Yearly | 181 | 6 | 11 | 11 |
| M1 | Quarterly | 203 | 8 | 10 | 6 |
| M1 | Monthly | 617 | 18 | 4 | 6 |
| M3 | Yearly | 645 | 6 | 8 | 15 |
| M3 | Quarterly | 756 | 8 | 9 | 15 |
| M3 | Monthly | 1,428 | 18 | 11 | 10 |
| M3 | Other | 174 | 8 | 11 | 15 |
| M4 | Weekly | 359 | 13 | 10 | 15 |
| M4 | Daily | 4,227 | 14 | 9 | 15 |
| M4 | Hourly | 414 | 48 | 5 | 11 |
| Tourism | Yearly | 518 | 4 | 8 | 6 |
| Tourism | Quarterly | 427 | 8 | 9 | 11 |
| Tourism | Monthly | 366 | 24 | 8 | 11 |
| NN5 | Daily | 111 | 56 | 11 | 10 |
| NN5 | Weekly | 111 | 8 | 11 | 15 |

A separate stress-test panel comprises Car Parts Monthly, FRED-MD Monthly, Hospital Monthly, COVID Deaths Daily, and San Francisco Traffic Weekly. These five tasks do not determine the principal ordering. The competition-only inclusion rule is based on benchmark provenance and was fixed independently of ranking outcomes; the additional tasks quantify sensitivity to expanding the benchmark scope, the concern raised by Roque et al. (2025).

## 3.2 Forecasting models and fixed point forecasts

The forecast pool contains 11 methods (Table 2). The fixed forecasts were generated under one terminal-holdout protocol rather than assembled from published leaderboards. Local statistical models were fitted with the R forecast package, version 8.12 (Hyndman & Khandakar, 2008); Pooled Regression used glmnet 4.0.2; the neural estimators used GluonTS 0.8.0 under Python 3.7.4 (Alexandrov et al., 2020).

**Table 2.** Forecasting methods in the fixed comparison panel. Implementation details are in the Supplementary Material.

| Family | Scope | Methods |
|:--|:--|:--|
| Classical statistical | Local (one model per series) | SES, Theta, ETS, ARIMA, TBATS |
| Regression | Global (one model per task) | Pooled Regression |
| Neural / deep learning | Global (one model per task) | FFNN, DeepAR, N-BEATS, WaveNet, Transformer |

For FFNN, DeepAR, N-BEATS, and Transformer the dataset-specific lag length was used as context length; WaveNet kept its default receptive field. Parameters not set in the experimental code kept package defaults. No validation set or model-specific hyperparameter search was used, and neural point forecasts are the median of 100 predictive sample paths. Models were not rerun over repeated seeds or rolling origins, and forecasts were not replaced by loss-specific optimal functionals.

These choices give a controlled fixed-forecast comparison. They do not remove differences in how strongly individual model classes adapt to the data: the local statistical procedures perform automatic series-specific selection, whereas the neural estimators run at defaults. The ranking results are conditional on this protocol and cannot separate evaluation sensitivity from differences in tuning effort. Section 6 describes the software with which the statistical part of the protocol can be rerun.

## 3.3 Evaluation-evidence screen

The candidate library contains 15 magnitude-based point-forecast measures in six families: absolute (MAE, MdAE), squared (MSE, RMSE), normalized (nMAE, nRMSE), scaled (MASE, RMSSE), percentage (MAPE, MdAPE, RMSPE, RMdSPE, sMAPE, MAAPE), and logarithmic (MSLE). Formulas and screening rules are in the Supplementary Material.

Each dataset-measure pair is classified as Primary, Secondary, or Excluded by rule-based domain and denominator checks. A measure that is undefined for the observed data, structurally unsuitable for cross-series aggregation, or incompatible with a model's output does not enter the task-level ranking; only Primary measures do. The thresholds are operational design parameters, not universal validity constants, and their influence is evaluated in Section 5.4.

Dataset-level measure status is kept separate from model-output compatibility. When several measures are compared, model ranks are recomputed on their task-level common support. Across the 15 tasks common support ranges from four to eleven models (Table 1). This common-support rule is the reference specification of the analysis. It is one of three ways to handle a measure that is undefined for a model: the others are to remove the measure from the task, or to remove the series on which it is undefined. Section 5.4 compares the three, and the choice turns out to decide the leading positions.

# 4 Ranking-reliability framework

Table 3 summarizes the role of each component. Primary components define the estimands; robustness components perturb one design choice at a time.

**Table 3.** Analysis components and their inferential roles.

| Component | Role | Rationale |
|:--|:--|:--|
| Family-balanced task score | Descriptive aggregation | Prevents repeated voting by closely related measures |
| Task-balanced Bradley-Terry | Primary global ordering | Handles incomplete coverage; equalizes task contribution |
| Skillings-Mack with permutation | Primary omnibus inference | Uses all incomplete blocks |
| Kendall's W | Metric-agreement diagnostic | Multi-ranking concordance on a 0-1 scale |
| Cross-metric selection regret | Decision-consequence diagnostic | Separates rank disagreement from its cost |
| Family omission, weighting, taxonomy, thresholds | Robustness: evaluation specification | Tests analyst dependence of the aggregation |
| Drop the model, the measure, or the series when a measure is undefined | Robustness: model-compatibility rule | Tests whether the exclusions are informative about accuracy |
| Competition omission, cluster bootstrap, scope expansion | Robustness: benchmark composition | Tests dependence on which tasks are included |
| Fixed task effects; complete-case Friedman | Secondary specification checks | Familiar reference procedures |

## 4.1 Metric-specific ranks and family-balanced task scores

For task $d$, Primary measure $m$, and model $j$, the $K_d$ eligible models are ranked by mean series-level loss. With $r_{dmj}$ the rank of model $j$, the normalized rank

$$\tilde r_{dmj} = \frac{r_{dmj}-1}{K_d-1} \tag{1}$$

maps the best observed model to zero and the worst to one. Ranks are recomputed on the common model support of the task before any cross-metric aggregation.

The ordinal transformation is deliberate. Raw losses from heterogeneous measures are not commensurable, whereas normalized ranks give a common scale that is invariant to monotone rescaling. The cost is that rank scores treat narrow and wide loss gaps alike. Magnitude information is therefore kept out of the global ordering and evaluated separately through selection regret (Section 4.4).

Closely related measures should not receive repeated votes merely because a family contains several formulas. Let $\mathcal F_d$ be the set of families represented among the Primary measures of task $d$ and $\mathcal M_{df}$ the Primary measures of family $f$. The family-balanced score is

$$s_{dj} = \frac{1}{|\mathcal F_d|}\sum_{f\in\mathcal F_d}\frac{1}{|\mathcal M_{df}|}\sum_{m\in\mathcal M_{df}}\tilde r_{dmj}. \tag{2}$$

It is a transparent descriptive summary, not a claim that all families have equal decision-theoretic importance.

## 4.2 Primary ordering: task-balanced Bradley-Terry

Because coverage differs across tasks, a mean over observed ranks confounds model quality with where a model happens to be available. The global ordering is therefore estimated from pairwise within-task comparisons of the scores $s_{dj}$. For each pair of models observed in task $d$, the lower score receives a win and an exact tie is split into half-wins. Each pair carries weight $\omega_d = \binom{K_d}{2}^{-1}$, so every task contributes total weight one regardless of how many models or series it contains.

The Bradley-Terry model (Bradley & Terry, 1952) assigns each model an ability $\pi_j>0$ with

$$\Pr(i \text{ beats } j) = \frac{\pi_i}{\pi_i+\pi_j}, \tag{3}$$

and the abilities maximize the weighted log-likelihood

$$\ell(\boldsymbol\pi)=\sum_{d}\omega_d\sum_{i<j\in\mathcal K_d}\Big[w_{dij}\log\frac{\pi_i}{\pi_i+\pi_j}+w_{dji}\log\frac{\pi_j}{\pi_i+\pi_j}\Big], \tag{4}$$

where $\mathcal K_d$ is the model set of task $d$ and $w_{dij}\in\{0,\tfrac12,1\}$ is the win credit of $i$ over $j$. Abilities are reported on the log scale, centered to mean zero. This use follows recent multi-dataset comparison work (Wainer, 2023; Grishina et al., 2026); the task-balanced weighting is specific to the present design.

Uncertainty is assessed by resampling the 15 tasks with replacement and re-estimating the ordering. With so few blocks, the resulting first-place and top-three frequencies are benchmark-composition sensitivity summaries, not population probabilities.

## 4.3 Omnibus inference with incomplete blocks

The Friedman test is a complete-block procedure. Restricting the panel to tasks with all 11 models would leave five tasks. The Skillings-Mack statistic extends the Friedman idea to arbitrary missing cells and reduces to it in a complete design (Skillings & Mack, 1981). With $k_d$ models observed in task $d$ and within-task ranks $r_{dj}$ of the scores $s_{dj}$, the adjusted treatment sums are

$$A_j=\sum_{d:\,j\in\mathcal K_d}\sqrt{\frac{12}{k_d+1}}\Big(r_{dj}-\frac{k_d+1}{2}\Big), \tag{5}$$

and the statistic is $SM=\mathbf A^{\top}\boldsymbol\Sigma^{-}\mathbf A$, where $\boldsymbol\Sigma$ is the null covariance matrix of $\mathbf A$ and $\boldsymbol\Sigma^{-}$ a generalized inverse.

Because the number of tasks is modest and the missingness irregular, the null distribution is obtained by permuting the observed scores within each task, which preserves each task's support and score structure. Simulation-based calibration is recommended for small or irregular Skillings-Mack designs (Chatfield & Mander, 2009). The number of blocks follows from the provenance rule, not from a power calculation, and tasks that share a competition are not assumed independent; Section 5.4 examines that dependence. A complete-case Friedman permutation test is kept as a secondary check.

## 4.4 Agreement across admissible measures and its consequence

Kendall's coefficient of concordance treats the $m$ Primary measures of a task as judges ranking the same $K$ common-support models:

$$W=\frac{12\,S}{m^{2}\,(K^{3}-K)}, \tag{6}$$

where $S$ is the sum of squared deviations of the models' rank sums from their mean. $W$ lies between zero and one, with one denoting perfect concordance; the standard tie correction is applied when needed (Kendall & Smith, 1939).

Rank movement need not change the selected model. Let $L_{db}(j)$ be the mean loss of model $j$ under measure $b$ in task $d$, and $j^{*}_{da}$ the model selected by measure $a$. The normalized regret of selecting with $a$ and evaluating with $b$ is

$$R_d(a\to b)=\frac{L_{db}(j^{*}_{da})-\min_{j}L_{db}(j)}{\max_{j}L_{db}(j)-\min_{j}L_{db}(j)}, \tag{7}$$

with minima and maxima taken over the common support. The task-level mean regret averages $R_d(a\to b)$ over all $m(m-1)$ ordered pairs of distinct Primary measures. No task-measure pair has a zero loss range in the present data.

## 4.5 Robustness design

Robustness is organized along three axes. *Evaluation-specification* checks re-estimate the ordering after omitting one measure family at a time, under three alternative weightings (equal measure weights, a finer source taxonomy, one representative per family), and over a $4\times4\times3$ grid of screening thresholds. *Model-compatibility* checks replace the common-support rule by one of two rules that keep every model: removing from a task any Primary measure that is undefined for some model, or keeping all Primary measures and restricting the task to the series on which they are defined for every model. *Benchmark-composition* checks omit one competition at a time, resample the five competitions as clusters, and expand the benchmark to the five-task stress panel; the competition omission is repeated under each compatibility rule. Bootstrap convergence is checked separately to quantify Monte Carlo error, and horizon-specific rank stability is examined as an exploratory analysis. Finally, Section 5.5 asks whether measured characteristics of the series account for instability, across tasks and across series within tasks.

# 5 Results

## 5.1 Comparison support

All tasks start from the same 11-model input set, but the evidence gate produces a sparse comparison matrix. The compatibility audit contains 1,848 eligible model-measure cells, 29 incompatible cells, and 15 missing cells. Common support is complete for M1 Yearly, M3 Monthly, M3 Other, NN5 Daily, and NN5 Weekly, lies between eight and ten models for most other tasks, and falls to five for M4 Hourly and four for M1 Monthly (Table 1; Figure S1). Under this rule, complete-case inference would use five tasks. The 29 incompatible cells have a single origin: all are MSLE cells in which at least one forecast is negative. Section 5.4 examines what follows from that.

The number of Primary measures ranges from six (M1 Quarterly, M1 Monthly, Tourism Yearly) to all 15 in several M3, M4, and NN5 tasks. These counts say which evidence is admissible for each task. They are not evidence that any one measure is universally preferable.

## 5.2 Global ordering under incomplete coverage (RQ1)

Under the common-support rule, the Bradley-Terry analysis places TBATS, ETS, and N-BEATS in the first three positions, followed by DeepAR and Theta (Table 4, Figure 1). DeepAR and Theta are effectively tied: their log-abilities are 0.540 and 0.521, and Theta has the higher top-three frequency.

**Table 4.** Competition-only global ranking under the common-support rule. Bootstrap frequencies and the 95% interval for the log-ability are based on 5,000 task resamples. The mean family-balanced score is not coverage-adjusted and is shown for orientation only; because it averages each model over its own task subset, its order need not match the Bradley-Terry order.

| Model | Task coverage | Mean score | BT rank | Log-ability | 95% bootstrap interval | P(rank 1) | P(top 3) |
|:--|--:|--:|--:|--:|:--|--:|--:|
| TBATS | 8 | 0.247 | 1 | 1.171 | [0.084, 2.854] | 0.493 | 0.795 |
| ETS | 9 | 0.279 | 2 | 1.051 | [0.296, 2.053] | 0.270 | 0.793 |
| N-BEATS | 11 | 0.311 | 3 | 0.845 | [0.032, 1.910] | 0.170 | 0.664 |
| DeepAR | 12 | 0.365 | 4 | 0.540 | [-0.236, 1.429] | 0.025 | 0.324 |
| Theta | 13 | 0.338 | 5 | 0.521 | [-0.367, 1.690] | 0.042 | 0.405 |
| FFNN | 12 | 0.474 | 6 | -0.144 | [-0.687, 0.438] | 0.000 | 0.003 |
| Pooled Regression | 11 | 0.521 | 7 | -0.194 | [-1.000, 0.478] | 0.000 | 0.015 |
| ARIMA | 15 | 0.557 | 8 | -0.542 | [-1.272, 0.015] | 0.000 | 0.001 |
| WaveNet | 15 | 0.591 | 9 | -0.792 | [-1.881, -0.052] | 0.000 | 0.000 |
| Transformer | 14 | 0.776 | 10 | -1.140 | [-2.490, -0.395] | 0.000 | 0.000 |
| SES | 15 | 0.755 | 11 | -1.315 | [-2.237, -0.577] | 0.000 | 0.000 |

Two features of Table 4 qualify the point ordering. First, no model is first in a majority of resamples: TBATS leads in 49.3%, ETS in 27.0%, and N-BEATS in 17.0%. Membership of the top three is more stable (79.5%, 79.3%, and 66.4%) than the identity of the leader. Second, coverage is lowest where the ranking is highest. TBATS is observed in 8 of 15 tasks and ETS in 9, and the TBATS interval is the widest in the table. The leading positions thus rest on the least evidence, which is exactly the situation in which an unadjusted mean score would be most misleading and in which the leader should not be reported without its coverage. Section 5.4 shows that the coincidence has a cause: the tasks on which the two models are unobserved are tasks on which they rank poorly.

Frequencies computed with 1,000, 2,500, 5,000, and 10,000 resamples differ by at most 0.02 (Table S13), so Monte Carlo error is small relative to the uncertainty induced by having 15 tasks.

The Skillings-Mack statistic over all 15 tasks is 45.94, with a permutation p-value of about $5.0\times10^{-5}$ from 20,000 within-task permutations. The complete-case Friedman test on the five tasks with all 11 models gives 30.30 with a permutation p-value of $8.0\times10^{-5}$. Both reject the null of no systematic rank differences; the Friedman result is secondary because it discards ten tasks. The answer to RQ1 is that a global ordering can be estimated from all tasks and that models do differ systematically, but the evidence identifies a leading group rather than a leader. Which models form the top of that group depends on how undefined evidence is handled (Section 5.4).

## 5.3 Agreement among admissible measures (RQ2)

Metric agreement is heterogeneous (Table 5, Figure 2). Kendall's W ranges from 0.591 in M1 Quarterly to 1.000 in M1 Monthly. The three lowest values occur in M1 Quarterly, Tourism Yearly, and M4 Daily, so material disagreement arises within established competition tasks and does not depend on the stress-test data.

**Table 5.** Metric agreement and cross-metric selection regret for selected tasks, ordered by Kendall's W. The full 15-task table is Table S5.

| Task | Primary measures | Common models | Kendall W | Mean regret | Maximum regret |
|:--|--:|--:|--:|--:|--:|
| M1 Quarterly | 6 | 10 | 0.591 | 0.134 | 1.000 |
| Tourism Yearly | 6 | 8 | 0.657 | 0.174 | 0.950 |
| M4 Daily | 15 | 9 | 0.720 | 0.124 | 0.985 |
| M3 Monthly | 10 | 11 | 0.903 | 0.060 | 0.493 |
| M4 Weekly | 15 | 10 | 0.939 | 0.000 | 0.000 |
| NN5 Weekly | 15 | 11 | 0.942 | 0.009 | 0.083 |
| NN5 Daily | 10 | 11 | 0.984 | 0.009 | 0.127 |
| M1 Monthly | 6 | 4 | 1.000 | 0.000 | 0.000 |

Disagreement and its consequence are related but not equivalent. The three low-concordance tasks have mean regret between 0.124 and 0.174, and a maximum regret near one: for some pairs of measures, the model chosen by one is the worst, or nearly the worst, under the other. By contrast M3 Quarterly has zero regret despite W = 0.820: the measures order the remaining models differently but select the same one. M3 Monthly shows the opposite case: concordance is high (W = 0.903), yet some measure switches cost almost half of the loss range.

Across the 15 tasks the Spearman correlation between disagreement ($1-W$) and mean regret is 0.717 ($p = 0.003$). This is a descriptive statistic, since the tasks are not a random sample of forecasting problems. On RQ2, the measures agree strongly on most tasks, disagree materially on a few, and where they disagree the choice of measure tends to change the selected model at a real cost.

## 5.4 Robustness (RQ3)

**Evaluation specification.** These checks, like Tables 6 and 7, use the common-support rule. Re-estimating the ordering after omitting one measure family at a time keeps TBATS, ETS, and N-BEATS in the top three, with Spearman correlations of 0.982-0.991 against the full ordering (Table 6). The same set is retained under equal measure weights, the finer source taxonomy, and one representative per family (Table S11). The leader is not retained: ETS replaces TBATS in first place under equal measure weights and under the finer taxonomy.

**Table 6.** Leave-one-family-out robustness of the competition-only Bradley-Terry ordering.

| Omitted family | Spearman vs full ordering | Top three |
|:--|--:|:--|
| Absolute | 0.991 | TBATS, ETS, N-BEATS |
| Squared | 0.991 | TBATS, ETS, N-BEATS |
| Normalized | 0.991 | TBATS, N-BEATS, ETS |
| Scaled | 0.991 | TBATS, ETS, N-BEATS |
| Percentage | 0.991 | TBATS, N-BEATS, ETS |
| Logarithmic | 0.982 | TBATS, ETS, N-BEATS |

Across the 48 screening-threshold configurations (near-zero threshold, scale-heterogeneity cutoff, denominator-tail criterion), the equal-weight ordering has a minimum Spearman correlation of 0.991 with its reference and the same three models lead in every configuration (Table S12). The set-level conclusion is not driven by one threshold combination.

**Benchmark composition.** Competition-level perturbations are more demanding (Table 7). The orderings stay strongly correlated with the full ordering, and TBATS and ETS remain in the top three whichever competition is omitted. N-BEATS does not: DeepAR takes the third position when M3 is omitted and Theta when M4 is omitted. The leader changes from TBATS to ETS in three of the five omissions.

**Table 7.** Leave-one-competition-out sensitivity of the Bradley-Terry ordering.

| Omitted competition | Tasks left | Spearman vs full | Top three | Leader |
|:--|--:|--:|:--|:--|
| M1 | 12 | 0.955 | TBATS, ETS, N-BEATS | TBATS |
| M3 | 11 | 0.973 | ETS, TBATS, DeepAR | ETS |
| M4 | 12 | 0.955 | ETS, TBATS, Theta | ETS |
| Tourism | 12 | 0.982 | TBATS, ETS, N-BEATS | TBATS |
| NN5 | 13 | 0.973 | ETS, TBATS, N-BEATS | ETS |

An exploratory bootstrap that resamples the five competitions as clusters gives the same picture: top-three frequencies are 0.927 for ETS, 0.750 for TBATS, and 0.689 for N-BEATS (Table S10). With five clusters these are sensitivity diagnostics, not inferential probabilities.

Adding the five non-competition data sets yields the 20-task ordering ETS, TBATS, N-BEATS, DeepAR, Theta, Pooled Regression, FFNN, ARIMA, WaveNet, SES, Transformer (Table S8). Its Spearman correlation with the competition-only ordering is 0.973. The expansion changes the first-ranked model and reverses FFNN and Pooled Regression but keeps the same three models at the top.

**Model-compatibility rule.** The checks above vary which measures vote and with what weight. They hold fixed the rule that makes the comparison matrix incomplete: a model leaves a task when a Primary measure of that task is undefined for it on any series. An audit of the per-series error files shows what this rule does here. All 29 incompatible cells of Section 5.1 involve a single measure, MSLE, which is undefined when a forecast is negative. The exclusions rest on few series. The median is three series per excluded model-task pair; 17 of the 29 pairs involve at most three series and seven a single one (TBATS leaves M3 Quarterly because of one series in 756); the largest share is 11% (ETS on M4 Hourly). The only other gap in the matrix is TBATS on M4 Daily, for which the archive holds no forecasts.

We therefore re-estimated the ordering under two rules that keep the model. *Drop the measure* removes MSLE from the ten tasks in which it is undefined for some model. *Drop the series* keeps MSLE and restricts each task to the series on which it is defined for every model, which removes 187 of the 10,537 series and is lenient toward the models whose negative forecasts caused the exclusion. Either rule fills 164 of the 165 model-task cells and leaves 14 complete tasks.

**Table 8.** Bradley-Terry ordering under three model-compatibility rules. Common support is the rule of the main analysis (Table 4). Frequencies are from 5,000 task resamples.

| Model | Common support: tasks | BT rank | Log-ability | Drop the measure: BT rank | Log-ability | P(rank 1) | P(top 3) | Drop the series: BT rank | Log-ability | P(rank 1) | P(top 3) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| TBATS | 8 | 1 | 1.171 | 5 | 0.452 | 0.110 | 0.347 | 5 | 0.296 | 0.042 | 0.238 |
| ETS | 9 | 2 | 1.051 | 3 | 0.696 | 0.164 | 0.660 | 3 | 0.704 | 0.164 | 0.660 |
| N-BEATS | 11 | 3 | 0.845 | 1 | 0.830 | 0.386 | 0.780 | 2 | 0.735 | 0.252 | 0.732 |
| DeepAR | 12 | 4 | 0.540 | 4 | 0.548 | 0.071 | 0.426 | 4 | 0.514 | 0.078 | 0.405 |
| Theta | 13 | 5 | 0.521 | 2 | 0.774 | 0.266 | 0.737 | 1 | 0.925 | 0.459 | 0.873 |
| FFNN | 12 | 6 | -0.144 | 6 | 0.039 | 0.000 | 0.003 | 7 | -0.007 | 0.000 | 0.001 |
| Pooled Regression | 11 | 7 | -0.194 | 7 | 0.038 | 0.003 | 0.042 | 6 | 0.146 | 0.004 | 0.088 |
| ARIMA | 15 | 8 | -0.542 | 8 | -0.329 | 0.000 | 0.002 | 9 | -0.537 | 0.000 | 0.001 |
| WaveNet | 15 | 9 | -0.792 | 9 | -0.412 | 0.000 | 0.001 | 8 | -0.281 | 0.000 | 0.004 |
| Transformer | 14 | 10 | -1.140 | 11 | -1.621 | 0.000 | 0.000 | 11 | -1.446 | 0.000 | 0.000 |
| SES | 15 | 11 | -1.315 | 10 | -1.017 | 0.000 | 0.000 | 10 | -1.048 | 0.000 | 0.000 |

The top of the ordering changes (Table 8). TBATS falls from first to fifth under both alternative rules, Theta rises from fifth to second or first, and N-BEATS or Theta leads. The Spearman correlation with the common-support ordering is 0.855 for the first rule and 0.818 for the second, lower than for any other perturbation in this section. The reason is that the exclusions were selective. Under the drop-the-measure rule, the mean family-balanced score of TBATS is 0.236 on the eight tasks where it was observed before and 0.645 on the six to which it is restored; for ETS the values are 0.285 on nine tasks and 0.462 on six. The common-support rule had removed the two leading models from tasks on which they rank poorly. A negative forecast for a non-negative series is plausibly a symptom of a poor fit, so whether MSLE is defined carries information about accuracy and the missing cells are not missing at random.

Some conclusions do not depend on the rule. The same five models (TBATS, ETS, N-BEATS, DeepAR, Theta) fill the first five positions under all three rules, and none of the other six has a top-three frequency above 0.09. ETS and N-BEATS are in the top three under every rule. The omnibus result stands: the Skillings-Mack statistic is 43.26 and 42.10 under the two alternative rules (permutation $p\approx5\times10^{-5}$ in both), and with 14 complete tasks the ordinary Friedman test rejects as well (47.87 and 45.96). Sensitivity to benchmark composition remains: under the drop-the-measure rule the leave-one-competition-out leader is N-BEATS, DeepAR, ETS, N-BEATS, or TBATS depending on the competition omitted, and only N-BEATS stays in the top three in all five cases.

Metric agreement is affected too. With MSLE removed and all models restored, the largest task-level mean regret falls from 0.174 to 0.067; Tourism Yearly moves from W = 0.657 to 0.923 with zero regret and M1 Quarterly from 0.591 to 0.763, while M4 Daily remains the least concordant task (0.659). The leave-one-family-out check of Table 6 did not reveal this dependence because it removed the votes of the logarithmic family while keeping the model support that MSLE had determined.

On RQ3, then, two conclusions hold across every perturbation: the models differ systematically, and the same five lead. Which three of the five come first is robust to how the admissible measures are grouped, weighted, and thresholded. It is not robust to the model-compatibility rule, and only partly to which tasks are included. The first position is robust to none of these.

## 5.5 Do series characteristics explain where rankings are unstable?

Metric agreement differs strongly across tasks (Section 5.3). A natural question is whether measurable properties of the series predict where it is low. We examine it first across tasks and then across series within tasks.

**Task level.** We examined this with the two feature sets distributed with the Monash archive tooling: the tsfeatures set (Hyndman et al., 2020) and the 22 catch22 features (Lubba et al., 2019), both computed in R for each of the 10,537 series. Each feature was summarized by its median over the series of a task and related to the two task-level instability measures, metric disagreement ($1-W$) and mean selection regret, by Spearman correlation with a permutation p-value (100,000 permutations).

Because there are 15 tasks and many features, the analysis was split into two layers before any correlation was computed. The confirmatory layer has six characteristics chosen for their link to why accuracy measures might disagree: strength of trend, strength of seasonality, spectral entropy, lag-one autocorrelation, the within-series coefficient of variation, and the spread of series levels within a task (interquartile range of $\log_{10}$ of the series means). Four design covariates were examined alongside them: the number of Primary measures, the number of common-support models, the horizon, and the number of series. The exploratory layer screens all 55 remaining features of both packages, with Benjamini-Hochberg adjustment.

**Table 9.** Spearman correlation between task-level characteristics and ranking instability (15 tasks). Permutation p-values in parentheses are unadjusted; none is below 0.05, and after Holm adjustment within each column all exceed 0.5.

| Characteristic (task median) | Metric disagreement, $1-W$ | Mean selection regret |
|:--|:--|:--|
| Trend strength | 0.39 (0.16) | 0.05 (0.86) |
| Seasonal strength | -0.32 (0.24) | -0.36 (0.19) |
| Spectral entropy | -0.35 (0.20) | -0.08 (0.78) |
| Lag-one autocorrelation | 0.29 (0.30) | 0.21 (0.45) |
| Coefficient of variation | 0.39 (0.16) | 0.20 (0.46) |
| Spread of series levels | 0.28 (0.31) | 0.38 (0.17) |
| *Design covariates* | | |
| Number of Primary measures | -0.11 (0.70) | -0.33 (0.22) |
| Number of common-support models | -0.27 (0.32) | -0.14 (0.61) |
| Horizon | -0.40 (0.14) | -0.21 (0.46) |
| Number of series ($\log_{10}$) | 0.36 (0.19) | 0.29 (0.29) |

No pre-specified characteristic is reliably related to either instability measure (Table 9). The largest absolute correlation is 0.40, and none approaches significance once the ten tests per outcome are taken into account. The signs are nonetheless coherent: disagreement tends to be higher for strongly trended, less seasonal, more variable series and at short horizons, which is the direction one would expect if measures diverge mostly where no dominant regular pattern anchors every method. Two of the null results are informative in themselves. The number of Primary measures is unrelated to disagreement ($\rho=-0.11$), so low concordance is not an artifact of how many measures passed the screen, and the number of common-support models is unrelated as well.

The exploratory screen gives the same answer. For selection regret, no feature has an unadjusted p-value below 0.05. For disagreement, six of 55 do, against about three expected by chance, and the smallest adjusted value is $q=0.125$. The strongest two are the catch22 features that locate outliers in time (`DN_OutlierInclude_n_001_mdrmd`, $\rho=-0.75$; its positive-outlier counterpart, $\rho=0.64$). We report this as a hypothesis for later work, namely that measures disagree more where extreme observations are concentrated in one part of the series, and not as a finding.

Two qualifications apply to the task-level analysis. With 15 tasks, only a correlation above roughly 0.5 in absolute value could be detected at the conventional level. And a task median discards the variation among series within a task. Seasonal strength was available for the eight tasks whose features were computed with a seasonal period and was set to zero elsewhere; restricted to those eight tasks its correlation with disagreement is $-0.55$ ($p=0.16$).

**Series level.** We therefore repeated the analysis with the series as the unit. For each of the 10,537 series, the common-support models of its task were ranked under each Primary measure of that task using the losses of that series alone, which gives a Kendall's W and a mean selection regret per series. One adjustment is needed. Within a single series, MAE, nMAE, and MASE differ only by a constant and rank the models identically, as do MSE, RMSE, nRMSE, and RMSSE; each of these two groups was counted as one judge, leaving between four and ten distinct judges per task. The squared-error group was represented by a measure on the RMSE scale, because MSE, although it ranks the models identically, gives a different normalized regret. Series within a task are not independent forecasting problems, so tasks remain the unit of inference: Spearman correlations were computed within each task, and the 15 within-task correlations were pooled on the Fisher z scale with a t interval across tasks.

**Table 10.** Within-task Spearman correlation between series characteristics and series-level instability, pooled over the 15 tasks (seasonal strength: 8 tasks). Brackets give the 95% interval across tasks; "tasks" is the number of tasks in which the correlation has the sign of the pooled value. Holm-adjusted p-values within each outcome: * below 0.01.

| Series characteristic | Metric disagreement, $1-W$ | Tasks | Mean selection regret | Tasks |
|:--|:--|--:|:--|--:|
| Trend strength | -0.27 [-0.35, -0.18] * | 14 / 15 | -0.21 [-0.27, -0.15] * | 14 / 15 |
| Lag-one autocorrelation | -0.24 [-0.33, -0.13] * | 14 / 15 | -0.17 [-0.25, -0.10] * | 13 / 15 |
| Spectral entropy | 0.21 [0.13, 0.29] * | 15 / 15 | 0.18 [0.11, 0.25] * | 14 / 15 |
| Coefficient of variation | 0.09 [0.02, 0.17] | 10 / 15 | 0.06 [-0.01, 0.13] | 9 / 15 |
| Seasonal strength | 0.02 [-0.13, 0.16] | 4 / 8 | -0.03 [-0.16, 0.11] | 2 / 8 |
| Series level ($\log_{10}$ mean) | -0.04 [-0.14, 0.07] | 9 / 15 | -0.04 [-0.11, 0.04] | 9 / 15 |

At this level the picture is clear and consistent (Table 10). Within a task, the accuracy measures disagree more, and the choice of measure costs more, for series that are harder to forecast: series with high spectral entropy, weak trend, and low autocorrelation. The direction is the same in 13 to 15 of the 15 tasks for each of the three characteristics. The effects are modest, with correlations between 0.17 and 0.27 in absolute value, so forecastability accounts for a small part of the variation and does not allow instability to be predicted for an individual series. The level of a series, its coefficient of variation, and its seasonal strength are unrelated to instability. The exploratory screen agrees: 37 of 55 features pass a 5% false-discovery threshold for disagreement, and the strongest are further descriptors of persistence and irregularity (the `spike` feature, the Hurst exponent, unit-root statistics, and the autocorrelation-based catch22 features), not a different kind of characteristic.

Three further observations change how the task-level results should be read. First, within a single series the measures mostly agree: the median series-level disagreement is 0.04, and for 61% of series every switch of measure selects a model with zero regret. Second, task-level disagreement is not an accumulation of series-level disagreement: across the 15 tasks, the two are uncorrelated ($\rho=-0.12$). Third, measures that are equivalent within every series still disagree once losses are averaged over series. Kendall's W among the task-level rankings of MAE, nMAE, and MASE alone has a median of 0.95 across tasks and falls to 0.79 in M4 Daily; among MSE, RMSE, nRMSE, and RMSSE the median is 0.92 and the minimum 0.60, again in M4 Daily, which is below the concordance of all 15 measures on that task (0.72).

Taken together, ranking instability has two distinct sources. One operates within series and is tied to forecastability: when there is little persistent signal, models are close and the measures separate them differently. The other operates between series and is a matter of aggregation: measures that scale or normalize losses differently give different series different weight in the task mean, and this alone can reorder models. The second source is consistent with the finding that aggregate properties of the series do not predict task-level disagreement. We have not decomposed task-level disagreement into the two sources, so their relative size remains open. The per-series outcomes, all correlation tables, and both scripts are part of the reproducibility package.

**Figure 1.** Competition-only Bradley-Terry log-ability estimates with 95% task-bootstrap intervals. Higher values indicate stronger pairwise ranking performance.

**Figure 2.** Agreement among admissible metric rankings (Kendall's W) and the corresponding mean cross-metric selection regret, by task.

# 6 Forecast Studio: an open workbench for reproducible forecast evaluation

## 6.1 Motivation and design goals

Every ranking in Section 5 is conditional on a forecast-generation protocol (Section 3.2). A reader can assess that condition only if the protocol can be rerun, and the original experiments require R, Python, two sets of pinned libraries, and scripts that were written for one machine. The same barrier applies to practitioners who want to apply the reporting recommendations of Section 7.2 to their own data. Forecast Studio is our response. It was designed around four goals.

- **No programming.** Data selection, model choice, parameter settings, execution, and inspection of results happen in a browser interface.
- **Local and offline.** The application runs on the user's computer, binds to the loopback address only, and sends no data elsewhere. Its interface layer uses only the Python standard library, and it is sized for an ordinary laptop (8 GB of memory, no GPU).
- **Fidelity to the archived protocol.** Catalogue defaults (lag, horizon, integer conversion) are extracted programmatically from the original experiment scripts, and the original R code path is retained as one of three interchangeable engines.
- **Provenance by default.** Every run writes a manifest with its configuration, seed, software versions, the SHA-256 hash of the data file, skipped series, and the number of series on which a method fell back to a simpler model.

## 6.2 Architecture and workflow

Forecast Studio is a Python package (`tsfstudio`, Python 3.10 or later) with a command-line interface and a browser interface that call the same core, so a run started in either way produces identical output files and manifest. Table 11 lists its components.

**Table 11.** Forecast Studio components.

| Component | Function |
|:--|:--|
| Data layer | Reader and writer for the Monash `.tsf` format; a catalogue of 43 archive files with study defaults; an Excel template and converter that reports input errors by cell address |
| R engine | The original local and global R models (forecast package), run through a worker process with fallback counting |
| Python engine | Built-in SES, Theta, naive benchmarks, Naive2, Comb S-H-D and pooled regression; StatsForecast adapters for ETS, ARIMA, TBATS, intermittent-demand and further classical methods (Garza et al., 2022); decomposition-based, bagged and temporal-aggregation methods; optional global CatBoost and LightGBM |
| C++ engine | A compiled library with counterparts of the Python engine's methods, among them line-by-line ports of the R routines used in the original study (`ets`, `auto.arima`, `tbats`, `stl`), called from Python and run on several threads; it gives R-equivalent forecasts without an R installation |
| Accuracy measures | Eight default measures that reproduce the archived error files (sMAPE, msMAPE, MASE, MAE, RMSE, WAPE, two nRMSE variants) and eight optional ones (RMSSE, MSE, MAPE, MAAPE, MedAE, RMSLE, ME, percentage bias) |
| Within-task ranking (`rank`) | Mean ranks, Friedman test, Nemenyi critical difference (Demšar, 2006), AvgRelMAE (Davydenko & Fildes, 2013), M4 OWA (Makridakis et al., 2020), and a series bootstrap that returns each model's first-place and top-$k$ frequency and the Kendall $\tau$ distribution against the full-sample ranking |
| Cross-task reliability (`reliability`) | The framework of Section 4: common-support ranks, family-balanced scores, task-balanced Bradley-Terry ordering with task bootstrap and model coverage, permutation-calibrated Skillings-Mack test, Kendall's W, cross-metric selection regret, and leave-one-family-out re-estimation |
| Series-level instability (`instability`) | The analysis of Section 5.5: per-series Kendall's W and selection regret with within-series-equivalent measures counted once, the concordance of such measures at the task level (the aggregation effect), six built-in series features, and their within-task association with instability, pooled over tasks; external feature tables such as tsfeatures or catch22 output can be supplied instead |
| Runner | Train-test split, missing-value policy, memory and run-time estimate before execution, output files, manifest |
| Interface | Five-step browser workflow; 39 selectable methods with explanatory margin notes and references |

The interface follows five steps. (1) *Data*: the user picks a catalogue dataset or uploads a `.tsf` or Excel file, and the application reports frequency, horizon, number and length of series, shares of missing and zero values, and the demand pattern. (2) *Model*: methods are grouped as local, intermittent-demand, and global; a method whose engine is not installed is marked and cannot be started. (3) *Settings*: defaults are the values of the original study, a changed field is highlighted, and one click restores the default. (4) *Run*: a pre-run check estimates memory and time, refuses a run that would exhaust memory, and asks for confirmation of long runs; progress is shown and the run can be cancelled. (5) *Results*: a summary table, forecast diagnostics, sample-series charts, the error distribution, a per-series error table, and file downloads. Past runs can be compared side by side per measure, with a warning when datasets or horizons differ.

Two interface decisions follow directly from the findings of this paper. Each model, parameter, and measure carries a margin note that states when it is appropriate and what to watch for, so that a measure is not selected without its domain restrictions in view (Section 3.3). And the forecast diagnostics report the share of zero forecasts and raise a warning when a method that forecasts only zeros would be rewarded by MAE, MASE, or WAPE on intermittent demand, a concrete instance of invalid evidence entering a ranking.

The interface was put through an inspection-based usability review. The complete flow was walked through in a browser on two synthetic data sets (smooth weekly and intermittent monthly series), at desktop, small-window, and phone widths and in light and dark colour schemes, and every finding was recorded with a severity. The review found five high-severity problems, all corrected in release 0.1.1: past runs could not be opened on a phone; reloading the page during a run lost its progress display and the cancel control; results of an earlier run stayed on screen after the selection changed, so that a table could be read as belonging to the wrong model; colours were fixed in code, with no dark scheme; and the selection lists could not be operated sensibly by keyboard or screen reader. The interface now recovers a running job after a reload, marks displayed results as stale when the data or model selection changes, follows the colour scheme of the operating system, and treats each list as a single keyboard stop. Two medium-severity findings that bear on research use were also corrected: the results now show the run configuration (parameters, seed, software versions, and data hash), and the comparison of past runs no longer marks a best value across different data sets or horizons. [TODO: the corrections are taken from the 0.1.1 release notes; confirm them in the released build.] This review is an inspection, not a study with users, and we make no claim about measured usability. [TODO: the walkthrough was carried out with an AI coding assistant; disclose this according to the journal's policy on AI tools.]

Because the application accepts requests from a browser, it validates the `Host` header, requires a per-session random token on every request, never accepts a file path from the client, and restricts downloads to files recorded in a run manifest. The repository contains an automated test suite of 355 cases covering the data format, the measures (against hand-computed values), the three engines, the runner, the three analysis modules, and these security properties. [TODO: restate the count and pass status at the tagged release, on a machine with R and the optional libraries installed.]

## 6.3 Validation against the archived forecasts

The software contribution matters for this paper only to the extent that the workbench regenerates the evidence the rankings were built on. We tested this on the two NN5 tasks (111 series each; horizons 8 and 56), for which the archived per-series forecasts and error files of Section 3.2 are available as reference.

*Measures.* For 11 models, five measures, and 111 series, the largest relative difference between the workbench's measure values and the archived R error files is about $5\times10^{-15}$.

*R engine.* Table 12 compares forecasts from the R engine with the archived forecasts. The engine ran R 4.6.2 with forecast 9.0.2, whereas the archive was produced with forecast 8.12. Agreement is at machine precision for SES, ETS, Theta, and Pooled Regression, and at optimizer tolerance for the two ARIMA variants; no series differs by more than $10^{-3}$ in relative terms. On NN5 Daily the maximum relative difference is at most $5\times10^{-15}$ for SES, ETS, and Theta and $9\times10^{-14}$ for Pooled Regression, and two consecutive runs gave identical output.

**Table 12.** R engine versus archived forecasts, NN5 Weekly ($h=8$, 111 series, no fallbacks). Relative differences are per forecast value.

| Method | Median relative difference | Maximum relative difference | Series differing by more than $10^{-3}$ |
|:--|--:|--:|--:|
| SES | $1\times10^{-15}$ | $5\times10^{-15}$ | 0 / 111 |
| ETS | $1\times10^{-15}$ | $4\times10^{-15}$ | 0 / 111 |
| Theta | $1\times10^{-15}$ | $5\times10^{-15}$ | 0 / 111 |
| ARIMA | $3\times10^{-15}$ | $1.6\times10^{-6}$ | 0 / 111 |
| ARIMA with Fourier terms | $4\times10^{-14}$ | $5\times10^{-8}$ | 0 / 111 |
| Pooled Regression | $3\times10^{-15}$ | $8\times10^{-15}$ | 0 / 111 |

*Python engine.* The Python engine removes the R dependency at the price of exact replication (Table 13). Pooled Regression is reproduced exactly. The built-in SES and Theta differ on a handful of series, all cases in which the optimal smoothing parameter lies on its lower bound and the two optimizers stop at different points. The StatsForecast implementations of ETS and ARIMA select different models for many series, yet the aggregate measures move by less than 1.6%.

**Table 13.** Python engine versus archived forecasts, NN5 Weekly. $\Delta$ is the relative change in the task-mean measure.

| Method | Median relative difference | Series differing by more than $10^{-3}$ | $\Delta$ sMAPE | $\Delta$ MASE |
|:--|--:|--:|--:|--:|
| Pooled Regression | $1\times10^{-15}$ | 0 / 111 | 0 | 0 |
| SES | $4\times10^{-6}$ | 6 / 111 | +0.27% | +0.36% |
| Theta | $4\times10^{-6}$ | 5 / 111 | +0.06% | +0.08% |
| ETS | $4.7\times10^{-5}$ | 21 / 111 | -1.55% | -1.38% |
| ARIMA | $2.4\times10^{-3}$ | 70 / 111 | -0.07% | -0.33% |
| ARIMA with Fourier terms | $4.2\times10^{-3}$ | 96 / 111 | -0.11% | -0.40% |

The two engines therefore support different claims. The R engine supports *exact reproduction* of the statistical forecasts in the archive, across a major version change of the underlying package. The Python engine supports *accuracy-equivalent replication*: series-level forecasts differ, aggregate accuracy does not differ materially. A study that uses the workbench should name the engine and version, which the manifest records. This distinction is itself an instance of the paper's theme, since an implementation change that leaves mean accuracy almost unchanged can still reorder closely ranked models.

*Reliability analysis.* The `reliability` command takes either a set of run folders labelled by task and model or a table of mean losses by task, measure, and model, and returns every quantity defined in Section 4. We checked it against the stored outputs of the analysis reported in Section 5. Starting from the 135 family-balanced task scores of the competition benchmark, it reproduces all eleven Bradley-Terry log-abilities of Table 4 to the three reported decimals, the coverage counts, and the Skillings-Mack statistic to six decimals (45.941534). On complete blocks its Skillings-Mack statistic equals the Friedman statistic. These checks are part of the test suite, so the results of this paper are regression tests of the released software.

*Series-level analysis.* The `instability` command computes the series-level quantities of Section 5.5 from run folders. The per-series values reported in this paper were produced by the same package code, and the suite checks all 111 NN5 Weekly series against the stored values, together with the three task-level concordances of that task (0.942 for all measures, 0.996 within the absolute-error group, 0.966 within the squared-error group). The command computes six features itself, so that it does not depend on R, and these are approximations of the tsfeatures definitions, not reimplementations. On the 111 NN5 Weekly series, lag-one autocorrelation, coefficient of variation, and level are identical to the R values; spectral entropy has a rank correlation of 0.91 with the R value and trend strength 0.64, because a periodogram and a moving-average decomposition replace the smoothed spectrum and STL; seasonal strength is reported only for series with at least three full periods. The feature results of Section 5.5 use the R features. A user who needs agreement with R can pass an R feature table to the command.

## 6.4 Ranking stability inside a single task

The `rank` command addresses a question complementary to Section 5. There, tasks are the blocks and the concern is the cross-task ordering. Within one task the series are the blocks, and the same diagnostics show how well a task-level winner is determined. As an illustration we ran 13 statistical and regression methods of the Python engine on NN5 Weekly and resampled the 111 series 2,000 times (Table 14). This pool differs from the 11-model panel of Section 3.2, so the numbers are not comparable with Table 4.

**Table 14.** Within-task ranking stability on NN5 Weekly (13 methods, 111 series, 2,000 series resamples). "Leader" is the method with the lowest mean rank.

| Measure | Friedman $\chi^2$ ($p$) | Leader | Leader P(first) | Runner-up P(first) | Median Kendall $\tau$ |
|:--|:--|:--|--:|--:|--:|
| MASE | 104.0 ($9\times10^{-17}$) | SCUM | 0.640 | 0.237 | 0.724 |
| sMAPE | 104.5 ($7\times10^{-17}$) | SCUM | 0.454 | 0.401 | 0.740 |
| RMSSE | 125.6 ($5\times10^{-21}$) | SCUM | 0.942 | 0.038 | 0.740 |

The omnibus test rejects equality overwhelmingly under every measure, and the same method has the lowest mean rank under all three. The confidence that it leads nevertheless ranges from little better than a coin flip against Pooled Regression under sMAPE to near certainty under RMSSE. Methods that tie in a resample share the first-place credit. Under MASE the mean ranks of the 11 best methods span 5.78 to 7.28, less than one Nemenyi critical difference (1.73). The pattern of Section 5.2, a clearly non-null ranking whose first position is weakly determined, thus reappears at the series level, and the workbench reports it with a single command (`rank`) rather than leaving it to a separate analysis.

## 6.5 Scope and availability

Forecast Studio currently covers the statistical and regression part of the forecast pool. The neural forecasts analyzed in Section 5 come from the archived GluonTS runs and cannot yet be regenerated in the workbench. The three analysis commands (`rank`, `reliability`, `instability`) are available from the command line and not yet from the browser interface. The rule-based evidence screen of Section 3.3 and the competition-level perturbations of Section 5.4 remain in the scripts of the reproducibility package: the workbench drops a model from a task when one of its losses is undefined, but it does not classify measures as Primary, Secondary, or Excluded. The comparison of compatibility rules in Section 5.4 is likewise a script of the reproducibility package; it calls the workbench's `reliability` functions, and the workbench itself applies the common-support rule only. The repository's validation record reports that the C++ ports reproduce the archived R forecasts of the two NN5 tasks for ETS, ARIMA, and TBATS to within optimizer tolerance on all 222 series. [TODO: these figures come from `cpp/README.md` and were not re-run for this draft; verify them, and decide whether to report them in Section 6.3 with a table.] The intermittent-demand methods have no counterpart in the archive. Probabilistic measures and rolling-origin evaluation are not included.

The source code, tests, and validation fixtures are available at https://github.com/MicheleCedolin/forecast-studio [TODO: confirm the repository is public; add an archived DOI; replace by an anonymized link if the review is blinded] under the GNU General Public License, version 3; a notice file lists the routines translated from R and other libraries. For users who do not want to install Python, release 0.1.1 provides a self-contained Windows package (about 224 MB, for 64-bit Windows 10 and 11) that starts the interface from a single executable. It is built by an automated workflow from the repository, and its SHA-256 checksum is published with it. A local build of the same code, unpacked into a fresh folder, reproduced the Python engine's Comb S-H-D result on NN5 Weekly (mean MASE 0.8929). The package is unsigned, so Windows shows a warning at first start. [TODO: at the time of this draft the `reliability` and `instability` commands, their tests, and the analysis scripts of Section 5.5 exist only in the working copy. They are not in the repository's main branch and not in release 0.1.1. Commit them and cut a new release before submission, otherwise Section 6 describes software a reader cannot obtain.]

# 7 Discussion

## 7.1 What the evidence supports

This study treats a forecast ranking as an evidence-dependent object and traces three sources of uncertainty through it: admissible evaluation choices, incomplete model coverage, and benchmark composition. The evidence is better summarized by a benchmark-conditional leading set with explicit local exceptions than by a single winner.

This framing is consistent with the closest accuracy-measure literature. Koutsandreas et al. (2022) find only small ranking discrepancies across common measures on average, and several tasks here have Kendall W above 0.90. Aggregate agreement does not rule out localized disagreement, as M1 Quarterly, Tourism Yearly, and M4 Daily show, and Spiliotis et al. (2023) similarly report that selection criteria can disagree with consequences for accuracy. What the present analysis adds is the separation of full-ranking disagreement from selection regret, both conditioned on common model support.

The statistical design follows the comparison graph that the evaluation protocol actually produces. Complete-block rank tests remain useful references (Demšar, 2006; Koning et al., 2005), but under the common-support rule only five tasks contain all 11 models, and under any rule one model-task cell has no forecasts. The Skillings-Mack test retains all 15 tasks, and the task-balanced Bradley-Terry model converts the same incomplete graph into an interpretable ordering, in line with Wainer (2023) and Grishina et al. (2026). Where the incompleteness shrinks to that single cell, the Friedman test on the 14 complete tasks and the Skillings-Mack test on all 15 give the same verdict.

The robustness axes behave differently, and this is the main practical finding. Choices about how the admissible measures are grouped, weighted, and thresholded moved the leader but never the leading three. Two other choices moved both. One is the selection of data sets, which echoes Roque et al. (2025). The other is less visible: what happens to a model when a measure cannot be computed for it. Here a single measure, undefined on a median of three series per case, removed each of the two common-support leaders from six tasks on which it ranks poorly, and that is what placed them first. Disputes over the "right" metric therefore matter less for the conclusions of a benchmark study than which data sets are included and how undefined evidence is handled.

The exclusions are not ignorable in the missing-data sense. A negative forecast for a non-negative series signals a poor forecast, so a model tends to be removed where it does badly. The Bradley-Terry model adjusts for where a model is observed. It cannot adjust for why a model is unobserved. An incomplete-block method is therefore a remedy for incompleteness that is unrelated to accuracy, such as the missing TBATS forecasts for M4 Daily, and not for incompleteness created by the evaluation rule. We do not conclude that one compatibility rule is correct. Dropping the measure discards a loss that may matter to the user, and dropping the series changes the task for every model. The three orderings should be reported together.

The feature analysis adds a distinction that the task-level results alone would hide. For a single series the admissible measures usually agree, and they agree less where the series is hard to forecast. Part of the disagreement seen at the task level arises afterwards, when losses are averaged over series and differently scaled measures weight the series differently. This matters for practice. A dispute between, say, MAE and MASE on a benchmark is then not a dispute about how to measure error. It is a decision about which series should dominate the average, and it can be made explicitly.

The Model Confidence Set also returns a set rather than a winner (Hansen et al., 2011). Under the common-support rule an all-model MCS would be driven by the five complete tasks unless the joint loss-difference process were reformulated on a changing support graph. That reformulation is a separate methodological problem. Task-bootstrap first-place and top-three frequencies serve as the uncertainty summaries here.

## 7.2 Reporting recommendations

The results suggest six items that a benchmark study should report alongside any ranking. Forecast Studio records items 1 and 4 in the manifest of every run and computes items 3 and 5 with its `reliability` command; its `instability` command reports how much of a task's disagreement arises from averaging over series.

1. **Forecast-generation protocol.** Software and versions, tuning effort per model class, seeds, and the forecast origin.
2. **Evidence screen.** Which measures were admissible for which task, and why the others were not; and what was done when an admissible measure was undefined for a model (the model, the measure, or the series removed), with the number of series responsible.
3. **Coverage.** The number of tasks on which each model is observed, next to its rank.
4. **Fallbacks and exclusions.** How many series were skipped or forecast by a substitute method.
5. **A stability summary, not only a point ranking.** First-place and top-$k$ frequencies under resampling of the benchmark units.
6. **Composition sensitivity.** The ranking after removing each benchmark source in turn.

For an analyst who has to choose a model rather than report a benchmark, the diagnostics translate into the reading in Table 15. The thresholds a user applies are a matter of judgment; the table states which quantity answers which question.

**Table 15.** Reading the diagnostics when a model has to be chosen.

| Observation | What it means | What to do |
|:--|:--|:--|
| A measure is undefined or structurally unsuitable for the data | It is not evidence for this task | Exclude it before ranking; do not average it in |
| High Kendall's W and regret near zero | Admissible measures select the same model | The choice of measure is immaterial; report one and say so |
| Measures that differ only in scaling (MAE, nMAE, MASE) rank models differently | The difference is in how much weight each series receives, not in how errors are measured | Decide which series should count more (large ones, or all equally) and choose the measure accordingly |
| High W but a large maximum regret | Measures agree overall but one switch is costly | Identify the pair of measures involved and decide which loss matches the use |
| Low W and high mean regret | The selected model depends on the measure | Choose the measure from the decision the forecast serves, then select; do not pool |
| First-place frequency below one half, top-$k$ frequency high | A leading group is identified, a leader is not | Report or deploy from the group; break ties on cost, simplicity, or run time |
| A leading model has low coverage | Its rank rests on few tasks | Treat its position as provisional, state the coverage, and find out why the model is missing |
| A model is missing from a task because a measure is undefined for it on a few series | The exclusion may be informative: the model may be doing badly there | Count the series responsible; re-rank with the measure or the series removed instead, and report the orderings together |
| Omnibus test rejects but mean ranks lie within one critical difference | Models differ somewhere, not necessarily at the top | Do not read the top of the ranking as significant |

## 7.3 Limitations

*Fixed forecasts and tuning effort.* The ordering is conditional on the protocol of Section 3.2. Local statistical procedures perform automatic series-specific selection, whereas the neural estimators use one GluonTS version, fixed context lengths, package defaults, no validation set, and a single seed. Tuning-effort heterogeneity cannot be separated from the fixed forecast pool, and the results should not be read as a contemporary leaderboard of optimally tuned algorithms.

*Few and dependent blocks.* The 15 tasks are not a probability sample of forecasting problems, and tasks from one competition share design and domain provenance. Bootstrap stability across 1,000-10,000 resamples shows adequate Monte Carlo precision; it does not remove the limitation of having 15 blocks in five clusters.

*Coverage of the leaders.* The two highest-ranked models are those observed on the fewest tasks. The Bradley-Terry model adjusts for where a model is observed but cannot supply information about tasks on which it is not. That information turns out to matter: when the measure that caused the exclusions is removed instead of the models, both fall in the ordering (Table 8), so the common-support ordering of the first three positions should not be read without that table.

*Compatibility rules.* The two alternative rules were examined after the common-support results were known and were not part of the original design. Neither is neutral: removing MSLE removes the only measure of the logarithmic family from ten tasks, and removing series changes the task (82 of the 414 M4 Hourly series and 63 of the 518 Tourism Yearly series). Intermediate rules, such as truncating negative forecasts at zero before computing the measure, were not tried.

*Horizon aggregation.* The analysis summarizes performance over each task's prescribed horizon. In an exploratory analysis the median first-to-final-horizon rank correlation is only 0.249, against 0.764 for adjacent horizons (Table S14). The results characterize full-horizon rankings, not horizon-invariant superiority (Strøm & Gundersen, 2024).

*Series characteristics.* The feature analysis of Section 5.5 is correlational. At the series level the relations with forecastability are consistent across tasks but modest, and they do not support predicting instability for an individual series. At the task level, 15 tasks could reveal only a strong dependence. The features were computed on complete series, including the test period, and seasonal strength was available for eight tasks only. The account of the aggregation source rests on the concordance of within-series-equivalent measures; we did not decompose task-level disagreement formally into its two sources.

*Software scope.* The workbench reproduces the statistical forecasts but not yet the neural ones, and its validation against the archive covers the two NN5 tasks (Section 6.5).

Future work should extend the framework to newer benchmark archives and to foundation models, for which overlap between pre-training corpora and benchmark data adds a further validity condition; to hierarchical or cluster-aware inference with more benchmark sources; to probabilistic forecasts; and to application-specific costs.

# 8 Conclusion

Forecasting-model rankings are conditional summaries of an evaluation design. This study develops a compact reliability analysis for the common situation in which the evidence is multi-metric and the model-by-task matrix is incomplete. A validity screen defines the admissible evidence, common support ensures that cross-metric comparisons refer to the same models, a task-balanced Bradley-Terry model supplies the global ordering, and a permutation-calibrated Skillings-Mack test provides omnibus inference. Kendall's W and cross-metric regret then separate ranking instability from instability that changes the selected model.

Across 15 competition tasks and 10,537 series, the evidence supports a benchmark-conditional leading structure rather than a universal winner. Five methods, TBATS, ETS, N-BEATS, DeepAR, and Theta, occupy the first five positions under every specification we tried. Their order is not determined. When a model is dropped from the tasks on which a measure is undefined for it, TBATS, ETS, and N-BEATS come first, and that membership survives every change to metric weighting, taxonomy, and screening thresholds. It does not survive a change of the dropping rule itself: the exclusions come from one measure and a few series, they remove the leaders where they do poorly, and without them TBATS is fifth. Changes to benchmark composition alter the leader under every rule. Accuracy measures agree strongly on most tasks and materially less on a few, and where they disagree the choice of measure tends to carry selection regret. Within a series they agree more than at the task level, and less for series with little persistent signal; part of the task-level disagreement is an effect of aggregation over series.

Forecast Studio makes the conditioning protocol executable. It reproduces the archived statistical forecasts to machine precision, records the provenance that a qualified ranking requires, and reports ranking stability next to the ranking itself. Forecasting benchmarks should report not only who ranks first but also the forecast-generation protocol, the validity and coverage conditions, the treatment of undefined evidence, the metric specification, the horizon, and the benchmark composition under which that ranking was obtained.

# Declarations

**Funding.** [TODO]

**Competing interests.** [TODO]

**Author contributions.** [TODO]

**Data availability.** The benchmark data are publicly available through the Monash Time Series Forecasting Archive and the original competition sources.

**Code availability.** A reproducibility package contains the fixed forecast-error inputs, admissibility and compatibility audits, common-support rankings, analysis code, outputs, tables, and figures for every analysis in Sections 5 and S5; it operates on stored forecast outputs and does not refit models. Forecast Studio, with its tests, validation fixtures, and a packaged Windows release, is available at https://github.com/MicheleCedolin/forecast-studio [TODO: DOI and license]. A blinded package is available for peer review.

# References

Alexandrov, A., Benidis, K., Bohlke-Schneider, M., Flunkert, V., Gasthaus, J., Januschowski, T., et al. (2020). GluonTS: Probabilistic and neural time series modeling in Python. Journal of Machine Learning Research, 21(116), 1-6. [new in v2: verify]

Andrawis, R.R., Atiya, A.F. & El-Shishiny, H. (2011). Forecast combinations of computational intelligence and linear models for the NN5 time series forecasting competition. International Journal of Forecasting, 27, 672-688.

Athanasopoulos, G., Hyndman, R.J., Song, H. & Wu, D.C. (2011). The tourism forecasting competition. International Journal of Forecasting, 27, 822-844.

Bradley, R.A. & Terry, M.E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. Biometrika, 39, 324-345.

Cerqueira, V., Roque, L. & Soares, C. (2025). ModelRadar: Aspect-based forecast evaluation. Machine Learning, 114, 229.

Chatfield, M. & Mander, A. (2009). The Skillings-Mack test (Friedman test when there are missing data). The Stata Journal, 9, 299-305.

Davydenko, A. & Fildes, R. (2013). Measuring forecasting accuracy: The case of judgmental adjustments to SKU-level demand forecasts. International Journal of Forecasting, 29, 510-522. [new in v2: verify]

Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research, 7, 1-30.

Garza, F., Mergenthaler Canseco, M., Challú, C. & Olivares, K.G. (2022). StatsForecast: Lightning fast forecasting with statistical and econometric models. PyCon, Salt Lake City. https://github.com/Nixtla/statsforecast [new in v2: verify]

Godahewa, R., Bergmeir, C., Webb, G.I., Hyndman, R.J. & Montero-Manso, P. (2021). Monash Time Series Forecasting Archive. Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, 1.

Grishina, E., Kuznetsov, S., Tsyganov, A., Ivanov, I., Korovaitceva, D., Rusanova, M., Parkina, U., Derevyagin, A., Frolov, E., Samsonov, S. & Lysenko, A. (2026). Bradley-Terry rankings for recommender systems across dataset taxonomies. Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining. https://doi.org/10.1145/3770855.3817890

Gusev, R. & Zaytsev, A. (2026). Benchmarking on tasks that matter: Dataset selection for preserving model rankings. Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining. https://doi.org/10.1145/3770855.3817569

Hansen, P.R., Lunde, A. & Nason, J.M. (2011). The model confidence set. Econometrica, 79, 453-497.

Hewamalage, H., Ackermann, K. & Bergmeir, C. (2023). Forecast evaluation for data scientists: Common pitfalls and best practices. Data Mining and Knowledge Discovery, 37, 788-832.

Hyndman, R.J., Kang, Y., Montero-Manso, P., Talagala, T., Wang, E., Yang, Y. & O'Hara-Wild, M. (2020). tsfeatures: Time series feature extraction. R package. https://CRAN.R-project.org/package=tsfeatures [new in v2: verify; state the version used]

Hyndman, R.J. & Khandakar, Y. (2008). Automatic time series forecasting: The forecast package for R. Journal of Statistical Software, 27(3), 1-22. [new in v2: verify]

Hyndman, R.J. & Koehler, A.B. (2006). Another look at measures of forecast accuracy. International Journal of Forecasting, 22, 679-688.

Kang, Y., Hyndman, R.J. & Smith-Miles, K. (2017). Visualising forecasting algorithm performance using time series instance spaces. International Journal of Forecasting, 33, 345-358.

Kendall, M.G. & Smith, B.B. (1939). The problem of m rankings. Annals of Mathematical Statistics, 10, 275-287.

Kolassa, S. (2016). Evaluating predictive count data distributions in retail sales forecasting. International Journal of Forecasting, 32, 788-803. [new in v2: verify]

Kolassa, S. (2020). Why the best point forecast depends on the error or accuracy measure. International Journal of Forecasting, 36, 208-211.

Koning, A.J., Franses, P.H., Hibon, M. & Stekler, H.O. (2005). The M3 competition: Statistical tests of the results. International Journal of Forecasting, 21, 397-409.

Kourentzes, N. (2013). Intermittent demand forecasts with neural networks. International Journal of Production Economics, 143, 198-206. [new in v2: cited by Reviewer 2; verify]

Koutsandreas, D., Spiliotis, E., Petropoulos, F. & Assimakopoulos, V. (2022). On the selection of forecasting accuracy measures. Journal of the Operational Research Society, 73, 937-954.

Lubba, C.H., Sethi, S.S., Knaute, P., Schultz, S.R., Fulcher, B.D. & Jones, N.S. (2019). catch22: CAnonical Time-series CHaracteristics. Data Mining and Knowledge Discovery, 33, 1821-1852. [new in v2: verify]

Makridakis, S., Andersen, A., Carbone, R., Fildes, R., Hibon, M., Lewandowski, R., et al. (1982). The accuracy of extrapolation (time series) methods: Results of a forecasting competition. Journal of Forecasting, 1, 111-153.

Makridakis, S. & Hibon, M. (2000). The M3-Competition: Results, conclusions and implications. International Journal of Forecasting, 16, 451-476.

Makridakis, S., Spiliotis, E. & Assimakopoulos, V. (2018). The M4 Competition: Results, findings, conclusion and way forward. International Journal of Forecasting, 34, 802-808.

Makridakis, S., Spiliotis, E. & Assimakopoulos, V. (2020). The M4 Competition: 100,000 time series and 61 forecasting methods. International Journal of Forecasting, 36, 54-74. [new in v2: verify]

Montero-Manso, P., Athanasopoulos, G., Hyndman, R.J. & Talagala, T.S. (2020). FFORMA: Feature-based forecast model averaging. International Journal of Forecasting, 36, 86-92. [new in v2: verify]

Petropoulos, F., Apiletti, D., Assimakopoulos, V., Babai, M.Z., Barrow, D.K., Ben Taieb, S., et al. (2022). Forecasting: Theory and practice. International Journal of Forecasting, 38, 705-871. [new in v2: verify]

Petropoulos, F., Makridakis, S., Assimakopoulos, V. & Nikolopoulos, K. (2014). Horses for courses in demand forecasting. European Journal of Operational Research, 237, 152-163.

Roque, L., Cerqueira, V., Soares, C. & Torgo, L. (2025). Cherry-picking in time series forecasting: How to select datasets to make your model shine. Proceedings of the AAAI Conference on Artificial Intelligence, 39, 20192-20199.

Shchur, O., Turkmen, C., Erickson, N., Shen, H., Shirkov, A., Hu, T. & Wang, Y. (2023). AutoGluon-TimeSeries: AutoML for probabilistic time series forecasting. Proceedings of the International Conference on Automated Machine Learning, PMLR 224. [new in v2: verify]

Skillings, J.H. & Mack, G.A. (1981). On the use of a Friedman-type statistic in balanced and unbalanced block designs. Technometrics, 23, 171-177.

Spiliotis, E., Petropoulos, F. & Assimakopoulos, V. (2023). On the disagreement of forecasting model selection criteria. Forecasting, 5, 487-498.

Syntetos, A.A., Nikolopoulos, K. & Boylan, J.E. (2010). Judging the judges through accuracy-implication metrics: The case of inventory forecasting. International Journal of Forecasting, 26, 134-143. [new in v2: verify]

Strøm, E. & Gundersen, O.E. (2024). Performance metrics for multi-step forecasting measuring win-loss, seasonal variance and forecast stability: An empirical study. Applied Intelligence, 54, 10490-10515.

Wainer, J. (2023). A Bayesian Bradley-Terry model to compare multiple machine learning algorithms on multiple data sets. Journal of Machine Learning Research, 24(341), 1-34.
