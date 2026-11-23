---
title: "Cover letter and response to the reviews of submission b84d9475-d98b-46c7-b5b0-2bb72564d1e0"
date: "Draft, 22 November 2026"
---

<!--
Draft for the authors. Section references are to manuscript/Manuscript_v2.md.
[TODO] marks facts that are not available in the repository. Response R1.2 describes a gap that is still open
(license, DOI, pushed commits); read it before sending.
-->

# Part A. Cover letter

[TODO: date]

The Editor, *Machine Learning*

Dear Editor,

I submit the manuscript "How Reliable Are Forecasting-Model Rankings? A Validity- and Coverage-Aware Benchmark Evaluation with an Open Workbench" for consideration in *Machine Learning*.

I want to be open about its history. An earlier, co-authored study on a related question, "Evaluating Forecasting Models under Non-Normal Error Distributions: A Metric-Sensitive Comparative Framework" (submission b84d9475-d98b-46c7-b5b0-2bb72564d1e0), was rejected by the journal. The two reviewers identified weaknesses that I accept: the study rested on one data set of 111 series, its novelty claim was narrow, the promised code and appendices were not available, and it offered no guidance beyond counting pairwise wins. I did not revise that paper. I redesigned the study around the question the reviewers pointed to, namely what an analyst should conclude when accuracy measures send different messages. The present manuscript is new in its data, its forecast pool, its statistical method, and its conclusions. Because the reviews shaped it, I attach a point-by-point account of how each comment is handled (Part B) and would welcome the same reviewers.

The main differences from the earlier submission are these.

- **Scope.** The benchmark has 15 dataset-frequency tasks from the M1, M3, M4, Tourism, and NN5 competitions, with 10,537 series. The earlier study used NN5 Daily only.
- **Question.** The paper no longer proposes a metric-based ranking procedure. It asks how reliable a ranking is once measure validity, incomplete model coverage, disagreement among admissible measures, and benchmark composition are made explicit.
- **Method.** A validity screen decides which measures count as evidence for each task. A task-balanced Bradley-Terry model and a permutation-calibrated Skillings-Mack test handle the incomplete model-by-task matrix that the screen produces. Kendall's W and a cross-metric selection regret separate ranking disagreement from its consequence for model choice.
- **Finding.** Models differ systematically, and five of the eleven lead under every specification, but their order is not determined. Choices about how measures are grouped, weighted, and thresholded change the leader and not the leading three. Two other choices change both: which tasks are included, and what is done when a measure is undefined for a model. Every model exclusion in the benchmark comes from one measure, most from at most three series, and the exclusions favor the models they remove (Section 5.4, Table 8).
- **Reproducibility.** A reproducibility package covers every reported analysis. In addition I release Forecast Studio, an open, locally run workbench. Its R engine reproduces the archived statistical forecasts to machine precision, and its analysis commands reproduce the Bradley-Terry ordering and the Skillings-Mack statistic reported in the paper.

I believe the manuscript fits the journal's interest in evaluation methodology, and it builds on recent work published there (Cerqueira et al., 2025).

I am the sole author of the present manuscript. The earlier submission had a co-author; the present study was designed, carried out, and written independently of it. [TODO: confirm this sentence describes the situation, and that the earlier co-author agrees.] The manuscript is not under consideration elsewhere. [TODO: confirm, and add statements on funding and competing interests.]

Yours sincerely,

Michele Cedolin
[TODO: affiliation, e-mail]

# Part B. Response to the reviews of the earlier submission

Reviewer comments are summarized in italics. "Now" describes the present manuscript.

## Reviewer 1

**R1.1** *The analysis rests on a single data set (NN5, 111 series, banking), yet the framework is presented as broadly applicable.*

I agree. The principal benchmark now contains 15 tasks from five competitions and 10,537 series, covering yearly to hourly frequencies and several domains (Section 3.1, Table 1). I also changed what is claimed. The paper states that a ranking is conditional on the benchmark, shows by leave-one-competition-out analysis how far the ranking moves when a source is removed (Table 7), and does not present the leading models as general winners.

**R1.2** *Data and code are said to be included but no link, repository, or supplementary material is provided; Appendix A is missing.*

This was a fault of the earlier submission. The present submission includes the Supplementary Material (15 tables) and a reproducibility package with the forecast-error inputs, the screening audits, the analysis code, and all outputs. The software is described in Section 6, and the paper's own results are part of its test suite. The code is at https://github.com/MicheleCedolin/forecast-studio, with a packaged Windows release that runs without installing Python. [TODO: before sending, confirm the repository is public, push the commits that add the `reliability` command and the manuscript's regression tests, add a license file, and archive a release to obtain a DOI. Without these the response overstates what a reviewer can access.]

**R1.3** *Recent work on automated machine learning, ensembles, and meta-learning is missing, as is the relation to existing ranking frameworks and model-selection criteria.*

Section 2.4 now discusses feature-based meta-learning, automated ensembling, and the selection and combination literature, and states how my work relates to them: those methods optimize under a given loss and benchmark and are themselves compared by leaderboards, whereas I study how far such leaderboards depend on the loss, coverage, and benchmark. Section 2.3 relates the method to established ranking procedures (Friedman-type tests, multiple comparisons with the best, the Model Confidence Set, Bradley-Terry models), and Section 7.1 explains why a complete-block procedure or a Model Confidence Set would use only five of the fifteen tasks. Disagreement among selection criteria is discussed in Sections 2.1 and 7.1.

**R1.4** *Insufficient detail on preprocessing, validation, and which hyperparameters were optimized.*

The forecast pool is different from that of the earlier study, and its protocol is reported in full in Section 3.2 and Supplement S1: software and versions, the terminal-holdout split, context lengths, and the fact that no validation set and no hyperparameter search were used. I do not claim an optimized comparison. Section 7.3 states that differences in tuning effort between model classes cannot be separated from the results. The statistical part of the protocol can be rerun in Forecast Studio, which writes the configuration, seed, and software versions of every run to a manifest (Section 6.1).

**R1.5** *Main concern: the framework reduces to "run all models and count wins" and gives no concrete guidance on when to prefer which model.*

I accept this criticism of the earlier paper. The present paper does not count wins. It reports, for each model, how often it is first and how often it is in the leading group when the benchmark is resampled, together with the number of tasks on which it is observed (Table 4). For guidance, Section 7.2 gives six reporting items and Table 15 states, for each diagnostic pattern, what it means and what to do, for example when admissible measures disagree and the disagreement changes the selected model.

I should be clear about what is still not offered. The paper does not predict the best model from series characteristics or business constraints; the feature analysis in Section 5.5 relates instability to forecastability within tasks, with effects too modest for prediction. It tells the analyst how much a given ranking can be trusted and which choice (measure or data) drives it. I state this limit in Section 7.3.

## Reviewer 2

**R2.0** *If a metric is not suitable for the properties of a series, why include it in the framework at all? Is the metric leading or following?*

This question changed the design. Suitability is now settled first. A rule-based screen classifies each measure for each task, and a measure that is undefined or structurally unsuitable does not enter the ranking (Section 3.3, Supplement S2). Only the measures that pass are compared. The Introduction states this ordering explicitly. The 48-configuration sensitivity analysis shows that the conclusions do not depend on the particular screening thresholds (Section 5.4).

The question also has a second half that I did not see at first. A measure can be suitable for a task and still be undefined for one model on a few series. Removing the model in that case, as the main analysis does, turned out to decide the leading positions: all 29 such exclusions come from one measure (MSLE, undefined for negative forecasts), and they remove models from tasks on which they rank poorly. Section 5.4 therefore reports the ordering under three rules side by side (Table 8), and the paper states which conclusions hold under all of them.

**R2.1** *The statement that competitions rely on a single metric should be corrected; acknowledge that competitions recognize the problem and that metrics often correlate weakly with decisions.*

The statement has been removed. Section 2.1 now notes that the M4 competition combined two measures and that multi-measure reporting is common, and it discusses the work showing that accuracy rankings and inventory outcomes can diverge. My selection regret is positioned as a step toward decision consequences that stays within accuracy measures, not as a substitute for a cost-based evaluation.

**R2.2** *Using several metrics with non-parametric multiple-comparison tests is standard (e.g. Kourentzes, 2013); the novelty claim is inaccurate.*

Agreed. The claim is withdrawn, and the Introduction says so directly: "Computing several accuracy measures and applying a non-parametric comparison test is by now routine in forecasting studies, and we claim no novelty for either." Kourentzes (2013) is cited.

**R2.3** *The contribution is framed too narrowly (why 11 metrics, why Mann-Whitney, why Dunn?). The real question is what to do when metrics give different messages.*

I adopted this framing. The contribution is no longer tied to a count of measures or a named test. The components are chosen for stated reasons: measures are grouped into families so that related formulas do not vote repeatedly, the Skillings-Mack test is used because the comparison matrix is incomplete, and the Bradley-Terry model is used because coverage differs by task (Sections 4.1 to 4.3). Alternative weightings and taxonomies are reported as robustness checks (Table S10).

**R2.4** *111 series is limited; a large and diverse set is needed, and time-series characteristics should be used to validate the framework.*

The first part is addressed (see R1.1). For the second part I added a feature analysis (Section 5.5, Table 9). The tsfeatures and catch22 features of all 10,537 series were summarized per task and related to metric disagreement and to selection regret. Six characteristics and four design covariates were fixed in advance; all other features were screened with a false-discovery adjustment. The analysis was run at two levels.

Across the 15 tasks the result is negative: no characteristic is reliably related to instability (largest absolute Spearman correlation 0.40; nothing survives adjustment), and the number of admissible measures is unrelated to disagreement, so low agreement is not an artifact of the screen.

Across series within tasks the result is positive and consistent (Table 10). Model rankings were computed per series under each admissible measure, and within-task correlations were pooled with tasks as the unit of inference. Measures disagree more, and the choice of measure costs more, for series that are harder to forecast: higher spectral entropy (pooled correlation 0.21, same sign in 15 of 15 tasks), weaker trend (-0.27, 14 of 15), and lower autocorrelation (-0.24, 14 of 15). Series level, coefficient of variation, and seasonal strength are unrelated. The effects are modest, and I say so.

The series-level analysis also showed something I had not expected. Within a single series the measures mostly agree (median disagreement 0.04), and measures that are equivalent within every series, such as MAE, nMAE, and MASE, still rank models differently once losses are averaged over series. Part of the task-level disagreement is therefore an aggregation effect: a question of which series dominate the average, not of how error is measured. This is now part of the discussion (Section 7.1) and of the practical guidance (Table 15).

Limits are stated in Section 7.3: the analysis is correlational, the features were taken from the archive tooling and computed on complete series, and the two sources of disagreement are not formally decomposed.

**R2.5** *Several methods exist in multiple versions; precise, referenced descriptions of the implementations are needed.*

Section 3.2 and Supplement S1 name the function and package version behind every method. In Forecast Studio each method carries a description with its references, and the manifest of each run records the versions actually used.

**R2.6** *The literature argues for relative and scaled measures such as MASE; none is used.*

MASE and RMSSE are now part of the measure library as a separate "scaled" family, together with two normalized measures (Section 3.3, Table S2).
