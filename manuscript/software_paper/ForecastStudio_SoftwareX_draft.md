<!--
Draft 2, 2026-11-25. Target: SoftwareX (Elsevier), article type "Original software publication".
Limits of that article type: 3,000 words of main text (metadata tables, captions and references excluded), at most six figures.
Scope decided by the author on 2026-11-25: an independent software paper. Its two messages are (1) a forecasting platform anyone can
use and (2) forecasting models implemented in C++ for fast solutions. It does not compare forecasting models and does not depend on
the ranking manuscript.
Open items are marked [TODO] and collected at the end of the file.
-->

# Forecast Studio: an open, no-code forecasting platform with a C++ core for the desktop and the browser

Michele Cedolin

[TODO: affiliation, postal address, e-mail, ORCID]

## Abstract

Forecast Studio is an open-source platform that makes time-series forecasting available to users who do not program. Data are loaded from an Excel sheet or a Monash `.tsf` file, a method is chosen from a list in which every entry is explained, and forecasts, accuracy measures and charts are returned together with a record of how they were produced. The platform has two editions: a desktop application with 45 methods, and a browser edition that needs no installation and runs on a phone, with the data never leaving the device. Both are built on tsfcore, a C++17 library in which the forecasting methods are implemented, among them ports of the automatic ETS, ARIMA and TBATS procedures of the R forecast package. The ports agree with their reference implementations to numerical precision and run SPEEDUP_ABSTRACT, which turns batch forecasting of thousands of series into an interactive task.

**Keywords:** time series forecasting; no-code software; C++; WebAssembly; open-source software; reproducibility

## Metadata

**Table 1.** Code metadata.

| Nr | Code metadata description | |
|:--|:--|:--|
| C1 | Current code version | v0.1.1 [TODO: cut a new release that contains the browser edition and the C++ run pipeline, and cite that version here] |
| C2 | Permanent link to code/repository used for this code version | https://github.com/MicheleCedolin/forecast-studio |
| C3 | Permanent link to reproducible capsule | [TODO: Zenodo DOI of the release archive] |
| C4 | Legal code license | GNU General Public License v3.0 |
| C5 | Code versioning system used | git |
| C6 | Software code languages, tools and services used | C++17, Python (3.10 or later), JavaScript, R; CMake, Emscripten, PyInstaller, GitHub Actions |
| C7 | Compilation requirements, operating environments and dependencies | Browser edition: any current browser with WebAssembly. Desktop edition: packaged build for 64-bit Windows 10 and 11; source installation on Windows, macOS and Linux with NumPy, SciPy, pandas, openpyxl, XlsxWriter, PyYAML; optional StatsForecast, CatBoost, LightGBM, scikit-learn, PyTorch, NeuralForecast, Chronos; optional R with the forecast package |
| C8 | Link to developer documentation/manual | https://github.com/MicheleCedolin/forecast-studio/tree/main/docs |
| C9 | Support email for questions | [TODO: e-mail] |

## 1. Motivation and significance

Forecasts of demand, admissions, load or arrivals are needed far beyond the community that develops forecasting methods. The established methods are freely available: automatic exponential smoothing and ARIMA in the forecast package for R [1], and a growing set of Python libraries such as StatsForecast [2], GluonTS [3], Darts [4] and sktime [5]. All of them are programming libraries. To obtain a forecast, a user has to install a language environment, resolve package versions, write a script, and assemble the results. This excludes most of the people who hold the data: planners and analysts in firms, hospitals and public bodies, and students who meet forecasting for the first time. Where installing software is not permitted, or the only device at hand is a phone, the libraries cannot be used at all.

A second barrier is time. The automatic procedures that make statistical forecasting usable without expert tuning search over many candidate models for every series. In their reference implementations this takes seconds per series for the heavier methods, so a collection of a few thousand series occupies a machine for minutes to hours and cannot be explored interactively. Faster implementations exist for some methods [2], but they are independent re-implementations whose results differ from the reference on individual series, which leaves a user unsure which numbers to trust.

Forecast Studio addresses both barriers. Its aims are the following.

- **A platform everyone can use.** No programming: data, method, settings, run and results are steps in a browser interface, and every method, parameter and accuracy measure carries a short explanation of how it works, when to use it and what to watch for. No installation: the browser edition is a web page. Any device: the page works on a phone. No disclosure: the desktop edition listens on the local machine only, and the browser edition computes inside the page, so confidential data are never uploaded.
- **Fast solutions through C++.** The forecasting methods are implemented in one C++ library, tsfcore, that processes all series of a job in a single call across the processor's cores. The same source is compiled natively for the desktop and to WebAssembly [6] for the browser.
- **Results that can be trusted.** The C++ implementations are ports, not approximations: each is checked against the implementation it was translated from, series by series, and the checks are kept in the test suite.
- **A record of every run.** Each run stores its configuration, random seed, software versions, the SHA-256 hash of the data file, the series that were skipped, and the series on which a method failed and a fallback forecast was used.

To our knowledge no other open tool combines a no-code interface, execution on the user's own device including phones, and compiled implementations that reproduce the reference R procedures.

## 2. Software description

### 2.1 Software architecture

The platform consists of a C++ core and two editions built on it (Fig. 1).

[Figure 1 about here. Architecture: tsfcore at the centre; native build used by the desktop edition next to the Python and R engines; WebAssembly build used by the browser edition; shared data format, accuracy measures and result files.]

**tsfcore** is a C++17 library with a plain C interface. It contains

- ports of the R routines behind `ets`, `auto.arima`, `tbats`, `stl`, `stlf`, `baggedETS`, `thief` and `mapa`, together with the numerical routines they rely on: R's Nelder-Mead, BFGS and L-BFGS-B optimisers, its Kalman-filter likelihood for ARIMA, the LINPACK and LAPACK paths that R takes, and R's random number generator;
- ports of the StatsForecast implementations of Theta variants, complex exponential smoothing, MSTL, MFLES and the intermittent-demand methods;
- global models: pooled regression, and LightGBM [7] and CatBoost [8] through their native libraries;
- the platform itself: a reader and writer for the Monash `.tsf` format [9], 16 accuracy measures, and a complete run pipeline (train-test split, missing-value policy, rolling origins, prediction intervals, result files, run record).

All series of a job are passed in one call and distributed over threads inside the library. A series on which a method fails receives the seasonal naive forecast and is flagged, so that one difficult series does not stop a run. A notice file lists every translated routine with its origin and licence.

The **browser edition** is tsfcore compiled to WebAssembly with Emscripten and driven by a static page. A web worker loads the module, runs the pipeline on a file the user selects or on one of nine small example datasets, and hands back the result files. Because nothing but static files is served, the page can be hosted anywhere; the public instance is served by GitHub Pages and rebuilt automatically on every change to the core.

The **desktop edition** is a Python package, `tsfstudio`, with a local web interface and a command-line interface that call the same runner. It loads tsfcore as its C++ engine and adds two further engines behind the same interface: an R engine that runs the original forecast package, and a Python engine with StatsForecast, gradient-boosting and optional neural models. The interface server uses only the Python standard library, validates the host header, requires a per-session token and never accepts a file path from the client. A packaged Windows build starts from one executable and needs no Python installation.

The repository holds 404 automated test cases covering the data format, the measures (against hand-computed values), the engines, the runner and the server. [TODO: restate the count and the pass status at the tagged release.]

### 2.2 Software functionalities

**Data.** A catalogue of 43 files of the Monash archive [9] with default horizons, any `.tsf` file, or an Excel workbook: a template is provided, and the converter reports input errors by cell address. Before any method is run, the platform reports frequency, number and length of series, shares of missing and zero values and the demand-pattern classes of Syntetos et al. [10], and can compute the catch22 [11] and tsfeatures [12] characteristics and draw a principal-component map of the series.

**Methods.** The desktop edition offers 45 methods (Table 2): 42 on the C++ engine, all 45 on the Python engine, and 25 on the R engine, which calls the functions of the R packages forecast, thief, MAPA and smooth directly. Running a method on the C++ engine and on its R or Python counterpart is therefore a matter of changing one setting. The browser edition offers the 37 local and intermittent-demand methods. A method whose engine is not installed is marked and cannot be started.

**Table 2.** Methods by group.

| Group | Count | Examples |
|:--|--:|:--|
| Local (one model per series) | 30 | naive benchmarks, SES, Holt, Holt-Winters, Theta [13], automatic ETS and ARIMA [1], TBATS [14], dynamic harmonic regression, STL-based methods, bagged ETS [15], temporal aggregation (THieF [16], MAPA [17]), CES, MFLES, combinations |
| Intermittent demand | 7 | Croston variants, SBA, TSB, ADIDA, IMAPA, zero forecast |
| Global (one model for all series) | 8 | pooled regression, CatBoost [8], three LightGBM variants [7], N-HiTS [18], N-BEATS [19], Chronos-Bolt [20] |

**Run.** The desktop interface follows five steps: data, model, settings, run, results. Defaults are shown, a changed setting is highlighted, and one click restores it. A pre-run check estimates memory and time, refuses a run that would exhaust memory, shows progress and allows cancellation. A run uses a single train-test split or several rolling origins. Prediction intervals are available for every method through scaled split-conformal prediction [21,22], which needs no model-specific variance formula.

**Results.** A summary table, charts of actual values and forecasts for any series, the distribution of the error over the series, a per-series table, and downloads as CSV and Excel files. Sixteen accuracy measures are computed per series, among them sMAPE, MASE, MAE, RMSE, WAPE and RMSSE [23]; a value that is undefined for a series is left out and counted, not replaced. The platform warns when a method forecasts only zeros and is rewarded for it on intermittent demand. Past runs can be reopened with their settings.

**Correctness of the C++ core.** Table 3 summarises the agreement of tsfcore with the implementations it was ported from. Where it ports an R routine it reproduces R at or near machine precision, and where it ports a StatsForecast model it reproduces that library. Reaching this level required reproducing details that an independent implementation would not: the optimiser's exact iteration, R's long-double summation, the singular-value routine R calls, and correctly rounded trigonometric functions for Fourier terms.

**Table 3.** Agreement of tsfcore with its reference implementation on NN5 Weekly and NN5 Daily (111 series each), from the validation record in the repository. Differences are relative, per forecast value. [TODO: these figures are taken from `cpp/README.md`; re-run the validation scripts at the tagged release.]

| Method | Reference | Agreement |
|:--|:--|:--|
| ETS | R forecast 9.0.2 | all 222 series within $10^{-10}$ |
| ARIMA (automatic) | R forecast 9.0.2 | same model on all 222 series; worst difference $1.5\times10^{-6}$ weekly, $9\times10^{-5}$ daily |
| TBATS | R forecast | all 222 series within $1.3\times10^{-9}$ |
| Bagged ETS | R forecast 9.0.2 | all 222 series within $2\times10^{-12}$ |
| Pooled regression | R | within $5\times10^{-14}$ |
| Croston variants, TSB, ADIDA, IMAPA | StatsForecast 2.1.1 | bit-identical |
| LightGBM, CatBoost | Python packages | bit-identical |
| Theta variants, MFLES | StatsForecast 2.1.1 | within $1.3\times10^{-14}$ |

The WebAssembly build was compared with the native build on 27 runs: 22 produced byte-identical result files, and the rest differed at relative magnitudes between $10^{-15}$ and $10^{-6}$, traced to the mathematical library. [TODO: these browser figures come from development notes; re-run and record them in the repository.]

## 3. Illustrative examples

**A forecast on a phone.** A user opens the public page, selects the example dataset "NN5 weekly" (111 series of weekly cash withdrawals) or a `.tsf` file of their own, looks at the demand-pattern classes and the map of the series, chooses a method, for instance ETS, and starts the run. The page shows the summary of the accuracy measures, a chart of actual values and forecast for each series, and the distribution of the error, and offers the result files as one download. No account, installation or upload is involved (Fig. 2).

[Figure 2 about here. Browser edition on a phone-sized screen: data characteristics, method selection and results of a run.]

**Speed.** Table 4 compares the run time of the same method on the three engines of the desktop edition, on a six-core laptop processor (Intel Core i5-8400, 8 GB of memory). The times are those of the forecasting step of an ordinary run started from the command line. R and Python were given one process and six processes, tsfcore one thread and six threads.

SPEED_TABLE

SPEED_TEXT

[Figure 3 about here. Desktop interface: the Results step of one of these runs, with the summary table, a sample-series chart and the run record.]

## 4. Impact

*Access.* The platform removes the three preconditions of current forecasting software: programming skill, an installed environment, and a computer. A planner can forecast from an Excel sheet; a lecturer can have a class run and inspect forecasts on their own phones; an analyst working under data-protection rules can use it because the data stay on the device. The explanations attached to every method and measure, and the Learn section of the browser edition, are written so that the platform can be used for teaching as well as for producing forecasts.

*Speed.* With run times of seconds instead of minutes, forecasting a collection of series becomes interactive: a user can try several methods, horizons or settings in one sitting, and procedures that multiply the number of model fits (rolling origins, conformal intervals, bagging, temporal aggregation) become practical on a laptop. The same speed is what makes execution inside a browser feasible at all.

*A reusable C++ implementation of the standard methods.* tsfcore provides automatic ETS, ARIMA, TBATS, STL-based and temporal-aggregation forecasting as a self-contained, multi-threaded library with a C interface and no dependency on R or Python. It can be called from any language and embedded where an R or Python runtime is not available: web pages, mobile applications, or production systems. Because it reproduces the reference results, existing studies and workflows built on the R forecast package can move to it without a change in their numbers.

*Reproducibility.* The run record and the published agreement between engines let a result be traced to the exact data, settings and software that produced it.

The software is new, and we make no claim about its user base. [TODO: add download or usage figures if available at submission.]

## 5. Conclusions

Forecast Studio makes established forecasting methods usable by anyone with a browser, and makes them fast by implementing them in C++ without changing their results. Current limits are the following. The browser edition runs the local and intermittent-demand methods with default settings and reads `.tsf` files of a few megabytes; global and neural models, adjustable settings and Excel input require the desktop edition. The packaged desktop build exists for Windows only and is unsigned, and the neural models require a source installation with PyTorch. Forecasts are univariate, without exogenous variables. The desktop edition still runs its pipeline in Python and calls tsfcore for the forecasts only. Planned work is to move the desktop edition fully onto the shared core, to bring Excel input and global models to the browser edition, and to offer the core as a library for other languages.

## CRediT authorship contribution statement

**Michele Cedolin:** Conceptualization, Methodology, Software, Validation, Writing – original draft, Writing – review & editing.

## Declaration of competing interest

The author declares no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Declaration of generative AI and AI-assisted technologies in the writing process

[TODO: Elsevier requires this statement. An AI coding assistant (Claude, Anthropic) was used in developing and testing the software and in drafting this manuscript; state this in the author's own words and confirm that the author reviewed the content and takes full responsibility for it.]

## Data availability

The example datasets are from the Monash Time Series Forecasting Archive (CC BY 4.0) and are distributed with the browser edition. The validation fixtures and the script behind Table 4 (`tools/engine_speed_study.py`) are in the repository.

## References

[TODO: check every entry against the publisher's record before submission; the entries were written from memory of the literature, and the repository's own verified list is in `src/tsfstudio/app/models_info.py` (REFS).]

1. Hyndman RJ, Khandakar Y. Automatic time series forecasting: the forecast package for R. J Stat Softw 2008;27(3):1–22.
2. Garza F, Mergenthaler Canseco M, Challu C, Olivares KG. StatsForecast: lightning fast forecasting with statistical and econometric models. PyCon, Salt Lake City; 2022.
3. Alexandrov A, Benidis K, Bohlke-Schneider M, et al. GluonTS: probabilistic and neural time series modeling in Python. J Mach Learn Res 2020;21(116):1–6.
4. Herzen J, Lässig F, Piazzetta SG, et al. Darts: user-friendly modern machine learning for time series. J Mach Learn Res 2022;23(124):1–6.
5. Löning M, Bagnall A, Ganesh S, Kazakov V, Lines J, Király FJ. sktime: a unified interface for machine learning with time series. arXiv:1909.07872; 2019.
6. Haas A, Rossberg A, Schuff DL, et al. Bringing the web up to speed with WebAssembly. In: Proc 38th ACM SIGPLAN Conf Programming Language Design and Implementation; 2017. p. 185–200.
7. Ke G, Meng Q, Finley T, et al. LightGBM: a highly efficient gradient boosting decision tree. In: Advances in Neural Information Processing Systems 30; 2017.
8. Prokhorenkova L, Gusev G, Vorobev A, Dorogush AV, Gulin A. CatBoost: unbiased boosting with categorical features. In: Advances in Neural Information Processing Systems 31; 2018.
9. Godahewa R, Bergmeir C, Webb GI, Hyndman RJ, Montero-Manso P. Monash time series forecasting archive. In: NeurIPS Datasets and Benchmarks Track; 2021.
10. Syntetos AA, Boylan JE, Croston JD. On the categorization of demand patterns. J Oper Res Soc 2005;56(5):495–503.
11. Lubba CH, Sethi SS, Knaute P, Schultz SR, Fulcher BD, Jones NS. catch22: CAnonical Time-series CHaracteristics. Data Min Knowl Discov 2019;33(6):1821–52.
12. Hyndman R, Kang Y, Montero-Manso P, et al. tsfeatures: time series feature extraction. R package; 2023.
13. Assimakopoulos V, Nikolopoulos K. The theta model: a decomposition approach to forecasting. Int J Forecast 2000;16(4):521–30.
14. De Livera AM, Hyndman RJ, Snyder RD. Forecasting time series with complex seasonal patterns using exponential smoothing. J Am Stat Assoc 2011;106(496):1513–27.
15. Bergmeir C, Hyndman RJ, Benítez JM. Bagging exponential smoothing methods using STL decomposition and Box–Cox transformation. Int J Forecast 2016;32(2):303–12.
16. Athanasopoulos G, Hyndman RJ, Kourentzes N, Petropoulos F. Forecasting with temporal hierarchies. Eur J Oper Res 2017;262(1):60–74.
17. Kourentzes N, Petropoulos F, Trapero JR. Improving forecasting by estimating time series structural components across multiple frequencies. Int J Forecast 2014;30(2):291–302.
18. Challu C, Olivares KG, Oreshkin BN, Garza Ramirez F, Mergenthaler Canseco M, Dubrawski A. N-HiTS: neural hierarchical interpolation for time series forecasting. In: Proc AAAI Conf Artificial Intelligence 2023;37(6):6989–97.
19. Oreshkin BN, Carpov D, Chapados N, Bengio Y. N-BEATS: neural basis expansion analysis for interpretable time series forecasting. In: Int Conf Learning Representations; 2020.
20. Ansari AF, Stella L, Turkmen C, et al. Chronos: learning the language of time series. Trans Mach Learn Res; 2024.
21. Vovk V, Gammerman A, Shafer G. Algorithmic learning in a random world. New York: Springer; 2005.
22. Stankevičiūtė K, Alaa AM, van der Schaar M. Conformal time-series forecasting. In: Advances in Neural Information Processing Systems 34; 2021.
23. Hyndman RJ, Koehler AB. Another look at measures of forecast accuracy. Int J Forecast 2006;22(4):679–88.

---

## Open items before submission

1. **Release.** v0.1.1 predates the browser edition, the C++ run pipeline, rolling origins, intervals and the feature sets. Cut a new release, archive it on Zenodo, and put the version and DOI in Table 1 (C1, C3).
2. **Relation to the ranking manuscript.** This paper is independent and does not cite it. The ranking manuscript (`manuscript/Manuscript_v2.md`, Section 6) cites this paper for the description of the software; update that citation when this paper is submitted. The ranking commands of the package (`rank`, `reliability`, `instability`) are deliberately not described here.
3. **Figures.** Figure 1 (architecture diagram) and Figures 2 and 3 (screenshots of the two editions) are still to be made.
4. **Numbers to re-run at the release:** test count and pass status (404 collected on 2026-11-24, not executed for this draft); Table 3 (from `cpp/README.md`; the bagged ETS row is the weaker of its weekly and daily figures); the browser-versus-native comparison (from development notes).
5. **Table 4** was measured on 2026-11-25 with tsfcore 0.4.0 by `tools/engine_speed_study.py` (one run per cell, not repeated). Repeat it at the release, preferably three times with the median reported.
6. **Repository state reviewers will see:** `README.md` still describes phases D and E as not started and omits the C++ engine and the browser edition; several documents under `docs/` are in Turkish. SoftwareX reviewers assess the repository and its documentation.
7. **Front matter:** affiliation, e-mail, ORCID, highlights (SoftwareX asks for three to five), AI-use statement, reference check.
