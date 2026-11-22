# Alternative forecasting models and error metrics — research report

Prepared by the `forecasting-researcher` agent, 2026-11-22. Prefer primary sources when citing in the manuscript; items
marked *not verified* must be checked before use.

## Confidence
- statsforecast signatures: taken from the current Nixtla models reference. The local venv has **statsforecast 2.1.1** —
  check with `inspect.signature` (the engine's `make_model` already filters unknown keywords and reports them).
- Foundation-model ranks: fev-bench mirror updated 2026-08-26. GIFT-Eval leaderboard could not be rendered.
- Moirai 2.0 licence: *not verified* (Moirai 1.x was CC-BY-NC-4.0).
- Metric formulas: standard definitions from the primary papers (papers not re-fetched).

## 1. Models

### 1a. Available now in statsforecast 2.x (`statsforecast.models`)

| Class (current docs) | Suits | R equivalent |
|---|---|---|
| `AutoCES(season_length=1, model='Z')` | general smooth series; part of SCUM | `smooth::auto.ces` |
| `AutoTheta(season_length=1, decomposition_type='multiplicative', model=None)` | chooses STM/OTM/DSTM/DOTM | `forecTheta` |
| `DynamicOptimizedTheta(season_length=1, decomposition_type='multiplicative')` | yearly/quarterly/monthly (M3/M4 benchmark) | `forecTheta::dotm` |
| `MSTL(season_length, trend_forecaster=AutoETS(model='ZZN'))`, `season_length` may be a list | multiple seasonality; rival to TBATS / DHR | `forecast::mstl` + `stlf` |
| `RandomWalkWithDrift()` | trended series benchmark | `forecast::rwf(drift=TRUE)` |
| `HistoricAverage()`, `WindowAverage(window_size)`, `SeasonalWindowAverage(season_length, window_size)` | simple benchmarks; NN5-type daily data | `meanf` |
| `CrostonClassic()`, `CrostonOptimized()`, `CrostonSBA()`, `ADIDA()`, `IMAPA()`, `TSB(alpha_d, alpha_p)` | intermittent demand (M5, car parts) | `tsintermittent` |
| `Holt(season_length=1, error_type='A')`, `HoltWinters(...)` | fixed ETS forms, Comb S-H-D components | `holt`, `hw` |
| `MFLES(...)`, `AutoMFLES(test_size, ...)` | boosted decomposition (needs scikit-learn) | – |
| `ZeroModel()` | M5 sanity baseline | – |

Docs mismatch: `AutoTBATS` signature says `season_length`, the parameter table says `seasonal_periods` (the engine uses
`season_length`, confirmed on the author's machine).

### 1b. Small numpy implementations
Notation: y₁…y_T training, h = 1…H, season m.
1. **Drift**: ŷ_{T+h} = y_T + h·(y_T − y₁)/(T − 1) (= `rwf(drift=TRUE)`, = statsforecast RWD).
2. **Naive2 (M4)**: seasonality test |r_m| > 1.645·√((1 + 2Σ_{k=1}^{m−1} r_k²)/T) (T ≥ 3m, m > 1); if seasonal, classical
   multiplicative decomposition, naive on y/S, reseasonalise. Needed for OWA.
3. **Croston / SBA / TSB**: SES (α = 0.1) on non-zero sizes z and intervals p; Croston ŷ = ẑ/p̂, SBA ŷ = (1 − α/2)ẑ/p̂
   (Syntetos & Boylan 2005); TSB smooths occurrence probability each period (Teunter, Syntetos & Babai 2011).
4. **Comb S-H-D (M3/M4)**: deseasonalise as Naive2, mean of SES, Holt and damped Holt, reseasonalise.
5. **SCUM** (Petropoulos & Svetunkov 2020): point-wise median of ETS, CES, ARIMA, DOTM.

### 1c. Heavier, optional
- **Global ML**: `mlforecast` + `lightgbm` (MIT, CPU, Windows wheels) — the M5-style global GBM; rival to global CatBoost.
- **Deep**: `neuralforecast` (Apache-2.0): NBEATS, NHITS, TiDE feasible on CPU for M4/NN5 sizes; TFT, PatchTST slow on
  CPU. GluonTS is in maintenance mode — keep only to reproduce Monash numbers.
- **Foundation models** (fev-bench, MASE skill vs Seasonal Naive, 2026-08-26):

| Rank | Model | Package | Licence | CPU |
|---|---|---|---|---|
| 1 | Chronos-2 (120M; small 28M) | `chronos-forecasting` | Apache-2.0 | feasible (small); Chronos-Bolt very fast |
| 2 | TiRex-2 | `tirex-2` | *not verified* (gated HF page) | `device="cpu"` |
| 3–5 | Toto-2.0 | `toto-2` | Apache-2.0 | 22M / 313M feasible |
| 9 | TimesFM-2.5 (200M) | `timesfm[torch]` | Apache-2.0 weights | *not verified* |

TimesFM 3.0 self-hosted weights are non-commercial; TimeGPT is a proprietary API (data leaves the machine — exclude from
a reproducible paper). **Leakage**: most foundation models were pre-trained on corpora that include Monash / M4 /
electricity / traffic; zero-shot results on these datasets may be contaminated. Report overlap per model.

## 2. Metrics
e_t = y_t − ŷ_t over the horizon; in-sample scales use n training points and season m.

| Metric | Formula | Scale-free | Zeros | Source |
|---|---|---|---|---|
| MSE | mean(e²) | no | ok | – |
| MAPE | 100·mean\|e/y\| | yes | undefined at y=0 | Hyndman & Koehler 2006 |
| MdAPE | median 100\|e/y\| | yes | undefined at y=0 | Armstrong & Collopy 1992 |
| sMdAPE | median 200\|e\|/(\|y\|+\|ŷ\|) | yes | 0/0 | M3 |
| MAAPE | mean arctan\|e/y\| ∈ [0, π/2] | yes | finite (π/2) | Kim & Kim 2016 |
| RMSLE | √mean(ln(1+ŷ) − ln(1+y))² | ~relative | needs ≥ 0 | – |
| MedAE | median\|e\| | no | ok | – |
| RMSSE | √(mean e² / ((1/(n−1))Σ(y_t − y_{t−1})²)) | yes | ok | M5 (Makridakis et al. 2022) |
| WRMSSE | Σ w_i RMSSE_i over 12 aggregation levels, dollar-sales weights | yes | built for M5 | needs M5 price/hierarchy files |
| OWA | ½(sMAPE/sMAPE_Naive2 + MASE/MASE_Naive2), dataset totals | yes | – | M4 (Makridakis et al. 2020) |
| AvgRelMAE | exp(Σ n_i ln r_i / Σ n_i), r_i = MAE_i/MAE_i^bench | yes | bench MAE 0 → drop | Davydenko & Fildes 2013 |
| Pinball / WQL / CRPS | quantile loss; WQL = 2ΣL_u/Σ\|y\| | WQL yes | ok | Gneiting & Raftery 2007 |
| MSIS, coverage | scaled interval score, α = 0.05 | yes | ok | M4 |
| ME, PBias, scaled ME | mean(ŷ−y); 100Σ(ŷ−y)/Σy; ME / in-sample naive MAE | PBias, sME | PBias unstable if Σy≈0 | – |

**Ranking-reliability tools**: Friedman (`scipy.stats.friedmanchisquare`), Nemenyi CD = q_α√(k(k+1)/(6N)) (Demšar 2006),
MCB (Koning et al. 2005), paired Wilcoxon + Holm, Diebold–Mariano, Model Confidence Set (Hansen, Lunde & Nason 2011),
rank stability via Kendall τ / Spearman ρ across bootstrap resamples, origins, metrics, and Python-vs-R engines.

## 3. Recommended order
**Models**: (1) AutoCES + DynamicOptimizedTheta → SCUM; (2) Naive2 + Comb S-H-D; (3) drift; (4) CrostonSBA, TSB, ADIDA,
IMAPA (+ zero baseline); (5) MSTL; (6) LightGBM via mlforecast, then Chronos-2-small / Chronos-Bolt; next N-HiTS.

**Metrics**: (1) RMSSE; (2) OWA (needs Naive2); (3) AvgRelMAE; (4) pinball/WQL + MSIS + coverage; (5) PBias / scaled ME;
(6) Friedman + Nemenyi/MCB + bootstrap Kendall τ. Postpone WRMSSE until M5 price/hierarchy files are loaded.

## Sources
- statsforecast models: https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html
- fev-bench: https://arxiv.org/abs/2509.26468 ; https://huggingface.co/spaces/autogluon/fev-bench
- GIFT-Eval: https://huggingface.co/spaces/Salesforce/GIFT-Eval
- Chronos: https://github.com/amazon-science/chronos-forecasting ; https://arxiv.org/abs/2510.15821
- TimesFM: https://github.com/google-research/timesfm
- TiRex-2: https://github.com/NX-AI/tirex-2 ; https://arxiv.org/abs/2607.01204
- Toto 2.0: https://pypi.org/project/toto-2/
- Moirai 2.0: https://arxiv.org/abs/2511.11698
- Papers (bibliographic details *not re-verified*): Hyndman & Koehler 2006 (IJF 22(4)); Makridakis et al. M4 (IJF 36(1),
  2020), M5 (IJF 38(4), 2022); Davydenko & Fildes 2013 (IJF 29(3)); Gneiting & Raftery 2007 (JASA 102); Kim & Kim 2016
  (IJF 32(3)); Koning et al. 2005 (IJF 21(3)); Demšar 2006 (JMLR 7); Petropoulos & Svetunkov 2020 (IJF 36(1)); Syntetos &
  Boylan 2005 (IJF 21(2)); Teunter, Syntetos & Babai 2011 (EJOR 214); Hansen, Lunde & Nason 2011 (Econometrica 79).
