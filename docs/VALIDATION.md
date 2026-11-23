# Doğrulama

## 1. Metrikler — kullanıcının gerçek R çıktılarına karşı (yapıldı)
NN5 haftalık (111 seri, eğitim 105, ufuk 8) ve NN5 günlük (111 seri, eğitim 735, ufuk 56) için kullanıcının R hata
dosyaları (`smape, msmape, mase, mae, rmse`; seri başına) ile `tsfstudio.metrics`:

* 11 model × 5 metrik × 111 seri: en büyük bağıl fark **≈5×10⁻¹⁵** (makine duyarlılığı).
* Genişletilmiş özet dosyasındaki `MAE, RMSE, WAPE, NRMSE_mean, NRMSE_range` ve `sMAPE` (×100) 23 model/veri çiftinde **≈5×10⁻¹⁶**.
* MASE gecikmesi: haftalık 52 (365,25/7'den), günlük 7.

Otomatik test: `tests/test_golden_nn5.py` (`tests/golden/nn5_weekly/`, 412 KB). Yeniden üretim: `tools/make_nn5_fixture.py`.

## 1b. Python motoru — kullanıcının R tahminlerine karşı (yapıldı, bu ortamda)
`python -m tsfstudio golden` / `tests/test_golden_python.py` (NN5 haftalık h=8 ve günlük h=56, 111 seri):

| Veri | Yöntem | ortanca bağıl fark | >1e-3 fark eden seri | ΔSMAPE (göreli) | ΔMASE (göreli) |
|---|---|---|---|---|---|
| haftalık | pooled_regression (lag 65) | 1×10⁻¹⁵ | 0/111 | 0 | 0 |
| haftalık | ses | 4×10⁻⁶ | 6/111 | +%0,27 | +%0,36 |
| haftalık | theta | 4×10⁻⁶ | 5/111 | +%0,06 | +%0,08 |
| günlük | pooled_regression (lag 9) | 0 | 0/111 | 0 | 0 |
| günlük | ses | 1,5×10⁻⁵ | 6/111 | +%0,18 | +%0,18 |
| günlük | theta | 8×10⁻⁶ | 4/111 | −%0,32 | −%0,39 |

Sapmanın nedeni: sapan **tüm** SES serilerinde Python'un en iyi α'sı alt sınırda (1×10⁻⁴), R'ınki ise daha yüksek SSE'de durmuş
(R'ın kendi koduyla yeniden ölçüm, 1e: haftalık 6 seride %0,07–1,5, günlük 6 seride %0,001–0,7 daha yüksek; bu belgenin önceki
sürümündeki "%0,2–41" değeri yeniden üretilemedi). Yani R'ın Nelder-Mead'i bu serilerde optimuma yakınsamıyor. Bu,
`test_ses_deviations_occur_only_at_alpha_lower_bound` ve `tests/test_r_reference_optimum.py` ile sınanır. Süre: 111 seri için her biri < 0,5 sn (R kaydınızda 7–11 sn).

statsforecast tabanlı yöntemler geliştirme ortamında **çalıştırılamadı**; adaptör mantığı (R zinciri, frekans kuralları,
Fourier regresörleri, yedek sayımı) sahte bir statsforecast modülüyle sınanır. Gerçek doğrulama kullanıcının makinesinde `golden` ile yapılır.

### 1c. statsforecast — kullanıcının Windows makinesinde, NN5 haftalık (h=8, 111 seri, Python 3.12, tek çekirdek)
Kaynak: kullanıcının `golden` çıktısı (2026-11-03). Tüm satırlarda yedek (SNaive) sayısı 0, **TBATS hariç**.

| Yöntem | ortanca bağıl fark | >1e-3 fark eden seri | ΔSMAPE (göreli) | ΔMASE (göreli) | süre (Python / R kaydı) |
|---|---|---|---|---|---|
| ses | 4,4×10⁻⁶ | 6/111 | +%0,27 | +%0,36 | 0,25 sn / 8 sn |
| theta | 4,0×10⁻⁶ | 5/111 | +%0,06 | +%0,08 | 0,29 sn / 9 sn |
| pooled_regression | 1,4×10⁻¹⁵ | 0/111 | ≈0 | ≈0 | 0,03 sn |
| ets (`AutoETS`) | 4,7×10⁻⁵ | 21/111 | −%1,55 | −%1,38 | 0,9 sn / 11 sn |
| arima (`AutoARIMA`) | 2,4×10⁻³ | 70/111 | −%0,07 | −%0,33 | 25 sn / 41 sn |
| dhr_arima | 4,2×10⁻³ | 96/111 | −%0,11 | −%0,40 | 50 sn / 42 sn |
| tbats | **GEÇERSİZ** | – | – | – | 111/111 seri yedeğe düştü |

Okuma: (i) SES/Theta/havuzlu regresyon sonuçları Windows + Python 3.12'de, geliştirme ortamındaki sayılarla **aynıdır** (platformdan bağımsız).
(ii) ETS/ARIMA/DHR-ARIMA tahminleri R'dan *seri düzeyinde* farklıdır (model seçimi/optimizasyon farkı: `statsforecast` ARIMA'sı çok sayıda
"possible convergence problem: minimize gave code 2" uyarısı verdi) ama *toplu doğruluk ölçüleri* %1,6'nın altında oynar ve işaret hep Python
lehinedir. Bu, "birebir yeniden üretim" değil "doğruluk açısından eşdeğer" demektir; makalede motor adı ve sürümü belirtilmelidir.
Seri düzeyinde eşdeğerlik için eşleştirilmiş bir test (ör. seri başına SMAPE farkında Wilcoxon) önerilir; henüz yapılmadı.
(iii) TBATS satırı bir **hataydı**: `AutoTBATS` imzası `season_length` ister, adaptör `seasonal_periods` gönderdi, anahtar süzüldü ve her seri
yedeğe düştü (SMAPE 0,164 = SNaive). Düzeltildi (`season_length`, tam sayı dönemler); golden artık yedek sayısı > 0 ise nedenini yazar.
Yeniden doğrulama gerekir. Günlük küme sonuçları (`golden_result.csv`) henüz incelenmedi.

### 1d. Naive2 and Comb S-H-D against R (done 2026-11-22, Windows, R 4.6.2, forecast 9.0.2)
Reference: `tools/r_reference/m4_benchmarks.R`, which follows the official M4 benchmark code (SeasonalityTest, `decompose`,
`naive`, `ses`, `holt`, `holt(damped = TRUE)`). Comparison: `tools/r_reference/compare_m4_benchmarks.py`. Golden test:
`tests/test_golden_m4_benchmarks.py` (fixtures `r_m4_naive2`, `r_m4_comb` in `tests/golden/nn5_*`).

| Data | Method | median rel. diff. | >1e-3 series | ΔSMAPE (rel.) | ΔMASE (rel.) |
|---|---|---|---|---|---|
| weekly (m 52, h 8; no series seasonal) | naive2 | 0 | 0/111 | 0 | 0 |
| weekly | comb_shd | 1.3×10⁻³ | 70/111 | −0.03% | −0.05% |
| daily (m 7, h 56; all 111 seasonal) | naive2 | 8×10⁻¹⁶ | 0/111 | 0 | 0 |
| daily | comb_shd | 5.5×10⁻⁴ | 55/111 | −0.31% | −0.35% |

- The seasonality test agrees with R on 222/222 series.
- **A first version of the Holt fit was wrong.** It optimised the initial level and trend numerically next to the smoothing
  parameters, with a hard penalty for beta > alpha. It stopped early (on one series exactly at its starting point). Daily Holt
  was 15% worse in SMAPE than R and Comb 5% worse. It was rewritten: for fixed (alpha, beta, phi) the initial state is solved
  in closed form, and beta is searched as alpha·u with u in [0, 1].
- After the fix, Python's in-sample MSE is lower than or equal to R's on every one of the 444 fits (111 series × 2 datasets ×
  Holt and damped Holt). It is up to 7.8% lower, because R's Nelder-Mead stops before the optimum. This is the same
  situation as SES in 1b. The remaining forecast differences are therefore R optimiser artefacts, not a different model.

### 1e. Direct checks against R and CRAN packages (done 2026-11-23, Linux, R 4.3.3, forecast 8.21.1)
Rule: a method that also exists in R is compared with R; a method without an R counterpart is not. R 4.3.3 / forecast 8.21.1
reproduce the study's stored R forecasts for `ses` and `theta` and the forecast 9.0.2 `r_m4_comb` fixture to ≤ 1e-14, so the
older version stands in for the study's. Data: NN5 weekly (h 8) and daily (h 56), 111 series each; the intermittent methods
on 200 synthetic series (`tools/r_reference/make_intermittent_series.py`, seed 0, h 12), because NN5 is not intermittent.
Scripts: `tools/r_reference/{basic,cran,ces,ses_optimum,holt_theta_optimum}_reference.R`. Tests: `tests/test_r_reference_optimum.py`.

**Is the SES / Theta / Comb deviation from R an optimiser artefact (1b, 1d)? Yes.**

| Check | weekly | daily |
|---|---|---|
| SES series that differ from R by > 1e-3 | 6 | 6 |
| ... of which the SSE optimum is at alpha = 1e-4 | 6 | 6 |
| ... R's SSE above the optimum | 0.07–1.5% | 0.001–0.7% |
| ... R `ets()` with alpha fixed at the optimum: SSE above the optimum | 1e-4% | ≤ 1.5e-4% |
| C++ SES forecast vs the exact optimum, worst | 7.5e-9 | 1.9e-8 |
| Theta series that differ from R by > 1e-3 (all with the optimum at the bound) | 5 | 4 |
| C++ Theta forecast vs the exact optimum, worst | 5.6e-9 | 6.5e-5 (1 series > 1e-6) |
| Holt / damped Holt fits where R's SSE exceeds the multi-start optimum by > 0.1% | 71 / 83 | 26 / 19 |
| ... largest excess (R is never below the optimum) | 8.5% / 7.6% | 1.9% / 0.6% |
| C++ Comb S-H-D vs the multi-start optimum: median, worst, series > 1e-3 | 7e-9, 4.5e-3, 1 | 7e-8, 3.1e-3, 1 |

Two qualifications. (i) "R cannot find the boundary optimum" is too strong: on 2 of the 12 deviating SES series R's alpha is at
the bound as well and it is the initial level that has not converged, and on 36 other series with the optimum at the bound R
converges. (ii) For Comb, one series per dataset differs from the multi-start reference by 0.3–0.5%; which of the two points
is the better optimum was not determined.

**Methods with an R counterpart**

| Method | R call | Result (relative difference) |
|---|---|---|
| snaive, naive | `forecast::snaive`, `forecast::naive` | 0 |
| drift | `forecast::rwf(drift = TRUE)` | ≤ 1.3e-14 |
| croston_classic | `forecast::croston(alpha = 0.1)`; `tsintermittent::crost(w = c(0.1, 0.1), init = "naive", init.opt = FALSE)` | ≤ 5e-15 |
| croston_sba | `tsintermittent::crost(type = "sba", ...)` (same settings) | ≤ 5e-15 |
| tsb | `tsintermittent::tsb(w = c(0.1, 0.1), init = "naive", init.opt = FALSE)` | ≤ 4e-15 |
| croston_optimized | `tsintermittent::crost(init = "naive", cost = "mse", init.opt = FALSE)` | not equivalent: median 8% |
| imapa | `tsintermittent::imapa()` | not equivalent: median 16% |
| dotm | `forecTheta::dotm()` 3.0.3 | not equivalent: median 1.5e-3 weekly, 2.3e-2 daily; SMAPE 12.01 vs 11.97, 21.86 vs 22.23 |
| auto_ces | `smooth::auto.ces()` 4.5.2 | not equivalent: median 1.9e-4 weekly, 1.9e-2 daily; SMAPE 12.36 vs 12.45, 20.81 vs 21.87 |

The four "not equivalent" methods are ports of statsforecast; the R packages are separate implementations and the source of
each difference was not analysed. The equalities hold for the fixed-parameter calls shown, not for the R functions' defaults
(which optimise the parameters). The C++ engine was used for all rows. Not compared: adida (no function in tsintermittent),
lightgbm (CRAN has 4.7.0, the app uses 4.6.0), catboost (not on CRAN), auto_theta (no single R function).

**Platform note (thief, mapa).** With the library built on Linux (GCC), 66 of the 67 `r_temporal` series equal the Windows R
fixtures to ≤ 1e-12; one quarterly series (18 observations) differs by 1.4e-6 (thief) and 1.2e-6 (mapa). R itself shows the
same: thief 0.3 / MAPA 2.1.0 run on Linux (R 4.3.3, forecast 8.21.1) differ from the Windows fixture (R 4.6.2, forecast 9.0.2)
by 1.4e-6 / 1.2e-6 on that series, and the Linux C++ forecasts equal the Linux R forecasts there to 1e-13. The difference
therefore comes from R's own platform or version dependence, not from the port; the two were not separated. Equality with R
to rounding is a same-platform statement.

## 2. R motoru — kullanıcının R tahminlerine karşı (yapıldı: Windows, yerel R, haftalık küme)
Ortam: R 4.6.2 (ucrt, kullanıcı klasörüne kurulu), `forecast` 9.0.2, `jsonlite` 2.0.0; `python -m tsfstudio golden --engine r`
(NN5 haftalık, h=8, 111 seri, 0 yedek). Referans: kullanıcının kendi R çıktıları (muhtemelen daha eski bir `forecast` sürümü).

| Yöntem | ortanca bağıl fark | maks. bağıl fark | >1e-3 fark eden seri | ΔSMAPE | süre |
|---|---|---|---|---|---|
| ses | 1×10⁻¹⁵ | 5×10⁻¹⁵ | 0/111 | ≈0 | 0,3 sn |
| ets | 1×10⁻¹⁵ | 4×10⁻¹⁵ | 0/111 | ≈0 | 1,8 sn |
| theta | 1×10⁻¹⁵ | 5×10⁻¹⁵ | 0/111 | ≈0 | 0,5 sn |
| arima | 3×10⁻¹⁵ | 1,6×10⁻⁶ | 0/111 | −5×10⁻⁶ % | 14 sn |
| dhr_arima | 4×10⁻¹⁴ | 5×10⁻⁸ | 0/111 | −6×10⁻⁸ % | 10 sn |
| pooled_regression (lag 65) | 3×10⁻¹⁵ | 8×10⁻¹⁵ | 0/111 | ≈0 | 0,2 sn |

Okuma: orijinal R yolu, paket sürümü farkına rağmen kullanıcının R sonuçlarını makine duyarlılığında (arima/dhr_arima'da 10⁻⁶–10⁻⁸,
yani en iyileyici toleransı düzeyinde) yeniden üretiyor. Bu, "birebir yeniden üretilebilirlik" iddiasını destekleyen asıl kanıttır.
Günlük küme (NN5 günlük, h=56, 111 seri, 0 yedek): ses/ets/theta maks. bağıl fark ≤ 5×10⁻¹⁵, pooled_regression (lag 9) 9×10⁻¹⁴, ΔSMAPE/ΔMASE ≈ 0
(ets 55 sn; aynı komut iki kez çalıştırıldı, çıktılar basamağına kadar aynı: belirlenimci). tbats (haftalık/günlük) ve günlük arima R-motoru koşuları henüz yapılmadı. R'da CatBoost paketi kurulu değil (isteğe bağlı).
`tests/test_r_engine_live.py` aynı karşılaştırmayı otomatik yapar. Toleranslar: ses/ets/theta/havuzlu regresyon `1e-6`, arima `1e-4`,
tbats ve dhr_arima `1e-3`.

Windows notu: `parallel::mclapply` Windows'ta çok çekirdeği desteklemez; R işçisi Windows'ta tek çekirdeğe indirilir (PSOCK ile çok çekirdek
henüz yok).

## 2b. C3 modelleri (aralıklı talep + ek klasik) — doğrulama durumu
R karşılıkları yoktur, bu yüzden altın karşılaştırma yapılamaz. Şu ana kadar yalnızca: (i) belgeye göre imzalı sahte statsforecast ile birim
testleri (anahtar geçişi, mevsimsellik kuralları, hata/yedek davranışı, not toplama); (ii) imzaları: CrostonClassic/Optimized, ADIDA, AutoTheta,
AutoCES, HoltWinters, AutoMFLES resmi belgeden; CrostonSBA, TSB, IMAPA, MSTL başka kaynaklardan (belge sayfası kesik). **Gerçek kütüphanede
henüz çalıştırılmadı**; ilk doğrulama kullanıcının makinesinde `doctor` olacaktır.

## 3. Burada (R'sız ortamda) doğrulananlar
İş protokolü (sütun-ana okuma, yedek/NA sayaçları, ilerleme, hata ve çökme mesajı, iptal) bir `Rscript` taklidiyle; koşucu, eksik
politikaları, bölme, manifest, bellek ve süre tahmincisi referans motorla sınanır. R'ın kendisi, PyTorch ve Docker bu ortamda
çalıştırılamaz (ağ ve yetki sınırı) — bu yüzden bölüm 2 ve `doctor` sizin makinenizde zorunlu adımlardır.

## 4. Henüz bilinmeyen / açık
* `rocker/r-ver:4.5.1` etiketi Docker Hub'da var (amd64 + arm64); CRAN anlık görüntüsünün tarihi ve `forecast` sürümü imaj oluşunca `doctor` çıktısında görünür.
* CatBoost R paketi `catboost-R-linux-x86_64-1.2.10.tgz` GitHub sürümünde var (doğrulandı); kurulum başarısı build günlüğünde.
* Süre öncülleri tek makinenin tek koşusudur (kaba kılavuz).
* Kullanıcının dosyalarındaki ek metrikler (`MSE, MAPE, RMSLE, MedAE, MdAPE, NRMSE_IQR, MASE-özet`) henüz modülde yok (bkz. sonraki adımlar).
