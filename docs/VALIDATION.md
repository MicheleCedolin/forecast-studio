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

Sapmanın nedeni: sapan **tüm** SES serilerinde Python'un en iyi α'sı alt sınırda (1×10⁻⁴), R'ınki ise daha yüksek SSE'de (haftalık
6 seride %0,2–41 daha yüksek) durmuş; yani R'ın Nelder-Mead'i sınır optimumunu bulamıyor. Bu, `test_ses_deviations_occur_only_at_alpha_lower_bound`
ile sınanır. Süre: 111 seri için her biri < 0,5 sn (R kaydınızda 7–11 sn).

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
