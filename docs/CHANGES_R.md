# R kodundaki değişiklikler

Kaynaklar `reference/` altında **değiştirilmeden** durur. `src/tsfstudio/r/` altındaki kopyalar modelleme kararlarını
(model çağrıları, parametreler, normalizasyon, özyineleme) korur; yalnızca aşağıdaki noktalar farklıdır.

| # | Orijinal | Düzeltme | Sonuca etkisi |
|---|---|---|---|
| 1 | `tryCatch(..., error = function(e) SNaive)` sessizdi | Her geri dönüş sayılır ve mesajı kaydedilir (`fallbacks.csv`, manifest) | Tahminler aynı; yalnızca raporlanır. Yedek oranı >%50 ise uyarı |
| 2 | NA tahminler sessizce 0 | Aynı davranış korunur ama sayılır (`na_replaced`) | Aynı |
| 3 | `warning(e)` günlüğü dolduruyordu | Kaldırıldı; mesaj yedek kaydında | Yok |
| 4 | `create_input_matrix`: döngü içi `rbind` (karesel), `print(i)` | Gömülü matrisler listede toplanıp tek `rbind` ile birleştirilir | Özdeş matris, çok daha hızlı |
| 5 | Çok kısa seride anlaşılmaz hata | Hangi serinin sorunlu olduğunu söyleyen hata | Yok |
| 6 | Büyük veride (M5) bellek patlaması | İsteğe bağlı `max_obs` (varsayılan kapalı = orijinal): her serinin son `max_obs` gözlemi; ortalama normalizasyonu bu pencere üzerinden | Varsayılanda yok |
| 7 | CatBoost `catboost.train` varsayılanlarıyla çalışıyordu | `params.catboost` ile geçersiz kılınabilir (boş = aynı varsayılanlar) | Varsayılanda yok |
| 8 | Dosya başında `set.seed(1)` | Tohum `job.seed` ile her koşuda açıkça ayarlanır (varsayılan 1) | Varsayılanda yok |
| 9 | `forecast_horizon <- 1` üzerine yazma, `rolling_origin.R` kaynaklanınca örnek koşma, `os.system("rm")`, sabit `BASE_DIR`, `distutils.strtobool` | Taşınmadı; yerine işçi protokolü ve Python tarafı | Yok |

## DHR-ARIMA ile ilgili açık nokta

Orijinal çağrı `fourier(ts, K = 1)` skaler K kullanır. Çok mevsimli `msts` (ör. saatlik: 24, 168, 8766) için `forecast`
paketinde K, mevsim sayısı kadar uzunlukta olmalıdır; aksi hâlde hata → **sessizce SNaive**. İşçi orijinal davranışı
korur ve `params.fourier_k` / `params.fourier_k_per_period` ile değiştirmeye izin verir. Kendi paket sürümlerinizde bunun
olup olmadığını `fallbacks.csv` gösterir; olursa çok mevsimli verideki DHR-ARIMA sonuçlarının bir kısmı aslında SNaive'dir.

## Python tarafı ile fark

`PythonBaselineEngine.snaive` ilk mevsimsel periyodu kullanır; R'daki `snaive(msts)` ise `floor(max(periyot))` frekansını.
Çok mevsimli veride sonuçlar farklıdır; karşılaştırmada R motoru referanstır.

## Python motorundaki karşılıklar
Ayrıntı: `docs/PYTHON_ENGINE.md`. Özet: SES, Theta ve havuzlu regresyon R algoritmasıyla birebir uygulanmıştır (NN5 altın testleri);
ETS/ARIMA/TBATS/DHR-ARIMA statsforecast ile benzer-ama-aynı-değil karşılıklardır. Python motorunda NaN/Inf çıktı 0 yapılmaz,
yedek tahmine düşülür; DHR-ARIMA'da skaler K çok mevsimli veride her periyoda uygulanır.

### TBATS dönemleri (Python motoru)
R `tbats()` gerçek değerli dönemleri (ör. 52,18; 365,25) kabul eder. statsforecast `AutoTBATS(season_length: int | list[int])` belgeye göre
tam sayı ister; adaptör dönemleri en yakın tam sayıya yuvarlar (52,18 → 52; 365,25 → 365). Bu bilinçli bir farktır; etkisi `golden` ile ölçülür.
