# Metrikler: orijinal R tanımları ve bu uygulamadaki farklar

Kaynak: `reference/utils/error_calculator_original.R` (SMAPE, mSMAPE, MASE, MAE, RMSE) ve
`reference/utils/error_calculator.R` (WAPE, nRMSE). Uygulama: `src/tsfstudio/metrics.py`.
Her formül elle hesaplanmış değerlerle `tests/test_metrics.py` içinde sınanır.

## Tanımlar (seri başına; ε = 0.1)

| Metrik | Tanım | Birim |
|---|---|---|
| SMAPE | mean( 2·\|f−a\| / (\|f\|+\|a\|) ) | oran, 0–2 aralığı (×100 **değil**) |
| mSMAPE | mean( 2·\|f−a\| / max(0.5+ε, \|f\|+\|a\|+ε) ) | oran |
| MASE | mean(\|f−a\|) / mean(\|Δ_m y_train\|), m = ⌊min(mevsimsellik)⌋; ölçek tanımsızsa m = 1 | – |
| MAE, RMSE | klasik | veri birimi |
| WAPE | Σ\|f−a\| / Σ\|a\| × 100 | **yüzde** |
| nRMSE_Mean / nRMSE_Range | RMSE / mean(a)  ;  RMSE / (max(a)−min(a)) | oran |

Not: SMAPE/mSMAPE oran (0–2), WAPE ise yüzde olarak orijinal kodda böyledir; karşılaştırma yaparken birimlere dikkat edin.

## Orijinalden bilinçli farklar

1. **Hizalama.** Orijinalde `calculate_mase` NaN/Inf serileri vektörden atıyor, diğer metrikler atmıyordu;
   `data.frame(...)` içinde sütunlar kayıyor (`error_calculator.R`) ya da geri dönüştürülüyordu. Burada tanımsız
   seri değeri **NaN olarak yerinde kalır**; özet (`N_valid`, `N_excluded`) kaç serinin dışarıda kaldığını gösterir.
   Ortalama/medyan yalnızca sonlu değerler üzerindendir — orijinal MASE özetiyle aynı sonuç.
2. **Payda 0.** WAPE/nRMSE'de R `Inf` üretir ve `mean(na.rm=TRUE)` özetini `Inf` yapar; burada `NaN` ve dışarıda kalır.
3. **NaN içeren çiftler.** MASE'te tahmin/gerçek ayrı ayrı süzülürse uzunluklar uyuşmayabilirdi; çiftler birlikte süzülür.
   NaN içermeyen veride sonuç özdeştir.
4. **MASE gecikmesi.** `diff(lag = min(mevsimsellik))` R'da kesirli gecikmeyi kısaltır (365,25/7 → 52); aynı şekilde kesilir.
   Mevsimsel ölçek `NaN` ise (seri gecikmeden kısa, ya da 0/0) lag 1 denenir — orijinaldeki `is.na(mase)` mantığı birebir.
5. **Yuvarlama.** Eski `smooth::MASE` sonucu 3 haneye yuvarlıyordu; güncel greybox yuvarlamaz. Burada yuvarlama yoktur
   (`mase_digits=3` ile isteğe bağlı açılabilir).

## Orijinal kodda bulunan ve burada giderilen sorunlar

| Dosya | Sorun | Çözüm |
|---|---|---|
| `fixed_horizon_functions.R` (yerel model) | Tahminler `.xlsx` yazılıyor, hata hesabı hâlâ `<veri>_<yöntem>.txt` okuyor → döngü bitince hata | R motoru düzeltilmiş sürücüyle (Aşama B) |
| `error_calculator.R` | `calculate_smape/msmape/mase/mae/rmse` tanımsız (yalnız `_original.R` içinde) | Tek Python modülü |
| `error_calculator_full.R` | `calculate_mape` tanımsız; `Theils_U` satır sayısı uyumsuz; naif taban kesirli/vektörel mevsimsellikte bozuk | Genişletilmiş 15'lik küme sonraki aşamada, bu hatalar giderilerek |
| `fixed_horizon_functions.R` | `forecast_horizon <- 1` atama sonraki serileri etkiliyor; sessiz SNaive yedeği ve NA→0 | Yedek kullanımı seri bazında sayılıp raporlanacak |
| `rolling_origin.R` | Kaynak olarak yüklenince örnek koşular çalışıyor; sabit Colab `BASE_DIR` | Fonksiyonlar ayrı, yol parametre |
| `data_loader.py` | `distutils.util.strtobool` Python ≥ 3.12'de yok | `tsfstudio.tsf.reader` yerel `strtobool` |
