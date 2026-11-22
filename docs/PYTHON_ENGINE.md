# Python motoru (`--engine python`)

R veya Docker gerektirmez. Yerel Windows/Linux/macOS'ta `pip` ile çalışır.

| Yöntem | Tür | Gerekli | R karşılığı | R ile ilişki | Doğrulama |
|---|---|---|---|---|---|
| `naive`, `snaive` | yerel | – | `snaive(msts)` | aynı (frekans = `floor(max(periyot))`) | birim test |
| `ses` | yerel | – | `forecast::ses` | **aynı algoritma** (ML, α ∈ [1e-4, 0.9999], başlangıç düzeyi kapalı formda) | NN5 altın: ortanca fark ~4×10⁻⁶ |
| `theta` | yerel | – | `forecast::thetaf` | **aynı algoritma** (ACF mevsim sınaması, çarpımsal ayrıştırma, SES + yarım eğim) | NN5 altın: ortanca fark ~4–8×10⁻⁶ |
| `pooled_regression` | global | – | `glm` | **aynı sonuç** (akış hâlinde QR; M5 L12 için bile bellek sorunu yok) | NN5 altın: 5×10⁻¹⁵ |
| `ets` | yerel | statsforecast | `forecast::ets` | benzer, **aynı değil** (AutoETS); frekans > 24 → mevsimsiz (R gibi) | kullanıcı makinesinde `golden` |
| `arima` | yerel | statsforecast | `auto.arima` | benzer (AutoARIMA); R zinciri: log → ham → mevsimsiz; `allowdrift`, `approximation` R gibi | `golden` |
| `dhr_arima` | yerel | statsforecast | `auto.arima(xreg=fourier)` | benzer; Fourier K=1 skaler, çok mevsimli veride her periyoda uygulanır (R hata verebilir) | `golden` |
| ~~`tbats`~~ | – | **devre dışı** (kullanıcı kararı, 2026-11-03) | `forecast::tbats` (R motorunda çalışır) | statsforecast `AutoTBATS` R'dan farklı model buldu (8 seri: ortanca tahmin farkı %3,1) | adaptör `py_sf.forecast_tbats` durur |
| `catboost` | global | catboost | `catboost.train` | aynı çekirdek ve varsayılanlar; `random_seed` varsayılanı 0 (R gibi) | `golden` yok (R'da NN5 CatBoost çıktınız yok) |

"Kurulu değil" yöntemler listede görünmez ve çağrılırsa açık bir hata verir.

## Python'a özgü modeller (R karşılığı yok) — statsforecast
Amaç R'ı Python'a çevirmek değil (kullanıcı kararı, 2026-11-03); bunlar statsforecast'ın kendi modelleridir ve `golden` ile sınanamaz.
Doğrulama: `doctor` (çalışıyor mu, sonlu mu, yedek var mı) ve kendi verinizde sonuçların incelenmesi. **Varsayılan değerlerden
"bizim seçimimiz" olarak işaretlenenler kanıtlı öneri değildir; verinizde ayarlayın.** Parametreler `--param anahtar=değer` ile verilir.

| Yöntem | Sınıf | Parametreler (varsayılan) | Not |
|---|---|---|---|
| `croston_classic` | `CrostonClassic` | – | Düz (sabit) tahmin: talep boyutu / talep aralığı |
| `croston_sba` | `CrostonSBA` | – | Croston'a Syntetos–Boylan sapma düzeltmesi |
| `croston_optimized` | `CrostonOptimized` | – | Düzleştirme parametresi kütüphanede optimize edilir |
| `tsb` | `TSB` | `alpha_d`=0,1; `alpha_p`=0,1 (bizim seçimimiz) | Talep olasılığını her dönem günceller (eskime riskine duyarlı) |
| `adida` | `ADIDA` | – | Zamansal toplulaştırma + ayrıştırma |
| `imapa` | `IMAPA` | – | Çoklu toplulaştırma (aralıklı MAPA) |
| `auto_theta` | `AutoTheta` | `decomposition_type`="multiplicative" | Theta varyantlarından otomatik seçim. Sıfır/negatif değer varsa toplamsala geçer (not düşer). Dönem için ≥ 2 dönem veri yoksa mevsimsiz |
| `auto_ces` | `AutoCES` | `model`="Z", `max_season`=24 (bizim seçimimiz) | Karmaşık üstel düzleştirme. Dönem > `max_season` ise mevsimsiz (R `ets` kuralına benzer; değiştirilebilir) |
| `holt_winters` | `HoltWinters` | `error_type`="A", `max_season`=24 (bizim seçimimiz) | Mevsimsizse Holt'a iner (not düşer). `error_type="M"` için tüm değerler > 0 |
| `mstl` | `MSTL` | – | Çok mevsimli ayrıştırma; dönemler tam sayıya yuvarlanır, ≥ 2 ve en az iki dönem veri olanlar kullanılır; hiçbiri yoksa seri yedeğe düşer (yıllık veride bu beklenir). Trend tahmincisi kütüphane varsayılanı (AutoETS, mevsimsiz) |
| `auto_mfles` | `AutoMFLES` | `n_windows`=2, `metric`="smape" | **scikit-learn gerekir** (statsforecast bildirmez; kurulu değilse yöntem listede görünmez). `test_size` = ufuk (kütüphane zorunlu kılar); hakemli referansı yok/doğrulanmadı |

* **Sessiz atma yok:** bu modellerde sürümünüzün tanımadığı bir anahtar hata verir (seri yedeğe düşer, nedeni yazılır).
* **Sıfırlar:** hep sıfır olan seride aralıklı modeller doğrudan 0 tahmin eder (kütüphane 0/0 üretebilir).
* **Ölçüt uyarısı:** aralıklı serilerde sMAPE/MAPE tanımsız ya da yanıltıcıdır; ölçek tabanlı ölçüler (MASE, RMSSE ve M5'te WRMSSE)
  kullanın. Eğitim kesiminin tamamı sıfırsa MASE paydası sıfırdır.
* **Süre:** yeni yöntemler için süre öncülü yok; `estimate`/koşu önce küçük bir seri örneğiyle denenmelidir (özellikle `auto_mfles`, `auto_ces`, `mstl`).
* **Kaynaklar (makalede kullanmadan önce künyeleri doğrulayın):** Croston (1972), *ORQ* 23(3); Syntetos & Boylan (2005), *IJF* 21(2);
  Teunter, Syntetos & Babai (2011), *EJOR* 214(3) [TSB]; Nikolopoulos vd. (2011), *JORS* 62(3) [ADIDA]; Petropoulos & Kourentzes (2015),
  *JORS* 66(6) [IMAPA]; Assimakopoulos & Nikolopoulos (2000), *IJF* 16(4) [Theta]; Svetunkov, Kourentzes & Ord (2022), *NRL* 69(8) [CES];
  Winters (1960), *Management Science* 6(3); Bandara, Hyndman & Bergmeir (2021), arXiv:2107.13462 [MSTL]. Kütüphane: Nixtla StatsForecast (açık kaynak, Apache-2.0).

## Tam sayıya yuvarlama ve aralıklı talep (önemli)
Katalogda sayım verisi olan kümelerde (car_parts, M5, vb.) `integer_conversion: true` (kendi çalışmanızdaki gibi): tahminler yuvarlanır.
Aralıklı talep yöntemleri ise düz bir talep **hızı** tahmin eder (ör. dönem başına 0,3 birim); yuvarlama bunu 0'a çevirir. İlk
`car_parts` denemesinde (`croston_sba`, 100 seri) görülen buydu: tüm tahminler 0 → sMAPE = 2,0 ve WAPE = %100 (hepsi tam). Aralıklı talep
yöntemleriyle `--integer-conversion off` kullanın; açık bırakırsanız koşucu uyarı yazar. Eski çalışmalarla karşılaştırma için `auto` (katalog) kalır.

**Ölçüt notu:** ortalama mutlak hata temelli ölçüler (MAE, MASE, WAPE) aralıklı seride ortanca tahmini (çoğu zaman 0) ödüllendirir; sıfır
tahmini bu ölçülerde iyi görünebilir ama envanter açısından anlamsızdır (Hyndman, 2006, *Foresight* 4(4), 43–46; künyeyi doğrulayın).
Aralıklı serilerde RMSE/RMSSE'ye ek olarak sapma (bias) ve envanter sonuçları (hizmet düzeyi, stok maliyeti) raporlanmalıdır; bu, proje
temasının (doğruluk yerine karar etkisi) tam olarak gerektirdiği analizdir ve v1 dışında bıraktığımız envanter eklentisinin ilk adayıdır.

## Bilinçli farklar
* **Hata semantiği:** model hata verirse seri `snaive` alır ve **sayılır** (`fallbacks.csv`). Model NaN/Inf üretirse R bunu 0 yapar;
  burada da yedek tahmine düşülür (0 uydurmak yerine). `na_replaced` Python motorunda her zaman 0'dır.
* **SES α sınırı:** R'ın Nelder-Mead'i bazı serilerde (α optimumu alt sınırdaysa) daha yüksek hatalı bir noktada durur; Python tam optimumu bulur.
  NN5'te 111 serinin 4–6'sında fark > 10⁻³ (en çok %17), **ama ortalama SMAPE/MASE'e etkisi < %0,4** (bkz. VALIDATION.md).
* **DHR-ARIMA:** çok mevsimli `msts`'te R `fourier(K=1)` sürüme göre hata verip SNaive'e düşebilir; Python'da terim üretilir.
* **ARIMA, mevsim > 200:** mevsimsize düşülür (`arima_max_season` ile değişir). R'ın bu konudaki davranışı doğrulanmadı.
* **Paralellik:** `--workers N` ile süreç havuzu (Windows'ta `spawn`; R'daki `mclapply` Windows'ta paralel çalışmaz).

## Doğrulama komutları
```powershell
.\scripts\windows\tsfs.ps1 doctor     # her yöntemin canlı mini testi
.\scripts\windows\tsfs.ps1 golden     # kendi R çıktılarınızla karşılaştırma (NN5 haftalık + günlük), --csv ile kaydeder
```
