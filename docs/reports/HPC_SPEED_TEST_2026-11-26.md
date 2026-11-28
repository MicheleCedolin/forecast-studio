# Forecast Studio – HPC motor hız testi (26.11.2026)

**Durum:** deneme testi. Makaleye girecek sonuç değildir: düğüm paylaşılıyordu, iki veri seti var ve Python ile R bir kez koştu.
**Amaç:** C++ (tsfcore), Python (statsforecast) ve R (forecast) motorlarının aynı yöntemlerdeki sürelerini karşılaştırmak; C++'ta thread, Python ve R'de işçi süreci sayısıyla ölçeklenmeyi görmek; kümedeki ölçüm hattının uçtan uca çalıştığını doğrulamak.

---

## 1. Ortam

| Öğe | Değer |
|:--|:--|
| Küme | SGE 8.1.9; kuyruk `all.q`; paralel ortam `smp` (`allocation_rule $pe_slots`); 9 hesaplama düğümü |
| Kullanıcı kotası | en çok 32 slot (`qquota`: `slots=1/32`) |
| Testin koştuğu düğüm | `compute-0-4.local`, iş başına **8 slot** |
| CPU | 2 × Intel Xeon E5-2698 v3 @ 2.30 GHz: 2 soket × 16 çekirdek × 2 HT = **32 fiziksel, 64 mantıksal** çekirdek; 125.6 GB bellek |
| Çekirdek bağlama | yok (`affinity_cpus = 64`: iş düğümün bütün mantıksal CPU'larını görüyor) |
| Düğüm yükü | 1 dk yük ortalaması başta 33.81, sonda 37.66. Düğüm **paylaşılıyordu**: o sırada diğer düğümlerde de 48–60/64 slot doluydu |
| İşletim sistemi | CentOS 7, Linux 3.10, glibc 2.17 (sistem g++ 4.8.5 kullanılmadı) |
| Ortam | conda `~/miniforge3/envs/tsfstudio`: Python 3.11.18, numpy 2.4.6, scipy 1.17.1, statsforecast 2.1.1, lightgbm 4.6.0 |
| R | conda R 4.5.3, forecast 9.0.2, jsonlite 2.0.0 (bozuk sistem R 3.4.3 kullanılmadı) |
| C++ | tsfcore 0.4.0, kümede conda derleyicisiyle (`x86_64-conda-linux-gnu-c++`) yeniden derlendi; `TSF_STATIC_RUNTIME=ON` |
| Kod | depo `MicheleCedolin/forecast-studio`, dal `claude/forecast-studio-hpc-speed-p4qixh`, paket commit `3b908f5` (`forecast-studio-hpc-speed-20261126.tar.gz`, `--no-data`) |
| Zaman | iş 08:16:07'de başladı, 09:33:47'de bitti (~78 dk, yeniden gönderimden sonraki kısım) |

## 2. Tasarım

| Etmen | Düzeyler |
|:--|:--|
| Veri setleri | `nn5_weekly` (111 seri, ufuk 8), `nn5_daily` (111 seri, 735 gözlem, ufuk 56), Monash `.tsf` dosyaları |
| Yöntemler | ses, theta, ets, arima |
| C++ | 1 ve 8 thread (`TSFSTUDIO_CPP_THREADS`), 3 tekrar, rapor edilen değer medyan |
| Python, R | 1 ve 8 işçi süreci (`--workers`), 1 tekrar |
| Diğer | süreç başına BLAS ve numba 1 thread; tek koşu sınırı 60 dk; motorlar aynı işte sırayla koştu |

**Ölçülen değer:** her koşunun `manifest.json` dosyasındaki `timing.engine_wall_sec`. Bu, motorun tüm serileri uydurup tahmin ettiği çağrının süresidir. Dosya okuma, hata ölçüleri ve yazma dahil değildir. R sürelerine Rscript'in başlatılması dahildir. Her hücre `python -m tsfstudio run` ile, yani bir kullanıcının çalıştıracağı biçimde koşturuldu.

**Koşu sırası:** her (veri seti, yöntem, motor) için önce büyük düzey koştu, sonra 1. Python ölçümden önce bir kez ısıtıldı (numba derlemesi).

## 3. Çalıştırılan komutlar

Login node:

```bash
# paket açma ve ön kontrol
cd ~ && tar -xzf forecast-studio-hpc-speed-20261126.tar.gz && cd ~/forecast-studio
bash scripts/hpc/preflight.sh 2>&1 | tee preflight.txt

# C++ çekirdeğinin yeniden derlenmesi ve motor kontrolü (çevrimdışı)
source scripts/hpc/activate.sh
rm -rf cpp/build && TSF_STATIC_RUNTIME=ON bash cpp/build.sh
python -m tsfstudio engines

# test (8 slot); yarıda bir kez qdel ile durduruldu, SPEED_REPEATS=1 ile aynı klasöre yeniden gönderildi
SPEED_DATASETS=nn5_weekly,nn5_daily SPEED_METHODS=ses,theta,ets,arima SPEED_THREADS=1,8 \
SPEED_REPEATS=3 SPEED_REF_REPEATS=1 SPEED_TIMEOUT_MIN=60 SPEED_OUT=results/speed_test \
qsub -V -pe smp 8 scripts/hpc/speed.sge
```

**Yeniden gönderim:** kaldığı yerden devam etti. Biten tekrarlar `runs.csv` dosyasında saklandı. Bu yüzden ilk işte tamamlanan hücrelerin (nn5_weekly'nin tamamı, nn5_daily'nin bir kısmı) C++ tekrar sayısı 3, ikinci işte koşanların 1–2.

**İlgili dosyalar:** `tools/engine_speed_study.py` (ölçüm ve rapor), `scripts/hpc/speed.sge` (SGE işi), `scripts/hpc/run_speed.sh` (başlatıcı), `scripts/hpc/activate.sh` (ortam), `scripts/hpc/preflight.sh` (ön kontrol), `docs/HPC_SPEED.md` (kullanım). Çıktılar `results/speed_test/` altında: `speed_table.md`, `summary.csv`, `ratios.csv`, `runs.csv`, `node.json`, `node_lscpu.txt`, `node_qhost.txt`, `runs/<veri>/<yöntem>/<motor>_x<düzey>/` (tahminler ve seri başına hatalar).

## 4. Sonuçlar

### 4.1 Motor süreleri (saniye, medyan)

| Veri seti | Yöntem | Python (1) | Python (8) | R (1) | R (8) | C++ (1) | C++ (8) |
|:--|:--|--:|--:|--:|--:|--:|--:|
| nn5_weekly | ses | 0.421 | 0.443 | 3.184 | 2.878 | 0.015 | 0.012 |
| nn5_weekly | theta | 0.574 | 0.608 | 3.776 | 3.250 | 0.013 | 0.009 |
| nn5_weekly | ets | 1.560 | 1.281 | 5.586 | 3.488 | 0.743 | 0.140 |
| nn5_weekly | arima | 45.2 | 31.9 | 32.4 | 12.0 | 9.042 | 2.155 |
| nn5_daily | ses | 1.272 | 1.195 | 4.241 | 3.182 | 0.080 | 0.020 |
| nn5_daily | theta | 1.212 | 0.979 | 6.020 | 3.630 | 0.072 | 0.018 |
| nn5_daily | ets | 61.5 | 46.8 | 111.2 | 24.4 | 46.7 | 6.790 |
| nn5_daily | arima | 243.1 | 164.3 | **2374** | 446.0 | 1387 | 222.7 |

C++'ta tekrarlar arası yayılım dar. Örnek: nn5_daily ets, 1 thread, toplam duvar süresi 48.5 / 50.5 / 49.7 s.

### 4.2 Motorlar arası oranlar (büyüdükçe C++ o kadar hızlı)

| Veri seti | Yöntem | Python/C++ (1) | R/C++ (1) | R/Python (1) | Python/C++ (8) | R/C++ (8) |
|:--|:--|--:|--:|--:|--:|--:|
| nn5_weekly | ses | 28.3 | 214 | 7.6 | 37 | 240 |
| nn5_weekly | theta | 42.9 | 282 | 6.6 | 68 | 361 |
| nn5_weekly | ets | 2.10 | 7.52 | 3.58 | 9.2 | 24.9 |
| nn5_weekly | arima | 5.00 | 3.58 | 0.72 | 14.8 | 5.6 |
| nn5_daily | ses | 16.0 | 53.2 | 3.3 | 60 | 159 |
| nn5_daily | theta | 16.9 | 84.0 | 5.0 | 54 | 202 |
| nn5_daily | ets | 1.32 | 2.38 | 1.81 | 6.9 | 3.6 |
| nn5_daily | arima | **0.175** | 1.71 | 9.77 | 0.74 | 2.0 |

1 birimlik sütunlar, araç tarafından tam duyarlıkla üretilen `ratios.csv` dosyasından alındı. 8 birimlik sütunlar 4.1'deki medyanlardan hesaplandı; özellikle 0.01 s civarındaki C++ sürelerinde yuvarlama payı var.

### 4.3 Ölçeklenme: 1'den 8'e hızlanma (parantez içinde verimlilik = hızlanma / 8)

| Veri seti | Yöntem | C++ | R | Python |
|:--|:--|--:|--:|--:|
| nn5_weekly | ses | 1.23 (0.15) | 1.11 | 0.95 |
| nn5_weekly | theta | 1.47 (0.18) | 1.16 | 0.94 |
| nn5_weekly | ets | 5.31 (0.66) | 1.60 | 1.22 |
| nn5_weekly | arima | 4.20 (0.52) | 2.70 | 1.42 |
| nn5_daily | ses | 4.09 (0.51) | 1.33 | 1.06 |
| nn5_daily | theta | 3.95 (0.49) | 1.66 | 1.24 |
| nn5_daily | ets | 6.87 (0.86) | 4.56 | 1.31 |
| nn5_daily | arima | 6.23 (0.78) | 5.32 | 1.48 |

### 4.4 Doğruluk tutarlılığı (MASE ortalaması)

| Veri seti | Yöntem | C++ | R | Python |
|:--|:--|--:|--:|--:|
| nn5_weekly | ses | 0.9063812 | 0.9031552 | 0.9063812 |
| nn5_weekly | theta | 0.8853564 | 0.8846909 | 0.8853564 |
| nn5_weekly | ets | 0.9106506 | 0.9106506 | 0.8981254 |
| nn5_weekly | arima | 1.0268575 | 1.0268575 | 1.0238356 |
| nn5_daily | ses | 1.5237475 | 1.5210052 | 1.5237475 |
| nn5_daily | theta | 0.8810485 | 0.8845194 | 0.8810485 |
| nn5_daily | ets | 0.8649048 | 0.8649048 | 0.8696301 |
| nn5_daily | arima | 0.9873295 | 0.9873299 | 0.9273948 |

Her motorun MASE değeri, her thread ve işçi sayısında 7. haneye kadar aynı.

## 5. Bulgular

1. **Ölçüm hattı çalışıyor.** 89 koşunun hepsi `ok` durumunda; hiçbir koşu başarısız olmadı, zaman aşımına uğramadı ya da atlanmadı. Düğüm bilgisi, yük ve her koşunun başlangıç yükü kayda geçti. Yarıda kesilen iş kaldığı yerden devam edebildi.
2. **Paralellik sonucu değiştirmiyor.** Her motor için MASE, thread ve işçi sayısından bağımsız.
3. **C++, ETS ve ARIMA'da R'yi yeniden üretiyor.** ETS'de MASE aynı, ARIMA'da fark ≤ 4×10⁻⁷. ses ve theta'da C++ Python ile aynı sonucu veriyor; R ile aradaki fark %0.1–0.4. Python'un ETS ve ARIMA'sı (statsforecast) model seçimini farklı yaptığı için başka sonuç veriyor. Bunlar daha önce bilinen motor farklarıyla tutarlı.
4. **Basit yöntemlerde C++ çok hızlı.** ses ve theta tek çekirdekte R'den 53–282 kat, Python'dan 16–43 kat hızlı. Bu yöntemlerde süreyi R'nin başlatma maliyeti belirliyor.
5. **ETS'de fark orta düzeyde.** Tek çekirdekte C++, R'den 2.4–7.5 kat, Python'dan 1.3–2.1 kat hızlı.
6. **ARIMA'da sonuç seriye bağlı:**
   - nn5_weekly'de C++ R'den 3.6 kat, Python'dan 5 kat hızlı.
   - nn5_daily'de (uzun, mevsimsel seriler) C++ R'den yalnızca 1.7 kat hızlı, Python'dan **5.7 kat yavaş** (1387 s'ye karşı 243 s).
   - C++ burada R'nin `auto.arima` aramasını birebir izliyor (aynı MASE). Python ise daha kısa bir aramayla farklı modellere ulaşıyor; bu veri setinde MASE değeri daha düşük (0.927, diğerlerinde 0.987).
   - Taslak makaledeki "ARIMA'da R'ye göre ~14 kat" ifadesi bu iki veri setinde gözlenmedi.
7. **C++, ağır işlerde 8 thread'e iyi ölçekleniyor.** ets ve arima'da hızlanma 4.2–6.9, verimlilik 0.52–0.86. Uzun serilerde (nn5_daily) verimlilik daha yüksek. ses ve theta'da iş milisaniyeler sürdüğü için ölçeklenme anlamlı değil.
8. **R, uzun işlerde işçi süreçleriyle ölçekleniyor.** nn5_daily ets ve arima'da 8 işçiyle hızlanma 4.6–5.3; kısa işlerde ~1–2.7.
9. **Python'un işçi ölçeklenmesi şüpheli derecede zayıf.** 8 işçiyle en fazla 1.48 kat hızlanıyor; 241 s'lik bir işte bile 1.48. Ya `--workers` Python motorunda işi gerçekten paylaştırmıyor, ya da süreç veya numba ek yükü baskın. Bu ayrıca incelenmeli.

## 6. Sınırlar ve uyarılar

- **Düğüm paylaşılıyordu:** yük ortalaması ~34–38/64 ve çekirdek bağlama yoktu. Mutlak süreler gürültülü. Motorlar aynı işte sırayla koştuğu için oranlar mutlak sürelerden daha güvenilir.
- **Python ve R birer kez koştu.** C++'ta tekrar sayısı hücreye göre 1–3, çünkü iş yeniden gönderildi.
- **Kapsam dar:** iki veri seti, ikisi de NN5 (111 seri), yalnızca 1 ve 8 düzeyi. Taslak makaledeki 16 veri seti medyanıyla doğrudan karşılaştırılamaz.
- **Klasörde bu teste ait olmayan satırlar var.** `results/speed_test/runs.csv` içinde nn5_weekly ses ve ets için 2, 4, 16 ve 32 thread ile 32 işçi satırları bulunuyor; bunlar aynı klasöre daha önce yazılmış başka bir koşudan geliyor. O koşunun slot sayısı bilinmiyor. Bu yüzden 32 thread'deki ets hızlanması (11.6, verimlilik 0.36) bu raporda kullanılmadı. Ayrıca araç kayıtlı tekrarları atladığı için nn5_weekly ses ve ets'nin 1 birimlik hücreleri (C++ 1. tekrar, Python 1, R 1) o önceki koşudan geliyor. Bu dört satırın medyanı iki koşunun karışımı. Aynı nedenle `speed_table.md`'deki "32 threads / workers" A tablosu da yalnızca o satırlara dayanıyor.
- **Ölçülen süre yalnızca motor çağrısı.** Duvar süresinde her koşuya 2.5–3 s başlatma ve yazma maliyeti ekleniyor.

## 7. Önerilen sonraki adımlar

1. **Python işçi ölçeklenmesi:** `--workers` değerinin Python motorunda gerçekten kullanıldığını küçük ölçekte doğrulamak, gerekirse düzeltmek.
2. **C++ ARIMA, uzun mevsimsel seriler:** nn5_daily'de neden yavaş kaldığını (arama adımları, model sayısı) profillemek.
3. **Gerçek 32 çekirdek testi:** yöneticiden ayrılmış bir düğüm istemek, ya da paylaşılan düğümde 32 slot ile koşup yükü rapor etmek. Bunun için `results/` altında yeni ve boş bir klasör kullanılmalı.
4. **Daha geniş kapsam:** m3_monthly ve m4_hourly ile tbats eklenirse, tahmini süre paylaşılan düğümde 5–8 saat.
