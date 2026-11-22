# Windows 11 + Docker Desktop kurulumu

Makine: i5-8400 (6 çekirdek), 8 GB RAM. Docker Desktop **WSL2 arka ucu** ile çalışır.

## 1. Bir kez yapılacaklar
1. Docker Desktop'ı kurun, bilgisayarı yeniden başlatın, Docker Desktop'ı açıp "Engine running" olmasını bekleyin.
2. WSL2 belleği: varsayılan RAM'in %50'sidir (~4 GB). `scripts\windows\wslconfig.example` içeriğini
   `C:\Users\<kullanıcı>\.wslconfig` olarak kaydedin (`memory=6GB`, `swap=8GB`, `processors=5`), sonra PowerShell'de
   `wsl --shutdown` çalıştırıp Docker Desktop'ı yeniden başlatın. (`setup.ps1` bunu sormadan yazmaz.)
3. Proje klasörünü açın ve `data\tsf\` içine 43 `.tsf` dosyasını koyun.
4. `powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1`
   (Betik engellenirse: `Get-ChildItem -Recurse | Unblock-File`.)

İlk imaj oluşturma 15–40 dk sürebilir (R paketleri + yaklaşık 2 GB veri katmanı). Sonraki kod değişikliklerinde veri
katmanı yeniden oluşturulmaz.

## 2. Günlük kullanım
```powershell
.\scripts\windows\run.ps1 doctor                                   # ortam + her R yöntemi için hızlı canlı test
.\scripts\windows\run.ps1 estimate --dataset m5_item_store --task global --method pooled_regression --lag 12
.\scripts\windows\run.ps1 run --dataset nn5_weekly --engine r --method ets
.\scripts\windows\golden-test.ps1
```
`estimate` koşmadan bellek ve süre öngörüsü verir; `run` uzun koşularda (>15 dk tahmini) onay ister (`--yes` ile atlanır).

## 3. Sorun giderme
| Belirti | Neden / çözüm |
|---|---|
| `docker: command not found` | Docker Desktop kurulu değil ya da yeniden başlatılmadı |
| "Docker calismiyor" | Docker Desktop'ı açın, "Engine running" bekleyin |
| `R işçisi ... kod 137` | Bellek yetmedi (OOM). `.wslconfig` belleğini artırın, `max_obs` ya da `--max-series` kullanın |
| Çok yavaş | `.wslconfig` `processors`, `--workers 4` (yerel modeller; Windows'ta R'ın kendisi paralel çalışmaz, Docker/Linux çalışır) |
| `doctor`: catboost UYARI | CatBoost R paketi kurulamadı; imajı `--build-arg CATBOOST_REQUIRED=1` ile yeniden kurup hata mesajına bakın |
| İmaj çok büyük | Veriyi `docker-compose.yml` içindeki bind mount ile bağlayın, `Dockerfile`'daki `COPY data/tsf` satırını silin |
