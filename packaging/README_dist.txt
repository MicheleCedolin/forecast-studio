Forecast Studio {version} (Windows, 64-bit)
=========================================

[TR] Python kurmadan calisir.
  1. Bu klasoru OneDrive disinda bir yere acin (ornek: C:\ForecastStudio).
  2. ForecastStudio.exe dosyasina cift tiklayin. Bir konsol penceresi ve tarayicida arayuz acilir.
     Ilk acilis 10-20 saniye surebilir.
  3. Kendi .tsf dosyalarinizi data\tsf klasorune koyun ya da arayuzden yukleyin (Excel de olur).
     Denemek icin: examples\nn5_weekly.tsf (NN5, 111 haftalik seri).
  4. Sonuclar runs\ klasorune yazilir (tahminler, metrics.xlsx, manifest.json).
  5. Kapatmak icin arayuzdeki "Quit application" dugmesi ya da konsolda Ctrl+C.
  Windows "bilinmeyen yayimci" uyarisi verirse: "Daha fazla bilgi" -> "Yine de calistir".

[EN] Runs without installing Python.
  1. Unzip this folder outside OneDrive (for example C:\ForecastStudio).
  2. Double-click ForecastStudio.exe. A console window opens and the interface opens in your browser.
     The first start can take 10-20 seconds.
  3. Put .tsf files in data\tsf or upload them (or an Excel file) in the interface.
     To try it: examples\nn5_weekly.tsf (NN5, 111 weekly series).
  4. Results are written to runs\ (forecasts, metrics.xlsx, manifest.json).
  5. To stop: the "Quit application" button in the interface, or Ctrl+C in the console.
  If Windows SmartScreen warns about an unknown publisher: "More info" -> "Run anyway".

Command line (same commands as `python -m tsfstudio`):
  ForecastStudio.exe run --file examples\nn5_weekly.tsf --method theta
  ForecastStudio.exe doctor
  ForecastStudio.exe engines

The R engine is used only if R is installed on this computer (Rscript on PATH).
Source code and documentation: https://github.com/MicheleCedolin/forecast-studio
