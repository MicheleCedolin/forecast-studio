# Yerel (seri başına) tek değişkenli modeller.
#
# KAYNAK: reference/models/local_univariate_models.R (kullanıcının orijinal kodu).
# Model çağrıları (forecast::ets, ses, thetaf, auto.arima, tbats, ...) ve parametreleri DEĞİŞTİRİLMEDİ.
# Tek fark: orijinaldeki sessiz "hata -> SNaive" geri dönüşü artık SAYILIR (note_fallback), böylece
# çalışmanın sonunda kaç serinin gerçekte model değil SNaive tahmini aldığı raporlanabilir.
# Orijinaldeki `warning(e)` çağrıları günlüğü doldurduğundan kaldırıldı; mesaj fallback kaydında tutulur.
#
# Her fonksiyon: (time_series: ts/msts, forecast_horizon) -> ts. ARIMA ve DHR-ARIMA ayrıca (tahmin, model)
# listesi döndürebilir (rolling origin için model yeniden kullanımı).

.fb <- new.env(parent = emptyenv())
.fb$msg <- NULL

reset_fallback <- function() { .fb$msg <- NULL }
note_fallback  <- function(model, e) { .fb$msg <- paste0(model, ": ", conditionMessage(e)) }
get_fallback   <- function() .fb$msg


# Calculate ets forecasts
get_ets_forecasts <- function(time_series, forecast_horizon){
  tryCatch(
    forecast(forecast:::ets(time_series), h = forecast_horizon)$mean
  , error = function(e) {
    note_fallback("ets", e)
    get_snaive_forecasts(time_series, forecast_horizon)
  })
}


# Calculate simple exponential smoothing forecasts
get_ses_forecasts <- function(time_series, forecast_horizon){
  tryCatch(
    forecast(forecast:::ses(time_series, h = forecast_horizon))$mean
  , error = function(e) {
    note_fallback("ses", e)
    get_snaive_forecasts(time_series, forecast_horizon)
  })
}


# Calculate theta forecasts
get_theta_forecasts <- function(time_series, forecast_horizon){
  tryCatch(
    forecast:::thetaf(y = time_series, h = forecast_horizon)$mean
  , error = function(e) {
    note_fallback("theta", e)
    get_snaive_forecasts(time_series, forecast_horizon)
  })
}


# Calculate auto.arima forecasts
# model: önceden uydurulmuş model (rolling origin'de yeniden kullanım); NULL ise yeni uydurulur.
get_arima_forecasts <- function(time_series, forecast_horizon, model = NULL){
  if(is.null(model)){
    fit <- NULL
    tryCatch({
      fit <- forecast:::auto.arima(time_series, lambda = 0)
    }, error = function(e) {
      tryCatch({
        fit <<- forecast:::auto.arima(time_series)
      }, error = function(e2){
        fit <<- forecast:::auto.arima(time_series, seasonal = FALSE)
      })
    })

    tryCatch({
      f <- forecast:::forecast.Arima(fit, h = forecast_horizon)$mean
      list(f, fit)
    }, error = function(e) {
      note_fallback("arima", e)
      f <- get_snaive_forecasts(time_series, forecast_horizon)
      list(f, fit)
    })
  }else{
    tryCatch(
      forecast(forecast:::Arima(time_series, model = model), h = forecast_horizon)$mean
    , error = function(e) {
      note_fallback("arima", e)
      get_snaive_forecasts(time_series, forecast_horizon)
    })
  }
}


# Calculate tbats forecasts
get_tbats_forecasts <- function(time_series, forecast_horizon){
  tryCatch(
    forecast(forecast:::tbats(time_series), h = forecast_horizon)$mean
  , error = function(e) {
    note_fallback("tbats", e)
    get_snaive_forecasts(time_series, forecast_horizon)
  })
}


# Calculate dynamic harmonic regression arima forecasts
# K: Fourier terim sayısı. Orijinal kod K = 1 (skaler) kullanır; msts nesnesinde birden çok mevsimsel
#    periyot varsa forecast::fourier bazı sürümlerde "periyot sayısı ile mertebe sayısı uyuşmuyor"
#    hatası verebilir ve bu durumda seri SNaive'e düşer. Düşme sayısı raporlanır; isterseniz K'yi
#    periyot sayısı kadar tekrarlatabilirsiniz (worker parametresi fourier_k_per_period).
get_dhr_arima_forecasts <- function(time_series, forecast_horizon, model = NULL, K = 1){
  if(is.null(model)){
    tryCatch({
      xreg <- forecast:::fourier(time_series, K = K)
      model <- forecast:::auto.arima(time_series, xreg = xreg, seasonal = FALSE)
      xreg1 <- forecast:::fourier(time_series, K = K, h = forecast_horizon)
      f <- forecast(model, xreg = xreg1)$mean
      list(f, model)
    }, error = function(e) {
      note_fallback("dhr_arima", e)
      f <- get_snaive_forecasts(time_series, forecast_horizon)
      list(f, model)
    })
  }else{
    tryCatch({
      xreg <- forecast:::fourier(time_series, K = K)
      xreg1 <- forecast:::fourier(time_series, K = K, h = forecast_horizon)
      forecast(forecast:::Arima(time_series, model = model, xreg = xreg), xreg = xreg1)$mean
    }, error = function(e) {
      note_fallback("dhr_arima", e)
      get_snaive_forecasts(time_series, forecast_horizon)
    })
  }
}


# Calculate snaive forecasts
get_snaive_forecasts <- function(time_series, forecast_horizon){
  forecast:::snaive(time_series, h = forecast_horizon)$mean
}
