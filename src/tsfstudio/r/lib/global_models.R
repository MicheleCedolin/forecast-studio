# Global modeller: havuzlanmış regresyon (glm) ve CatBoost.
#
# KAYNAK: reference/models/global_models.R ve reference/utils/global_model_helper.R.
# Model kurulumu (formül `y ~ Lag1 + ... + 0`, glm, catboost.train varsayılanları, özyinelemeli tahmin,
# ortalama normalizasyonu) DEĞİŞTİRİLMEDİ. Farklar:
#   * create_input_matrix: döngü içi rbind (karesel maliyet) ve print(i) kaldırıldı; sonuç özdeş.
#   * Çok kısa seri için hangi serinin sorunlu olduğunu söyleyen açık hata.
#   * max_obs (varsayılan NULL = orijinal): bellek sınırlı büyük veride (ör. M5) her serinin son max_obs
#     gözlemi kullanılır; ortalama normalizasyonu bu pencere üzerinden yapılır.
#   * catboost eğitim parametreleri `cb_params` listesiyle verilebilir (boş = catboost.train varsayılanları).
#   * set.seed orijinaldeki gibi (varsayılan 1) ama worker tarafından job$seed ile ayarlanır.

failed_loading_catboost <- FALSE
tryCatch(suppressPackageStartupMessages(library(catboost)), error = function(err) { failed_loading_catboost <<- TRUE })


# Creating embedded matrix and final lags to train the global models for a given lag
create_input_matrix <- function(dataset, lag, max_obs = NULL){
  n <- length(dataset)
  series_means <- numeric(n)
  embedded_list <- vector("list", n)
  final_lags <- matrix(NA_real_, nrow = n, ncol = lag)

  for (i in seq_len(n)) {
    time_series <- as.numeric(dataset[[i]])

    if (!is.null(max_obs) && length(time_series) > max_obs)
      time_series <- tail(time_series, max_obs)

    if (length(time_series) < lag + 1)
      stop(sprintf("Seri %d, lag = %d için çok kısa (uzunluk %d; en az %d gerekir).", i, lag, length(time_series), lag + 1))

    mean_value <- mean(time_series)

    # Mean normalisation
    if(mean_value == 0)
      mean_value <- 1 # Avoid division by zero

    time_series <- time_series / mean_value
    series_means[i] <- mean_value

    # Embed the series
    embedded_list[[i]] <- embed(time_series, lag + 1)

    # Creating the test set
    final_lags[i, ] <- rev(tail(time_series, lag))
  }

  embedded_series <- as.data.frame(do.call(rbind, embedded_list))
  colnames(embedded_series) <- c("y", paste("Lag", 1:lag, sep = ""))

  final_lags <- as.data.frame(final_lags)
  colnames(final_lags) <- paste("Lag", 1:lag, sep = "")

  list(embedded_series, final_lags, series_means)
}


# Forecasting with different lags
start_forecasting <- function(dataset, lag, forecast_horizon, method, max_obs = NULL, cb_params = list()){
  # Creating embedded matrix (for model training) and test set
  result <- create_input_matrix(dataset, lag, max_obs)

  embedded_series <- result[[1]] # Embedded matrix
  final_lags <- result[[2]] # Test set
  series_means <- result[[3]] # Mean value of each series

  fit_model(embedded_series, lag, final_lags, forecast_horizon, series_means, method, cb_params)
}


# Fit and forecast from a global model
fit_model <- function(fitting_data, lag, final_lags, forecast_horizon, series_means, method, cb_params = list()) {
  # Create the formula
  formula <- "y ~ "
  for(predictor in 2:ncol(fitting_data)){
    if(predictor != ncol(fitting_data)){
      formula <- paste0(formula, colnames(fitting_data)[predictor], " + ")
    }else{
      formula <- paste0(formula, colnames(fitting_data)[predictor])
    }
  }

  formula <- paste(formula, "+ 0", sep="")
  formula <- as.formula(formula)

  # Fit global models
  if(method == "pooled_regression"){
    # Fit the pooled regression model
    model <- glm(formula = formula, data = fitting_data)
  }else if(method == "catboost"){
    if (failed_loading_catboost) stop("Error when loading catboost, cannot run global model based on catboost")
    # Fit the CatBoost model (cb_params boşsa catboost.train varsayılanları)
    train_pool <- catboost.load_pool(data = as.matrix(fitting_data[-1]), label = as.matrix(fitting_data[,1]))
    model <- catboost.train(train_pool, params = cb_params)
  }else{
    stop(paste0("Bilinmeyen global yöntem: ", method))
  }

  # Do forecasting
  forec_recursive(lag, model, final_lags, forecast_horizon, series_means, method)
}


# Recursive forecasting of the series until a given horizon
forec_recursive <- function(lag, model, final_lags, forecast_horizon, series_means, method){

  # This will store the predictions corresponding with each horizon
  predictions <- NULL

  for (i in 1:forecast_horizon){
    # Get predictions for the current horizon
    if(method == "pooled_regression")
      new_predictions <- predict.glm(object = model, newdata = as.data.frame(final_lags))
    else if(method == "catboost"){
      catboost_final_lags <- catboost.load_pool(final_lags)
      new_predictions <- catboost.predict(model, catboost_final_lags)
    }

    # Adding the current forecasts to the final predictions matrix
    predictions <- cbind(predictions, new_predictions)

    # Updating the test set for the next horizon
    if(i < forecast_horizon){
      final_lags <- final_lags[-lag]
      final_lags <- cbind(new_predictions, final_lags)
      colnames(final_lags)[1:lag] <- paste("Lag", 1:lag, sep="")
      final_lags <- as.data.frame(final_lags)
    }
  }

  # Renormalise the predictions
  true_predictions <- predictions * as.vector(series_means)
  true_predictions
}
