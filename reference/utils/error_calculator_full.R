library(smooth)
library(openxlsx)

calculate_errors <- function(forecasts, test_set, training_set, seasonality, output_file_name){
  # Baseline naive forecasts
  naive_forecasts <- matrix(NA, nrow = nrow(test_set), ncol = ncol(test_set))
  for(i in 1:nrow(test_set)){
    if(length(training_set[[i]]) >= seasonality){
      naive_forecasts[i, ] <- rep(tail(training_set[[i]], seasonality)[1], ncol(test_set))
    } else {
      naive_forecasts[i, ] <- rep(tail(training_set[[i]], 1), ncol(test_set))
    }
  }

  # Standard errors
  smape_per_series <- calculate_smape(forecasts, test_set)
  msmape_per_series <- calculate_msmape(forecasts, test_set)
  mase_per_series <- calculate_mase(forecasts, test_set, training_set, seasonality)
  mae_per_series <- calculate_mae(forecasts, test_set)
  rmse_per_series <- calculate_rmse(forecasts, test_set)
  wape_per_series <- calculate_wape(forecasts, test_set)
  nrmse_mean_per_series <- calculate_nrmse_mean(forecasts, test_set)
  nrmse_range_per_series <- calculate_nrmse_range(forecasts, test_set)

  # Scaled errors
  naive_mae <- calculate_mae(naive_forecasts, test_set)
  naive_rmse <- calculate_rmse(naive_forecasts, test_set)
  naive_mape <- calculate_mape(naive_forecasts, test_set) # You'll need to define this if not already

  sMAE <- mae_per_series / naive_mae
  sRMSE <- rmse_per_series / naive_rmse
  sMSE <- (rmse_per_series^2) / (naive_rmse^2)
  RelMAE <- sMAE
  RelRMSE <- sRMSE
  RelMAPE <- msmape_per_series / naive_mape

  # Theil's U
  theils_u <- sqrt(colMeans((forecasts - test_set)^2, na.rm = TRUE)) /
              (sqrt(colMeans((naive_forecasts - test_set)^2, na.rm = TRUE)) + 1e-10)

  # Combine errors
  error_df <- data.frame(
    SMAPE = smape_per_series,
    mSMAPE = msmape_per_series,
    MASE = mase_per_series,
    MAE = mae_per_series,
    RMSE = rmse_per_series,
    WAPE = wape_per_series,
    nRMSE_Mean = nrmse_mean_per_series,
    nRMSE_Range = nrmse_range_per_series,
    sMAE = sMAE,
    sRMSE = sRMSE,
    sMSE = sMSE,
    RelMAE = RelMAE,
    RelRMSE = RelRMSE,
    RelMAPE = RelMAPE,
    Theils_U = theils_u
  )

  # Summary
  metric_names <- colnames(error_df)
  summary_stats <- data.frame(
    Metric = metric_names,
    Mean = sapply(error_df, function(x) mean(x, na.rm = TRUE)),
    Median = sapply(error_df, function(x) median(x, na.rm = TRUE))
  )

  # Excel output
  wb <- createWorkbook()
  addWorksheet(wb, "Errors_Per_Series")
  addWorksheet(wb, "Summary")

  writeData(wb, "Errors_Per_Series", error_df)
  writeData(wb, "Summary", summary_stats)
  saveWorkbook(wb, paste0(output_file_name, "_metrics.xlsx"), overwrite = TRUE)

  print(summary_stats)
}