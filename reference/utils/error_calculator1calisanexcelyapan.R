library(smooth)
library(openxlsx)

calculate_errors <- function(forecasts, test_set, training_set, seasonality, output_file_name){
  # Calculate all error metrics
  smape_per_series <- calculate_smape(forecasts, test_set)
  msmape_per_series <- calculate_msmape(forecasts, test_set)
  mase_per_series <- calculate_mase(forecasts, test_set, training_set, seasonality)
  mae_per_series <- calculate_mae(forecasts, test_set)
  rmse_per_series <- calculate_rmse(forecasts, test_set)

  # Combine all per-series errors into a data frame
  error_df <- data.frame(
    SMAPE = smape_per_series,
    mSMAPE = msmape_per_series,
    MASE = mase_per_series,
    MAE = mae_per_series,
    RMSE = rmse_per_series
  )

  # Create summary statistics
  summary_stats <- data.frame(
    Metric = c("SMAPE", "mSMAPE", "MASE", "MAE", "RMSE"),
    Mean = c(mean(smape_per_series, na.rm = TRUE),
             mean(msmape_per_series, na.rm = TRUE),
             mean(mase_per_series, na.rm = TRUE),
             mean(mae_per_series, na.rm = TRUE),
             mean(rmse_per_series, na.rm = TRUE)),
    Median = c(median(smape_per_series, na.rm = TRUE),
               median(msmape_per_series, na.rm = TRUE),
               median(mase_per_series, na.rm = TRUE),
               median(mae_per_series, na.rm = TRUE),
               median(rmse_per_series, na.rm = TRUE))
  )

  # Create and write Excel workbook
  wb <- createWorkbook()
  addWorksheet(wb, "Errors_Per_Series")
  addWorksheet(wb, "Summary")

  writeData(wb, sheet = "Errors_Per_Series", error_df)
  writeData(wb, sheet = "Summary", summary_stats)

  saveWorkbook(wb, paste0(output_file_name, "_metrics.xlsx"), overwrite = TRUE)

  # Print summary
  print(summary_stats)
}