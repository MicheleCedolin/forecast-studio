# Detect whether script is being run interactively (e.g. from source()) or via commandArgs
if (interactive() || length(commandArgs(trailingOnly = TRUE)) == 0) {
  cat("⚠️ No command-line arguments detected. Skipping execution of error_calc_helper.R\n")
} else {
  args <- commandArgs(trailingOnly = TRUE)
  BASE_DIR <- args[1]
  forecasts_path <- args[2]
  test_set_path <- args[3]
  training_set_path <- args[4]
  seasonality <- args[5]
  output_file_name <- args[6]

  source(file.path(BASE_DIR, "utils", "error_calculator.R", fsep = "/"))

  forecasts <- read.csv(forecasts_path, header = FALSE)
  test_set <- read.csv(test_set_path, header = FALSE)
  training_set <- readLines(training_set_path)
  training_set <- strsplit(training_set, ',')
  output_file_name <- file.path(BASE_DIR, "results", "fixed_horizon_errors", output_file_name, fsep = "/")

  calculate_errors(as.matrix(forecasts), as.matrix(test_set), training_set, as.numeric(seasonality), output_file_name)
}