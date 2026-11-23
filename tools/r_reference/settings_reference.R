# R reference for the C++ engine's non-default settings (tests/test_cpp_engine.py::test_non_default_settings_reproduce_r),
# forecast 9.0.2. Writes, next to the golden training data:
#   nn5_weekly/r_bagged_ets_n20_s5.csv : set.seed(5); forecast(baggedETS(y, bootstrapped_series = bld.mbb.bootstrap(y, 20)), 8)
#                                        on msts(y, 52.17857142857143), first 10 series
#   nn5_weekly/r_ets_man.csv           : forecast(ets(y, model = "MAN"), 8), first 10 series (Holt, error_type M)
#   nn5_daily/r_ets_mam.csv            : forecast(ets(ts(y, frequency = 7), model = "MAM"), 8) on the last 200 values of
#                                        the first 6 series whose last 200 values are all positive (Holt-Winters, error_type M)
# Usage: Rscript tools/r_reference/settings_reference.R tests/golden
suppressPackageStartupMessages(library(forecast))
g <- commandArgs(trailingOnly = TRUE)[1]
out <- function(M, f) writeLines(apply(M, 1, function(r) paste(sprintf("%.17g", r), collapse = ",")), f)
W <- as.matrix(read.csv(file.path(g, "nn5_weekly", "train.csv"), header = FALSE))
wk <- lapply(1:10, function(i) W[i, !is.na(W[i, ])])
out(t(sapply(wk, function(v) { y <- msts(v, seasonal.periods = 52.17857142857143); set.seed(5)
  as.numeric(forecast(baggedETS(y, bootstrapped_series = bld.mbb.bootstrap(y, 20)), h = 8)$mean) })),
  file.path(g, "nn5_weekly", "r_bagged_ets_n20_s5.csv"))
out(t(sapply(wk, function(v) as.numeric(forecast(ets(ts(v), model = "MAN"), h = 8)$mean))),
  file.path(g, "nn5_weekly", "r_ets_man.csv"))
D <- as.matrix(read.csv(file.path(g, "nn5_daily", "train.csv.gz"), header = FALSE))
dy <- lapply(seq_len(nrow(D)), function(i) { v <- D[i, !is.na(D[i, ])]; tail(v, 200) })
dy <- Filter(function(v) min(v) > 0, dy)[1:6]
out(t(sapply(dy, function(v) as.numeric(forecast(ets(ts(v, frequency = 7), model = "MAM"), h = 8)$mean))),
  file.path(g, "nn5_daily", "r_ets_mam.csv"))
