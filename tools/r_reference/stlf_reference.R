# R reference for the C++ stl_ets / stl_arima: forecast(mstl(msts(y, periods)), method = ...)$mean, as the R engine
# builds the series (worker.R: msts(series, seasonal.periods = seasonality)).
# Usage: Rscript stlf_reference.R train.csv "7,30.4375" h out.csv ets|arima
suppressPackageStartupMessages(library(forecast))
a <- commandArgs(trailingOnly = TRUE)
X <- as.matrix(read.csv(a[1], header = FALSE)); per <- as.numeric(strsplit(a[2], ",")[[1]]); h <- as.integer(a[3])
method <- if (length(a) >= 5) a[5] else "ets"
FC <- matrix(NA, nrow(X), h)
t0 <- Sys.time()
for (i in seq_len(nrow(X))) {
  y <- msts(X[i, !is.na(X[i, ])], seasonal.periods = per)
  f <- tryCatch(if (method == "ets") stlf(y, h = h) else forecast(mstl(y), method = "arima", h = h),
                error = function(e) NULL)
  if (!is.null(f)) FC[i, ] <- f$mean
}
cat(sprintf("R stlf (%s): %.1f s\n", method, as.numeric(difftime(Sys.time(), t0, units = "secs"))))
write.table(FC, a[4], sep = ",", row.names = FALSE, col.names = FALSE)
