# Reference for the C++ dhr_arima: the study's get_dhr_arima_forecasts (src/tsfstudio/r/lib/local_models.R):
# xreg <- fourier(y, K); auto.arima(y, xreg = xreg, seasonal = FALSE); forecast(fit, xreg = fourier(y, K, h))$mean.
# Usage: Rscript dhr_reference.R <train.csv[.gz]> <period(s), comma-separated> <horizon> <K> <out_prefix>
# Writes <out>_forecast.csv and <out>_order.csv (p,d,q,drift,mean,ok per series). A failed series gets NA (the app
# falls back to snaive).
suppressPackageStartupMessages(library(forecast))
a <- commandArgs(trailingOnly = TRUE)
f <- a[1]; per <- as.numeric(strsplit(a[2], ",")[[1]]); h <- as.integer(a[3]); K <- as.integer(a[4]); out <- a[5]
con <- if (grepl("\\.gz$", f)) gzfile(f) else file(f)
X <- as.matrix(read.csv(con, header = FALSE))
n <- nrow(X)
FC <- matrix(NA, n, h)
ORD <- matrix(NA, n, 6, dimnames = list(NULL, c("p", "d", "q", "drift", "mean", "ok")))
t0 <- Sys.time()
for (i in seq_len(n)) {
  y <- X[i, ]; y <- msts(y[!is.na(y)], seasonal.periods = per)
  r <- tryCatch({
    fit <- auto.arima(y, xreg = fourier(y, K = K), seasonal = FALSE)
    list(fc = forecast(fit, xreg = fourier(y, K = K, h = h))$mean, fit = fit)
  }, error = function(e) NULL)
  if (is.null(r)) { ORD[i, "ok"] <- 0; next }
  FC[i, ] <- r$fc
  o <- arimaorder(r$fit); cf <- names(coef(r$fit))
  ORD[i, ] <- c(o[1:3], "drift" %in% cf, "intercept" %in% cf, 1)
}
secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
write.table(FC, paste0(out, "_forecast.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
write.table(ORD, paste0(out, "_order.csv"), sep = ",", row.names = FALSE, col.names = TRUE)
cat(sprintf("done: %d series in %.1f s, failed %d, forecast %s\n", n, secs, sum(ORD[, "ok"] == 0), as.character(packageVersion("forecast"))))
