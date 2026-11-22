# Reference for the C++ auto.arima: the study's call chain (auto.arima(lambda = 0) -> auto.arima() -> seasonal = FALSE).
# Usage: Rscript arima_reference.R <train.csv[.gz]> <frequency> <horizon> <out_prefix>
# Writes <out>_forecast.csv, <out>_order.csv (p,d,q,P,D,Q,period,drift,mean,step per series), <out>_fit.csv (loglik, aicc, sigma2).
suppressPackageStartupMessages(library(forecast))
a <- commandArgs(trailingOnly = TRUE)
f <- a[1]; freq <- as.numeric(a[2]); h <- as.integer(a[3]); out <- a[4]
con <- if (grepl("\\.gz$", f)) gzfile(f) else file(f)
X <- as.matrix(read.csv(con, header = FALSE))
n <- nrow(X)
FC <- matrix(NA, n, h)
ORD <- matrix(NA, n, 10, dimnames = list(NULL, c("p", "d", "q", "P", "D", "Q", "period", "drift", "mean", "step")))
FIT <- matrix(NA, n, 3, dimnames = list(NULL, c("loglik", "aicc", "sigma2")))
t0 <- Sys.time()
for (i in seq_len(n)) {
  y <- X[i, ]; y <- ts(y[!is.na(y)], frequency = freq)
  step <- 1
  fit <- tryCatch(auto.arima(y, lambda = 0), error = function(e) NULL)
  if (is.null(fit)) { step <- 2; fit <- tryCatch(auto.arima(y), error = function(e) NULL) }
  if (is.null(fit)) { step <- 3; fit <- auto.arima(y, seasonal = FALSE) }
  FC[i, ] <- forecast(fit, h = h)$mean
  o <- arimaorder(fit)
  cf <- names(coef(fit))
  ORD[i, ] <- c(o["p"], o["d"], o["q"], if (length(o) > 3) o[c("P", "D", "Q", "Frequency")] else c(0, 0, 0, NA),
                "drift" %in% cf, "intercept" %in% cf, step)
  FIT[i, ] <- c(fit$loglik, fit$aicc, fit$sigma2)
}
secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
write.table(FC, paste0(out, "_forecast.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
write.table(ORD, paste0(out, "_order.csv"), sep = ",", row.names = FALSE, col.names = TRUE)
write.table(FIT, paste0(out, "_fit.csv"), sep = ",", row.names = FALSE, col.names = TRUE)
cat(sprintf("done: %d series in %.1f s, forecast %s\n", n, secs, as.character(packageVersion("forecast"))))
lab <- apply(ORD, 1, function(r) sprintf("(%d,%d,%d)(%s,%s,%s)%s%s", r[1], r[2], r[3], r[4], r[5], r[6],
                                          if (r[8] == 1) "+drift" else "", if (r[9] == 1) "+mean" else ""))
print(sort(table(lab), decreasing = TRUE)[1:min(12, length(unique(lab)))])
cat("fallback steps:", table(ORD[, "step"]), "\n")
