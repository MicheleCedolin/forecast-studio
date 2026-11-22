# Fixed-order stats::arima reference (checks the C++ port of stats::arima before auto.arima).
# Usage: Rscript arima_fixed_reference.R <out.csv>
# One row per (case, series, model): loglik, aic, sigma2, coefficients, first 8 forecasts.
G <- "tests/golden"
rd <- function(case) { f <- file.path(G, case, "train.csv"); if (!file.exists(f)) f <- gzfile(paste0(f, ".gz")); as.matrix(read.csv(f, header = FALSE)) }
specs <- list(
  list(case = "nn5_weekly", p = 0, d = 1, q = 1, P = 0, D = 0, Q = 0, s = 52, drift = TRUE,  mean = TRUE,  method = "CSS-ML"),
  list(case = "nn5_weekly", p = 2, d = 1, q = 1, P = 0, D = 1, Q = 0, s = 52, drift = FALSE, mean = TRUE,  method = "CSS-ML"),
  list(case = "nn5_weekly", p = 1, d = 0, q = 0, P = 0, D = 0, Q = 0, s = 52, drift = FALSE, mean = TRUE,  method = "CSS-ML"),
  list(case = "nn5_weekly", p = 1, d = 1, q = 2, P = 0, D = 0, Q = 0, s = 52, drift = FALSE, mean = TRUE,  method = "CSS"),
  list(case = "nn5_daily",  p = 1, d = 0, q = 0, P = 1, D = 1, Q = 0, s = 7,  drift = TRUE,  mean = TRUE,  method = "CSS-ML"),
  list(case = "nn5_daily",  p = 1, d = 0, q = 1, P = 2, D = 0, Q = 0, s = 7,  drift = FALSE, mean = TRUE,  method = "CSS-ML"),
  list(case = "nn5_daily",  p = 1, d = 1, q = 1, P = 0, D = 0, Q = 2, s = 7,  drift = FALSE, mean = TRUE,  method = "CSS-ML"),
  list(case = "nn5_daily",  p = 2, d = 0, q = 2, P = 2, D = 1, Q = 0, s = 7,  drift = TRUE,  mean = TRUE,  method = "CSS"))
rows <- list()
for (sp in specs) {
  X <- rd(sp$case)
  for (i in 1:5) {
    y <- X[i, ]; y <- y[!is.na(y)]
    xreg <- if (sp$drift) cbind(drift = seq_along(y)) else NULL
    fit <- tryCatch(stats::arima(y, order = c(sp$p, sp$d, sp$q), seasonal = list(order = c(sp$P, sp$D, sp$Q), period = sp$s),
                                 xreg = xreg, include.mean = sp$mean, method = sp$method), error = function(e) NULL)
    if (is.null(fit)) { rows[[length(rows) + 1]] <- c(sp$case, i, sp$p, sp$d, sp$q, sp$P, sp$D, sp$Q, sp$s, sp$drift, sp$mean, sp$method, "ERROR", rep(NA, 27)); next }
    nx <- if (sp$drift) cbind(drift = length(y) + 1:8) else NULL
    fc <- predict(fit, n.ahead = 8, newxreg = nx)$pred
    cf <- c(coef(fit), rep(NA, 16))[1:16]
    vd <- suppressWarnings(any(is.nan(sqrt(diag(fit$var.coef)))))
    rows[[length(rows) + 1]] <- c(sp$case, i, sp$p, sp$d, sp$q, sp$P, sp$D, sp$Q, sp$s, sp$drift, sp$mean, sp$method, "OK",
                                  fit$loglik, ifelse(is.na(fit$aic), NA, fit$aic), fit$sigma2, vd, cf, fc)
  }
}
M <- do.call(rbind, rows)
colnames(M) <- c("case", "series", "p", "d", "q", "P", "D", "Q", "s", "drift", "mean", "method", "status", "loglik", "aic", "sigma2", "nanvar",
                 paste0("c", 1:16), paste0("f", 1:8))
write.csv(M, commandArgs(trailingOnly = TRUE)[1], row.names = FALSE)
cat("rows:", nrow(M), "\n")
