# Reference for the C++ ets: R forecast::ets() with default settings (model "ZZZ", AICc selection).
# Usage: Rscript ets_reference.R <train.csv[.gz]> <frequency> <horizon> <out_prefix>
# Writes <out>_forecast.csv (n x h), <out>_model.csv (one ETS(x,y,z) label per series), <out>_fit.csv
# (loglik, aicc, alpha, beta, gamma, phi per series).
suppressPackageStartupMessages(library(forecast))
a <- commandArgs(trailingOnly = TRUE)
f <- a[1]; freq <- as.numeric(a[2]); h <- as.integer(a[3]); out <- a[4]
con <- if (grepl("\\.gz$", f)) gzfile(f) else file(f)
X <- as.matrix(read.csv(con, header = FALSE))
n <- nrow(X)
FC <- matrix(NA, n, h); MOD <- character(n)
FIT <- matrix(NA, n, 6, dimnames = list(NULL, c("loglik", "aicc", "alpha", "beta", "gamma", "phi")))
t0 <- Sys.time()
for (i in seq_len(n)) {
  y <- X[i, ]; y <- ts(y[!is.na(y)], frequency = freq)
  fit <- suppressWarnings(ets(y))
  FC[i, ] <- forecast(fit, h = h)$mean
  MOD[i] <- fit$method
  p <- fit$par
  FIT[i, ] <- c(fit$loglik, fit$aicc, p["alpha"], if ("beta" %in% names(p)) p["beta"] else NA,
                if ("gamma" %in% names(p)) p["gamma"] else NA, if ("phi" %in% names(p)) p["phi"] else NA)
}
secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
write.table(FC, paste0(out, "_forecast.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
writeLines(MOD, paste0(out, "_model.csv"))
write.table(FIT, paste0(out, "_fit.csv"), sep = ",", row.names = FALSE, col.names = TRUE)
cat(sprintf("done: %d series in %.1f s, forecast %s\n", n, secs, as.character(packageVersion("forecast"))))
print(table(MOD))
