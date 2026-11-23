# R reference for the small-sample branch of ets() (HoltWintersZZ + L-BFGS-B): for each row of train.csv,
# forecast(ets(ts(y, frequency = fr), model = model), h)$mean, written with 17 significant digits (NA on error).
# Usage: Rscript ets_small_reference.R train.csv fr h out.csv [model]
suppressPackageStartupMessages(library(forecast))
a <- commandArgs(trailingOnly = TRUE)
X <- as.matrix(read.csv(a[1], header = FALSE)); fr <- as.numeric(a[2]); h <- as.integer(a[3])
model <- if (length(a) >= 5) a[5] else "ZZZ"
out <- character(nrow(X))
for (i in seq_len(nrow(X))) {
  y <- ts(X[i, !is.na(X[i, ])], frequency = fr)
  f <- tryCatch(suppressWarnings(forecast(ets(y, model = model), h = h)$mean), error = function(e) rep(NA_real_, h))
  out[i] <- paste(sprintf("%.17g", f), collapse = ",")
}
writeLines(out, a[4])
