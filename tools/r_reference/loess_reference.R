# R reference for the C++ / Python loess port: for each row of train.csv (series y of length n),
# loess(y ~ seq_along(y), span = 6 / n, degree = 1)$fitted, written with 17 significant digits.
# Usage: Rscript loess_reference.R train.csv out.csv
a <- commandArgs(trailingOnly = TRUE)
X <- as.matrix(read.csv(a[1], header = FALSE))
out <- character(nrow(X))
for (i in seq_len(nrow(X))) {
  y <- X[i, !is.na(X[i, ])]
  trend <- seq_along(y)
  f <- tryCatch(suppressWarnings(loess(ts(y, frequency = 1) ~ trend, span = 6 / length(y), degree = 1))$fitted,
                error = function(e) rep(NA_real_, length(y)))
  out[i] <- paste(sprintf("%.17g", f), collapse = ",")
}
writeLines(out, a[2])
