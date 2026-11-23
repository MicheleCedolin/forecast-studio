# R forecasts for the closed-form benchmarks: forecast::snaive, naive and rwf(drift = TRUE).
# Usage: Rscript basic_reference.R <train.csv[.gz]> <frequency> <h> <out_prefix>   (writes <out_prefix>_<method>.csv)
suppressMessages(library(forecast))
a <- commandArgs(TRUE); f <- a[1]; ppy <- as.integer(a[2]); fh <- as.integer(a[3]); out <- a[4]
con <- if (grepl("gz$", f)) gzfile(f) else file(f)
X <- as.matrix(read.csv(con, header = FALSE, na.strings = c("nan", "NaN", "NA", "")))
fs <- list(snaive = function(x) snaive(x, h = fh)$mean, naive = function(x) naive(x, h = fh)$mean,
           drift = function(x) rwf(x, h = fh, drift = TRUE)$mean)
for (k in names(fs)) { M <- t(sapply(1:nrow(X), function(i) { y <- unname(X[i, ]); as.numeric(fs[[k]](ts(y[!is.na(y)], frequency = ppy))) }))
  write.table(M, paste0(out, "_", k, ".csv"), sep = ",", row.names = FALSE, col.names = FALSE) }
