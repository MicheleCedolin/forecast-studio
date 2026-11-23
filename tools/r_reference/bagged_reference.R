# R reference for the C++ bagged_ets: set.seed(seed) before each series, then
# forecast(baggedETS(msts(y, periods)), h)$mean (forecast 9.0.2).
# Usage: Rscript bagged_reference.R train.csv "7" h out.csv [seed] [first_n_series]
#        Rscript bagged_reference.R --boot train.csv freq row seed num out.csv   (bld.mbb.bootstrap of one series)
suppressPackageStartupMessages(library(forecast))
a <- commandArgs(trailingOnly = TRUE)
if (a[1] == "--boot") {
  X <- as.matrix(read.csv(a[2], header = FALSE)); r <- as.integer(a[4])
  y <- ts(X[r, !is.na(X[r, ])], frequency = as.numeric(a[3]))
  set.seed(as.integer(a[5]))
  b <- bld.mbb.bootstrap(y, as.integer(a[6]))
  M <- do.call(rbind, lapply(b, as.numeric))
  writeLines(apply(M, 1, function(r) paste(sprintf("%.17g", r), collapse = ",")), a[7])
  quit(save = "no")
}
X <- as.matrix(read.csv(a[1], header = FALSE)); per <- as.numeric(strsplit(a[2], ",")[[1]]); h <- as.integer(a[3])
seed <- if (length(a) >= 5) as.integer(a[5]) else 0L
rows <- if (length(a) >= 6) seq_len(min(nrow(X), as.integer(a[6]))) else seq_len(nrow(X))
FC <- matrix(NA, length(rows), h)
t0 <- Sys.time()
for (i in rows) {
  y <- msts(X[i, !is.na(X[i, ])], seasonal.periods = per)
  set.seed(seed)
  f <- tryCatch(forecast(baggedETS(y), h = h), error = function(e) NULL)
  if (!is.null(f)) FC[i, ] <- f$mean
}
cat(sprintf("R baggedETS: %.1f s\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
writeLines(apply(FC, 1, function(r) paste(sprintf("%.17g", r), collapse = ",")), a[4])
