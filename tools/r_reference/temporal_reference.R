# R reference for the C++ thief / mapa: for each row of train.csv, on msts(y, periods) as the R engine builds it,
#   thief: thief::thief(y, h = h, usemodel = "ets")$mean
#   mapa:  MAPA::mapa(y, fh = h, ifh = 0, comb = "w.mean", hybrid = TRUE, model = "ZZZ", type = "ets")$outfor
# written with 17 significant digits (NA on error).
# Usage: Rscript temporal_reference.R train.csv "12" h out.csv thief|mapa
suppressPackageStartupMessages({library(forecast); library(thief); library(MAPA)})
a <- commandArgs(trailingOnly = TRUE)
X <- as.matrix(read.csv(a[1], header = FALSE)); per <- as.numeric(strsplit(a[2], ",")[[1]]); h <- as.integer(a[3])
method <- a[5]
out <- character(nrow(X))
t0 <- Sys.time()
for (i in seq_len(nrow(X))) {
  y <- msts(X[i, !is.na(X[i, ])], seasonal.periods = per)
  f <- tryCatch(suppressWarnings(
         if (method == "thief") thief(y, h = h, usemodel = "ets")$mean
         else mapa(y, fh = h, ifh = 0, comb = "w.mean", hybrid = TRUE, model = "ZZZ", type = "ets")$outfor),
       error = function(e) rep(NA_real_, h))
  out[i] <- paste(sprintf("%.17g", f), collapse = ",")
}
cat(sprintf("R %s: %.1f s\n", method, as.numeric(difftime(Sys.time(), t0, units = "secs"))))
writeLines(out, a[4])
