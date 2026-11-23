# CRAN counterparts of methods that the app takes from statsforecast: forecast::croston, tsintermittent (crost, tsb,
# imapa; tsintermittent 2.0) and forecTheta::dotm (forecTheta 3.0.3). The fixed-parameter calls (w = 0.1, init = "naive")
# are the ones that equal statsforecast's CrostonClassic / CrostonSBA / TSB; see docs/VALIDATION.md 1e.
# Usage: Rscript cran_reference.R <train.csv[.gz]> <frequency> <h> <out_prefix> <method,method,...>
#   methods: fc_croston, ti_croston, ti_sba, ti_croston_opt, ti_tsb, ti_imapa, ft_dotm
if (dir.exists("rlib")) .libPaths(c("rlib", .libPaths()))   # optional local package library; suppressMessages({library(forecast); library(tsintermittent); library(forecTheta)})
a <- commandArgs(TRUE); f <- a[1]; ppy <- as.integer(a[2]); fh <- as.integer(a[3]); out <- a[4]; which <- strsplit(a[5], ",")[[1]]
con <- if (grepl("gz$", f)) gzfile(f) else file(f)
X <- as.matrix(read.csv(con, header = FALSE, na.strings = c("nan", "NaN", "NA", "")))
fs <- list(
  fc_croston = function(y) croston(ts(y), h = fh, alpha = 0.1)$mean,
  ti_croston = function(y) crost(y, h = fh, w = c(0.1, 0.1), init = "naive", type = "croston", init.opt = FALSE)$frc.out,
  ti_sba     = function(y) crost(y, h = fh, w = c(0.1, 0.1), init = "naive", type = "sba", init.opt = FALSE)$frc.out,
  ti_croston_opt = function(y) crost(y, h = fh, init = "naive", type = "croston", cost = "mse", init.opt = FALSE)$frc.out,
  ti_tsb     = function(y) tsb(y, h = fh, w = c(0.1, 0.1), init = "naive", init.opt = FALSE)$frc.out,
  ti_imapa   = function(y) imapa(y, h = fh)$frc.out,
  ft_dotm    = function(y) dotm(ts(y, frequency = ppy), h = fh)$mean)
for (k in which) { t0 <- Sys.time()
  M <- t(sapply(1:nrow(X), function(i) { y <- unname(X[i, ]); y <- y[!is.na(y)]
    r <- tryCatch({ setTimeLimit(elapsed = 30, transient = TRUE); as.numeric(fs[[k]](y)) }, error = function(e) rep(NA_real_, fh))
    if (length(r) != fh) rep(NA_real_, fh) else r }))
  write.table(M, paste0(out, "_", k, ".csv"), sep = ",", row.names = FALSE, col.names = FALSE)
  cat(k, "NA series:", sum(is.na(M[, 1])), "sec:", round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1), "\n") }
