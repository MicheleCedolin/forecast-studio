# R reference for the C++ port of R's svd() (cpp/src/lapack_svd.cpp) and of forecast::fourier (autoarima.cpp), and for
# dhr_arima with K > 1 (tests/test_cpp_engine.py::test_r_svd_and_fourier_bit_identical_to_r, ::test_dhr_arima_k3_reproduces_r).
# Writes tests/golden/r_svd/{x,v,d}<i>.csv (the matrix, svd(x)$v and svd(x)$d with all 17 digits) and
# tests/golden/nn5_weekly/r_dhr_k3.csv (forecast(auto.arima(y, xreg = fourier(y, 3), seasonal = FALSE), xreg = fourier(y, 3, 8)),
# first 20 series). R 4.6.2 (LAPACK 3.12.1), forecast 9.0.2.
# Usage: Rscript tools/r_reference/svd_reference.R tests/golden
suppressPackageStartupMessages(library(forecast))
g <- commandArgs(trailingOnly = TRUE)[1]
dir.create(file.path(g, "r_svd"), showWarnings = FALSE)
out <- function(M, f) writeLines(apply(as.matrix(M), 1, function(r) paste(sprintf("%.17g", r), collapse = ",")), f)
y105 <- ts(numeric(105), frequency = 52)
set.seed(1)
mats <- list(
  fourier(y105, K = 3),                                        # path 3, a five-fold singular value
  cbind(1, fourier(y105, K = 3)),                              # with intercept
  cbind(seq_len(105), fourier(y105, K = 5)),                   # with drift
  fourier(ts(numeric(735), frequency = 7), K = 3),             # daily
  fourier(ts(numeric(10), frequency = 7), K = 3),              # path 5 (n < 11k / 6)
  matrix(rnorm(8 * 5), 8, 5),                                  # path 5, random
  matrix(rnorm(60 * 7), 60, 7))                                # path 3, random
for (i in seq_along(mats)) {
  s <- svd(mats[[i]])
  out(mats[[i]], file.path(g, "r_svd", sprintf("x%d.csv", i - 1)))
  out(s$v, file.path(g, "r_svd", sprintf("v%d.csv", i - 1)))
  out(t(s$d), file.path(g, "r_svd", sprintf("d%d.csv", i - 1)))
}
W <- as.matrix(read.csv(file.path(g, "nn5_weekly", "train.csv"), header = FALSE))
F <- t(sapply(1:20, function(i) {
  y <- ts(W[i, !is.na(W[i, ])], frequency = 52)
  m <- auto.arima(y, xreg = fourier(y, K = 3), seasonal = FALSE)
  as.numeric(forecast(m, xreg = fourier(y, K = 3, h = 8))$mean)
}))
out(F, file.path(g, "nn5_weekly", "r_dhr_k3.csv"))
