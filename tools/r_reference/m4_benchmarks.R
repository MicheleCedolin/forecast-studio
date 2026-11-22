# Reference forecasts for Naive2 and Comb S-H-D, following the official M4 benchmark code
# (Makridakis, Spiliotis & Assimakopoulos, M4 competition GitHub repository, Benchmarks_and_Evaluation.R).
# Usage: Rscript m4_benchmarks.R <train.csv[.gz]> <period m> <horizon h> <out_prefix>
# train: one series per row, comma separated, no header. Writes <out_prefix>_naive2.csv, _comb.csv, _ses.csv,
# _holt.csv, _damped.csv (one row per series, h columns) and <out_prefix>_seasonal.csv (TRUE/FALSE per series).
suppressPackageStartupMessages(library(forecast))

args <- commandArgs(trailingOnly = TRUE)
train_file <- args[1]; ppy <- as.integer(args[2]); fh <- as.integer(args[3]); out <- args[4]

SeasonalityTest <- function(input, ppy) {           # verbatim logic of the M4 benchmark code
  tcrit <- 1.645
  if (length(input) < 3 * ppy) {
    test_seasonal <- FALSE
  } else {
    xacf <- acf(input, plot = FALSE)$acf[-1, 1, 1]
    clim <- tcrit / sqrt(length(input)) * sqrt(cumsum(c(1, 2 * xacf^2)))
    test_seasonal <- (abs(xacf[ppy]) > clim[ppy])
    if (is.na(test_seasonal) == TRUE) test_seasonal <- FALSE
  }
  test_seasonal
}

con <- if (grepl("\\.gz$", train_file)) gzfile(train_file) else file(train_file)
X <- as.matrix(read.csv(con, header = FALSE))
n <- nrow(X)
res <- list(naive2 = matrix(NA, n, fh), ses = matrix(NA, n, fh), holt = matrix(NA, n, fh),
            damped = matrix(NA, n, fh), comb = matrix(NA, n, fh))
seasonal <- logical(n)
for (i in seq_len(n)) {
  y <- X[i, ]; y <- y[!is.na(y)]
  input <- ts(y, frequency = ppy)
  ST <- FALSE
  if (ppy > 1) ST <- SeasonalityTest(input, ppy)
  if (ST) {
    Dec <- decompose(input, type = "multiplicative")
    des_input <- input / Dec$seasonal
    SIout <- head(rep(Dec$seasonal[(length(Dec$seasonal) - ppy + 1):length(Dec$seasonal)], fh), fh)
  } else {
    des_input <- input; SIout <- rep(1, fh)
  }
  seasonal[i] <- ST
  res$naive2[i, ] <- naive(des_input, h = fh)$mean * SIout
  res$ses[i, ]    <- ses(des_input, h = fh)$mean * SIout
  res$holt[i, ]   <- holt(des_input, h = fh, damped = FALSE)$mean * SIout
  res$damped[i, ] <- holt(des_input, h = fh, damped = TRUE)$mean * SIout
  res$comb[i, ]   <- (res$ses[i, ] + res$holt[i, ] + res$damped[i, ]) / 3
}
for (k in names(res)) write.table(res[[k]], paste0(out, "_", k, ".csv"), sep = ",", row.names = FALSE, col.names = FALSE)
write.table(seasonal, paste0(out, "_seasonal.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
cat(sprintf("done: %d series, seasonal %d, forecast %s\n", n, sum(seasonal), as.character(packageVersion("forecast"))))
