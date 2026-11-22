suppressPackageStartupMessages(library(forecast))
a <- commandArgs(trailingOnly = TRUE)
X <- as.matrix(read.csv(a[1], header = FALSE)); fr <- as.numeric(a[2]); h <- as.integer(a[3])
FC <- matrix(NA, nrow(X), h); MD <- character(nrow(X))
t0 <- Sys.time()
for (i in seq_len(nrow(X))) {
  y <- ts(X[i, !is.na(X[i, ])], frequency = fr)
  mdl <- if (length(a) >= 5) a[5] else "AAA"
  f <- tryCatch(ets(y, model = mdl), error = function(e) NULL)
  if (!is.null(f)) { FC[i, ] <- forecast(f, h = h)$mean; MD[i] <- f$method }
}
cat(sprintf("R ets: %.1f s\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
write.table(FC, a[4], sep = ",", row.names = FALSE, col.names = FALSE)
writeLines(MD, paste0(a[4], ".model"))
