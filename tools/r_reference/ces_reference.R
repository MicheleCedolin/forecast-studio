# smooth::auto.ces (smooth 4.5.2) forecasts, for comparison with the app's auto_ces (statsforecast AutoCES); the two are
# not equivalent, see docs/VALIDATION.md 1e. Usage: Rscript ces_reference.R <train.csv[.gz]> <frequency> <h> <out_prefix>
if (dir.exists("rlib")) .libPaths(c("rlib", .libPaths()))   # optional local package library; suppressMessages(library(smooth))
a <- commandArgs(TRUE); f <- a[1]; ppy <- as.integer(a[2]); fh <- as.integer(a[3]); out <- a[4]
con <- if (grepl("gz$", f)) gzfile(f) else file(f)
X <- as.matrix(read.csv(con, header = FALSE, na.strings = c("nan", "NaN", "NA", ""))); mods <- character(nrow(X)); t0 <- Sys.time()
M <- t(sapply(1:nrow(X), function(i) { y <- unname(X[i, ]); y <- y[!is.na(y)]
  tryCatch({ setTimeLimit(elapsed = 120, transient = TRUE); m <- auto.ces(ts(y, frequency = ppy), h = fh, holdout = FALSE, silent = TRUE)
    mods[i] <<- as.character(modelType(m)); as.numeric(m$forecast) }, error = function(e) { mods[i] <<- conditionMessage(e); rep(NA_real_, fh) }) }))
write.table(M, paste0(out, "_ces.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
cat("NA:", sum(is.na(M[, 1])), "sec:", round(as.numeric(difftime(Sys.time(), t0, units = "secs"))), "\n"); print(table(mods))
