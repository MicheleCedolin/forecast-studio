# Is R's SES fit at the optimum? For every series: forecast::ses() as the study calls it, the exact SSE optimum
# (initial level solved in closed form for each alpha, alpha searched on [1e-4, 0.9999]) and R's own ets() with alpha
# fixed at that optimum. Columns: alpha_r, l0_r, sse_r, fc_r, sse_at_r_alpha, alpha_opt, sse_opt, fc_opt, sse_r_fixed,
# fc_r_fixed. Result of 2026-11-23 (R 4.3.3, forecast 8.21.1): docs/VALIDATION.md 1e.
# Usage: Rscript ses_optimum_reference.R <train.csv[.gz]> <frequency> <h> <out.csv>
suppressMessages(library(forecast))
a <- commandArgs(TRUE); f <- a[1]; freq <- as.numeric(a[2]); h <- as.integer(a[3]); out <- a[4]
con <- if (grepl("gz$", f)) gzfile(f) else file(f)
X <- as.matrix(read.csv(con, header = FALSE, na.strings = c("nan", "NaN", "NA", "")))
prof <- function(y, alpha) {            # SSE with the initial level solved exactly for this alpha
  n <- length(y); e0 <- numeric(n); l <- 0
  for (t in 1:n) { e0[t] <- y[t] - l; l <- l + alpha * e0[t] }
  c <- (1 - alpha)^(0:(n - 1)); l0 <- sum(e0 * c) / sum(c * c)
  e <- e0 - l0 * c; l <- l0; for (t in 1:n) l <- l + alpha * (y[t] - l)
  c(sse = sum(e^2), l0 = l0, fc = l)
}
res <- t(sapply(1:nrow(X), function(i) {
  y <- unname(X[i, ]); y <- y[!is.na(y)]; x <- ts(y, frequency = freq)
  fit <- ses(x, h = h); p <- fit$model$par
  sse_r <- sum(residuals(fit$model)^2)
  grid <- c(10^seq(-4, -1, length.out = 400), seq(0.1, 0.9999, length.out = 600))
  s <- sapply(grid, function(al) prof(y, al)["sse"]); k <- which.min(s)
  o <- optimize(function(al) prof(y, al)["sse"], c(grid[max(k - 1, 1)], grid[min(k + 1, length(grid))]), tol = 1e-12)
  al <- if (o$objective < s[k]) o$minimum else grid[k]; b <- prof(y, al)
  chk <- prof(y, p["alpha"])              # my recursion at R's alpha (profiled l0) -> must be <= sse_r
  fx <- tryCatch(ets(x, "ANN", alpha = max(al, 1.01e-4), opt.crit = "mse"), error = function(e) NULL)
  c(alpha_r = unname(p["alpha"]), l0_r = unname(p["l"]), sse_r = sse_r, fc_r = as.numeric(fit$mean[1]),
    sse_at_r_alpha = unname(chk["sse"]), alpha_opt = al, sse_opt = unname(b["sse"]), fc_opt = unname(b["fc"]),
    sse_r_fixed = if (is.null(fx)) NA else sum(residuals(fx)^2), fc_r_fixed = if (is.null(fx)) NA else as.numeric(forecast(fx, h = 1)$mean))
}))
write.csv(res, out, row.names = FALSE)
