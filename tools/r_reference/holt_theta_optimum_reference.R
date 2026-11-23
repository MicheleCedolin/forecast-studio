# Are R's Holt / damped Holt (Comb S-H-D) and Theta fits at the optimum? Writes R's forecasts (<out>_r_comb, _r_theta),
# the forecasts at the optimum of the same SSE objective (<out>_opt_comb: multi-start L-BFGS-B with the initial state
# solved by least squares; <out>_opt_theta: exact SES optimum) and <out>_stats.csv (SSE of R and of the optimum).
# Result of 2026-11-23 (R 4.3.3, forecast 8.21.1): docs/VALIDATION.md 1e.
# Usage: Rscript holt_theta_optimum_reference.R <train.csv[.gz]> <frequency> <h> <out_prefix>
suppressMessages(library(forecast))
a <- commandArgs(TRUE); f <- a[1]; ppy <- as.integer(a[2]); fh <- as.integer(a[3]); out <- a[4]
con <- if (grepl("gz$", f)) gzfile(f) else file(f)
X <- as.matrix(read.csv(con, header = FALSE, na.strings = c("nan", "NaN", "NA", "")))
sprof <- function(y, alpha) {                      # SES: exact initial level for this alpha
  n <- length(y); e0 <- numeric(n); l <- 0
  for (t in 1:n) { e0[t] <- y[t] - l; l <- l + alpha * e0[t] }
  cc <- (1 - alpha)^(0:(n - 1)); l0 <- sum(e0 * cc) / sum(cc * cc)
  l <- l0; for (t in 1:n) l <- l + alpha * (y[t] - l)
  list(sse = sum((e0 - l0 * cc)^2), fc = l)
}
sopt <- function(y) {
  grid <- c(10^seq(-4, -1, length.out = 300), seq(0.1, 0.9999, length.out = 400))
  s <- sapply(grid, function(al) sprof(y, al)$sse); k <- which.min(s)
  o <- optimize(function(al) sprof(y, al)$sse, c(grid[max(k - 1, 1)], grid[min(k + 1, length(grid))]), tol = 1e-12)
  al <- if (o$objective < s[k]) o$minimum else grid[k]; c(alpha = al, sprof(y, al))
}
hprof <- function(y, al, be, phi, h = 0) {         # Holt: exact initial level and slope for these parameters
  n <- length(y); l <- c(0, 1, 0); b <- c(0, 0, 1); E <- matrix(0, n, 3)
  for (t in 1:n) { e <- c(y[t], 0, 0) - (l + phi * b); E[t, ] <- e; l <- l + phi * b + al * e; b <- phi * b + be * e }
  A <- -E[, 2:3]; s0 <- qr.solve(A, E[, 1]); e <- E[, 1] - A %*% s0
  r <- list(sse = sum(e^2))
  if (h > 0) { L <- l[1] + s0[1] * l[2] + s0[2] * l[3]; B <- b[1] + s0[1] * b[2] + s0[2] * b[3]; r$fc <- L + cumsum(phi^(1:h)) * B }
  r
}
hopt <- function(y, damped, start) {
  obj <- function(p) { al <- p[1]; be <- max(1e-4, p[2] * al); ph <- if (damped) p[3] else 1
    tryCatch(hprof(y, al, be, ph)$sse, error = function(e) 1e300) }
  lo <- c(1e-4, 0, if (damped) 0.8); up <- c(0.9999, 1, if (damped) 0.98)
  S <- rbind(start, expand.grid(c(0.001, 0.05, 0.3, 0.8), c(0.01, 0.3, 0.9))); best <- NULL
  for (i in 1:nrow(S)) { p0 <- c(S[i, 1], S[i, 2], if (damped) (if (i == 1) start[3] else 0.9))
    p0 <- pmin(pmax(p0, lo), up)
    o <- tryCatch(optim(p0, obj, method = "L-BFGS-B", lower = lo, upper = up, control = list(factr = 1e3, maxit = 500)), error = function(e) NULL)
    if (!is.null(o) && (is.null(best) || o$value < best$value)) best <- o }
  p <- best$par; ph <- if (damped) p[3] else 1; be <- max(1e-4, p[2] * p[1])
  c(list(alpha = p[1], beta = be, phi = ph), hprof(y, p[1], be, ph, fh))
}
ST_test <- function(input, ppy) { if (length(input) < 3 * ppy) return(FALSE)
  xacf <- acf(input, plot = FALSE)$acf[-1, 1, 1]; clim <- 1.645 / sqrt(length(input)) * sqrt(cumsum(c(1, 2 * xacf^2)))
  isTRUE(abs(xacf[ppy]) > clim[ppy]) }
n <- nrow(X); M <- function() matrix(NA, n, fh)
R <- list(r_comb = M(), opt_comb = M(), r_theta = M(), opt_theta = M()); stats <- NULL
for (i in 1:n) {
  y <- unname(X[i, ]); y <- y[!is.na(y)]; input <- ts(y, frequency = ppy); N <- length(y)
  # --- Comb S-H-D (M4 benchmark code)
  if (ppy > 1 && ST_test(input, ppy)) { Dec <- decompose(input, type = "multiplicative"); des <- input / Dec$seasonal
    SI <- head(rep(Dec$seasonal[(N - ppy + 1):N], fh), fh) } else { des <- input; SI <- rep(1, fh) }
  d <- as.numeric(des)
  fs <- ses(des, h = fh); fh1 <- holt(des, h = fh, damped = FALSE); fd <- holt(des, h = fh, damped = TRUE)
  R$r_comb[i, ] <- (fs$mean + fh1$mean + fd$mean) / 3 * SI
  so <- sopt(d); ph <- fh1$model$par; pd <- fd$model$par
  ho <- hopt(d, FALSE, c(ph["alpha"], ph["beta"] / ph["alpha"])); do <- hopt(d, TRUE, c(pd["alpha"], pd["beta"] / pd["alpha"], pd["phi"]))
  R$opt_comb[i, ] <- (so$fc + ho$fc + do$fc) / 3 * SI
  # --- Theta (forecast::thetaf)
  ft <- thetaf(input, h = fh); R$r_theta[i, ] <- ft$mean
  x <- input; seas <- FALSE
  if (ppy > 1 && N > 2 * ppy) { r <- as.numeric(acf(x, lag.max = ppy, plot = FALSE)$acf)[-1]
    seas <- abs(r[ppy]) / sqrt((1 + 2 * sum(r[-ppy]^2)) / N) > qnorm(0.95) }
  if (seas) { dc <- decompose(x, type = "multiplicative"); x <- seasadj(dc) }
  xs <- as.numeric(x); to <- sopt(xs); b0 <- lsfit(0:(N - 1), xs)$coef[2]
  tf <- to$fc + b0 / 2 * (0:(fh - 1) + (1 - (1 - to$alpha)^N) / to$alpha)
  if (seas) tf <- tf * rep(tail(dc$seasonal, ppy), trunc(1 + fh / ppy))[1:fh]
  R$opt_theta[i, ] <- tf
  sr <- function(m) sum(residuals(m$model)^2)
  stats <- rbind(stats, c(ses_r = sr(fs), ses_opt = so$sse, ses_alpha_r = unname(fs$model$par["alpha"]), ses_alpha_opt = so$alpha,
    holt_r = sr(fh1), holt_opt = ho$sse, damped_r = sr(fd), damped_opt = do$sse, theta_alpha_opt = to$alpha))
}
for (k in names(R)) write.table(R[[k]], paste0(out, "_", k, ".csv"), sep = ",", row.names = FALSE, col.names = FALSE)
write.csv(stats, paste0(out, "_stats.csv"), row.names = FALSE)
