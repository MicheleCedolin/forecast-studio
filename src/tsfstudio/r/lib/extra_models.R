# Further local models for the R engine (added 2026-11-25): the R counterparts of methods that the C++ and Python
# engines already offer. local_models.R holds the nine methods of the original study and is left as it is.
#
# Two kinds, as stated per function:
#   REFERENCE  the C++ method is a port of exactly this R call and was checked against it (cpp/README.md,
#              tools/r_reference/*.R); the R engine now runs that call.
#   LIBRARY    an R library function for the same method; the C++ / Python engines follow another implementation
#              (statsforecast), so forecasts are close, not equal.
#
# Each function: (time_series: ts/msts as worker.R builds it, forecast_horizon, params) -> numeric vector of length h.
# Unlike local_models.R these functions do not catch errors: worker.R's run_local records the failure and gives the
# series the seasonal naive forecast, the same rule as the C++ and Python engines.

# R packages a method needs beyond forecast (worker.R stops with a clear message when one is missing;
# r_engine.py reads the same table to list only the methods that can run).
extra_packages <- list(thief = "thief", mapa = "MAPA", auto_ces = "smooth", croston_sba = "tsintermittent", croston_optimized = "tsintermittent", tsb = "tsintermittent", imapa = "tsintermittent", dotm = "forecTheta", scum = c("smooth", "forecTheta"), stm = "forecTheta", otm = "forecTheta", dstm = "forecTheta")

extra_methods <- c("naive", "drift", "historic_average", "naive2", "comb_shd", "holt_ets", "holt_winters", "stl_ets",
                   "stl_arima", "bagged_ets", "thief", "mapa", "comb_eat", "comb_eat_median", "auto_ces",
                   "croston_classic", "tslm", "croston_sba", "croston_optimized", "tsb", "imapa", "dotm", "scum",
                   "stm", "otm", "dstm", "bats", "struct_ts", "dshw", "nnetar")

# msts() keeps a single period as the ts frequency; with several periods the frequency is the largest one.
# Single-season methods use floor() of it, as forecast::ets and the other engines do (py_models.r_frequency).
.freq1 <- function(x) max(1, floor(frequency(x)))
.plain <- function(x, m = .freq1(x)) stats::ts(as.numeric(x), frequency = m)
.par <- function(params, key, default) if (is.null(params[[key]])) default else params[[key]]


# ---- LIBRARY: simple benchmarks (forecast package) ------------------------------------------------------------
get_naive_forecasts <- function(time_series, forecast_horizon, params = list())
  as.numeric(forecast::naive(time_series, h = forecast_horizon)$mean)

get_drift_forecasts <- function(time_series, forecast_horizon, params = list())
  as.numeric(forecast::rwf(time_series, h = forecast_horizon, drift = TRUE)$mean)

get_historic_average_forecasts <- function(time_series, forecast_horizon, params = list())
  as.numeric(forecast::meanf(time_series, h = forecast_horizon)$mean)


# ---- REFERENCE: M4 benchmarks (the official benchmark code; tools/r_reference/m4_benchmarks.R) ----------------
.m4_seasonality_test <- function(input, ppy) {
  tcrit <- 1.645
  if (length(input) < 3 * ppy) return(FALSE)
  xacf <- stats::acf(input, plot = FALSE)$acf[-1, 1, 1]
  clim <- tcrit / sqrt(length(input)) * sqrt(cumsum(c(1, 2 * xacf^2)))
  test_seasonal <- (abs(xacf[ppy]) > clim[ppy])
  if (is.na(test_seasonal)) FALSE else test_seasonal
}

.m4_deseasonalise <- function(time_series, fh) {
  ppy <- .freq1(time_series)
  input <- .plain(time_series, ppy)
  if (ppy > 1 && .m4_seasonality_test(input, ppy)) {
    dec <- stats::decompose(input, type = "multiplicative")
    list(des = input / dec$seasonal,
         si = head(rep(dec$seasonal[(length(dec$seasonal) - ppy + 1):length(dec$seasonal)], fh), fh))
  } else {
    list(des = input, si = rep(1, fh))
  }
}

get_naive2_forecasts <- function(time_series, forecast_horizon, params = list()) {
  d <- .m4_deseasonalise(time_series, forecast_horizon)
  as.numeric(forecast::naive(d$des, h = forecast_horizon)$mean) * d$si
}

get_comb_shd_forecasts <- function(time_series, forecast_horizon, params = list()) {
  d <- .m4_deseasonalise(time_series, forecast_horizon)
  f_ses <- as.numeric(forecast::ses(d$des, h = forecast_horizon)$mean)
  f_holt <- as.numeric(forecast::holt(d$des, h = forecast_horizon, damped = FALSE)$mean)
  f_damp <- as.numeric(forecast::holt(d$des, h = forecast_horizon, damped = TRUE)$mean)
  (f_ses + f_holt + f_damp) / 3 * d$si
}


# ---- REFERENCE: ETS with a fixed form (tools/r_reference/hw_reference.R) --------------------------------------
# holt_ets: ets(y, model = "AAN"); the damped and the undamped trend are compared by AICc. error_type "M" -> "MAN".
get_holt_ets_forecasts <- function(time_series, forecast_horizon, params = list()) {
  model <- if (identical(.par(params, "error_type", "A"), "M")) "MAN" else "AAN"
  as.numeric(forecast::forecast(forecast::ets(time_series, model = model), h = forecast_horizon)$mean)
}

# holt_winters: ets(y, model = "AAA") on one seasonal period. As in the other engines the period must be at most
# max_season (ets itself refuses more than 24) and the series must hold two full periods; otherwise the series has no
# usable season and falls back, with the reason.
get_holt_winters_forecasts <- function(time_series, forecast_horizon, params = list()) {
  m <- .freq1(time_series)
  max_season <- as.numeric(.par(params, "max_season", 24))
  if (m <= 1) stop("Nonseasonal data")
  if (m > max_season) stop(sprintf("period %d > max_season (%g)", m, max_season))
  if (length(time_series) < 2 * m) stop(sprintf("period %d needs at least %d observations", m, 2 * m))
  model <- if (identical(.par(params, "error_type", "A"), "M")) "MAM" else "AAA"
  as.numeric(forecast::forecast(forecast::ets(.plain(time_series, m), model = model), h = forecast_horizon)$mean)
}


# ---- REFERENCE: STL decomposition + ETS / ARIMA (tools/r_reference/stlf_reference.R) --------------------------
get_stl_ets_forecasts <- function(time_series, forecast_horizon, params = list())
  as.numeric(forecast::stlf(time_series, h = forecast_horizon)$mean)

get_stl_arima_forecasts <- function(time_series, forecast_horizon, params = list())
  as.numeric(forecast::forecast(forecast::mstl(time_series), method = "arima", h = forecast_horizon)$mean)


# ---- REFERENCE: bagged ETS (tools/r_reference/bagged_reference.R) ---------------------------------------------
# set.seed(seed) before every series, then forecast(baggedETS(y), h). The Mersenne-Twister generator is selected
# explicitly: with several workers the worker process runs L'Ecuyer streams, which would change the bootstrap.
get_bagged_ets_forecasts <- function(time_series, forecast_horizon, params = list()) {
  n_boot <- as.integer(.par(params, "n_boot", 100))
  seed <- as.integer(.par(params, "seed", 0))
  RNGkind("Mersenne-Twister", "Inversion", "Rejection")
  set.seed(seed)
  fit <- forecast::baggedETS(time_series, bootstrapped_series = forecast::bld.mbb.bootstrap(time_series, n_boot))
  as.numeric(forecast::forecast(fit, h = forecast_horizon)$mean)
}


# ---- REFERENCE: temporal aggregation (tools/r_reference/temporal_reference.R) ---------------------------------
get_thief_forecasts <- function(time_series, forecast_horizon, params = list())
  as.numeric(suppressWarnings(thief::thief(time_series, h = forecast_horizon, usemodel = "ets"))$mean)

get_mapa_forecasts <- function(time_series, forecast_horizon, params = list())
  as.numeric(suppressWarnings(MAPA::mapa(time_series, fh = forecast_horizon, ifh = 0, comb = "w.mean", hybrid = TRUE,
                                         model = "ZZZ", type = "ets"))$outfor)


# ---- Combinations of the R engine's own ETS, ARIMA and Theta --------------------------------------------------
# The rule of the other engines: a component that fails or is not finite is left out; fewer than two -> error.
# ARIMA is the study's chain (local_models.R). The C++ engine combines its ports of the same R functions.
.comb_eat <- function(time_series, h, fun) {
  parts <- list(
    ets = function() forecast::forecast(forecast::ets(time_series), h = h)$mean,
    arima = function() {
      fit <- tryCatch(forecast::auto.arima(time_series, lambda = 0), error = function(e)
        tryCatch(forecast::auto.arima(time_series), error = function(e2) forecast::auto.arima(time_series, seasonal = FALSE)))
      forecast::forecast(fit, h = h)$mean
    },
    theta = function() forecast::thetaf(y = time_series, h = h)$mean)
  fc <- lapply(parts, function(p) tryCatch(as.numeric(p()), error = function(e) NULL))
  fc <- Filter(function(f) !is.null(f) && length(f) == h && all(is.finite(f)), fc)
  if (length(fc) < 2) stop("fewer than two of ETS, ARIMA and Theta gave a forecast")
  apply(do.call(rbind, fc), 2, fun)
}

get_comb_eat_forecasts <- function(time_series, forecast_horizon, params = list())
  .comb_eat(time_series, forecast_horizon, mean)

get_comb_eat_median_forecasts <- function(time_series, forecast_horizon, params = list())
  .comb_eat(time_series, forecast_horizon, stats::median)


# ---- LIBRARY: complex exponential smoothing (smooth::auto.ces; Svetunkov, Kourentzes & Ord 2022) --------------
# The other engines run statsforecast's AutoCES, a translation of this function. One seasonal period, used when it
# is at most max_season and the series holds two periods (the engines' rule); otherwise the non-seasonal model.
get_auto_ces_forecasts <- function(time_series, forecast_horizon, params = list()) {
  m <- .freq1(time_series)
  max_season <- as.numeric(.par(params, "max_season", 24))
  if (m > max_season || length(time_series) < 2 * m) m <- 1
  model <- .par(params, "model", "Z")
  seasonality <- switch(model, Z = c("none", "simple", "partial", "full"), N = "none", S = "simple", P = "partial",
                        F = "full", stop(paste0("unknown CES model: ", model)))
  if (m == 1) seasonality <- "none"
  y <- .plain(time_series, m)
  fit <- if (length(seasonality) > 1) smooth::auto.ces(y, seasonality = seasonality, h = forecast_horizon, holdout = FALSE, silent = TRUE)
         else smooth::ces(y, seasonality = seasonality, h = forecast_horizon, holdout = FALSE, silent = TRUE)
  as.numeric(fit$forecast)
}


# ---- LIBRARY: Croston's method (forecast::croston) ------------------------------------------------------------
# alpha = 0.1 for demand sizes and intervals, as statsforecast's CrostonClassic. A series without any demand gets
# zeros (the engines' rule).
get_croston_classic_forecasts <- function(time_series, forecast_horizon, params = list()) {
  if (all(as.numeric(time_series) == 0)) return(rep(0, forecast_horizon))
  as.numeric(forecast::croston(time_series, h = forecast_horizon, alpha = 0.1)$mean)
}


# ---- LIBRARY: intermittent demand (tsintermittent; Kourentzes & Petropoulos) -----------------------------------
# A series without any demand gets zeros (the engines' rule; the package refuses such a series).
# croston_sba: smoothing 0.1 for sizes and intervals, first values as initial states, nothing optimised, which is
# what statsforecast's CrostonSBA does. croston_optimized, imapa: the package's own optimisation, which is not the
# one statsforecast uses. tsb: smoothing 0.1 / 0.1 (the defaults of the other engines).
.intermittent <- function(time_series, h, fun) {
  x <- as.numeric(time_series)
  if (all(x == 0)) return(rep(0, h))
  as.numeric(fun(x)$frc.out)
}

get_croston_sba_forecasts <- function(time_series, forecast_horizon, params = list())
  .intermittent(time_series, forecast_horizon, function(x)
    tsintermittent::crost(x, h = forecast_horizon, w = c(0.1, 0.1), init = "naive", nop = 2, type = "sba", init.opt = FALSE))

get_croston_optimized_forecasts <- function(time_series, forecast_horizon, params = list())
  .intermittent(time_series, forecast_horizon, function(x) tsintermittent::crost(x, h = forecast_horizon, type = "croston"))

get_tsb_forecasts <- function(time_series, forecast_horizon, params = list())
  .intermittent(time_series, forecast_horizon, function(x)
    tsintermittent::tsb(x, h = forecast_horizon, w = c(0.1, 0.1), init = "naive", init.opt = FALSE))

get_imapa_forecasts <- function(time_series, forecast_horizon, params = list())
  .intermittent(time_series, forecast_horizon, function(x) tsintermittent::imapa(x, h = forecast_horizon))


# ---- LIBRARY: dynamic optimised Theta (forecTheta::dotm; Fiorucci et al. 2016) and SCUM ------------------------
# One seasonal period, used when the series holds two periods (the engines' rule). The multiplicative seasonal
# adjustment needs positive data; with zeros or negative values the additive one is used, as in the other engines.
.dotm <- function(time_series, h, params) {
  m <- .freq1(time_series)
  if (length(time_series) < 2 * m) m <- 1
  y <- .plain(time_series, m)
  s_type <- .par(params, "decomposition_type", "multiplicative")
  if (any(as.numeric(y) <= 0)) s_type <- "additive"
  as.numeric(forecTheta::dotm(y, h = h, s_type = s_type)$mean)
}

get_dotm_forecasts <- function(time_series, forecast_horizon, params = list())
  .dotm(time_series, forecast_horizon, params)

# SCUM (Petropoulos & Svetunkov 2020): the median of ETS, CES, ARIMA and DOTM, the four R functions the paper used.
# A component that fails or is not finite is left out; fewer than two -> error (the engines' rule).
get_scum_forecasts <- function(time_series, forecast_horizon, params = list()) {
  h <- forecast_horizon
  parts <- list(
    ets = function() forecast::forecast(forecast::ets(time_series), h = h)$mean,
    ces = function() get_auto_ces_forecasts(time_series, h),
    arima = function() {
      fit <- tryCatch(forecast::auto.arima(time_series, lambda = 0), error = function(e)
        tryCatch(forecast::auto.arima(time_series), error = function(e2) forecast::auto.arima(time_series, seasonal = FALSE)))
      forecast::forecast(fit, h = h)$mean
    },
    dotm = function() .dotm(time_series, h, list()))
  fc <- lapply(parts, function(p) tryCatch(as.numeric(p()), error = function(e) NULL))
  fc <- Filter(function(f) !is.null(f) && length(f) == h && all(is.finite(f)), fc)
  if (length(fc) < 2) stop("fewer than two of ETS, CES, ARIMA and DOTM gave a forecast")
  apply(do.call(rbind, fc), 2, stats::median)
}


# ---- REFERENCE: linear trend + seasonal dummies (forecast::tslm; cpp/src/tslm.cpp) ----------------------------
# tslm(y ~ trend + season) on one seasonal period. The dummies are used when the period is at most 366 and the series
# holds two full periods; otherwise the model is y ~ trend (tslm itself refuses a seasonal factor for frequency 1).
get_tslm_forecasts <- function(time_series, forecast_horizon, params = list()) {
  m <- .freq1(time_series)
  y <- .plain(time_series, m)
  if (length(y) < 3) stop("tslm needs at least 3 observations")
  fit <- if (m > 1 && m <= 366 && length(y) >= 2 * m) forecast::tslm(y ~ trend + season) else forecast::tslm(y ~ trend)
  as.numeric(forecast::forecast(fit, h = forecast_horizon)$mean)
}


# ---- LIBRARY: standard, optimised and dynamic standard Theta (forecTheta::stm / otm / dstm; Fiorucci et al. 2016) --
# The same rules as dotm above: one seasonal period when the series holds two periods, additive seasonal adjustment
# for data with zeros or negative values.
.theta_variant <- function(fun, time_series, h, params) {
  m <- .freq1(time_series)
  if (length(time_series) < 2 * m) m <- 1
  y <- .plain(time_series, m)
  s_type <- .par(params, "decomposition_type", "multiplicative")
  if (any(as.numeric(y) <= 0)) s_type <- "additive"
  as.numeric(fun(y, h = h, s_type = s_type)$mean)
}

get_stm_forecasts <- function(time_series, forecast_horizon, params = list())
  .theta_variant(forecTheta::stm, time_series, forecast_horizon, params)

get_otm_forecasts <- function(time_series, forecast_horizon, params = list())
  .theta_variant(forecTheta::otm, time_series, forecast_horizon, params)

get_dstm_forecasts <- function(time_series, forecast_horizon, params = list())
  .theta_variant(forecTheta::dstm, time_series, forecast_horizon, params)


# ---- REFERENCE: BATS without a seasonal period (forecast::bats; cpp/src/tbats.cpp) -----------------------------
# Box-Cox transformation, ARMA errors, trend and damping, each chosen by AIC, fitted to the series as a non-seasonal
# one: the model tbats() compares its seasonal candidates with. The seasonal BATS model is not offered (tbats covers
# seasonal data). use.parallel = FALSE: the default starts a cluster for series longer than 1000 values, with the
# same result.
get_bats_forecasts <- function(time_series, forecast_horizon, params = list()) {
  y <- stats::ts(as.numeric(time_series))
  if (forecast:::is.constant(y)) return(rep(y[length(y)], forecast_horizon))
  as.numeric(forecast::forecast(forecast::bats(y, use.parallel = FALSE), h = forecast_horizon)$mean)
}


# ---- REFERENCE: structural time series (stats::StructTS; cpp/src/struct_ts.cpp) --------------------------------
# The basic structural model (level, slope, seasonal) when the period is at most 24 and the series holds two full
# periods, otherwise the local linear trend model; variances by maximum likelihood (optim L-BFGS-B, as StructTS).
get_struct_ts_forecasts <- function(time_series, forecast_horizon, params = list()) {
  m <- .freq1(time_series)
  if (length(time_series) < 3) stop("StructTS needs at least 3 observations")
  seasonal <- m > 1 && m <= 24 && length(time_series) >= 2 * m
  y <- .plain(time_series, if (seasonal) m else 1)
  fit <- suppressWarnings(stats::StructTS(y, type = if (seasonal) "BSM" else "trend"))
  as.numeric(forecast::forecast(fit, h = forecast_horizon)$mean)
}


# ---- REFERENCE: double seasonal Holt-Winters (forecast::dshw; Taylor 2003; cpp/src/dshw.cpp) --------------------
# Needs two seasonal periods, the longer a multiple of the shorter, and positive data. With more than two periods the
# two shortest are used. The periods are passed as numbers, not integers: dshw()'s seasonal indices test
# identical(p %% 2, 0), which an integer period never satisfies.
get_dshw_forecasts <- function(time_series, forecast_horizon, params = list()) {
  per <- sort(floor(as.numeric(attr(time_series, "msts"))))
  if (length(per) < 2) stop("dshw needs two seasonal periods")
  as.numeric(forecast::dshw(as.numeric(time_series), period1 = per[1], period2 = per[2], h = forecast_horizon)$mean)
}


# ---- REFERENCE: neural network autoregression (forecast::nnetar with nnet::nnet; cpp/src/nnetar.cpp) -----------
# NNAR(p, P, k)[m] on one seasonal period with nnetar()'s defaults: p from ar(), P = 1, k = round((p + P + 1) / 2),
# `repeats` networks from random starting weights, averaged. set.seed(seed) before every series, with the
# Mersenne-Twister generator selected explicitly (see bagged ETS above), so a series gives the same forecast wherever
# it stands in the job. nnetar() needs the nnet package, which is part of every R installation (recommended).
get_nnetar_forecasts <- function(time_series, forecast_horizon, params = list()) {
  y <- .plain(time_series)
  if (length(y) < 3) stop("Not enough data to fit a model")
  if (forecast:::is.constant(y)) return(rep(y[length(y)], forecast_horizon))
  RNGkind("Mersenne-Twister", "Inversion", "Rejection")
  set.seed(as.integer(.par(params, "seed", 0)))
  fit <- forecast::nnetar(y, repeats = as.integer(.par(params, "repeats", 20)))
  as.numeric(forecast::forecast(fit, h = forecast_horizon)$mean)
}
