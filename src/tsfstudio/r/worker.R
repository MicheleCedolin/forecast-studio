#!/usr/bin/env Rscript
# Forecast Studio R işçisi.
#
# Kullanım: Rscript --vanilla worker.R <iş_klasörü>
#
# Girdi (Python tarafından yazılır, bkz. tsfstudio/engines/r_engine.py):
#   job.json    : task ("local"|"global"), method, horizon, horizons (veya null), seasonality, lengths,
#                 integer_conversion, seed, workers, lag, params
#   series.bin  : tüm eğitim serileri art arda, little-endian float64 (eksik = NaN)
# Çıktı (aynı klasör):
#   forecasts.bin  : (n_seri x horizon) float64, SÜTUN-ana sıralı (R'ın as.vector(matrix) düzeni)
#   summary.json   : süre, sürümler, yedek (fallback) kayıtları, NA→0 sayısı
#   progress.json  : ilerleme (çalışırken güncellenir)
#   error.txt      : hata olursa mesaj (çıkış kodu 2)

options(warn = 1)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("Kullanım: worker.R <iş_klasörü>")
job_dir <- normalizePath(args[1], mustWork = TRUE)

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
r_dir <- dirname(normalizePath(sub("^--file=", "", file_arg[1])))

suppressPackageStartupMessages({
  library(jsonlite)
  library(forecast)
})
source(file.path(r_dir, "lib", "local_models.R"))
source(file.path(r_dir, "lib", "extra_models.R"))
source(file.path(r_dir, "lib", "global_models.R"))


write_progress <- function(stage, done, total) {
  write_json(list(stage = stage, done = done, total = total, time = as.numeric(Sys.time())),
             file.path(job_dir, "progress.json"), auto_unbox = TRUE)
}

read_series <- function(job) {
  lengths <- as.numeric(job$lengths)
  con <- file(file.path(job_dir, "series.bin"), "rb")
  on.exit(close(con))
  all_vals <- readBin(con, what = "double", n = sum(lengths), size = 8, endian = "little")
  if (length(all_vals) != sum(lengths)) stop("series.bin beklenenden kısa/uzun.")
  ends <- cumsum(lengths)
  starts <- c(1, head(ends, -1) + 1)
  lapply(seq_along(ends), function(i) all_vals[starts[i]:ends[i]])
}

write_matrix <- function(m) {
  con <- file(file.path(job_dir, "forecasts.bin"), "wb")
  on.exit(close(con))
  writeBin(as.vector(m), con, size = 8, endian = "little")
}

versions <- function() {
  v <- list(R = R.version.string,
            forecast = as.character(packageVersion("forecast")),
            jsonlite = as.character(packageVersion("jsonlite")))
  for (p in unique(unlist(extra_packages))) if (p %in% loadedNamespaces()) v[[p]] <- as.character(packageVersion(p))
  v$catboost <- if (exists("failed_loading_catboost") && !failed_loading_catboost) as.character(packageVersion("catboost")) else NA_character_
  v
}


run_local <- function(job, series) {
  n <- length(series)
  H <- as.integer(job$horizon)
  hs <- if (is.null(job$horizons)) rep(H, n) else as.integer(job$horizons)
  method <- job$method
  fun_name <- paste0("get_", method, "_forecasts")
  if (!exists(fun_name, mode = "function")) stop(paste0("Bilinmeyen yerel yöntem: ", method))
  fun <- get(fun_name, mode = "function")
  seasonality <- as.numeric(job$seasonality)
  workers <- max(1L, as.integer(job$workers))
  if (.Platform$OS.type == "windows" && workers > 1L) {
    # parallel::mclapply Windows'ta mc.cores > 1'i desteklemez (hata verir). PSOCK kümesi henüz eklenmedi -> tek çekirdek.
    message("UYARI: Windows'ta R motoru tek çekirdekte çalışır (mclapply desteklenmiyor); işçi sayısı 1'e indirildi.")
    workers <- 1L
  }
  integer_conversion <- isTRUE(job$integer_conversion)

  # Methods of lib/extra_models.R: (series, h, params); an error gives the series the seasonal naive forecast and is
  # counted. A missing R package stops the job at once instead of turning every series into a fallback.
  is_extra <- method %in% extra_methods
  for (p in extra_packages[[method]]) {
    if (!requireNamespace(p, quietly = TRUE))
      stop(paste0("The method '", method, "' needs the R package '", p, "': install.packages(\"", p, "\")"))
  }
  params <- if (is.null(job$params)) list() else as.list(job$params)

  # DHR-ARIMA: Fourier terim sayısı (orijinal: K = 1)
  fourier_k <- if (is.null(job$params$fourier_k)) 1 else job$params$fourier_k
  K <- if (isTRUE(job$params$fourier_k_per_period)) rep(fourier_k, length(seasonality)) else fourier_k

  one <- function(i) {
    reset_fallback()
    h <- hs[i]
    ts_i <- forecast::msts(series[[i]], seasonal.periods = seasonality)
    f <- tryCatch({
      if (is_extra) {
        tryCatch(fun(ts_i, h, params), error = function(e) {
          note_fallback(method, e)
          get_snaive_forecasts(ts_i, h)
        })
      } else if (method == "dhr_arima") fun(ts_i, h, K = K) else fun(ts_i, h)
    }, error = function(e) {
      # Model işlevinin kendi yedeği de başarısız olduysa: son değeri taşı (orijinal kod burada çökerdi)
      note_fallback(paste0(method, "+naive"), e)
      rep(tail(as.numeric(ts_i), 1), h)
    })
    if (is.list(f)) f <- f[[1]]
    f <- as.numeric(f)
    na_n <- sum(is.na(f))
    f[is.na(f)] <- 0
    if (integer_conversion) f <- round(f)
    if (length(f) < H) f <- c(f, rep(NA_real_, H - length(f)))
    if (length(f) > H) f <- f[1:H]
    list(f = f, fb = get_fallback(), na = na_n)
  }

  chunk_size <- max(workers * 10L, 50L)
  idx <- seq_len(n)
  chunks <- split(idx, ceiling(idx / chunk_size))
  results <- vector("list", n)
  prescheduled <- method %in% c("ses", "ets", "theta")
  write_progress("forecasting", 0, n)
  done <- 0L
  for (ch in chunks) {
    res <- if (workers > 1L) parallel::mclapply(ch, one, mc.cores = workers, mc.preschedule = prescheduled) else lapply(ch, one)
    bad <- vapply(res, function(r) inherits(r, "try-error") || is.null(r), logical(1))
    if (any(bad)) stop("Bir işçi süreç öldü (muhtemelen bellek yetersiz). Paralel işçi sayısını azaltın.")
    results[ch] <- res
    done <- done + length(ch)
    write_progress("forecasting", done, n)
  }

  fmat <- do.call(rbind, lapply(results, function(r) r$f))
  fb_idx <- which(vapply(results, function(r) !is.null(r$fb), logical(1)))
  fallbacks <- lapply(fb_idx, function(i) list(index = i - 1L, message = results[[i]]$fb))   # 0 tabanlı (Python)
  list(forecasts = fmat, fallbacks = fallbacks, na_replaced = sum(vapply(results, function(r) r$na, numeric(1))))
}


run_global <- function(job, series) {
  H <- as.integer(job$horizon)
  lag <- as.integer(job$lag)
  max_obs <- if (is.null(job$params$max_obs)) NULL else as.integer(job$params$max_obs)
  cb_params <- if (is.null(job$params$catboost)) list() else as.list(job$params$catboost)
  write_progress("fitting", 0, 1)
  fmat <- start_forecasting(series, lag, H, job$method, max_obs = max_obs, cb_params = cb_params)
  fmat <- as.matrix(fmat)
  na_n <- sum(is.na(fmat))
  fmat[is.na(fmat)] <- 0
  if (isTRUE(job$integer_conversion)) fmat <- round(fmat)
  if (!is.null(job$horizons)) {
    hs <- as.integer(job$horizons)
    for (i in seq_len(nrow(fmat))) {
      d <- H - hs[i]
      if (d > 0) fmat[i, (H - d + 1):H] <- NA_real_
    }
  }
  write_progress("fitting", 1, 1)
  list(forecasts = fmat, fallbacks = list(), na_replaced = na_n)
}


main <- function() {
  job <- fromJSON(file.path(job_dir, "job.json"), simplifyVector = TRUE)
  set.seed(as.integer(job$seed))
  if (as.integer(job$workers) > 1L) RNGkind("L'Ecuyer-CMRG")
  if (as.integer(job$workers) > 1L) set.seed(as.integer(job$seed))

  start_time <- Sys.time()
  series <- read_series(job)
  out <- switch(job$task,
                local = run_local(job, series),
                global = run_global(job, series),
                stop(paste0("Bilinmeyen görev: ", job$task)))
  elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

  write_matrix(out$forecasts)
  summary <- list(
    n_series = length(series),
    horizon = ncol(out$forecasts),
    elapsed_sec = elapsed,
    na_replaced = out$na_replaced,
    n_fallbacks = length(out$fallbacks),
    fallbacks = out$fallbacks,
    versions = versions()
  )
  write_json(summary, file.path(job_dir, "summary.json"), auto_unbox = TRUE, digits = NA, null = "null", na = "null")
}

tryCatch(main(), error = function(e) {
  writeLines(conditionMessage(e), file.path(job_dir, "error.txt"))
  message("HATA: ", conditionMessage(e))
  quit(save = "no", status = 2)
})
