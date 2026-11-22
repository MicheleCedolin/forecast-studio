// Phase 3b: a port of R forecast::auto.arima() (stepwise search, KPSS ndiffs, STL-based nsdiffs, approximation
// with offset, CSS-ML refit) and of the study's call chain auto.arima(lambda = 0) -> auto.arima() -> seasonal = FALSE.
#pragma once

#include <string>

#include "arima.hpp"

namespace tsf {

// forecast::is.constant: isTRUE(all.equal(x, rep(x[1], n))).
bool is_constant(const Vec& x);

// urca::ur.kpss(type = "mu", use.lag = trunc(3 sqrt(n) / 13))@teststat and forecast::ndiffs(test = "kpss").
double kpss_stat(const Vec& y);
int ndiffs(Vec x, int max_d = 2);

// stats::stl(x, s.window = 11) with R's defaults (s.degree 0, t/l windows and jumps, robust = FALSE).
struct StlOut { Vec seasonal, trend; };
StlOut stl(const Vec& x, int period, int s_window);

// forecast::seas.heuristic (via mstl) and nsdiffs(test = "seas", max.D = 1).
double seas_heuristic(const Vec& x, int period);
int nsdiffs(const Vec& x, int period);

struct AutoArimaResult {
    Vec forecast;                   // back-transformed when lambda = 0 was used
    int p = 0, d = 0, q = 0, P = 0, D = 0, Q = 0, period = 1;
    bool drift = false, mean = false;
    double loglik = std::numeric_limits<double>::quiet_NaN();
    double aicc = std::numeric_limits<double>::quiet_NaN();
    double sigma2 = std::numeric_limits<double>::quiet_NaN();
    int step = 0;                   // auto_arima_study: which call of the chain succeeded (1, 2, 3)
};

// auto.arima(y, seasonal, lambda = log_transform ? 0 : NULL) with frequency(y) = freq; forecast(fit, h)$mean.
// Throws std::runtime_error where R stops.
AutoArimaResult auto_arima(const Vec& y, int freq, bool seasonal, bool log_transform, int h);

// The study's chain: auto.arima(y, lambda = 0), else auto.arima(y), else auto.arima(y, seasonal = FALSE).
AutoArimaResult auto_arima_study(const Vec& y, int freq, int h);

}  // namespace tsf
