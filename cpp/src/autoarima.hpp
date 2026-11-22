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

// The stl.f procedure with explicit windows, degrees, jumps and inner iterations (statsmodels' STL uses other defaults).
StlOut stl_core(const Vec& x, int period, int s_window, int t_window, int l_window, int s_deg, int t_deg, int l_deg,
                int s_jump, int t_jump, int l_jump, int inner);

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

// auto.arima(y, seasonal, lambda = log_transform ? 0 : NULL, xreg) with frequency(y) = freq;
// forecast(fit, h, xreg = newxreg)$mean. xreg / newxreg: columns of length n / h. Throws where R stops.
AutoArimaResult auto_arima(const Vec& y, int freq, bool seasonal, bool log_transform, int h,
                           const std::vector<Vec>& xreg = {}, const std::vector<Vec>& newxreg = {}, int fixed_d = -1);

// The study's chain: auto.arima(y, lambda = 0), else auto.arima(y), else auto.arima(y, seasonal = FALSE).
AutoArimaResult auto_arima_study(const Vec& y, int freq, int h);

// forecast::fourier(x, K) terms for one period at times t0 + 1 .. t0 + len: columns S1, C1, ..., SK, CK (sine columns
// that are identically zero are dropped), computed with R's sinpi / cospi.
std::vector<Vec> fourier_terms(double period, int K, int t0, int len);

// The study's dynamic harmonic regression: auto.arima(y, xreg = fourier(y, K), seasonal = FALSE), forecast with
// fourier(y, K, h). n_periods > 1 fails as in R (a scalar K with several periods), so the caller falls back.
AutoArimaResult dhr_arima(const Vec& y, int freq, int n_periods, int K, int h);

}  // namespace tsf
