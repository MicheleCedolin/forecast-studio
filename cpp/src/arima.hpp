// Phase 3a: a port of R's stats::arima (CSS, CSS-ML, ML; Gardner 1980 initial covariance) and its forecast.
#pragma once

#include <complex>
#include <vector>

#include "util.hpp"

namespace tsf {

using CVec = std::vector<std::complex<double>>;

// R polyroot(z): roots of z[0] + z[1] x + ... + z[n] x^n (trailing zeros must be removed by the caller).
CVec polyroot(const Vec& z);

// Stationary state covariance Q0 of the ARMA part (r x r, column-major): method 0 = Gardner et al. (1980) as
// stats:::getQ0, 1 = from the autocovariances. arima_fit uses Gardner for r <= 8 and the autocovariance form above.
Vec getQ0_acov(const Vec& phi, const Vec& theta);
Vec arima_q0(const Vec& phi, const Vec& theta, int method);

struct ArimaModel {                 // makeARIMA()
    Vec phi, theta, Delta;          // theta is padded to r - 1 by makeARIMA, not by upARIMA (as in R)
    int r = 1, d = 0, rd = 1;
    Vec a, P, Pn;                   // state and covariances (rd x rd, column-major)
};

enum class ArimaMethod { CSS, CSS_ML, ML };

struct ArimaSpec {
    int p = 0, d = 0, q = 0, P = 0, D = 0, Q = 0, period = 1;
    bool include_mean = true;       // ignored when d + D > 0 (as stats::arima)
    bool drift = false;             // regressor 1..n (forecast::Arima's include.drift / auto.arima's drift)
    ArimaMethod method = ArimaMethod::CSS_ML;
};

struct ArimaFit {
    bool ok = false;
    Vec coef;                       // ar, ma, sar, sma, [intercept | drift]
    int narma = 0, ncxreg = 0;
    bool has_intercept = false, has_drift = false;
    double sigma2 = 0, loglik = 0, aic = INF;
    int n_used = 0, convergence = 0;
    Vec residuals;
    Vec var_diag;                   // diag of var.coef (NaN where R's sqrt(diag) would be NaN)
    bool var_ok = false;            // var.coef could be computed (solve succeeded)
    ArimaModel model;               // filtered state after the final pass (for forecasting)
    int n = 0;
};

// stats::arima(x, order, seasonal, xreg = drift?, include.mean, method). Throws std::runtime_error where R errors.
ArimaFit arima_fit(const Vec& x, const ArimaSpec& spec);

// predict.Arima(fit, n.ahead = h)$pred (including the intercept / drift term).
Vec arima_forecast(const ArimaFit& fit, int h);

}  // namespace tsf
