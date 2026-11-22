// Shared numerical helpers for tsfcore (C++17). Each helper mirrors the numpy/scipy call used by the Python engine, so the
// C++ methods reproduce the Python ones (see tests/test_cpp_engine.py).
#pragma once

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tsf {

using Vec = std::vector<double>;
constexpr double ALPHA_LO = 1e-4, ALPHA_HI = 0.9999;
constexpr double INF = std::numeric_limits<double>::infinity();

// forecast::msts frame frequency: floor(max(periods)), at least 1.
int r_frequency(const double* periods, int n_periods);

double ptp(const Vec& x);
double mean(const Vec& x);

// R acf(type = "correlation"): mean removed, denominator n, lags 1..nlags.
Vec acf(const Vec& x, int nlags);

// R decompose(x, "multiplicative")$seasonal, recycled to length n (ts starts at season position 1).
// Returns an empty vector if the seasonal figure cannot be computed.
Vec decompose_multiplicative(const Vec& x, int m);

// numpy.linspace(start, stop, num)
Vec linspace(double start, double stop, int num);

// scipy.optimize.minimize_scalar(method="bounded") (fminbound): Brent's method on [lo, hi]. Port of
// scipy.optimize._optimize._minimize_scalar_bounded, so it visits the same points.
double fminbound(const std::function<double(double)>& f, double lo, double hi, double xatol = 1e-5, int maxiter = 500);

// Nelder-Mead minimiser (unconstrained; callers clamp inside the objective). Returns the best point; `fbest` gets its value.
Vec nelder_mead(const std::function<double(const Vec&)>& f, Vec x0, const Vec& step, double& fbest,
                int maxiter = 2000, double ftol = 1e-12, double xtol = 1e-10);

// Least-squares slope of x on t = 0..n-1 (numpy.polyfit(t, x, 1)[0]).
double ls_slope(const Vec& x);

}  // namespace tsf
