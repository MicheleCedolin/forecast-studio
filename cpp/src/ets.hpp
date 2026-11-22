// Phase 2: automatic ETS, a port of R forecast::ets() with its default settings
// (model "ZZZ", opt.crit "lik", bounds "both", ic "aicc", restrict = TRUE, allow.multiplicative.trend = FALSE).
#pragma once

#include <string>

#include "util.hpp"

namespace tsf {

struct EtsResult {
    Vec forecast;
    std::string model;          // e.g. "ETS(M,Ad,N)"
    double lik = INF;           // R's e$lik: n*log(sum e^2) (+ 2*sum log|f| for multiplicative error)
    double aicc = INF;
    double alpha = 0, beta = 0, gamma = 0, phi = 1;
};

// Fits every admissible ETS model to y (frequency m; m > 24 -> non-seasonal, as R) and returns the AICc-best forecast.
// Throws std::runtime_error when no model can be fitted.
EtsResult ets_auto(const Vec& y, int m, int h);

// Diagnostic for tests: initial state (R initstate) and fitted smoothing parameters of one fixed model; returns its lik.
// error/trend/season: 1 = additive, 2 = multiplicative (trend/season 0 = none).
double ets_debug(const Vec& y, int m, int error, int trend, int season, bool damped, Vec& init, Vec& fitted);

}  // namespace tsf
