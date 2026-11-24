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
    double sigma2 = 0;          // R's model$sigma2 (used by forecast.ets for class 3 models)
    // components and states as R's ets object holds them: error / trend / season (0 none, 1 additive, 2
    // multiplicative), damped, the state frequency (1 for non-seasonal), initstate (l, [b], s1..sm) and the final
    // state (states[n + 1, ]); hw = the fit came from R's small-sample HoltWintersZZ branch (n <= npars + 4).
    int err = 1, trend = 0, season = 0, m = 1;
    bool damped = false, hw = false;
    Vec init, fin;
};

// Fits every admissible ETS model to y (frequency m; m > 24 -> non-seasonal, as R) and returns the AICc-best forecast.
// Small samples (n <= npars + 4) follow R's HoltWintersZZ branch (L-BFGS-B fits of SES / Holt, smaller sigma2).
// Throws std::runtime_error when no model can be fitted.
EtsResult ets_auto(const Vec& y, int m, int h);

// ets(y, model = "ZZN") (no_season) or "ZZZ" on a ts of frequency m, as MAPA calls it.
EtsResult ets_zz(const Vec& y, int m, int h, bool no_season);

// ets(y, model = fit, use.initial.values = TRUE)$states[n + 1, ]: the states recomputed from fit$initstate with
// fit's parameters (alpha floored at 1e-10) by the ETS recursions, as R does when a fit is reused.
Vec ets_refit_final(const Vec& y, const EtsResult& fit);

// Holt-Winters as the Python engine's statsforecast HoltWinters(error_type = "A"): ets(y, model = "AAA") with damped
// and undamped trend compared by AICc (the R port above); season length as usable_season(n, m, 24), else "Nonseasonal
// data" (the caller falls back).
// mult_error: HoltWinters(error_type = "M"), i.e. ets(model = "MAM"); max_season replaces 24 in usable_season.
EtsResult ets_holt_winters(const Vec& y, int m, int h, bool mult_error = false, int max_season = 24);

// Holt as statsforecast Holt(error_type = "A"): ets(y, model = "AAN") (undamped / damped trend by AICc), non-seasonal.
// mult_error: Holt(error_type = "M"), ets(model = "MAN").
EtsResult ets_holt(const Vec& y, int h, bool mult_error = false);

// Diagnostic for tests: initial state (R initstate) and fitted smoothing parameters of one fixed model; returns its lik.
// error/trend/season: 1 = additive, 2 = multiplicative (trend/season 0 = none).
double ets_debug(const Vec& y, int m, int error, int trend, int season, bool damped, Vec& init, Vec& fitted);

}  // namespace tsf
