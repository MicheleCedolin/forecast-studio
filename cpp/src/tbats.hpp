// Phase 5: TBATS as the study runs it, forecast(forecast::tbats(y), h)$mean with R forecast 9.0.2: the trigonometric
// seasonal model with Box-Cox, trend / damping, ARMA errors (auto.arima on the residuals), the harmonic search and the
// non-seasonal BATS alternative, R's Nelder-Mead / BFGS, Guerrero's Box-Cox lambda. GPL-3 (translated from forecast).
#pragma once

#include "util.hpp"

namespace tsf {

// periods: one period -> a ts of frequency floor(p) (what msts() gives for a single period); several -> msts with the
// sorted exact periods. Throws where R's tbats() stops (the caller falls back to snaive, as the study's R code does).
Vec tbats_forecast(const Vec& y, const Vec& periods, int h);

}  // namespace tsf
