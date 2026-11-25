// Linear regression on a time trend and seasonal dummies: R forecast::tslm(y ~ trend + season) followed by
// forecast() (forecast 9.0.2, GPL-3), with lm()'s least squares (rlinpack.hpp) and predict.lm()'s summation order.
#pragma once

#include "util.hpp"

namespace tsf {

// Longest period that gets seasonal dummies (a daily series with a yearly period); the R engine applies the same rule.
constexpr int TSLM_MAX_SEASON = 366;

// m = floor(largest period). Seasonal dummies are used when 1 < m <= TSLM_MAX_SEASON and the series holds two full
// periods; otherwise the model is y ~ trend. Throws for fewer than 3 observations.
Vec tslm_forecast(const Vec& y, int h, int m);

}  // namespace tsf
