// Structural time series models: R stats::StructTS() and predict.StructTS() (R 4.6, GPL-2 or later), as the R engine
// calls them through forecast::forecast() (src/tsfstudio/r/lib/extra_models.R: get_struct_ts_forecasts).
#pragma once

#include "util.hpp"

namespace tsf {

// Longest period fitted with a seasonal component (the state has m + 1 elements and every likelihood evaluation costs
// n * (m + 1)^3 operations); the R engine applies the same rule.
constexpr int STRUCT_TS_MAX_SEASON = 24;

// m = floor(largest period). The basic structural model (type = "BSM": level, slope and dummy-variable seasonal) when
// 1 < m <= STRUCT_TS_MAX_SEASON and the series holds two full periods, otherwise the local linear trend
// (type = "trend"). Throws for fewer than 3 observations, non-finite data and where optim() stops in R.
Vec struct_ts_forecast(const Vec& y, int h, int m);

}  // namespace tsf
