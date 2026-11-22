// Phase 4e: MSTL as the Python engine runs it, statsforecast 2.1.1 MSTL(season_length, trend_forecaster =
// AutoETS(model = "ZZN")): mstl() with statsmodels' STL (seasonal_deg 0, 5 inner iterations, s.window 11, 15, ...; iterate 2 for several
// periods), seasonal naive forecasts of each seasonal component, and the trend forecaster on trend + remainder.
// The trend forecaster here is the C++ ets (R's ets, model "ZZN"), as everywhere in the C++ engine.
#pragma once

#include "util.hpp"

namespace tsf {

struct MstlParts {
    Vec xsa;                        // trend + remainder (the seasonally adjusted series)
    Vec seas_fc;                    // sum of the seasonal naive forecasts of the seasonal components
};

// Periods after the Python engine's valid_periods(n, periods): sorted unique round(p) with p >= 2 and 2p <= n
// (none -> error).
MstlParts mstl_parts(const Vec& y, const Vec& periods, int h);

Vec mstl_forecast(const Vec& y, const Vec& periods, int h);

}  // namespace tsf
