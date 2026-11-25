// Double seasonal Holt-Winters (Taylor 2003): R forecast::dshw() 9.0.2 (GPL-3), as the R engine calls it
// (src/tsfstudio/r/lib/extra_models.R: get_dshw_forecasts).
#pragma once

#include "util.hpp"

namespace tsf {

// periods: the job's seasonal periods; the two shortest, floored, are period1 < period2. Throws where dshw() stops
// (fewer than two periods, periods not nested, zero or negative data, fewer than 2 * period2 observations).
Vec dshw_forecast(const Vec& y, const Vec& periods, int h);

}  // namespace tsf
