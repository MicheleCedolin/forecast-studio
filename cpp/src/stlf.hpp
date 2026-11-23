// STL forecasts as R forecast 9.0.2 makes them: stlf(y, h) = forecast(mstl(y), method = "ets") and
// forecast(mstl(y), method = "arima"), with y built as the R engine builds it (msts(y, seasonal.periods)).
//   mstl: one period -> a ts of frequency floor(p), one stl() pass; several periods -> msts (periods sorted, those
//   >= n/2 dropped), two passes over the periods; stl(s.window = 7 + 4 i) with R's defaults (robust = FALSE).
//   seasadj = trend + remainder is forecast by ets(model = "ZZN") or auto.arima(seasonal = FALSE); each seasonal
//   component's last floor(p) values are repeated and added.
// Errors as in R (not seasonal, fewer than two periods of data) throw, so the engine falls back.
#pragma once

#include "util.hpp"

namespace tsf {

struct StlfParts {
    Vec xsa;                        // seasonally adjusted series (trend + remainder)
    Vec seas_fc;                    // seasonal part of the forecast (h values)
    int freq = 1;                   // frequency(x) of the ts / msts object
};

StlfParts r_mstl_parts(const Vec& y, const Vec& periods, int h);
Vec stlf_ets(const Vec& y, const Vec& periods, int h);
Vec stlf_arima(const Vec& y, const Vec& periods, int h);

}  // namespace tsf
