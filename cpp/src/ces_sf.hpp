// Phase 4c: complex exponential smoothing as the Python engine runs it, statsforecast 2.1.1 AutoCES(model = "Z")
// (ces.py: initial states, parameter set-up, AICc selection over N / S / P / F; src/ces.cpp: the three-pass fit,
// objective n log(SSE), forecasts; nelder_mead.h). Apache-2.0, Nixtla.
#pragma once

#include "util.hpp"

namespace tsf {

// AutoCES(season_length = m, model = "Z").forecast(y, h)["mean"], with the engine's usable_season(n, m, 24) rule.
Vec ces_forecast(const Vec& y, int m, int h);

}  // namespace tsf
