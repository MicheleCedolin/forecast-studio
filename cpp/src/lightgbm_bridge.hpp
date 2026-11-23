// Global LightGBM model driven from C++: the same data pipeline as py_global.lightgbm_forecast (mean-normalised lag
// rows of all series, float64, row-major; float32 labels as the Python package sets them), training and recursive
// prediction through LightGBM's C API (lib_lightgbm.dll / lib_lightgbm.so, loaded at run time). With the same
// parameter string the result equals the Python engine's.
#pragma once

#include <string>
#include <vector>

#include "util.hpp"

namespace tsf {

struct LightgbmSetup {
    std::string lib;                // lib_lightgbm.dll / lib_lightgbm.so
    std::string params;             // LightGBM parameter string ("key=value key=value ..."), used for the Dataset and the Booster
    int rounds = 100;               // boosting iterations
};

std::vector<Vec> lightgbm_forecast(const std::vector<Vec>& series, int lag, int h, int max_obs, const LightgbmSetup& s);

// py_global.lightgbm_feat_forecast: lags 1..lag, seasonal lags m and 2m, rolling mean / population sd over windows
// w1 = max(4, m_eff) and w2 = 2 w1 (sequential sums, oldest to newest), position t mod m; m_eff = m when the shortest
// series holds three seasons, else 1 (no seasonal features). Recursive forecasts.
std::vector<Vec> lightgbm_feat_forecast(const std::vector<Vec>& series, int lag, int h, int m, int max_obs,
                                        const LightgbmSetup& s);

}  // namespace tsf
