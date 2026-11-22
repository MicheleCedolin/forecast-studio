// Global models: pooled regression (R global_models.R / Python py_global.py): every series divided by its mean, lag
// rows from embed(x, lag + 1) stacked over all series, y ~ Lag1 + ... + Lag_lag + 0 by least squares (R glm, gaussian),
// linearly dependent lags aliased as R does, recursive forecasts scaled back by the series means.
#pragma once

#include <string>
#include <vector>

#include "util.hpp"

namespace tsf {

struct PooledResult {
    std::vector<Vec> forecasts;     // one row per series
    std::vector<int> aliased;       // 1-based lag numbers
    long long n_rows = 0;
};

// max_obs > 0: only the last max_obs observations of each series. Throws std::runtime_error (too short series, too few
// rows).
PooledResult pooled_regression(const std::vector<Vec>& series, int lag, int h, int max_obs);

}  // namespace tsf
