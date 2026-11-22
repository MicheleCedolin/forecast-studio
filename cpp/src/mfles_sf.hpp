// Phase 4f: MFLES as the Python engine runs it, statsforecast 2.1.1 AutoMFLES(test_size = h, season_length,
// n_windows = 2, metric = "smape"): the MFLES boosting fit and predict (mfles.py), its default configuration grid and
// rolling-origin selection, statsforecast's get_basis / siegel_repeated_medians (src/mfles.cpp), scikit-learn 1.9's
// Lasso coordinate descent (_cd_fast.pyx, gap-safe screening) and coreforecast's EWM / rolling mean.
#pragma once

#include <vector>

#include "util.hpp"

namespace tsf {

// periods: the Python engine's valid_periods(n, periods) (sorted unique round(p) with p >= 2 and 2p <= n), applied
// here; none -> season_length None.
Vec mfles_forecast(const Vec& y, const Vec& periods, int h);

// sklearn Lasso(alpha, fit_intercept = False, tol, max_iter).fit(X, y).coef_ for the columns X (exposed for tests).
Vec sk_lasso(const std::vector<Vec>& cols, const Vec& y, double alpha, double tol = 1e-3, int max_iter = 10000);

}  // namespace tsf
