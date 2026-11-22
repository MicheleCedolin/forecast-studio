// Phase 1 methods: C++ counterparts of src/tsfstudio/engines/py_models.py.
#pragma once

#include "util.hpp"

namespace tsf {

Vec naive(const Vec& y, int h);
Vec snaive(const Vec& y, int h, int m);
Vec drift(const Vec& y, int h);

struct SesFit { double alpha, level; };
SesFit ses_fit(const Vec& y);
Vec ses_forecast(const Vec& y, int h);

Vec theta_forecast(const Vec& y, int h, int m);

bool m4_seasonality_test(const Vec& x, int m);
Vec naive2(const Vec& y, int h, int m);
Vec holt_forecast(const Vec& y, int h, bool damped);
Vec comb_shd(const Vec& y, int h, int m);

// In-sample SSE of Holt/damped Holt at its optimum (for tests: compare the fit quality with Python/R).
double holt_best_sse(const Vec& y, bool damped);

}  // namespace tsf
