// Simple benchmarks as the Python engine runs them (statsforecast 2.1.1, Apache-2.0, Nixtla): HistoricAverage,
// WindowAverage, SeasonalWindowAverage, SeasonalExponentialSmoothingOptimized, ZeroModel, with the engine's
// usable_season(n, m, None) rule. A series too short for the window / season gives NaN (the caller falls back).
#pragma once

#include "util.hpp"

namespace tsf {

Vec historic_average(const Vec& y, int h);
Vec window_average(const Vec& y, int h, int window_size = 4);
Vec seasonal_window_average(const Vec& y, int h, int m, int window_size = 2);
Vec seasonal_es_optimized(const Vec& y, int h, int m);
Vec zero_model(int h);

}  // namespace tsf
