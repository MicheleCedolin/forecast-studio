// Phase 4b: the dynamic optimised theta model as the Python engine runs it, statsforecast 2.1.1
// DynamicOptimizedTheta (auto_theta(model = "DOTM") in theta.py, the state recursion and Nelder-Mead of src/theta.cpp
// and include/statsforecast/nelder_mead.h, statsmodels' acf / seasonal_decompose for the seasonal step).
#pragma once

#include "util.hpp"

namespace tsf {

// DynamicOptimizedTheta(season_length = m, decomposition_type).forecast(y, h)["mean"]. m is the season length after
// the engine's usable_season(n, m, None) rule (applied here); multiplicative switches to additive for data <= 0.
Vec dotm_forecast(const Vec& y, int m, int h, bool multiplicative = true);

// AutoTheta(season_length = m, decomposition_type): STM, OTM, DSTM and DOTM fitted, lowest in-sample mse kept.
Vec auto_theta_forecast(const Vec& y, int m, int h, bool multiplicative = true);

// StandardTheta, OptimizedTheta and DynamicTheta of statsforecast: the same code with one model fitted
// (theta fixed at 2 for STM and DSTM), same seasonal step and engine rules as dotm.
Vec stm_forecast(const Vec& y, int m, int h, bool multiplicative = true);
Vec otm_forecast(const Vec& y, int m, int h, bool multiplicative = true);
Vec dstm_forecast(const Vec& y, int m, int h, bool multiplicative = true);

}  // namespace tsf
