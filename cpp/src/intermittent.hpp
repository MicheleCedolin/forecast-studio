// Phase 4: intermittent-demand methods as the Python engine runs them (statsforecast 2.1.1: CrostonClassic,
// CrostonSBA, CrostonOptimized, TSB, ADIDA, IMAPA; src/ses.cpp and models.py), including the engine's rule that an
// all-zero series is forecast as zeros.
#pragma once

#include "util.hpp"

namespace tsf {

Vec croston_classic(const Vec& y, int h);
Vec croston_sba(const Vec& y, int h);
Vec croston_optimized(const Vec& y, int h);
Vec tsb(const Vec& y, int h, double alpha_d = 0.1, double alpha_p = 0.1);
Vec adida(const Vec& y, int h);
Vec imapa(const Vec& y, int h);

// statsforecast's golden-section search for the SES smoothing parameter on [lower, upper] (minimises the raw SSE).
double golden_section_ses(const Vec& x, double lower = 0.1, double upper = 0.3);

}  // namespace tsf
