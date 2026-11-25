// Neural network autoregression: R forecast::nnetar() 9.0.2 (GPL-3) with nnet::nnet() 7.3 (GPL-2 | GPL-3, W. N.
// Venables and B. D. Ripley), as the R engine calls them (src/tsfstudio/r/lib/extra_models.R: get_nnetar_forecasts).
#pragma once

#include <cstdint>

#include "util.hpp"

namespace tsf {

// forecast(nnetar(ts(y, frequency = m), repeats = repeats), h)$mean after set.seed(seed): NNAR(p, P, k)[m] with p from
// ar() (on the seasonally adjusted series for m > 1), P = 1, k = round((p + P + 1) / 2), `repeats` networks started
// from R's runif(-0.7, 0.7) weights and averaged, forecasts fed back recursively. m = floor(largest period).
// Throws for fewer than 3 observations, non-finite data and more than 1000 weights (nnet's MaxNWts).
Vec nnetar_forecast(const Vec& y, int h, int m, int repeats = 20, uint32_t seed = 0);

}  // namespace tsf
