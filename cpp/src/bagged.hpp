// Bagged ETS (Bergmeir, Hyndman & Benítez 2016) as R forecast 9.0.2 computes it after set.seed(seed):
//   forecast(baggedETS(y), h)$mean = rowMeans of forecast(ets(x_b), h)$mean over x_1 = y and x_2..x_num from
//   bld.mbb.bootstrap(y, num): Box-Cox (Guerrero lambda in [0, 1] if min(y) > 1e-6, else 1), stl(x, "periodic") for
//   frequency > 1 (and n > 2 frequency), else loess(span = 6/n, degree = 1), and a moving-block bootstrap of the
//   remainder (block 2 frequency, or min(8, n/2)) with R's Mersenne-Twister and sample() reproduced exactly.
// The series is the R engine's msts(y, periods): frequency floor(max(periods)).
#pragma once

#include <cstdint>
#include <vector>

#include "util.hpp"

namespace tsf {

// bld.mbb.bootstrap(ts(y, frequency = freq), num) after set.seed(seed); element 0 is y itself.
std::vector<Vec> bld_mbb_bootstrap(const Vec& y, int freq, int num, uint32_t seed);

Vec bagged_ets(const Vec& y, const Vec& periods, int h, int num = 100, uint32_t seed = 0);

}  // namespace tsf
