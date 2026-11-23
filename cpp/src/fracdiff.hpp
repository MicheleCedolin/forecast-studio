#pragma once
#include <vector>

namespace tsf {
// fracdiff::fracdiff(x, nar = 0, nma = 0, M)$d: the fractional differencing parameter in [0, 0.5].
// Throws std::runtime_error on a gamma-function failure or a series shorter than 2.
double fracdiff_d(const std::vector<double>& x, int M = 100);
}  // namespace tsf
