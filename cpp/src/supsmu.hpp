#pragma once
#include <vector>

namespace tsf {
// stats::supsmu(x, y)$y for increasing x without ties (span = "cv", bass = 0, unit weights).
std::vector<double> supsmu(const std::vector<double>& x, const std::vector<double>& y);
}  // namespace tsf
