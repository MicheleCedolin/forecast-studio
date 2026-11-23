// Temporal aggregation forecasts, as R computes them on msts(y, periods) (frequency m = floor(max(periods))):
//  - thief_ets: thief::thief(y, usemodel = "ets", comb = "struc") (Athanasopoulos, Hyndman, Kourentzes & Petropoulos
//    2017): tsaggregates() over the divisors of m, forecast(ets()) at every level, structural-scaling reconciliation
//    (hts::combinef with weights 1/nsum; solved here by a dense Cholesky, so it equals R's sparse LU to rounding).
//  - mapa_ets: MAPA::mapa(y, ppy = m, fh = h, comb = "w.mean", hybrid = TRUE, model = "ZZZ", type = "ets")$outfor
//    (Kourentzes, Petropoulos & Trapero 2014): ets() at aggregation levels 1..m, the final states translated into
//    level / trend / season forecasts, combined (mean, mean, 1/AL-weighted season) and averaged with level 1's ETS.
// Errors where R stops (m <= 1, fewer than two periods for thief) throw.
#pragma once

#include "util.hpp"

namespace tsf {

Vec thief_ets(const Vec& y, const Vec& periods, int h);
Vec mapa_ets(const Vec& y, const Vec& periods, int h);

}  // namespace tsf
