// See mstl_sf.hpp.
#include "mstl_sf.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

#include "autoarima.hpp"
#include "ets.hpp"

namespace tsf {

MstlParts mstl_parts(const Vec& y, const Vec& periods_in, int h) {
    const int n = static_cast<int>(y.size());
    std::set<int> ps;
    for (double p : periods_in) ps.insert(static_cast<int>(std::nearbyint(p)));   // int(round(p)), half to even
    std::vector<int> per;
    for (int p : ps) if (p >= 2 && 2 * p <= n) per.push_back(p);
    if (per.empty())
        throw std::runtime_error("There is no valid seasonal period for MSTL (a period >= 2 and at least two periods of "
                                 "data are required)");
    for (double v : y) if (std::isnan(v)) throw std::runtime_error("mstl cannot handle missing values");
    const int k = static_cast<int>(per.size());
    const int iterate = (k == 1) ? 1 : 2;
    std::vector<Vec> seas(k, Vec(n, 0.0));
    Vec deseas = y, trend;
    for (int it = 0; it < iterate; ++it)
        for (int i = 0; i < k; ++i) {
            for (int t = 0; t < n; ++t) deseas[t] = deseas[t] + seas[i][t];
            const int p = per[i], sw = 7 + 4 * (i + 1);
            // statsmodels STL defaults: trend = next odd of ceil(1.5 p / (1 - 1.5 / seasonal)), low_pass = next odd
            // above p, degrees (0, 1, 1), all jumps 1; fit(): 5 inner iterations when not robust (R's stl uses 2)
            int tw = static_cast<int>(std::ceil(1.5 * p / (1.0 - 1.5 / sw)));
            tw += (tw % 2 == 0);
            int lw = p + 1;
            lw += (lw % 2 == 0);
            const StlOut fit = stl_core(deseas, p, sw, tw, lw, 0, 1, 1, 1, 1, 1, 5);
            seas[i] = fit.seasonal;
            for (int t = 0; t < n; ++t) deseas[t] = deseas[t] - seas[i][t];
            trend = fit.trend;
        }
    MstlParts out;
    out.xsa.resize(n);
    for (int t = 0; t < n; ++t) {
        const double rem = deseas[t] - trend[t];
        out.xsa[t] = trend[t] + rem;
    }
    out.seas_fc.assign(h, 0.0);
    for (int j = 0; j < h; ++j) {
        double s = -0.0;                                // numpy sum over the components (fewer than 8)
        for (int i = 0; i < k; ++i) s += seas[i][n - per[i] + (j % per[i])];
        out.seas_fc[j] = s;
    }
    return out;
}

Vec mstl_forecast(const Vec& y, const Vec& periods, int h) {
    const MstlParts parts = mstl_parts(y, periods, h);
    Vec f = ets_auto(parts.xsa, 1, h).forecast;         // ets(model = "ZZN")
    for (int j = 0; j < h; ++j) f[j] += parts.seas_fc[j];
    return f;
}

}  // namespace tsf
