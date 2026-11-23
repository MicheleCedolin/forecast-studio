// See stlf.hpp.
#include "stlf.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "autoarima.hpp"
#include "ets.hpp"

namespace tsf {

StlfParts r_mstl_parts(const Vec& y, const Vec& periods, int h) {
    const int n = static_cast<int>(y.size());
    for (double v : y) if (std::isnan(v)) throw std::runtime_error("stlf: missing values are not supported");
    StlfParts out;
    out.freq = r_frequency(periods.data(), static_cast<int>(periods.size()));
    if (out.freq <= 1) throw std::runtime_error("y is not a seasonal ts object");     // stlf
    Vec per;
    int iterate = 1;
    if (periods.size() <= 1) {
        per.push_back(static_cast<double>(out.freq));   // msts() with one period: a plain ts of frequency floor(p)
    } else {
        for (double p : periods) if (p < n / 2.0) per.push_back(p);   // mstl: drop periods >= n / 2
        std::sort(per.begin(), per.end());
        if (per.empty()) throw std::runtime_error("mstl: no seasonal period shorter than half the series");
        if (per[0] <= 1) throw std::runtime_error("mstl: a seasonal period <= 1 is not supported");
        iterate = 2;
    }
    const int k = static_cast<int>(per.size());
    std::vector<Vec> seas(k, Vec(n, 0.0));
    Vec deseas = y, trend;
    for (int it = 0; it < iterate; ++it)
        for (int i = 0; i < k; ++i) {
            for (int t = 0; t < n; ++t) deseas[t] = deseas[t] + seas[i][t];
            const StlOut fit = stl(deseas, per[i], 7 + 4 * (i + 1));   // s.window = 7 + 4 * seq(6)
            seas[i] = fit.seasonal;
            for (int t = 0; t < n; ++t) deseas[t] = deseas[t] - seas[i][t];
            trend = fit.trend;
        }
    out.xsa.resize(n);
    for (int t = 0; t < n; ++t) out.xsa[t] = trend[t] + (deseas[t] - trend[t]);   // trendcycle + remainder
    out.seas_fc.assign(h, 0.0);
    for (int j = 0; j < h; ++j) {
        double s = 0.0;                                  // rowSums over the components
        for (int i = 0; i < k; ++i) {
            const int mp = static_cast<int>(per[i]);       // seq_len(mp)
            s += seas[i][n - mp + (j % mp)];
        }
        out.seas_fc[j] = s;
    }
    return out;
}

Vec stlf_ets(const Vec& y, const Vec& periods, int h) {
    const StlfParts p = r_mstl_parts(y, periods, h);
    Vec f = ets_auto(p.xsa, 1, h).forecast;              // ets(model = "ZZN", allow.multiplicative.trend = FALSE)
    for (int j = 0; j < h; ++j) f[j] += p.seas_fc[j];
    return f;
}

Vec stlf_arima(const Vec& y, const Vec& periods, int h) {
    const StlfParts p = r_mstl_parts(y, periods, h);
    Vec f = auto_arima(p.xsa, p.freq, false, false, h).forecast;   // auto.arima(x.sa, seasonal = FALSE)
    for (int j = 0; j < h; ++j) f[j] += p.seas_fc[j];
    return f;
}

}  // namespace tsf
