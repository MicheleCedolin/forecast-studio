// forecast::tslm(y ~ trend + season) and forecast.lm() (forecast 9.0.2), as the R engine calls them
// (src/tsfstudio/r/lib/extra_models.R: get_tslm_forecasts).
//
// tslm() builds the data frame  trend = 1..n,  season = factor(cycle(y))  and calls lm(), whose model matrix is
//   (Intercept), trend, season2, ..., seasonm        (treatment contrasts: season 1 is the base level)
// and whose coefficients come from LINPACK dqrdc2 / dqrsl (rlinpack.hpp: dqrls_coef). forecast.lm() predicts with
// trend = n + 1..n + h and the season continuing the cycle; predict.lm() computes X %*% beta, which R hands to the
// reference BLAS dgemv: the terms are added column by column, so a forecast is (b0 + b_trend * t) + b_season.
#include "tslm.hpp"

#include <stdexcept>

#include "rlinpack.hpp"

namespace tsf {

Vec tslm_forecast(const Vec& y, int h, int m) {
    const int n = static_cast<int>(y.size());
    if (n < 3) throw std::runtime_error("tslm needs at least 3 observations");
    const bool seasonal = m > 1 && m <= TSLM_MAX_SEASON && n >= 2 * m;
    std::vector<Vec> cols;
    cols.reserve(seasonal ? m + 1 : 2);
    cols.emplace_back(n, 1.0);
    Vec trend(n);
    for (int i = 0; i < n; ++i) trend[i] = static_cast<double>(i + 1);
    cols.push_back(std::move(trend));
    if (seasonal)
        for (int s = 1; s < m; ++s) {               // season s + 1 (the ts starts at season position 1)
            Vec d(n, 0.0);
            for (int i = s; i < n; i += m) d[i] = 1.0;
            cols.push_back(std::move(d));
        }
    const Vec b = rl::dqrls_coef(cols, y);
    for (double v : b)
        if (std::isnan(v)) throw std::runtime_error("tslm: rank-deficient regression");
    Vec fc(h);
    for (int k = 0; k < h; ++k) {
        const int t = n + k;                        // 0-based time index of the forecast
        double f = 0.0;
        f = f + b[0];
        f = f + b[1] * static_cast<double>(t + 1);
        const int s = t % m;
        if (seasonal && s > 0) f = f + b[1 + s];
        fc[k] = f;
    }
    return fc;
}

}  // namespace tsf
