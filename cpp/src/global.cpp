#include "global.hpp"

#include <cmath>
#include <stdexcept>

#include "arima.hpp"
#include "sf_common.hpp"

namespace tsf {

PooledResult pooled_regression(const std::vector<Vec>& series, int lag, int h, int max_obs) {
    if (lag < 1) throw std::runtime_error("lag must be >= 1");
    const int p = lag, nc = lag + 1;
    std::vector<Vec> norm;
    Vec means;
    for (size_t i = 0; i < series.size(); ++i) {
        Vec x = series[i];
        if (max_obs > 0 && static_cast<int>(x.size()) > max_obs) x = Vec(x.end() - max_obs, x.end());
        if (static_cast<int>(x.size()) < lag + 1)
            throw std::runtime_error("Series " + std::to_string(i + 1) + " is too short for lag = " + std::to_string(lag) +
                                     " (length " + std::to_string(x.size()) + "; at least " + std::to_string(lag + 1) +
                                     " required).");
        double mu = np_mean(x.data(), x.size());
        if (mu == 0.0) mu = 1.0;
        for (double& v : x) v /= mu;
        norm.push_back(std::move(x));
        means.push_back(mu);
    }
    // R factor of [Lag1..Lag_lag | y] by Givens row updates (rows stream in R's rbind order)
    std::vector<Vec> R(nc, Vec(nc, 0.0));
    Vec row(nc);
    long long nrows = 0;
    for (const Vec& x : norm) {
        const int n = static_cast<int>(x.size());
        for (int i = 0; i + lag < n; ++i) {
            for (int j = 1; j <= lag; ++j) row[j - 1] = x[i + lag - j];
            row[lag] = x[i + lag];
            ++nrows;
            for (int k = 0; k < nc; ++k) {
                if (row[k] == 0.0) continue;
                const double a = R[k][k], b = row[k];
                const double r = std::hypot(a, b), c = a / r, s = b / r;
                R[k][k] = r;
                for (int j = k + 1; j < nc; ++j) {
                    const double rk = R[k][j], xj = row[j];
                    R[k][j] = c * rk + s * xj;
                    row[j] = -s * rk + c * xj;
                }
                row[k] = 0.0;
            }
        }
    }
    if (nrows < nc) throw std::runtime_error("Not enough rows for the regression (" + std::to_string(nrows) + " < " +
                                             std::to_string(nc) + ").");
    // aliasing (dqrdc2 logic, as py_global._solve_with_aliasing): a column whose residual norm after projecting on the
    // kept columns is below 1e-11 of its own norm is dropped (NA coefficient -> 0)
    std::vector<Vec> cols(p, Vec(nc)), Q;
    Vec yv(nc);
    for (int r = 0; r < nc; ++r) { for (int j = 0; j < p; ++j) cols[j][r] = R[r][j]; yv[r] = R[r][p]; }
    std::vector<int> keep;
    PooledResult out;
    for (int j = 0; j < p; ++j) {
        const Vec& c = cols[j];
        double nrm = 0.0;
        for (double v : c) nrm += v * v;
        nrm = std::sqrt(nrm);
        Vec res = c;
        for (int pass = 0; pass < 2; ++pass)
            for (const Vec& qv : Q) {
                double d = 0.0;
                for (int r = 0; r < nc; ++r) d += qv[r] * res[r];
                for (int r = 0; r < nc; ++r) res[r] -= d * qv[r];
            }
        double rn = 0.0;
        for (double v : res) rn += v * v;
        rn = std::sqrt(rn);
        if (nrm == 0.0 || rn < 1e-11 * nrm) { out.aliased.push_back(j + 1); continue; }
        keep.push_back(j);
        for (double& v : res) v /= rn;
        Q.push_back(res);
    }
    Vec beta(p, 0.0);
    if (!keep.empty()) {
        std::vector<Vec> kc;
        for (int j : keep) kc.push_back(cols[j]);
        Vec coef, se;
        lm_qr(yv, kc, coef, se);
        for (size_t i = 0; i < keep.size(); ++i) beta[keep[i]] = coef[i];
    }
    out.n_rows = nrows;
    out.forecasts.assign(norm.size(), Vec(h));
    for (size_t s = 0; s < norm.size(); ++s) {
        const Vec& x = norm[s];
        Vec lags(p);
        for (int j = 0; j < p; ++j) lags[j] = x[x.size() - 1 - j];      // rev(tail(x, lag))
        for (int k = 0; k < h; ++k) {
            double pr = 0.0;
            for (int j = 0; j < p; ++j) pr += lags[j] * beta[j];
            out.forecasts[s][k] = pr * means[s];
            for (int j = p - 1; j > 0; --j) lags[j] = lags[j - 1];
            lags[0] = pr;
        }
    }
    return out;
}

}  // namespace tsf
