// Pieces shared by the statsforecast ports (theta_sf, ces_sf, intermittent): statsforecast 2.1.1's bounded Nelder-Mead
// (include/statsforecast/nelder_mead.h, Apache-2.0, Nixtla), numpy's pairwise summation and statsmodels'
// seasonal_decompose seasonal component.
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>

#include "util.hpp"

namespace tsf {

// np.sum / np.mean over a contiguous float64 array (pairwise summation, blocks of 8 up to 128)
double np_sum(const double* a, size_t n);
inline double np_mean(const double* a, size_t n) { return np_sum(a, n) / static_cast<double>(n); }

// statsmodels seasonal_decompose(x, model, period).seasonal: two-sided moving-average trend, per-position mean of the
// detrended series (NaN ends skipped), normalised to mean 1 (multiplicative) or 0 (additive), tiled to length n.
Vec sm_seasonal(const Vec& x, int period, bool multiplicative);

// nm::NelderMead with statsforecast's settings (init_step 0.05, zero_pert 1e-4, reflection 1, adaptive coefficients,
// max_iter 1000, population-SD stop 1e-4, projection onto [lower, upper]); returns the best vertex of the last ordering.
template <typename F>
Vec sf_nelder_mead(const F& fn, Vec x, const Vec& lower, const Vec& upper) {
    const double init_step = 0.05, zero_pert = 1e-4, alpha = 1.0, tol_std = 1e-4;
    const int max_iter = 1000;
    const int n = static_cast<int>(x.size());
    auto clamp = [&](Vec v) { for (int k = 0; k < n; ++k) v[k] = std::min(std::max(v[k], lower[k]), upper[k]); return v; };
    x = clamp(x);
    const double gamma = 1.0 + 2.0 / n, rho = 0.75 - 1.0 / (2 * n), sigma = 1.0 - 1.0 / n;
    std::vector<Vec> S(n + 1, x);
    for (int i = 0; i < n; ++i) {
        double v = S[i][i];
        v = (v == 0) ? zero_pert : v * (1.0 + init_step);
        S[i][i] = std::clamp(v, lower[i], upper[i]);
    }
    Vec fs(n + 1);
    for (int i = 0; i <= n; ++i) fs[i] = fn(S[i]);
    int best = 0;
    for (int it = 0; it < max_iter; ++it) {
        std::vector<int> ord(n + 1);
        std::iota(ord.begin(), ord.end(), 0);
        std::sort(ord.begin(), ord.end(), [&](int a, int b) { return fs[a] < fs[b]; });
        best = ord[0];
        const int worst = ord[n], second = ord[n - 1];
        double mean = 0.0;
        for (double v : fs) mean += v;
        mean /= (n + 1);
        double var = 0.0;
        for (double v : fs) var += (v - mean) * (v - mean);
        if (std::sqrt(var / (n + 1)) < tol_std) break;
        Vec xo(n, 0.0);
        for (int k = 0; k < n; ++k) {
            double s = 0.0;
            for (int r = 0; r <= n; ++r) s += S[r][k];
            xo[k] = (s - S[worst][k]) / n;
        }
        Vec xr(n);
        for (int k = 0; k < n; ++k) xr[k] = xo[k] + alpha * (xo[k] - S[worst][k]);
        xr = clamp(xr);
        const double fr = fn(xr);
        if (fs[best] <= fr && fr < fs[second]) { S[worst] = xr; fs[worst] = fr; continue; }
        if (fr < fs[best]) {
            Vec xe(n);
            for (int k = 0; k < n; ++k) xe[k] = xo[k] + gamma * (xr[k] - xo[k]);
            xe = clamp(xe);
            const double fe = fn(xe);
            if (fe < fr) { S[worst] = xe; fs[worst] = fe; } else { S[worst] = xr; fs[worst] = fr; }
            continue;
        }
        if (fs[second] <= fr && fr < fs[worst]) {
            Vec xc(n);
            for (int k = 0; k < n; ++k) xc[k] = xo[k] + rho * (xr[k] - xo[k]);
            xc = clamp(xc);
            const double fc = fn(xc);
            if (fc <= fr) { S[worst] = xc; fs[worst] = fc; continue; }
        } else {
            Vec xc(n);
            for (int k = 0; k < n; ++k) xc[k] = xo[k] - rho * (xr[k] - xo[k]);
            xc = clamp(xc);
            const double fc = fn(xc);
            if (fc < fs[worst]) { S[worst] = xc; fs[worst] = fc; continue; }
        }
        for (int j = 0; j <= n; ++j) {
            if (j == best) continue;
            for (int k = 0; k < n; ++k) S[j][k] = S[best][k] + sigma * (S[j][k] - S[best][k]);
            S[j] = clamp(S[j]);
            fs[j] = fn(S[j]);
        }
    }
    return S[best];
}

}  // namespace tsf
