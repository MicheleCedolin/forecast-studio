// Port of statsforecast 2.1.1's DOTM (Apache-2.0, Nixtla). See theta_sf.hpp.
#include "theta_sf.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace tsf {

namespace {

enum Model { STM, OTM, DSTM, DOTM };
constexpr double NA = -99999.0, TOL = 1e-10;

using States = std::vector<std::array<double, 5>>;    // level, meany, An, Bn, mu

std::array<double, 5> init_state(const Vec& y, Model mt, double init_smoothed, double alpha, double theta) {
    double An, Bn, mu;
    if (mt == DSTM || mt == DOTM) {
        An = y[0]; Bn = 0.0; mu = y[0];
    } else {
        const double n = static_cast<double>(y.size());
        double s = 0.0, w = 0.0;
        for (size_t i = 0; i < y.size(); ++i) { s += y[i]; w += y[i] * static_cast<double>(i + 1); }
        const double y_mean = s / n, weighted_avg = w / n;
        Bn = (6 * (2 * weighted_avg - (n + 1) * y_mean)) / (n * n - 1);
        An = y_mean - (n + 1) * Bn / 2;
        mu = init_smoothed + (1 - 1 / theta) * (An + Bn);
    }
    return {alpha * y[0] + (1 - alpha) * init_smoothed, y[0], An, Bn, mu};
}

// update(states, i, ...) of theta.cpp reads only row i - 1: prev -> cur
void update_row(const std::array<double, 5>& prev, std::array<double, 5>& cur, size_t i, Model mt, double alpha,
                double theta, double y, bool usemu) {
    const double level = prev[0], meany = prev[1], An = prev[2], Bn = prev[3];
    auto& st_i = cur;
    const double di = static_cast<double>(i);
    st_i[4] = level + (1 - 1 / theta) * (An * std::pow(1 - alpha, di) + Bn * (1 - std::pow(1 - alpha, di + 1)) / alpha);
    if (usemu) y = st_i[4];
    st_i[0] = alpha * y + (1 - alpha) * level;
    st_i[1] = (di * meany + y) / (di + 1);
    if (mt == DSTM || mt == DOTM) {
        st_i[3] = ((di - 1) * Bn + 6 * (y - meany) / (di + 1)) / (di + 2);
        st_i[2] = st_i[1] - st_i[3] * (di + 2) / 2;
    } else {
        st_i[2] = An;
        st_i[3] = Bn;
    }
}

void update(States& st, size_t i, Model mt, double alpha, double theta, double y, bool usemu) {
    update_row(st[i - 1], st[i], i, mt, alpha, theta, y, usemu);
}

// forecast(states, i, ...): h steps from row i - 1 (statsforecast copies rows 0..i-1 first; only i - 1 is read)
void forecast(const States& st, size_t i, Model mt, Vec& f, double alpha, double theta) {
    std::array<double, 5> prev = st[i - 1], cur{0, 0, 0, 0, 0};
    for (size_t j = 0; j < f.size(); ++j) {
        cur = {0, 0, 0, 0, 0};
        update_row(prev, cur, i + j, mt, alpha, theta, 0.0, true);
        f[j] = cur[4];
        prev = cur;
    }
}

double calc(const Vec& y, States& st, Model mt, double init_smoothed, double alpha, double theta, Vec& e, size_t nmse) {
    Vec denom(nmse, 0.0), f(nmse, 0.0), amse(nmse, 0.0);
    st[0] = init_state(y, mt, init_smoothed, alpha, theta);
    e[0] = y[0] - st[0][4];
    const size_t n = y.size();
    for (size_t i = 1; i < n; ++i) {
        forecast(st, i, mt, f, alpha, theta);
        if (std::fabs(f[0] - NA) < TOL) return NA;
        e[i] = y[i] - f[0];
        for (size_t j = 0; j < nmse; ++j)
            if (i + j < n) {
                denom[j] += 1.0;
                const double tmp = y[i + j] - f[j];
                amse[j] = (amse[j] * (denom[j] - 1.0) + tmp * tmp) / denom[j];
            }
        update(st, i, mt, alpha, theta, y[i], false);
    }
    double mean_y = 0.0;
    for (double v : y) mean_y += std::fabs(v);
    mean_y /= static_cast<double>(n);
    if (mean_y < TOL) mean_y = TOL;
    double ss = 0.0;
    for (size_t i = 3; i < n; ++i) ss += e[i] * e[i];
    return ss / mean_y;
}

struct Target {
    const Vec& y;
    Model mt;
    double init_level, init_alpha, init_theta;
    bool opt_level, opt_alpha, opt_theta;
    size_t nmse;
    double operator()(const Vec& p) const {
        size_t j = 0;
        const double level = opt_level ? p[j++] : init_level;
        const double alpha = opt_alpha ? p[j++] : init_alpha;
        const double theta = opt_theta ? p[j++] : init_theta;
        States st(y.size());
        Vec e(y.size(), 0.0);
        double mse = calc(y, st, mt, level, alpha, theta, e, nmse);
        mse = std::max(mse, -1e10);
        if (std::isnan(mse) || std::fabs(mse + 99999) < 1e-7) mse = -INFINITY;
        return mse;
    }
};

// nm::NelderMead (statsforecast): bounded by projection, adaptive coefficients, population-SD stopping rule.
Vec nelder_mead(const Target& F, Vec x, const Vec& lower, const Vec& upper) {
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
    for (int i = 0; i <= n; ++i) fs[i] = F(S[i]);
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
        const double fr = F(xr);
        if (fs[best] <= fr && fr < fs[second]) { S[worst] = xr; fs[worst] = fr; continue; }
        if (fr < fs[best]) {
            Vec xe(n);
            for (int k = 0; k < n; ++k) xe[k] = xo[k] + gamma * (xr[k] - xo[k]);
            xe = clamp(xe);
            const double fe = F(xe);
            if (fe < fr) { S[worst] = xe; fs[worst] = fe; } else { S[worst] = xr; fs[worst] = fr; }
            continue;
        }
        if (fs[second] <= fr && fr < fs[worst]) {
            Vec xc(n);
            for (int k = 0; k < n; ++k) xc[k] = xo[k] + rho * (xr[k] - xo[k]);
            xc = clamp(xc);
            const double fc = F(xc);
            if (fc <= fr) { S[worst] = xc; fs[worst] = fc; continue; }
        } else {
            Vec xc(n);
            for (int k = 0; k < n; ++k) xc[k] = xo[k] - rho * (xr[k] - xo[k]);
            xc = clamp(xc);
            const double fc = F(xc);
            if (fc < fs[worst]) { S[worst] = xc; fs[worst] = fc; continue; }
        }
        for (int j = 0; j <= n; ++j) {
            if (j == best) continue;
            for (int k = 0; k < n; ++k) S[j][k] = S[best][k] + sigma * (S[j][k] - S[best][k]);
            S[j] = clamp(S[j]);
            fs[j] = F(S[j]);
        }
    }
    return S[best];
}

// statsmodels seasonal_decompose(x, model, period).seasonal (two-sided moving-average trend, NaN-mean per position)
Vec seasonal_component(const Vec& x, int period, bool mult) {
    const int n = static_cast<int>(x.size());
    Vec filt;
    if (period % 2 == 0) { filt.assign(period + 1, 1.0 / period); filt.front() = filt.back() = 0.5 / period; }
    else filt.assign(period, 1.0 / period);
    const int L = static_cast<int>(filt.size());
    const int head = (L + 1) / 2 - 1;
    Vec trend(n, NAN);
    for (int j = 0; j + L <= n; ++j) {
        double s = 0.0;
        for (int k = 0; k < L; ++k) s += x[j + k] * filt[L - 1 - k];
        trend[j + head] = s;
    }
    Vec avg(period);
    for (int i = 0; i < period; ++i) {
        double s = 0.0;
        int c = 0;
        for (int t = i; t < n; t += period)
            if (!std::isnan(trend[t])) { s += mult ? x[t] / trend[t] : x[t] - trend[t]; ++c; }
        avg[i] = s / c;
    }
    double m = 0.0;
    for (double v : avg) m += v;
    m /= period;
    for (double& v : avg) v = mult ? v / m : v - m;
    Vec seas(n);
    for (int t = 0; t < n; ++t) seas[t] = avg[t % period];
    return seas;
}

bool seasonality_test(const Vec& y, int m) {           // acf(y, nlags = m, fft = False); 90% one-sided test
    const int n = static_cast<int>(y.size());
    double mean = 0.0;
    for (double v : y) mean += v;
    mean /= n;
    Vec d(n);
    for (int i = 0; i < n; ++i) d[i] = y[i] - mean;
    double c0 = 0.0;
    for (double v : d) c0 += v * v;
    Vec r(m);
    for (int k = 1; k <= m; ++k) {
        double c = 0.0;
        for (int i = k; i < n; ++i) c += d[i] * d[i - k];
        r[k - 1] = (c / n) / (c0 / n);
    }
    double s2 = 0.0;
    for (int k = 0; k < m - 1; ++k) s2 += r[k] * r[k];
    const double stat = std::sqrt((1 + 2 * s2) / n);
    return std::fabs(r[m - 1]) / stat > 1.6448536269514722;   // norm.ppf(0.95)
}

}  // namespace

Vec dotm_forecast(const Vec& y0, int m, int h, bool multiplicative) {
    const int n0 = static_cast<int>(y0.size());
    if (m <= 1 || n0 < 2 * m) m = 1;                   // usable_season(n, m, None)
    double ymin = INFINITY;
    for (double v : y0) ymin = std::min(ymin, v);
    if (multiplicative && ymin <= 0) multiplicative = false;   // the Python engine switches before statsforecast
    Vec y = y0;
    bool decompose = false;
    if (m >= 4 && n0 >= 2 * m) decompose = seasonality_test(y, m);
    Vec seas;
    if (decompose) {
        bool mult = multiplicative && ymin > 0;
        seas = seasonal_component(y, m, mult);
        if (mult) for (double v : seas) if (v < 0.01) { mult = false; break; }
        if (!mult && multiplicative) seas = seasonal_component(y, m, false);
        multiplicative = mult;
        for (int i = 0; i < n0; ++i) y[i] = mult ? y[i] / seas[i] : y[i] - seas[i];
    }
    const int n = static_cast<int>(y.size());
    if (n <= 3) throw std::runtime_error("tiny datasets");
    const Target F{y, DOTM, y[0] / 2, 0.5, 2.0, true, true, true, 3};
    const Vec par = nelder_mead(F, {y[0] / 2, 0.5, 2.0}, {-1e10, 0.1, 1.0}, {1e10, 0.99, 1e10});
    States st(n);
    Vec e(n, 0.0);
    const double mse = calc(y, st, DOTM, par[0], par[1], par[2], e, 3);
    if (std::isnan(mse) || std::fabs(mse + 99999) < 1e-7) throw std::runtime_error("no model able to be fitted");
    Vec f(h);
    forecast(st, n, DOTM, f, par[1], par[2]);
    if (decompose)
        for (int j = 0; j < h; ++j) {
            const double s = seas[n - m + (j % m)];     // seasonal naive of the seasonal component, repeated
            f[j] = multiplicative ? f[j] * s : f[j] + s;
        }
    return f;
}

}  // namespace tsf
