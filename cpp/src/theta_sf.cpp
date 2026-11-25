// Port of statsforecast 2.1.1's DOTM (Apache-2.0, Nixtla). See theta_sf.hpp.
#include "theta_sf.hpp"

#include "sf_common.hpp"

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

namespace {

// auto_theta(y, m, model, decomposition_type) + forecast_theta: models fitted in order, lowest mse kept.
Vec theta_select(const Vec& y0, int m, int h, bool multiplicative, std::initializer_list<Model> models) {
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
        seas = sm_seasonal(y, m, mult);
        if (mult) for (double v : seas) if (v < 0.01) { mult = false; break; }
        if (!mult && multiplicative) seas = sm_seasonal(y, m, false);
        multiplicative = mult;
        for (int i = 0; i < n0; ++i) y[i] = mult ? y[i] / seas[i] : y[i] - seas[i];
    }
    const int n = static_cast<int>(y.size());
    if (n <= 3) throw std::runtime_error("tiny datasets");
    double best_ic = INFINITY;
    Model bm = DOTM;
    Vec bpar;
    States bst;
    for (Model mt : models) {
        // initparamtheta: level y[0] / 2 and alpha 0.5 optimised; theta 2 optimised for OTM / DOTM, fixed otherwise
        const bool opt_theta = (mt == OTM || mt == DOTM);
        const Target F{y, mt, y[0] / 2, 0.5, 2.0, true, true, opt_theta, 3};
        Vec par;
        if (opt_theta) par = sf_nelder_mead(F, {y[0] / 2, 0.5, 2.0}, {-1e10, 0.1, 1.0}, {1e10, 0.99, 1e10});
        else { par = sf_nelder_mead(F, {y[0] / 2, 0.5}, {-1e10, 0.1}, {1e10, 0.99}); par.push_back(2.0); }
        States st(n);
        Vec e(n, 0.0);
        double mse = calc(y, st, mt, par[0], par[1], par[2], e, 3);
        if (!std::isnan(mse) && std::fabs(mse + 99999) < 1e-7) mse = NAN;
        if (!std::isnan(mse) && mse < best_ic) { best_ic = mse; bm = mt; bpar = par; bst = st; }
    }
    if (std::isinf(best_ic)) throw std::runtime_error("no model able to be fitted");
    Vec f(h);
    forecast(bst, n, bm, f, bpar[1], bpar[2]);
    if (decompose)
        for (int j = 0; j < h; ++j) {
            const double s = seas[n - m + (j % m)];     // seasonal naive of the seasonal component, repeated
            f[j] = multiplicative ? f[j] * s : f[j] + s;
        }
    return f;
}

}  // namespace

Vec dotm_forecast(const Vec& y, int m, int h, bool multiplicative) { return theta_select(y, m, h, multiplicative, {DOTM}); }

Vec auto_theta_forecast(const Vec& y, int m, int h, bool multiplicative) {
    return theta_select(y, m, h, multiplicative, {STM, OTM, DSTM, DOTM});
}

// StandardTheta / OptimizedTheta / DynamicTheta: auto_theta(model = "STM" | "OTM" | "DSTM"), one model fitted.
Vec stm_forecast(const Vec& y, int m, int h, bool multiplicative) { return theta_select(y, m, h, multiplicative, {STM}); }
Vec otm_forecast(const Vec& y, int m, int h, bool multiplicative) { return theta_select(y, m, h, multiplicative, {OTM}); }
Vec dstm_forecast(const Vec& y, int m, int h, bool multiplicative) { return theta_select(y, m, h, multiplicative, {DSTM}); }
}  // namespace tsf
