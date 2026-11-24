// Port of statsforecast 2.1.1's AutoCES. See ces_sf.hpp.
#include "ces_sf.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "sf_common.hpp"

namespace tsf {

namespace {

enum Season { NONE = 0, SIMPLE = 1, PARTIAL = 2, FULL = 3 };
constexpr double TOL = 1.0e-10, NA = -99999.0;

struct Mat {                                            // row-major states (rows x cols)
    int rows = 0, cols = 0;
    Vec v;
    Mat() = default;
    Mat(int r, int c) : rows(r), cols(c), v(static_cast<size_t>(r) * c, 0.0) {}
    double& operator()(long i, int j) { return v[static_cast<size_t>(i) * cols + j]; }
    double operator()(long i, int j) const { return v[static_cast<size_t>(i) * cols + j]; }
};

void cesupdate(Mat& st, long i, int m, int season, double a0, double a1, double b0, double b1, double y) {
    double e;
    if (season == NONE || season == PARTIAL || season == FULL) e = y - st(i - 1, 0);
    else e = y - st(i - m, 0);
    if (season > SIMPLE) e -= st(i - m, 2);
    if (season == NONE || season == PARTIAL || season == FULL) {
        const double s0 = st(i - 1, 0), s1 = st(i - 1, 1);
        st(i, 0) = s0 - (1.0 - a1) * s1 + (a0 - a1) * e;
        st(i, 1) = s0 + (1.0 - a0) * s1 + (a0 + a1) * e;
    } else {
        const double s0 = st(i - m, 0), s1 = st(i - m, 1);
        st(i, 0) = s0 - (1.0 - a1) * s1 + (a0 - a1) * e;
        st(i, 1) = s0 + (1.0 - a0) * s1 + (a0 + a1) * e;
    }
    if (season == PARTIAL) st(i, 2) = st(i - m, 2) + b0 * e;
    if (season == FULL) {
        const double s2 = st(i - m, 2), s3 = st(i - m, 3);
        st(i, 2) = s2 - (1.0 - b1) * s3 + (b0 - b1) * e;
        st(i, 3) = s2 + (1.0 - b0) * s3 + (b0 + b1) * e;
    }
}

// cesfcst_buf: buf rows 0..m-1 = states rows i-m..i-1, then h forecast steps (buf must have >= m + h rows)
void cesfcst_buf(Mat& buf, const Mat& st, long i, int m, int season, Vec& f, int h, double a0, double a1, double b0,
                 double b1) {
    for (int r = 0; r < m; ++r) for (int c = 0; c < st.cols; ++c) buf(r, c) = st(i - m + r, c);
    for (int r = m; r < m + h; ++r) for (int c = 0; c < buf.cols; ++c) buf(r, c) = 0.0;
    for (int ih = m; ih < m + h; ++ih) {
        if (season == NONE || season == PARTIAL || season == FULL) f[ih - m] = buf(ih - 1, 0);
        else f[ih - m] = buf(ih - m, 0);
        if (season > SIMPLE) f[ih - m] += buf(ih - m, 2);
        cesupdate(buf, ih, m, season, a0, a1, b0, b1, f[ih - m]);
    }
}

void inline_forecast(const Mat& st, long i, int m, int season, Vec& f, int nmse, double a0, double a1) {
    const double ca1 = 1.0 - a1, ca0 = 1.0 - a0;
    if (season == SIMPLE) {
        for (int j = 0; j < nmse; ++j) f[j] = st(i - m + j, 0);
    } else if (season == NONE) {
        double s0 = st(i - 1, 0), s1 = st(i - 1, 1);
        f[0] = s0;
        for (int j = 1; j < nmse; ++j) {
            const double n0 = s0 - ca1 * s1, n1 = s0 + ca0 * s1;
            s0 = n0; s1 = n1;
            f[j] = s0;
        }
    } else {
        double s0 = st(i - 1, 0), s1 = st(i - 1, 1);
        f[0] = s0 + st(i - m, 2);
        for (int j = 1; j < nmse; ++j) {
            const double n0 = s0 - ca1 * s1, n1 = s0 + ca0 * s1;
            s0 = n0; s1 = n1;
            f[j] = s0 + st(i - m + j, 2);
        }
    }
}

double pass(const Vec& y, Mat& st, int m, int season, double a0, double a1, double b0, double b1, Vec& e, Vec& amse,
            Vec& denom, int nmse, Vec& f) {
    const long n = static_cast<long>(y.size());
    double lik = 0.0;
    const bool use_inline = (season <= SIMPLE || nmse <= m);
    Mat fb;
    if (!use_inline) fb = Mat(m + nmse, st.cols);
    for (long i = m; i < n + m; ++i) {
        if (use_inline) inline_forecast(st, i, m, season, f, nmse, a0, a1);
        else cesfcst_buf(fb, st, i, m, season, f, nmse, a0, a1, b0, b1);
        if (std::fabs(f[0] - NA) < TOL) return NA;
        e[i - m] = y[i - m] - f[0];
        for (int j = 0; j < nmse; ++j)
            if (i - m + j < n) {
                denom[j] += 1.0;
                const double tmp = y[i - m + j] - f[j];
                amse[j] = (amse[j] * (denom[j] - 1.0) + tmp * tmp) / denom[j];
            }
        cesupdate(st, i, m, season, a0, a1, b0, b1, y[i - m]);
        lik += e[i - m] * e[i - m];
    }
    return lik;
}

void trailing(Mat& st, long n, int m, int season, Vec& f, double a0, double a1, double b0, double b1, Mat& buf) {
    cesfcst_buf(buf, st, n + m, m, season, f, m, a0, a1, b0, b1);
    for (int r = 0; r < m; ++r) for (int c = 0; c < st.cols; ++c) st(st.rows - m + r, c) = buf(m + r, c);
}

void reverse_rows(Mat& st) {
    for (int i = 0; i < st.rows / 2; ++i)
        for (int c = 0; c < st.cols; ++c) std::swap(st(i, c), st(st.rows - 1 - i, c));
}

// cescalc with backfit = 1: forward, backward, forward; returns n log(SSE of the last pass) or NA
double cescalc(const Vec& y, Mat& st, int m, int season, double a0, double a1, double b0, double b1, Vec& e, int nmse) {
    Vec denom(nmse, 0.0), amse(nmse, 0.0);
    const int me = (season == NONE) ? 1 : m;
    Vec f(std::max(nmse, me), 0.0);
    const long n = static_cast<long>(y.size());
    Vec ym = y;
    Mat buf(2 * me, st.cols);
    double lik = pass(ym, st, me, season, a0, a1, b0, b1, e, amse, denom, nmse, f);
    if (std::fabs(lik - NA) < TOL) return NA;
    trailing(st, n, me, season, f, a0, a1, b0, b1, buf);
    std::reverse(ym.begin(), ym.end()); reverse_rows(st); std::reverse(e.begin(), e.end());
    lik = pass(ym, st, me, season, a0, a1, b0, b1, e, amse, denom, nmse, f);
    if (std::fabs(lik - NA) < TOL) return NA;
    trailing(st, n, me, season, f, a0, a1, b0, b1, buf);
    std::reverse(ym.begin(), ym.end()); reverse_rows(st); std::reverse(e.begin(), e.end());
    lik = pass(ym, st, me, season, a0, a1, b0, b1, e, amse, denom, nmse, f);
    if (std::fabs(lik - NA) < TOL) return NA;
    trailing(st, n, me, season, f, a0, a1, b0, b1, buf);
    return static_cast<double>(n) * std::log(lik);
}

struct Fit { double aicc = INFINITY; Mat st; int season = NONE, m = 1; double a0 = 0, a1 = 0, b0 = 0, b1 = 0; bool ok = false; };

Fit cesmodel(const Vec& y, int m, int season) {
    if (season == NONE) m = 1;
    const int n = static_cast<int>(y.size());
    const int comps = 2 + (season == PARTIAL) + 2 * (season == FULL);
    const int lags = season == NONE ? 1 : m;
    Mat init(lags, comps);
    if (season == NONE) {
        const int idx = std::min(std::max(10, m), n);
        const double mu = np_mean(y.data(), idx);
        init(0, 0) = mu; init(0, 1) = mu / 1.1;
    } else if (season == SIMPLE) {
        for (int r = 0; r < lags; ++r) { init(r, 0) = y[r]; init(r, 1) = y[r] / 1.1; }
    } else {
        const double mu = np_mean(y.data(), lags);
        const Vec sd = sm_seasonal(y, lags, false);
        for (int r = 0; r < lags; ++r) {
            init(r, 0) = mu; init(r, 1) = mu / 1.1; init(r, 2) = sd[r];
            if (season == FULL) init(r, 3) = sd[r] / 1.1;
        }
    }
    // initparamces: optimised parameters in the order alpha_0, alpha_1, [beta_0, [beta_1]]; positional bounds
    double a0 = 1.3, a1 = 1.0, b0 = NAN, b1 = NAN;
    Vec x0{a0, a1};
    if (season == PARTIAL) { b0 = 0.1; x0.push_back(b0); }
    if (season == FULL) { b0 = 1.3; b1 = 1.0; x0.push_back(b0); x0.push_back(b1); }
    const Vec lo_all{0.01, 0.01, 0.01, 0.01}, up_all{1.8, 1.9, 1.5, 1.5};
    const Vec lower(lo_all.begin(), lo_all.begin() + x0.size()), upper(up_all.begin(), up_all.begin() + x0.size());
    const bool ob0 = season == PARTIAL || season == FULL, ob1 = season == FULL;
    auto make_states = [&]() {
        Mat st(n + 2 * m, comps);
        for (int r = 0; r < lags; ++r) for (int c = 0; c < comps; ++c) st(r, c) = init(r, c);
        return st;
    };
    auto target = [&](const Vec& p) {
        size_t j = 0;
        const double A0 = p[j++], A1 = p[j++];
        const double B0 = ob0 ? p[j++] : b0, B1 = ob1 ? p[j++] : b1;
        Mat st = make_states();
        Vec e(n, NAN);
        double lik = cescalc(y, st, m, season, A0, A1, B0, B1, e, 3);
        if (lik < -1e10) lik = -1e10;
        if (std::isnan(lik)) lik = -INFINITY;
        if (std::fabs(lik + 99999) < 1e-7) lik = -INFINITY;
        return lik;
    };
    const Vec par = sf_nelder_mead(target, x0, lower, upper);
    size_t j = 0;
    a0 = par[j++]; a1 = par[j++];
    if (ob0) b0 = par[j++];
    if (ob1) b1 = par[j++];
    Fit fit;
    fit.st = make_states();
    Vec e(n, NAN);
    double lik = cescalc(y, fit.st, m, season, a0, a1, b0, b1, e, 3);
    if (!std::isnan(lik) && std::fabs(lik + 99999) < 1e-7) lik = NAN;
    const int np_ = comps + 1;
    const double aic = lik + 2 * np_;
    fit.aicc = (n - np_ - 1 != 0) ? aic + 2.0 * np_ * (np_ + 1) / (n - np_ - 1) : INFINITY;
    fit.season = season; fit.m = m; fit.a0 = a0; fit.a1 = a1; fit.b0 = b0; fit.b1 = b1;
    fit.ok = true;
    return fit;
}

}  // namespace

Vec ces_forecast(const Vec& y, int m, int h, char model, int max_season) {
    const int n = static_cast<int>(y.size());
    if (m <= 1 || m > max_season || n < 2 * m) m = 1;  // usable_season(n, m, max_season)
    // statsforecast auto_ces(model): "Z" tries N, S, P, F; another letter fits that type only; "N" without a season
    std::vector<int> types;
    char st = (m < 1 || n < 2 * m || m == 1) ? 'N' : model;
    switch (st) {
        case 'Z': types = {NONE, SIMPLE, PARTIAL, FULL}; break;
        case 'N': types = {NONE}; break;
        case 'S': types = {SIMPLE}; break;
        case 'P': types = {PARTIAL}; break;
        case 'F': types = {FULL}; break;
        default: throw std::invalid_argument("auto_ces: model must be Z, N, S, P or F");
    }
    if (model != 'Z' && model != 'N' && model != 'S' && model != 'P' && model != 'F')
        throw std::invalid_argument("auto_ces: model must be Z, N, S, P or F");
    int npars = 2;                                      // "P" -> 3; "F" / "Z" -> 4
    if (st == 'P') npars += 1;
    if (st == 'F' || st == 'Z') npars += 2;
    if (n <= npars) throw std::runtime_error("tiny datasets");
    Fit best;
    double best_ic = INFINITY;
    for (int type : types) {
        Fit f = cesmodel(y, m, type);
        if (!std::isnan(f.aicc) && f.aicc < best_ic) { best_ic = f.aicc; best = std::move(f); }
    }
    if (std::isinf(best_ic)) throw std::runtime_error("no model able to be fitted");
    const int me = best.season == NONE ? 1 : best.m;
    Mat buf(me + h, best.st.cols);
    Vec f(std::max(h, me), 0.0);
    cesfcst_buf(buf, best.st, me + n, me, best.season, f, h, best.a0, best.a1, best.b0, best.b1);
    f.resize(h);
    return f;
}

}  // namespace tsf
