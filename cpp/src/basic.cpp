#include "basic.hpp"

#include <algorithm>
#include <array>
#include <numeric>

namespace tsf {

Vec naive(const Vec& y, int h) { return Vec(h, y.back()); }

Vec snaive(const Vec& y, int h, int m) {
    const int n = static_cast<int>(y.size());
    if (!(m >= 1 && n >= m)) m = 1;
    Vec f(h);
    for (int i = 0; i < h; ++i) f[i] = y[n - m + (i % m)];
    return f;
}

Vec drift(const Vec& y, int h) {
    const int n = static_cast<int>(y.size());
    const double slope = n > 1 ? (y[n - 1] - y[0]) / (n - 1) : 0.0;
    Vec f(h);
    for (int i = 0; i < h; ++i) f[i] = y[n - 1] + slope * (i + 1);
    return f;
}

// ------------------------------------------------------------------------------------------- SES
namespace {
struct SesProfile { double sse, l0, level_n; };

// For a given alpha: SSE, the closed-form initial level and the final level (py_models._ses_profile).
// Two passes, no allocation: pass 1 finds l0 = <a,u>/<a,a>, pass 2 sums the squared errors u_t - a_t*l0.
// a_t = (1-alpha)^t is carried as a running product instead of pow() (relative difference ~1e-13 for n ~ 1000).
SesProfile ses_profile(const Vec& y, double alpha) {
    const size_t n = y.size();
    const double beta = 1.0 - alpha;
    double b = 0.0, a = 1.0, aa = 0.0, au = 0.0;
    for (size_t t = 0; t < n; ++t) {
        const double u = y[t] - b;                      // b is the level before y_t (lfilter state)
        b = alpha * y[t] + beta * b;
        aa += a * a;
        au += a * u;
        a *= beta;
    }
    const double l0 = au / aa;
    const double level_n = a * l0 + b;                  // a == beta^n here
    double sse = 0.0;
    b = 0.0; a = 1.0;
    for (size_t t = 0; t < n; ++t) {
        const double e = (y[t] - b) - a * l0;
        sse += e * e;
        b = alpha * y[t] + beta * b;
        a *= beta;
    }
    return {sse, l0, level_n};
}
}  // namespace

SesFit ses_fit(const Vec& y) {
    const int n = static_cast<int>(y.size());
    if (n < 2 || ptp(y) == 0.0) return {ALPHA_HI, y.back()};
    const Vec grid = linspace(ALPHA_LO, ALPHA_HI, 60);
    Vec sses(grid.size());
    for (size_t i = 0; i < grid.size(); ++i) sses[i] = ses_profile(y, grid[i]).sse;
    const int i = static_cast<int>(std::min_element(sses.begin(), sses.end()) - sses.begin());
    const double lo = grid[std::max(i - 1, 0)], hi = grid[std::min(i + 1, static_cast<int>(grid.size()) - 1)];
    double alpha = fminbound([&](double a) { return ses_profile(y, a).sse; }, lo, hi, 1e-10);
    if (ses_profile(y, alpha).sse > sses[i]) alpha = grid[i];
    return {alpha, ses_profile(y, alpha).level_n};
}

Vec ses_forecast(const Vec& y, int h) { return Vec(h, ses_fit(y).level); }

// ------------------------------------------------------------------------------------------- Theta (forecast::thetaf)
Vec theta_forecast(const Vec& y, int h, int m) {
    Vec x = y;
    const int n = static_cast<int>(x.size());
    bool seasonal = false;
    if (m > 1 && ptp(x) != 0.0 && n > 2 * m) {
        const Vec r = acf(x, m);
        double s = 0.0;
        for (int k = 0; k < m - 1; ++k) s += r[k] * r[k];
        const double stat = std::sqrt((1.0 + 2.0 * s) / n);
        seasonal = std::fabs(r[m - 1]) / stat > 1.6448536269514722;   // norm.ppf(0.95)
    }
    Vec seas;
    if (seasonal) {
        seas = decompose_multiplicative(x, m);
        bool bad = seas.empty();
        for (double v : seas) if (std::fabs(v) < 1e-10 || std::isnan(v)) bad = true;
        if (bad) seas.clear();
        else for (int t = 0; t < n; ++t) x[t] /= seas[t];
    }
    SesFit sf = ses_fit(x);
    const double slope = n > 1 ? ls_slope(x) : 0.0;
    const double alpha = std::max(1e-10, sf.alpha);
    const double c = (1.0 - std::pow(1.0 - alpha, n)) / alpha;
    Vec f(h);
    for (int i = 0; i < h; ++i) f[i] = sf.level + (slope / 2.0) * (i + c);
    if (!seas.empty())
        for (int i = 0; i < h; ++i) f[i] *= seas[n - m + (i % m)];
    return f;
}

// ------------------------------------------------------------------------------------------- M4 benchmarks
bool m4_seasonality_test(const Vec& x, int m) {
    const int n = static_cast<int>(x.size());
    if (m <= 1 || n < 3 * m || ptp(x) == 0.0) return false;
    const int lag_max = std::min(static_cast<int>(std::floor(10.0 * std::log10(static_cast<double>(n)))), n - 1);
    if (m > lag_max) return false;
    const Vec r = acf(x, lag_max);
    double s = 0.0;
    for (int k = 0; k < m - 1; ++k) s += r[k] * r[k];
    const double clim = 1.645 / std::sqrt(static_cast<double>(n)) * std::sqrt(1.0 + 2.0 * s);
    return std::fabs(r[m - 1]) > clim;
}

namespace {
// (deseasonalised series, seasonal indices for the horizon); indices are 1 when not seasonal.
std::pair<Vec, Vec> m4_deseasonalise(const Vec& y, int m, int h) {
    const int n = static_cast<int>(y.size());
    if (m4_seasonality_test(y, m)) {
        const Vec seas = decompose_multiplicative(y, m);
        bool ok = !seas.empty();
        for (double v : seas) if (!std::isfinite(v) || std::fabs(v) < 1e-10) ok = false;
        if (ok) {
            Vec x(n), si(h);
            for (int t = 0; t < n; ++t) x[t] = y[t] / seas[t];
            for (int i = 0; i < h; ++i) si[i] = seas[n - m + (i % m)];
            return {x, si};
        }
    }
    return {y, Vec(h, 1.0)};
}

// Holt in error-correction form: f = l + phi*b; e = y - f; l <- f + alpha*e; b <- phi*b + beta*e.
struct HoltProfile { double sse, l0, b0; };

// For fixed (alpha, beta, phi): SSE with the initial level/trend solved by least squares (py_models._holt_profile).
// Errors are affine in (l0, b0): e = e0 - C x0, where e0 uses x0 = 0 and column j of C is minus the errors of the
// zero series started from unit state j.
// One pass, no allocation: the three recursions (data from state 0, zero series from unit level, zero series from unit
// trend) run side by side and only the 2x2 normal equations are accumulated, in long double. The SSE is then
// e0'e0 - x'(C'e0). If the two state columns are (nearly) collinear the trend start is fixed at 0, as a rank-deficient
// least-squares solve would do.
HoltProfile holt_profile(const Vec& y, double alpha, double beta, double phi) {
    using LD = long double;
    double l0 = 0.0, b0 = 0.0, l1 = 1.0, b1 = 0.0, l2 = 0.0, b2 = 1.0;
    LD s11 = 0, s12 = 0, s22 = 0, s1e = 0, s2e = 0, see = 0;
    for (double yt : y) {
        const double f0 = l0 + phi * b0, e0 = yt - f0;
        const double f1 = l1 + phi * b1, c1 = f1;           // c = -(error of the zero series) = forecast
        const double f2 = l2 + phi * b2, c2 = f2;
        s11 += (LD)c1 * c1; s12 += (LD)c1 * c2; s22 += (LD)c2 * c2;
        s1e += (LD)c1 * e0; s2e += (LD)c2 * e0; see += (LD)e0 * e0;
        l0 = f0 + alpha * e0; b0 = phi * b0 + beta * e0;
        l1 = f1 - alpha * c1; b1 = phi * b1 - beta * c1;
        l2 = f2 - alpha * c2; b2 = phi * b2 - beta * c2;
    }
    if (!(s11 > 0) || !std::isfinite((double)see)) return {INF, 0.0, 0.0};
    const LD det = s11 * s22 - s12 * s12;
    LD x1, x2;
    if (det > 1e-14L * s11 * s22) {
        x1 = (s22 * s1e - s12 * s2e) / det;
        x2 = (s11 * s2e - s12 * s1e) / det;
    } else {
        x1 = s1e / s11;
        x2 = 0;
    }
    double sse = (double)(see - x1 * s1e - x2 * s2e);
    if (!std::isfinite(sse)) return {INF, 0.0, 0.0};
    if (sse < 0.0) sse = 0.0;
    return {sse, (double)x1, (double)x2};
}

struct HoltFit { double alpha, beta, phi, sse; };

HoltFit holt_fit(const Vec& y, bool damped) {
    auto unpack = [damped](const Vec& p, double& a, double& b, double& ph) {
        a = std::clamp(p[0], ALPHA_LO, ALPHA_HI);
        b = std::clamp(a * std::clamp(p[1], 0.0, 1.0), ALPHA_LO, a);
        ph = damped ? std::clamp(p[2], 0.8, 0.98) : 1.0;
    };
    auto obj = [&](const Vec& p) {
        double a, b, ph;
        unpack(p, a, b, ph);
        // tiny penalty outside the box keeps Nelder-Mead from drifting on the flat (clamped) region
        double pen = 0.0;
        if (p[0] < ALPHA_LO) pen += ALPHA_LO - p[0];
        if (p[0] > ALPHA_HI) pen += p[0] - ALPHA_HI;
        if (p[1] < 0.0) pen -= p[1];
        if (p[1] > 1.0) pen += p[1] - 1.0;
        if (damped) { if (p[2] < 0.8) pen += 0.8 - p[2]; if (p[2] > 0.98) pen += p[2] - 0.98; }
        const double s = holt_profile(y, a, b, ph).sse;
        return s * (1.0 + pen);
    };
    // A finer start grid than Python's (evaluations are cheap here): the result is then refined from the best 6 points.
    const std::array<double, 12> alphas{0.002, 0.01, 0.03, 0.06, 0.1, 0.15, 0.22, 0.3, 0.42, 0.6, 0.8, 0.98};
    const std::array<double, 8> us{0.0005, 0.003, 0.01, 0.03, 0.1, 0.25, 0.5, 0.9};
    const Vec phis = damped ? Vec{0.8, 0.85, 0.9, 0.94, 0.98} : Vec{1.0};
    std::vector<std::pair<double, Vec>> grid;
    for (double a : alphas)
        for (double u : us)
            for (double ph : phis) {
                Vec p = damped ? Vec{a, u, ph} : Vec{a, u};
                grid.emplace_back(obj(p), p);
            }
    std::sort(grid.begin(), grid.end(), [](const auto& l, const auto& r) { return l.first < r.first; });
    Vec best = grid[0].second;
    double fbest = grid[0].first;
    const Vec step = damped ? Vec{0.05, 0.1, 0.02} : Vec{0.05, 0.1};
    for (int k = 0; k < 6 && k < static_cast<int>(grid.size()); ++k) {
        double fv;
        Vec p = nelder_mead(obj, grid[k].second, step, fv, 800, 1e-11, 1e-9);
        if (fv < fbest) { fbest = fv; best = p; }
    }
    // one restart from the overall best (standard remedy for Nelder-Mead collapsing early)
    double fv2;
    Vec p2 = nelder_mead(obj, best, step, fv2, 800, 1e-12, 1e-10);
    if (fv2 < fbest) { fbest = fv2; best = p2; }
    double a, b, ph;
    unpack(best, a, b, ph);
    return {a, b, ph, holt_profile(y, a, b, ph).sse};
}
}  // namespace

double holt_best_sse(const Vec& y, bool damped) { return holt_fit(y, damped).sse; }

Vec holt_forecast(const Vec& y, int h, bool damped) {
    const int n = static_cast<int>(y.size());
    if (n < 3 || ptp(y) == 0.0) return n > 1 ? drift(y, h) : naive(y, h);
    const HoltFit fit = holt_fit(y, damped);
    const HoltProfile pr = holt_profile(y, fit.alpha, fit.beta, fit.phi);
    double l = pr.l0, b = pr.b0;
    for (int t = 0; t < n; ++t) {
        const double f = l + fit.phi * b;
        const double e = y[t] - f;
        l = f + fit.alpha * e;
        b = fit.phi * b + fit.beta * e;
    }
    Vec out(h);
    double steps = 0.0, ph = 1.0;
    for (int i = 0; i < h; ++i) {
        if (damped) { ph *= fit.phi; steps += ph; } else { steps = i + 1; }
        out[i] = l + steps * b;
    }
    return out;
}

Vec naive2(const Vec& y, int h, int m) {
    auto [x, si] = m4_deseasonalise(y, m, h);
    Vec f = naive(x, h);
    for (int i = 0; i < h; ++i) f[i] *= si[i];
    return f;
}

Vec comb_shd(const Vec& y, int h, int m) {
    auto [x, si] = m4_deseasonalise(y, m, h);
    const Vec a = ses_forecast(x, h), b = holt_forecast(x, h, false), c = holt_forecast(x, h, true);
    Vec f(h);
    for (int i = 0; i < h; ++i) f[i] = (a[i] + b[i] + c[i]) / 3.0 * si[i];
    return f;
}

}  // namespace tsf
