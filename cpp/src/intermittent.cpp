// Port of statsforecast 2.1.1's intermittent-demand models (Apache-2.0, Nixtla): the SES kernel, the golden-section
// search and chunk_forecast follow src/ses.cpp; the model functions follow models.py (_croston_classic, _croston_sba,
// _croston_optimized, _tsb, _adida, _imapa), with numpy's pairwise summation and round-half-to-even where it matters.
#include "intermittent.hpp"

#include <cmath>
#include <stdexcept>

namespace tsf {

namespace {

// numpy's pairwise summation (np.sum / np.mean over a contiguous float64 array)
double np_sum(const double* a, size_t n) {
    if (n < 8) {
        double res = -0.0;
        for (size_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return np_sum(a, n2) + np_sum(a + n2, n - n2);
}

double np_mean(const Vec& v) { return np_sum(v.data(), v.size()) / static_cast<double>(v.size()); }

double ses_sse(double alpha, const Vec& x) {
    if (x.size() < 2) return 0.0;
    const double complement = 1.0 - alpha;
    double forecast = x[0], sse = 0.0;
    for (size_t i = 1; i < x.size(); ++i) {
        forecast = alpha * x[i - 1] + complement * forecast;
        const double err = x[i] - forecast;
        sse += err * err;
    }
    return sse;
}

double ses_forecast(const Vec& x, double alpha) {      // one-step forecast (statsforecast ses_forecast)
    if (x.empty()) throw std::invalid_argument("ses_forecast requires at least 1 data point");
    if (x.size() == 1) return x[0];
    const double complement = 1.0 - alpha;
    double f = x[0];                                    // fitted[0] = x[0] during the recursion
    for (size_t i = 1; i < x.size(); ++i) f = alpha * x[i - 1] + complement * f;
    return alpha * x.back() + complement * f;
}

double optimized_ses_forecast(const Vec& x) { return ses_forecast(x, golden_section_ses(x, 0.1, 0.3)); }

double chunk_forecast(const Vec& y, int level) {
    if (level <= 0) throw std::invalid_argument("chunk_forecast: aggregation_level must be >= 1");
    const int n = static_cast<int>(y.size());
    const int lost = n % level, n_cut = n - lost;
    if (n_cut < level) return y[n - 1];
    const int n_chunks = n_cut / level;
    Vec sums(n_chunks);
    for (int i = 0; i < n_chunks; ++i) {
        double s = 0.0;
        const int base = lost + i * level;
        for (int j = 0; j < level; ++j) s += y[base + j];
        sums[i] = s;
    }
    if (n_chunks <= 1) return sums[0];
    return ses_forecast(sums, golden_section_ses(sums));
}

Vec demand(const Vec& y) {                              // x[x > 0]
    Vec o;
    for (double v : y) if (v > 0) o.push_back(v);
    return o;
}

Vec intervals(const Vec& y) {                           // np.diff(nonzero_idxs + 1, prepend=0)
    Vec o;
    long prev = 0;
    for (size_t i = 0; i < y.size(); ++i)
        if (y[i] != 0) { const long k = static_cast<long>(i) + 1; o.push_back(static_cast<double>(k - prev)); prev = k; }
    return o;
}

bool all_zero(const Vec& y) {
    for (double v : y) if (v != 0) return false;
    return true;
}

double croston_mean(const Vec& y, bool optimized) {
    const Vec yd = demand(y);
    if (yd.empty()) return y.back();                    // _naive
    const Vec yi = intervals(y);
    const double ydp = optimized ? optimized_ses_forecast(yd) : ses_forecast(yd, 0.1);
    const double yip = optimized ? optimized_ses_forecast(yi) : ses_forecast(yi, 0.1);
    return yip != 0.0 ? ydp / yip : ydp;
}

}  // namespace

double golden_section_ses(const Vec& x, double lower, double upper) {
    const double gr = (std::sqrt(5.0) + 1.0) / 2.0;
    double a = lower, b = upper;
    double c = b - (b - a) / gr, d = a + (b - a) / gr;
    double fc = ses_sse(c, x), fd = ses_sse(d, x);
    while (std::fabs(b - a) >= 1e-12) {
        if (fc < fd) {
            b = d; d = c; fd = fc;
            c = b - (b - a) / gr;
            fc = ses_sse(c, x);
        } else if (fd < fc) {
            a = c; c = d; fc = fd;
            d = a + (b - a) / gr;
            fd = ses_sse(d, x);
        } else {
            break;
        }
    }
    return (b + a) / 2.0;
}

// The Python engine returns zeros for an all-zero series before calling statsforecast (py_sf.forecast_spec).
Vec croston_classic(const Vec& y, int h) {
    if (all_zero(y)) return Vec(h, 0.0);
    return Vec(h, croston_mean(y, false));
}

Vec croston_sba(const Vec& y, int h) {
    if (all_zero(y)) return Vec(h, 0.0);
    return Vec(h, croston_mean(y, false) * 0.95);
}

Vec croston_optimized(const Vec& y, int h) {
    if (all_zero(y)) return Vec(h, 0.0);
    return Vec(h, croston_mean(y, true));
}

Vec tsb(const Vec& y, int h, double alpha_d, double alpha_p) {
    if (all_zero(y)) return Vec(h, 0.0);
    Vec yp(y.size());
    for (size_t i = 0; i < y.size(); ++i) yp[i] = (y[i] != 0) ? 1.0 : 0.0;
    const double ypf = ses_forecast(yp, alpha_p);
    const double ydf = ses_forecast(demand(y), alpha_d);   // throws if no positive value, as statsforecast
    return Vec(h, ypf * ydf);
}

Vec adida(const Vec& y, int h) {
    if (all_zero(y)) return Vec(h, 0.0);
    const int level = static_cast<int>(std::nearbyint(np_mean(intervals(y))));   // round half to even
    return Vec(h, chunk_forecast(y, level) / level);
}

Vec imapa(const Vec& y, int h) {
    if (all_zero(y)) return Vec(h, 0.0);
    const int max_level = static_cast<int>(std::nearbyint(np_mean(intervals(y))));
    Vec forecasts(max_level);
    const int n = static_cast<int>(y.size());
    for (int level = 1; level <= max_level; ++level) {
        const int lost = n % level;
        const int n_chunks = (n - lost) / level;
        Vec sums(n_chunks);
        for (int i = 0; i < n_chunks; ++i) sums[i] = np_sum(y.data() + lost + static_cast<size_t>(i) * level, level);
        forecasts[level - 1] = optimized_ses_forecast(sums) / level;
    }
    return Vec(h, np_mean(forecasts));
}

}  // namespace tsf
