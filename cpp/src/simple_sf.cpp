#include "simple_sf.hpp"

#include <cmath>

#include "intermittent.hpp"
#include "sf_common.hpp"

namespace tsf {

namespace {

int usable_season(int n, int m) { return (m <= 1 || n < 2 * m) ? 1 : m; }

double ses_forecast(const Vec& x, double alpha) {      // statsforecast src/ses.cpp ses_forecast (one-step forecast)
    if (x.size() == 1) return x[0];
    const double complement = 1.0 - alpha;
    double f = x[0];
    for (size_t i = 1; i < x.size(); ++i) f = alpha * x[i - 1] + complement * f;
    return alpha * x.back() + complement * f;
}

}  // namespace

Vec historic_average(const Vec& y, int h) { return Vec(h, np_mean(y.data(), y.size())); }

Vec window_average(const Vec& y, int h, int w) {
    if (static_cast<int>(y.size()) < w) return Vec(h, NAN);
    return Vec(h, np_mean(y.data() + y.size() - w, static_cast<size_t>(w)));
}

Vec seasonal_window_average(const Vec& y, int h, int m, int w) {
    const int n = static_cast<int>(y.size());
    const int s = usable_season(n, m), min_samples = s * w;
    if (n < min_samples) return Vec(h, NAN);
    Vec avg(s, 0.0);
    const double* base = y.data() + (n - min_samples);
    for (int j = 0; j < s; ++j) avg[j] = base[j];       // reshape(w, s).mean(axis = 0): rows added in order
    for (int r = 1; r < w; ++r)
        for (int j = 0; j < s; ++j) avg[j] += base[r * s + j];
    for (double& v : avg) v /= w;
    Vec out(h);
    for (int k = 0; k < h; ++k) out[k] = avg[k % s];
    return out;
}

Vec seasonal_es_optimized(const Vec& y, int h, int m) {
    const int n = static_cast<int>(y.size());
    const int s = usable_season(n, m);
    if (n < s) return Vec(h, NAN);
    Vec vals(s);
    for (int i = 0; i < s; ++i) {
        Vec x;
        for (int t = i + n % s; t < n; t += s) x.push_back(y[t]);
        vals[i] = ses_forecast(x, golden_section_ses(x, 0.01, 0.99));
    }
    Vec out(h);
    for (int k = 0; k < h; ++k) out[k] = vals[k % s];
    return out;
}

Vec zero_model(int h) { return Vec(h, 0.0); }

}  // namespace tsf
