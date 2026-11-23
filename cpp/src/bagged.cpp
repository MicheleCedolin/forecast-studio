// See bagged.hpp.
#include "bagged.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "autoarima.hpp"
#include "ets.hpp"
#include "rstats.hpp"

namespace tsf {

namespace {

int r_nextodd(double v) { const int r = static_cast<int>(std::nearbyint(v)); return (r % 2 == 0) ? r + 1 : r; }

// stl(ts(x, frequency = period), s.window = "periodic")$time.series: (seasonal, trend, remainder)
void stl_periodic(const Vec& x, int period, Vec& seasonal, Vec& trend, Vec& remainder) {
    const int n = static_cast<int>(x.size());
    if (period < 2 || n <= 2 * period) throw std::runtime_error("series is not periodic or has less than two periods");
    const double s_window = 10.0 * n + 1;
    auto win = [](double k) { return std::max(3, r_nextodd(k)); };
    const int tw = win(std::ceil(1.5 * period / (1.0 - 1.5 / s_window)));
    const int sw = win(s_window), lw = win(r_nextodd(period));
    auto jump = [](int w) { return std::max(1, static_cast<int>(std::ceil(w / 10.0))); };
    const StlOut z = stl_core(x, period, sw, tw, lw, 0, 1, 1, jump(sw), jump(tw), jump(lw), 2);
    // periodic: seasonal <- tapply(seasonal, cycle(x), mean)[cycle(x)]
    Vec cyc(period);
    for (int c = 0; c < period; ++c) {
        Vec g;
        for (int t = c; t < n; t += period) g.push_back(z.seasonal[t]);
        cyc[c] = rs::mean(g.data(), g.size());
    }
    seasonal.resize(n);
    remainder.resize(n);
    for (int t = 0; t < n; ++t) {
        seasonal[t] = cyc[t % period];
        remainder[t] = x[t] - seasonal[t] - z.trend[t];
    }
    trend = z.trend;
}

// forecast:::MBB(x, window_size)
Vec mbb(const Vec& x, int w, rs::RRng& rng) {
    const int n = static_cast<int>(x.size());
    const int nb = n / w + 2;
    Vec bx(static_cast<size_t>(nb) * w, 0.0);
    for (int i = 0; i < nb; ++i) {
        const int c = rng.sample1(n - w + 1);           // sample(1:(n - w + 1), 1)
        for (int j = 0; j < w; ++j) bx[static_cast<size_t>(i) * w + j] = x[c - 1 + j];
    }
    const int start = rng.sample1(w);                   // sample(0:(w - 1), 1) + 1
    return Vec(bx.begin() + (start - 1), bx.begin() + (start - 1) + n);
}

}  // namespace

std::vector<Vec> bld_mbb_bootstrap(const Vec& x, int freq_in, int num, uint32_t seed) {
    const int n = static_cast<int>(x.size());
    if (n <= 1) return std::vector<Vec>(num, x);
    rs::RRng rng(seed);
    int freq = freq_in;
    if (n <= 2 * freq) freq = 1;
    const int block = (freq > 1) ? 2 * freq : std::min(8, n / 2);
    std::vector<Vec> xs;
    xs.push_back(x);
    if (num <= 1) return xs;
    double lambda = 1.0;
    if (*std::min_element(x.begin(), x.end()) > 1e-6) lambda = rs::box_cox_lambda(x, 0.0, 1.0, freq_in);
    Vec xbc(n);
    for (int t = 0; t < n; ++t) xbc[t] = rs::box_cox(x[t], lambda);
    Vec seasonal, trend, remainder;
    if (freq > 1) {
        stl_periodic(xbc, freq, seasonal, trend, remainder);
    } else {
        throw std::runtime_error("bagged_ets: non-seasonal series (frequency 1 or fewer than two periods) are not "
                                 "supported yet (R uses loess here)");
    }
    for (int i = 1; i < num; ++i) {
        const Vec r = mbb(remainder, block, rng);
        Vec b(n);
        for (int t = 0; t < n; ++t) b[t] = rs::inv_box_cox(trend[t] + seasonal[t] + r[t], lambda);
        xs.push_back(std::move(b));
    }
    return xs;
}

Vec bagged_ets(const Vec& y, const Vec& periods, int h, int num, uint32_t seed) {
    for (double v : y) if (std::isnan(v)) throw std::runtime_error("bagged_ets: missing values are not supported");
    const int freq = r_frequency(periods.data(), static_cast<int>(periods.size()));
    const std::vector<Vec> xs = bld_mbb_bootstrap(y, freq, num, seed);
    std::vector<long double> acc(h, 0.0L);
    for (const Vec& x : xs) {
        for (double v : x) if (!std::isfinite(v)) throw std::runtime_error("bagged_ets: a bootstrapped series is not finite");
        const Vec f = ets_auto(x, freq, h).forecast;
        for (int j = 0; j < h; ++j) acc[j] += f[j];
    }
    Vec out(h);                                           // rowMeans: long double sum / count
    for (int j = 0; j < h; ++j) out[j] = static_cast<double>(acc[j] / static_cast<long double>(xs.size()));
    return out;
}

}  // namespace tsf
