// Port of R forecast 9.0.2's dshw(), par_dshw(), dshw.mse() and seasindex() (R/dshw.R, GPL-3, Rob J Hyndman and
// contributors), with zoo's rollmean() (cumulative sums in long double, as R's cumsum) and optim()'s Nelder-Mead
// (r_optim.hpp). See dshw.hpp.
//
//   yhat[i] = (s + t) * I[i] * w[i]                       I: period1 indices, w: period2 indices
//   s' = alpha * y / (I w) + (1 - alpha)(s + t),  t' = beta (s' - s) + (1 - beta) t
//   I[i + p1] = gamma * y / (s' w) + (1 - gamma) I[i],  w[i + p2] = omega * y / (s' I) + (1 - omega) w[i]
//   AR(1) adjustment of the errors with phi; (alpha, beta, gamma, omega, phi) minimise the in-sample mse.
#include "dshw.hpp"

#include <algorithm>
#include <stdexcept>

#include "r_optim.hpp"
#include "rstats.hpp"

namespace tsf {

namespace {

// zoo::rollmean(x, k): y[1] = sum(x[1:k]), y[i] = x[k + i - 1] - x[i - 1]; cumsum(y) / k
Vec rollmean(const Vec& x, int k) {
    const int n = static_cast<int>(x.size());
    Vec out(n - k + 1);
    long double s = 0;
    for (int i = 0; i < k; ++i) s += x[i];              // sum(): long double
    const double y1 = static_cast<double>(s);
    long double cs = 0;
    for (int i = 0; i <= n - k; ++i) {
        const double yi = (i == 0) ? y1 : x[k + i - 1] - x[i - 1];
        cs += yi;                                       // cumsum(): long double running sum, stored as double
        out[i] = static_cast<double>(cs) / k;
    }
    return out;
}

// seasindex(y, p): ratio-to-moving-average seasonal indices of the first two periods (p as a double in R, so
// identical(p %% 2, 0) holds for an even period)
Vec seasindex(const Vec& y, int p) {
    const int n2 = 2 * p;
    const Vec shorty(y.begin(), y.begin() + n2);
    Vec average(std::max<size_t>(y.size(), static_cast<size_t>(n2)), 0.0);
    const Vec simplema = rollmean(shorty, p);
    Vec si(p);
    if (p % 2 == 0) {
        const Vec centeredma = rollmean(Vec(simplema.begin(), simplema.begin() + (n2 - p + 1)), 2);
        for (int j = 1; j <= p; ++j) average[p / 2 + j - 1] = shorty[p / 2 + j - 1] / centeredma[j - 1];
        int o = 0;
        for (int j = 1; j <= p / 2; ++j) si[o++] = average[p + j - 1];
        for (int j = 1 + p / 2; j <= p; ++j) si[o++] = average[j - 1];
    } else {
        const int q = (p - 1) / 2;
        for (int j = 1; j <= p; ++j) average[q + j - 1] = shorty[q + j - 1] / simplema[j - 1];
        int o = 0;
        for (int j = 1; j <= q; ++j) si[o++] = average[p + j - 1];
        for (int j = 1 + q; j <= p; ++j) si[o++] = average[j - 1];
    }
    return si;
}

struct Start {                                          // what does not depend on the parameters
    Vec I, w;
    double t = 0, s = 0;
};

Start make_start(const Vec& y, int p1, int p2) {
    Start S;
    S.I = seasindex(y, p1);
    S.w = seasindex(y, p2);
    for (int j = 0; j < p2; ++j) S.w[j] = S.w[j] / S.I[j % p1];        // wstart / rep(I, ratio)
    Vec v(p2);
    for (int j = 0; j < p2; ++j) {
        const double x = (j == 0) ? 0.0 : y[j] - y[j - 1];             // c(0, diff(y[1:period2]))
        v[j] = ((y[j] - y[p2 + j]) / p2) + x;
    }
    S.t = rs::mean(v.data(), v.size()) / 2;
    S.s = rs::mean(y.data(), static_cast<size_t>(2 * p2)) - (p2 + 0.5) * S.t;
    return S;
}

// one pass of dshw() with given parameters; returns the mse, fills fc (h values) when it is not null
double run(const Vec& y, int p1, int p2, const Start& S0, double alpha, double beta, double gamma, double omega,
           double phi, bool armethod, Vec* fc, int h) {
    const int n = static_cast<int>(y.size());
    if (!armethod) phi = 0;
    Vec I(n + p1), w(n + p2), yhat(n), e(n);
    std::copy(S0.I.begin(), S0.I.end(), I.begin());
    std::copy(S0.w.begin(), S0.w.end(), w.begin());
    double s = S0.s, t = S0.t;
    for (int i = 0; i < n; ++i) {
        yhat[i] = (s + t) * I[i] * w[i];
        const double snew = alpha * (y[i] / (I[i] * w[i])) + (1 - alpha) * (s + t);
        const double tnew = beta * (snew - s) + (1 - beta) * t;
        I[i + p1] = gamma * (y[i] / (snew * w[i])) + (1 - gamma) * I[i];
        w[i + p2] = omega * (y[i] / (snew * I[i])) + (1 - omega) * w[i];
        s = snew;
        t = tnew;
    }
    for (int i = 0; i < n; ++i) e[i] = y[i] - yhat[i];
    if (fc) {
        fc->resize(h);
        for (int k = 1; k <= h; ++k) {
            double f = (s + k * t) * I[n + (k - 1) % p1] * w[n + (k - 1) % p2];
            if (armethod) f = f + rs::r_pow(phi, static_cast<double>(k)) * e[n - 1];   // the error before the AR step
            (*fc)[k - 1] = f;
        }
    }
    if (armethod) {                                     // yhat <- yhat + phi * c(0, e[-n]); e <- y - yhat
        for (int i = n - 1; i >= 1; --i) yhat[i] = yhat[i] + phi * e[i - 1];
        yhat[0] = yhat[0] + phi * 0.0;
        for (int i = 0; i < n; ++i) e[i] = y[i] - yhat[i];
    }
    for (double& v : e) v = v * v;
    return rs::mean(e.data(), e.size());                // mse <- mean(e^2)
}

}  // namespace

Vec dshw_forecast(const Vec& y, const Vec& periods, int h) {
    const int n = static_cast<int>(y.size());
    Vec per;
    for (double p : periods) if (std::isfinite(p)) per.push_back(std::floor(p));
    std::sort(per.begin(), per.end());
    if (per.size() < 2) throw std::runtime_error("dshw needs two seasonal periods");
    const double d1 = per[0], d2 = per[1];
    for (double v : y) {
        if (!std::isfinite(v)) throw std::runtime_error("dshw: missing values are not supported");
        if (v <= 0) throw std::runtime_error("dshw not suitable when data contain zeros or negative numbers");
    }
    if (d2 > 1e8 || n < 2 * d2) throw std::runtime_error("Insufficient data to estimate model");
    if (d1 < 1 || d1 == d2) throw std::runtime_error("Inappropriate periods");
    const double ratio = d2 / d1;
    if (ratio - std::trunc(ratio) > 1e-10) throw std::runtime_error("Seasonal periods are not nested");
    const int p1 = static_cast<int>(d1), p2 = static_cast<int>(d2);
    const Start S = make_start(y, p1, p2);
    auto mse = [&](const Vec& par) -> double {          // dshw.mse
        const double mx = *std::max_element(par.begin(), par.end()), mn = *std::min_element(par.begin(), par.end());
        if (mx > 0.99 || mn < 0 || par[4] > 0.9) return INF;
        return run(y, p1, p2, S, par[0], par[1], par[2], par[3], par[4], std::fabs(par[4]) > 1e-07, nullptr, 1);
    };
    double fmin = 0;
    int fail = 0;
    const Vec par = nmmin(mse, Vec{0.1, 0.01, 0.001, 0.001, 0.0}, fmin, 500, fail);     // par_dshw: optim() defaults
    Vec fc;
    run(y, p1, p2, S, par[0], par[1], par[2], par[3], par[4], true, &fc, h);
    return fc;
}

}  // namespace tsf
