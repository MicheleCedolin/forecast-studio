// The tsfeatures set of the ranking study for tsfcore (C++17): a port of src/tsfstudio/tsfeat.py (itself a port of the R
// package tsfeatures as the study called it) and of features.tsfeatures_table. Same definitions, the same fallbacks
// (a value that is NA is tried again at frequency 1, then with width / lags 1) and the same rule that a feature some
// series lack is left out for the whole dataset. The heavy parts were already here: forecast::ets "AAN"
// (holt_parameters), fracdiff (hurst), robust stl() and supsmu() (stl_features).
// tools/validate_tsfeat.py compares this file with the Python module; the two agree to rounding (the least-squares
// fits use another factorisation than NumPy's LAPACK, and dot products are summed in another order).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "autoarima.hpp"
#include "ets.hpp"
#include "fracdiff.hpp"
#include "metrics.hpp"
#include "rstats.hpp"
#include "supsmu.hpp"
#include "util.hpp"

namespace tsf {
namespace tsfeat {

namespace {

const double NA = std::numeric_limits<double>::quiet_NaN();
const double PI = 3.14159265358979323846;

using Row = std::vector<std::pair<std::string, double>>;
struct Raise {};                          // where the Python module raises (the caller then falls back)

double np_sum(const Vec& a) { return metrics::pairwise_sum(a.data(), a.size()); }
double np_mean(const Vec& a) { return np_sum(a) / static_cast<double>(a.size()); }

double dot(const double* a, const double* b, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

// tsfeat.r_var: var() of the non-missing values
double r_var(const Vec& x0) {
    Vec x;
    x.reserve(x0.size());
    for (double v : x0)
        if (!std::isnan(v)) x.push_back(v);
    if (x.size() < 2) return NA;
    const double m = np_mean(x);
    Vec d(x.size());
    for (size_t i = 0; i < x.size(); ++i) d[i] = (x[i] - m) * (x[i] - m);
    return np_sum(d) / static_cast<double>(x.size() - 1);
}

bool constant(const Vec& x) {
    for (double v : x)
        if (!(v == x[0])) return false;
    return true;
}

// tsfeatures:::scalets, with R's long double sums (capi.cpp, tsf_r_scale)
Vec scalets(const Vec& x) {
    if (constant(x)) return x;
    const size_t n = x.size();
    long double s = 0.0L;
    for (size_t i = 0; i < n; ++i) s += x[i];
    const double mean = static_cast<double>(s / n);
    Vec out(n);
    long double ss = 0.0L;
    for (size_t i = 0; i < n; ++i) {
        out[i] = x[i] - mean;
        ss += static_cast<long double>(out[i] * out[i]);
    }
    const double sd = std::sqrt(static_cast<double>(ss) / static_cast<double>(n - 1));
    for (size_t i = 0; i < n; ++i) out[i] /= sd;
    return out;
}

Vec diff(const Vec& x, int times = 1) {
    Vec d = x;
    for (int t = 0; t < times; ++t) {
        if (d.size() < 2) return Vec();
        Vec e(d.size() - 1);
        for (size_t i = 0; i + 1 < d.size(); ++i) e[i] = d[i + 1] - d[i];
        d.swap(e);
    }
    return d;
}

// GCC 16.2 at -O2 / -O3 clones r_pacf for the constant lag 5 (interprocedural constant propagation) and then calls
// that clone for the seasonal lag as well, so seas_pacf came out missing; -fno-ipa-cp or this attribute avoids it.
#if defined(__GNUC__) && !defined(__clang__)
#define TSFEAT_NOIPA __attribute__((noipa))
#else
#define TSFEAT_NOIPA
#endif

// stats::acf(x, lag.max)$acf, lags 0..min(lag.max, n - 1)
TSFEAT_NOIPA Vec r_acf(const Vec& x, long lag_max) {
    const long n = static_cast<long>(x.size());
    lag_max = std::min(lag_max, n - 1);
    const double m = np_mean(x);
    Vec d(n);
    for (long i = 0; i < n; ++i) d[i] = x[i] - m;
    Vec acov(lag_max + 1);
    for (long k = 0; k <= lag_max; ++k) acov[k] = dot(d.data(), d.data() + k, static_cast<size_t>(n - k)) / static_cast<double>(n);
    const double a0 = acov[0];
    for (double& v : acov) v /= a0;
    return acov;
}

// stats::pacf(x, lag.max)$acf (Durbin-Levinson, R's uacf2pacf)
TSFEAT_NOIPA Vec r_pacf(const Vec& x, long lag_max) {
    const Vec rho = r_acf(x, lag_max);
    const long nlag = static_cast<long>(rho.size()) - 1;
    Vec out(std::max<long>(nlag, 0), 0.0);
    if (nlag < 1) return out;
    Vec phi(nlag + 1, 0.0), next(nlag + 1, 0.0);
    phi[1] = rho[1];
    out[0] = rho[1];
    for (long ll = 2; ll <= nlag; ++ll) {
        double a = rho[ll], b = 1.0;
        for (long i = 1; i < ll; ++i) {
            a -= phi[i] * rho[ll - i];
            b -= phi[i] * rho[i];
        }
        const double c = a / b;
        out[ll - 1] = c;
        for (long i = 1; i < ll; ++i) next[i] = phi[i] - c * phi[ll - i];
        next[ll] = c;
        phi = next;
    }
    return out;
}

double seq_get(const Vec& a, long k) { return (k >= 1 && k <= static_cast<long>(a.size())) ? a[k - 1] : NA; }
double sumsq_first(const Vec& a, long k) {
    if (static_cast<long>(a.size()) < k) return NA;
    Vec s(a.begin(), a.begin() + k);
    for (double& v : s) v *= v;
    return np_sum(s);
}
double sumsq(const Vec& a) { return sumsq_first(a, static_cast<long>(a.size())); }

// R's dnorm(x, mean, sd) (dnorm4, with its accurate branch for |z| >= 5)
double dnorm(double x, double mean, double sd) {
    const double z = std::fabs((x - mean) / sd);
    double v = 0.0;
    if (z < 5) {
        v = std::exp(-0.5 * z * z);
    } else if (z <= std::sqrt(-2 * std::log(2.0) * (-1021 + 1 - 53))) {
        const double x1 = std::ldexp(std::nearbyint(std::ldexp(z, 16)), -16);
        const double x2 = z - x1;
        v = std::exp(-0.5 * x1 * x1) * std::exp((-0.5 * x2 - x1) * x2);
    }
    return v * (1.0 / std::sqrt(2 * PI)) / sd;
}

double quantile7(const Vec& sorted, double p) {
    const double h = (static_cast<double>(sorted.size()) - 1) * p;
    const double lo = std::floor(h);
    const size_t i = static_cast<size_t>(lo), j = std::min(i + 1, sorted.size() - 1);
    return sorted[i] + (h - lo) * (sorted[j] - sorted[i]);
}

double bw_nrd0(const Vec& x) {
    const double hi = std::sqrt(r_var(x));
    Vec s = x;
    std::sort(s.begin(), s.end());
    double lo = std::min(hi, (quantile7(s, 0.75) - quantile7(s, 0.25)) / 1.34);
    if (!(lo != 0)) lo = hi != 0 ? hi : (std::fabs(x[0]) != 0 ? std::fabs(x[0]) : 1.0);
    return 0.9 * lo * std::pow(static_cast<double>(x.size()), -0.2);
}

// Minimum-norm least squares, as numpy.linalg.lstsq(X, y, rcond = None): a one-sided Jacobi SVD of the columns;
// singular values below eps * max(n, k) times the largest count as zero.
Vec lstsq(const std::vector<Vec>& cols, const Vec& y) {
    const size_t k = cols.size(), n = y.size();
    std::vector<Vec> U = cols, V(k, Vec(k, 0.0));
    for (size_t j = 0; j < k; ++j) V[j][j] = 1.0;
    for (int sweep = 0; sweep < 80; ++sweep) {
        bool rotated = false;
        for (size_t p = 0; p + 1 < k; ++p) {
            for (size_t q = p + 1; q < k; ++q) {
                const double alpha = dot(U[p].data(), U[p].data(), n), beta = dot(U[q].data(), U[q].data(), n);
                const double gamma = dot(U[p].data(), U[q].data(), n);
                if (gamma == 0.0 || !(std::fabs(gamma) > 1e-15 * std::sqrt(alpha * beta))) continue;
                rotated = true;
                const double zeta = (beta - alpha) / (2.0 * gamma);
                const double t = (zeta >= 0 ? 1.0 : -1.0) / (std::fabs(zeta) + std::sqrt(1.0 + zeta * zeta));
                const double c = 1.0 / std::sqrt(1.0 + t * t), s = c * t;
                for (size_t i = 0; i < n; ++i) {
                    const double a = U[p][i], b = U[q][i];
                    U[p][i] = c * a - s * b;
                    U[q][i] = s * a + c * b;
                }
                for (size_t i = 0; i < k; ++i) {
                    const double a = V[p][i], b = V[q][i];
                    V[p][i] = c * a - s * b;
                    V[q][i] = s * a + c * b;
                }
            }
        }
        if (!rotated) break;
    }
    Vec sig2(k);
    double smax2 = 0.0;
    for (size_t j = 0; j < k; ++j) {
        sig2[j] = dot(U[j].data(), U[j].data(), n);
        smax2 = std::max(smax2, sig2[j]);
    }
    const double rcond = std::numeric_limits<double>::epsilon() * static_cast<double>(std::max(n, k));
    Vec beta(k, 0.0);
    for (size_t j = 0; j < k; ++j) {
        if (!(std::sqrt(sig2[j]) > rcond * std::sqrt(smax2))) continue;
        const double w = dot(U[j].data(), y.data(), n) / sig2[j];
        for (size_t i = 0; i < k; ++i) beta[i] += V[j][i] * w;
    }
    return beta;
}

Vec residuals(const std::vector<Vec>& cols, const Vec& y, const Vec& beta) {
    Vec r(y.size());
    for (size_t i = 0; i < y.size(); ++i) {
        double f = 0.0;
        for (size_t j = 0; j < cols.size(); ++j) f += cols[j][i] * beta[j];
        r[i] = y[i] - f;
    }
    return r;
}

// ------------------------------------------------------------------------------------------------------ features
Row lumpiness_stability(const Vec& x, long width, bool lump) {
    const char* name = lump ? "lumpiness" : "stability";
    const long nr = static_cast<long>(x.size());
    if (nr < 2 * width) return {{name, 0.0}};
    const long nseg = nr / width;
    Vec stat(nseg);
    for (long s = 0; s < nseg; ++s) {
        const Vec seg(x.begin() + s * width, x.begin() + (s + 1) * width);
        stat[s] = lump ? r_var(seg) : np_mean(seg);
    }
    return {{name, r_var(stat)}};
}

// RcppRoll's rolling mean / variance (sums in sequential order); empty when the series is shorter than the window
bool roll(const Vec& x, long width, bool variance, Vec& out) {
    const long n = static_cast<long>(x.size());
    if (n < width) return false;
    out.assign(n - width + 1, NA);
    for (long i = 0; i + width <= n; ++i) {
        double s = x[i];
        for (long k = 1; k < width; ++k) s += x[i + k];
        const double mu = s / static_cast<double>(width);
        if (!variance) { out[i] = mu; continue; }
        if (width < 2) continue;
        double v = (x[i] - mu) * (x[i] - mu);
        for (long k = 1; k < width; ++k) v += (x[i + k] - mu) * (x[i + k] - mu);
        out[i] = v / static_cast<double>(width - 1);
    }
    return true;
}

void max_shift(const Vec& stat, long width, double& mx, double& tm) {
    const long n = static_cast<long>(stat.size());
    if (n <= width) { mx = 0.0; tm = NA; return; }
    mx = NA; tm = NA;
    for (long i = 0; i + width < n; ++i) {
        const double d = std::fabs(stat[i + width] - stat[i]);
        if (std::isnan(d)) continue;
        if (std::isnan(mx) || d > mx) { mx = d; tm = static_cast<double>(i + 2); }      // which.max + 1 (tsfeatures 1.0.x)
    }
}

Row max_level_var_shift(const Vec& x, long width, bool variance) {
    const char* a = variance ? "max_var_shift" : "max_level_shift";
    const char* b = variance ? "time_var_shift" : "time_level_shift";
    Vec stat;
    if (!roll(x, width, variance, stat)) return {{a, NA}, {b, NA}};
    double mx, tm;
    max_shift(stat, width, mx, tm);
    return {{a, mx}, {b, tm}};
}

Row max_kl_shift(const Vec& x, long width) {
    const int gw = 100;
    const long n = static_cast<long>(x.size());
    const double lo = *std::min_element(x.begin(), x.end()), hi = *std::max_element(x.begin(), x.end());
    const double step = (hi - lo) / (gw - 1);
    Vec xgrid(gw);
    for (int g = 0; g < gw; ++g) xgrid[g] = lo + g * step;
    xgrid[gw - 1] = hi;
    const double grid = xgrid[1] - xgrid[0];
    const double bw = bw_nrd0(x);
    if (n <= 2 * width) return {{"max_kl_shift", NA}, {"time_kl_shift", NA}};
    const double floor_d = dnorm(38.0, 0.0, 1.0);
    Vec dens(static_cast<size_t>(n) * gw);
    for (long i = 0; i < n; ++i)
        for (int g = 0; g < gw; ++g) dens[static_cast<size_t>(i) * gw + g] = std::max(dnorm(xgrid[g], x[i], bw), floor_d);
    // rmean[r] (r >= width - 1): mean of the densities of observations r - width + 1 .. r
    const long nw = n - width + 1;
    Vec rmean(static_cast<size_t>(nw) * gw);
    for (long w = 0; w < nw; ++w) {
        for (int g = 0; g < gw; ++g) {
            double s = dens[static_cast<size_t>(w) * gw + g];
            for (long k = 1; k < width; ++k) s += dens[static_cast<size_t>(w + k) * gw + g];
            rmean[static_cast<size_t>(w) * gw + g] = s / static_cast<double>(width);
        }
    }
    const long k = n - width;
    Vec kl(k, 0.0), terms(gw);
    for (long i = width - 1; i < k; ++i) {
        const double* a = &rmean[static_cast<size_t>(i - (width - 1)) * gw];
        const double* b = &rmean[static_cast<size_t>(i + 1) * gw];
        for (int g = 0; g < gw; ++g) {
            const double t = a[g] * (std::log(a[g]) - std::log(b[g])) * grid;
            terms[g] = std::isnan(t) ? 0.0 : t;
        }
        kl[i] = np_sum(terms);
    }
    if (k < 2) return {{"max_kl_shift", 0.0}, {"time_kl_shift", NA}};
    double mx = kl[1] - kl[0];
    long at = 0;
    for (long i = 1; i + 1 < k; ++i) {
        const double d = kl[i + 1] - kl[i];
        if (d > mx) { mx = d; at = i; }
    }
    return {{"max_kl_shift", mx}, {"time_kl_shift", static_cast<double>(at + 2)}};
}

Row crossing_points(const Vec& x) {
    Vec s = x;
    std::sort(s.begin(), s.end());
    const size_t n = s.size();
    const double med = n % 2 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2.0;
    long c = 0;
    for (size_t i = 0; i + 1 < n; ++i) c += (x[i] <= med) != (x[i + 1] <= med);
    return {{"crossing_points", static_cast<double>(c)}};
}

Row flat_spots(const Vec& x) {
    const double lo = *std::min_element(x.begin(), x.end()), hi = *std::max_element(x.begin(), x.end());
    double dx = hi - lo;
    double breaks[11];
    if (dx == 0) {
        dx = lo != 0 ? std::fabs(lo) : 1.0;
        for (int i = 0; i < 11; ++i) breaks[i] = lo - dx / 1000 + i * ((hi + dx / 1000) - (lo - dx / 1000)) / 10;
    } else {
        for (int i = 0; i < 11; ++i) breaks[i] = lo + i * (dx / 10);
        breaks[0] = lo - dx / 1000;
        breaks[10] = hi + dx / 1000;
    }
    auto code = [&](double v) { return static_cast<int>(std::lower_bound(breaks, breaks + 11, v) - breaks); };
    long run = 1, best = 1;
    int prev = code(x[0]);
    for (size_t i = 1; i < x.size(); ++i) {
        const int c = code(x[i]);
        run = c == prev ? run + 1 : 1;
        best = std::max(best, run);
        prev = c;
    }
    return {{"flat_spots", static_cast<double>(best)}};
}

// x_acf1 and x_acf10 of acf_features (all that stl_features needs of the remainder)
void acf_head(const Vec& x, long m, Vec& acfx) {
    acfx = x.size() > 1 ? r_acf(x, std::max<long>(10, m)) : Vec{NA, NA};
    acfx.erase(acfx.begin());
}

Row acf_features(const Vec& x, long m) {
    const size_t n = x.size();
    Vec acfx, d1{NA}, d2{NA};
    acf_head(x, m, acfx);
    if (n > 10) { d1 = r_acf(diff(x), 10); d1.erase(d1.begin()); }
    if (n > 11) { d2 = r_acf(diff(x, 2), 10); d2.erase(d2.begin()); }
    Row out{{"x_acf1", seq_get(acfx, 1)}, {"x_acf10", sumsq_first(acfx, 10)}, {"diff1_acf1", seq_get(d1, 1)},
            {"diff1_acf10", sumsq_first(d1, 10)}, {"diff2_acf1", seq_get(d2, 1)}, {"diff2_acf10", sumsq_first(d2, 10)}};
    if (m > 1) out.push_back({"seas_acf1", seq_get(acfx, m)});
    return out;
}

Row pacf_features(const Vec& x, long m) {
    const size_t n = x.size();
    const Vec p = n > 1 ? r_pacf(x, std::max<long>(5, m)) : Vec{NA};
    Row out{{"x_pacf5", n > 5 ? sumsq_first(p, 5) : NA}, {"diff1x_pacf5", n > 6 ? sumsq(r_pacf(diff(x), 5)) : NA},
            {"diff2x_pacf5", n > 7 ? sumsq(r_pacf(diff(x, 2), 5)) : NA}};
    if (m > 1) out.push_back({"seas_pacf", seq_get(p, m)});
    return out;
}

Row arch_stat(const Vec& x, long lags) {
    const long n = static_cast<long>(x.size());
    if (n <= 13) return {{"ARCH.LM", NA}};                      // tsfeatures 1.0.x: fixed at 13, whatever `lags` is
    const double mu = np_mean(x);
    Vec x2(n);
    for (long i = 0; i < n; ++i) x2[i] = (x[i] - mu) * (x[i] - mu);
    const long rows = n - lags;
    const Vec y(x2.begin() + lags, x2.end());
    std::vector<Vec> X;
    X.emplace_back(rows, 1.0);
    for (long j = 1; j <= lags; ++j) X.emplace_back(x2.begin() + (lags - j), x2.begin() + (n - j));
    const Vec res = residuals(X, y, lstsq(X, y));
    Vec fitted(rows);
    for (long i = 0; i < rows; ++i) fitted[i] = y[i] - res[i];
    const double fm = np_mean(fitted);
    double mss = 0.0, rss = 0.0;
    for (long i = 0; i < rows; ++i) {
        mss += (fitted[i] - fm) * (fitted[i] - fm);
        rss += res[i] * res[i];
    }
    const double r2 = (mss + rss) != 0 ? mss / (mss + rss) : NA;
    return {{"ARCH.LM", std::isnan(r2) ? 1.0 : r2}};
}

Row unitroot_kpss(const Vec& x) {
    const long n = static_cast<long>(x.size());
    const long lmax = static_cast<long>(std::trunc(4 * std::pow(n / 100.0, 0.25)));
    const double mu = np_mean(x);
    Vec res(n);
    for (long i = 0; i < n; ++i) res[i] = x[i] - mu;
    double cum = 0.0, nom = 0.0, s2 = 0.0;
    for (long i = 0; i < n; ++i) {
        cum += res[i];
        nom += cum * cum;
        s2 += res[i] * res[i];
    }
    nom /= static_cast<double>(n) * static_cast<double>(n);
    s2 /= static_cast<double>(n);
    double den = s2;
    if (lmax > 0) {
        double w = 0.0;
        for (long k = 1; k <= lmax; ++k) w += (1 - static_cast<double>(k) / (lmax + 1)) * dot(res.data() + k, res.data(), static_cast<size_t>(n - k));
        den = s2 + 2.0 / n * w;
    }
    if (den == 0) throw Raise();
    return {{"unitroot_kpss", nom / den}};
}

Row unitroot_pp(const Vec& x) {
    const Vec y(x.begin() + 1, x.end()), yl(x.begin(), x.end() - 1);
    const long n = static_cast<long>(y.size());
    const long lmax = static_cast<long>(std::trunc(4 * std::pow(n / 100.0, 0.25)));
    const std::vector<Vec> X{Vec(n, 1.0), yl};
    const Vec beta = lstsq(X, y);
    const Vec res = residuals(X, y, beta);
    const double s = dot(res.data(), res.data(), n) / n;
    const double ym = np_mean(y);
    double myybar = 0.0;
    for (long i = 0; i < n; ++i) myybar += (y[i] - ym) * (y[i] - ym);
    myybar /= static_cast<double>(n) * static_cast<double>(n);
    double w = 0.0;
    if (lmax >= 1) {
        for (long k = 1; k <= lmax; ++k) w += (1 - static_cast<double>(k) / (lmax + 1)) * dot(res.data() + k, res.data(), static_cast<size_t>(n - k));
    } else {
        w = (1 - 1.0) * dot(res.data() + 1, res.data(), static_cast<size_t>(n - 1)) + dot(res.data(), res.data(), n);
    }
    const double sig = s + 2.0 / n * w;
    const double lam = 0.5 * (sig - s);
    if (myybar == 0) throw Raise();
    return {{"unitroot_pp", n * (beta[1] - 1) - lam / myybar}};
}

// 10 * tseries::terasvirta.test(x, type = "Chisq")$stat / length(x)
Row nonlinearity(const Vec& x) {
    const long n = static_cast<long>(x.size());
    Vec z = x;
    if (!constant(x)) {
        const double mu = np_mean(x), sd = std::sqrt(r_var(x));
        for (double& v : z) v = (v - mu) / sd;
    }
    const Vec xs(z.begin() + 1, z.end()), y(z.begin(), z.end() - 1);
    const long t = static_cast<long>(xs.size());
    const std::vector<Vec> X1{Vec(t, 1.0), y};
    const Vec u = residuals(X1, xs, lstsq(X1, xs));
    const double ssr0 = dot(u.data(), u.data(), t);
    Vec y2(t), y3(t);
    for (long i = 0; i < t; ++i) { y2[i] = y[i] * y[i]; y3[i] = y[i] * y[i] * y[i]; }
    const std::vector<Vec> X2{Vec(t, 1.0), y, y2, y3};
    const Vec v = residuals(X2, u, lstsq(X2, u));
    const double ssr = dot(v.data(), v.data(), t);
    const double stat = (ssr > 0 && ssr0 > 0) ? n * std::log(ssr0 / ssr) : NA;
    return {{"nonlinearity", 10 * stat / n}};
}

// Spectral entropy from stats::spec.ar(x, method = "burg", n.freq = ceiling(n / 2 + 1)) (R's C_Burg)
Row entropy(const Vec& x) {
    const long n = static_cast<long>(x.size());
    const double mu = np_mean(x);
    Vec xc(n);
    for (long i = 0; i < n; ++i) xc[i] = x[i] - mu;
    const long pmax = std::min(n - 1, static_cast<long>(std::floor(10 * std::log10(static_cast<double>(n)))));
    if (pmax < 1) return {{"entropy", NA}};
    std::vector<Vec> coefs(pmax, Vec(pmax, 0.0));
    Vec u(xc.rbegin(), xc.rend()), v = u, var1(pmax + 1, 0.0);
    var1[0] = dot(xc.data(), xc.data(), n) / n;
    for (long p = 1; p <= pmax; ++p) {
        const size_t len = static_cast<size_t>(n - p);
        const double sm = dot(v.data() + p, u.data() + p - 1, len);
        const double d = dot(v.data() + p, v.data() + p, len) + dot(u.data() + p - 1, u.data() + p - 1, len);
        if (d == 0) throw Raise();
        const double phii = 2 * sm / d;
        coefs[p - 1][p - 1] = phii;
        for (long j = 1; j < p; ++j) coefs[p - 1][j - 1] = coefs[p - 2][j - 1] - phii * coefs[p - 2][p - j - 1];
        const Vec u0 = u;
        for (long i = p; i < n; ++i) {
            u[i] = u0[i - 1] - phii * v[i];
            v[i] = v[i] - phii * u0[i - 1];
        }
        var1[p] = var1[p - 1] * (1 - phii * phii);
    }
    long order = -1;
    double best = 0.0;
    bool any_finite = false;
    for (long p = 0; p <= pmax; ++p) {
        const double aic = n * std::log(var1[p]) + 2.0 * p + 2;
        if (std::isfinite(aic)) any_finite = true;
        if (std::isnan(aic)) continue;
        if (order < 0 || aic < best) { order = p; best = aic; }
    }
    if (!any_finite) return {{"entropy", NA}};
    const double vp = var1[order];                               // ar.burg: var.pred is not rescaled
    const long nf = static_cast<long>(std::ceil(n / 2.0 + 1));
    Vec spec(nf);
    for (long i = 0; i < nf; ++i) {
        const double f = i == nf - 1 ? 0.5 : i * (0.5 / (nf - 1));
        double cs = 0.0, sn = 0.0;
        for (long k = 1; k <= order; ++k) {
            cs += std::cos(2 * PI * (f * k)) * coefs[order - 1][k - 1];
            sn += std::sin(2 * PI * (f * k)) * coefs[order - 1][k - 1];
        }
        spec[i] = vp / ((1 - cs) * (1 - cs) + sn * sn);
    }
    Vec fx;
    fx.reserve(2 * nf - 1);
    for (long i = nf - 1; i >= 1; --i) fx.push_back(spec[i] / n);
    for (long i = 0; i < nf; ++i) fx.push_back(spec[i] / n);
    const double total = np_sum(fx);
    const double logn = std::log(static_cast<double>(n));
    Vec e(fx.size());
    for (size_t i = 0; i < fx.size(); ++i) {
        const double f = 0.999 * (fx[i] / total) + 0.001 / static_cast<double>(fx.size());
        e[i] = f * std::log(f) / logn;
    }
    const double ent = -np_sum(e);
    return {{"entropy", ent < 1.0 ? ent : 1.0}};
}

Row holt_parameters(const Vec& x) {
    try {
        const EtsResult f = ets_holt(x, 1);
        return {{"alpha", f.alpha}, {"beta", f.beta}};
    } catch (...) {
        return {{"alpha", NA}, {"beta", NA}};
    }
}

Row hurst(const Vec& x) {
    try {
        return {{"hurst", fracdiff_d(x, 100) + 0.5}};
    } catch (...) {
        return {{"hurst", NA}};
    }
}

// ------------------------------------------------------------------------------------------------- the study's calls
enum Func { KL, LEVEL, VAR, ACF, ARCH, CROSS, ENTROPY, FLAT, HOLT, HURST, LUMP, NONLIN, PACF, STAB, KPSS, PP, N_FUNCS };

bool is_shift(int f) { return f == KL || f == LEVEL || f == VAR; }

Row one(int f, const Vec& z, long m, long width = -1, long lags = 12) {
    const long w = width >= 0 ? width : (m > 1 ? m : 10);
    switch (f) {
        case KL: return max_kl_shift(z, w);
        case LEVEL: return max_level_var_shift(z, w, false);
        case VAR: return max_level_var_shift(z, w, true);
        case ACF: return acf_features(z, m);
        case ARCH: return arch_stat(z, lags);
        case CROSS: return crossing_points(z);
        case ENTROPY: return entropy(z);
        case FLAT: return flat_spots(z);
        case HOLT: return holt_parameters(z);
        case HURST: return hurst(z);
        case LUMP: return lumpiness_stability(z, w, true);
        case NONLIN: return nonlinearity(z);
        case PACF: return pacf_features(z, m);
        case STAB: return lumpiness_stability(z, w, false);
        case KPSS: return unitroot_kpss(z);
        default: return unitroot_pp(z);
    }
}

bool has_na(const Row& r) {
    for (const auto& kv : r)
        if (std::isnan(kv.second)) return true;
    return false;
}

// forecast::mstl(x, s.window, robust). kind: 0 "msts" (several periods), 1 "ts" (one frequency), 2 plain.
void mstl(const Vec& x, int kind, const Vec& periods, bool periodic, bool robust, Vec& trend, std::vector<Vec>& seas,
          Vec& used, Vec& remainder) {
    const size_t n = x.size();
    Vec msts;
    int iterate = 1;
    if (kind == 0) {
        for (double p : periods)
            if (p < n / 2.0) msts.push_back(p);
        std::sort(msts.begin(), msts.end());
        if (msts.empty()) throw Raise();
        iterate = 2;
    } else {
        msts.push_back(kind == 1 ? periods[0] : 1.0);
    }
    Vec deseas = x;
    seas.clear();
    if (msts[0] > 1) {
        static const int swin[6] = {11, 15, 19, 23, 27, 31};
        seas.assign(msts.size(), Vec(n, 0.0));
        for (int it = 0; it < iterate; ++it) {
            for (size_t i = 0; i < msts.size(); ++i) {
                for (size_t t = 0; t < n; ++t) deseas[t] += seas[i][t];
                StlOut f;
                try {
                    f = r_stl(deseas, msts[i], periodic, i < 6 ? swin[i] : 31, robust);
                } catch (...) {
                    throw Raise();
                }
                seas[i] = f.seasonal;
                trend = f.trend;
                for (size_t t = 0; t < n; ++t) deseas[t] -= seas[i][t];
            }
        }
        used = msts;
    } else {
        used.clear();
        Vec t(n);
        for (size_t i = 0; i < n; ++i) t[i] = i + 1.0;
        trend = supsmu(t, x);
    }
    remainder.resize(n);
    for (size_t t = 0; t < n; ++t) remainder[t] = deseas[t] - trend[t];
}

// coef(lm(y ~ poly(seq(n), degree = 2)))[2:3]: the projections on the orthonormal linear and quadratic polynomials
void poly_coefs(const Vec& y, double& lin, double& curv) {
    const size_t n = y.size();
    lin = curv = NA;
    if (n < 2) return;
    Vec t(n);
    for (size_t i = 0; i < n; ++i) t[i] = i + 1.0;
    const double tm = np_mean(t);
    Vec q1(n), q2(n);
    for (size_t i = 0; i < n; ++i) { q1[i] = t[i] - tm; q2[i] = q1[i] * q1[i]; }
    const double n1 = std::sqrt(dot(q1.data(), q1.data(), n));
    for (double& v : q1) v /= n1;
    lin = dot(q1.data(), y.data(), n);
    if (n < 3) return;
    for (int pass = 0; pass < 2; ++pass) {                      // Gram-Schmidt, twice for accuracy
        const double a = np_sum(q2) / static_cast<double>(n), b = dot(q2.data(), q1.data(), n);
        for (size_t i = 0; i < n; ++i) q2[i] -= a + b * q1[i];
    }
    const double n2 = std::sqrt(dot(q2.data(), q2.data(), n));
    for (double& v : q2) v /= n2;
    curv = dot(q2.data(), y.data(), n);
}

std::string num_suffix(size_t i, size_t count) { return count == 1 ? std::string() : std::to_string(i + 1); }

// tsfeatures::stl_features on the scaled series x. kind 0 "msts", 1 "ts".
Row stl_features(const Vec& x, int kind, const Vec& periods, long startx, bool periodic, bool robust) {
    const size_t n = x.size();
    const double pmax = *std::max_element(periods.begin(), periods.end());
    if (kind == 0 && n <= 2 * pmax) throw Raise();
    Vec msts_out;
    double nperiods;
    if (kind == 0) {
        msts_out = periods;
        nperiods = static_cast<double>(periods.size());
    } else {
        msts_out = Vec{periods[0]};
        nperiods = periods[0] > 1 ? 1.0 : 0.0;
        if (n <= 2 * periods[0]) kind = 2;                      // x <- c(x): mstl sees no seasonality
    }
    Vec trend0, used, remainder;
    std::vector<Vec> seas;
    mstl(x, kind, periods, periodic, robust, trend0, seas, used, remainder);
    Vec deseason = x;
    if (!seas.empty()) {
        Vec tot = seas[0];
        for (size_t i = 1; i < seas.size(); ++i)
            for (size_t t = 0; t < n; ++t) tot[t] += seas[i][t];
        for (size_t t = 0; t < n; ++t) deseason[t] = x[t] - tot[t];
    }
    const double varx = r_var(x), vare = r_var(remainder), vardes = r_var(deseason);
    double trend;
    if (varx < std::numeric_limits<double>::epsilon() || vardes / varx < 1e-10) trend = 0.0;
    else {
        if (vardes == 0) throw Raise();
        trend = std::max(0.0, std::min(1.0, 1 - vare / vardes));
    }
    const double rm = np_mean(remainder);
    Vec varloo(n);
    for (size_t t = 0; t < n; ++t) {
        const double d = (remainder[t] - rm) * (remainder[t] - rm);
        varloo[t] = (vare * static_cast<double>(n - 1) - d) / static_cast<double>(n - 2);
    }
    double lin, curv;
    poly_coefs(trend0, lin, curv);
    const long m_rem = kind == 1 ? static_cast<long>(periods[0]) : (kind == 0 ? std::max<long>(1, static_cast<long>(std::floor(pmax))) : 1);
    Vec acfr;
    acf_head(remainder, m_rem, acfr);
    Row out{{"nperiods", nperiods}};
    for (size_t i = 0; i < msts_out.size(); ++i) out.push_back({"seasonal_period" + num_suffix(i, msts_out.size()), msts_out[i]});
    out.push_back({"trend", trend});
    out.push_back({"spike", r_var(varloo)});
    out.push_back({"linearity", lin});
    out.push_back({"curvature", curv});
    out.push_back({"e_acf1", seq_get(acfr, 1)});
    out.push_back({"e_acf10", sumsq_first(acfr, 10)});
    for (size_t i = 0; i < seas.size(); ++i) {
        Vec rs(n);
        for (size_t t = 0; t < n; ++t) rs[t] = remainder[t] + seas[i][t];
        const double v = r_var(rs);
        if (v == 0) throw Raise();
        out.push_back({"seasonal_strength" + num_suffix(i, seas.size()), std::max(0.0, std::min(1.0, 1 - vare / v))});
    }
    for (size_t i = 0; i < seas.size(); ++i) {
        const double p = used[i];
        const long amax = static_cast<long>(std::max_element(seas[i].begin(), seas[i].end()) - seas[i].begin());
        const long amin = static_cast<long>(std::min_element(seas[i].begin(), seas[i].end()) - seas[i].begin());
        const double pk = std::fmod(static_cast<double>(startx + amax + 1), p), th = std::fmod(static_cast<double>(startx + amin + 1), p);
        out.push_back({"peak" + num_suffix(i, seas.size()), pk == 0 ? p : pk});
        out.push_back({"trough" + num_suffix(i, seas.size()), th == 0 ? p : th});
    }
    return out;
}

Row study_stl_features(const Vec& z, const Vec& periods, long startx) {
    const int kind = periods.size() > 1 ? 0 : 1;
    const Vec per = kind == 0 ? periods : Vec{static_cast<double>(std::max<long>(1, static_cast<long>(std::floor(periods[0]))))};
    for (int attempt = 0; attempt < 2; ++attempt) {
        try {
            return stl_features(z, kind, per, startx, attempt == 0, attempt == 0);
        } catch (const Raise&) {
        }
    }
    return stl_features(z, 1, Vec{1.0}, 0, true, true);
}

// tsfeat.study_features(y, periods, stl = True, startx)
Row study_features(const Vec& y, const Vec& periods, long startx) {
    const long m = std::max<long>(1, static_cast<long>(std::floor(*std::max_element(periods.begin(), periods.end()))));
    Row out{{"mean", np_mean(y)}, {"var", r_var(y)}};
    const Vec z = scalets(y);
    for (int f = 0; f < N_FUNCS; ++f) {
        Row r;
        bool ok = true;
        try { r = one(f, z, m); } catch (const Raise&) { ok = false; }
        if (!ok || has_na(r)) {
            ok = true;
            try { r = one(f, z, 1); } catch (const Raise&) { ok = false; }
            if (ok && has_na(r) && (is_shift(f) || f == ARCH)) r = is_shift(f) ? one(f, z, m, 1) : one(f, z, m, -1, 1);
        }
        if (ok) out.insert(out.end(), r.begin(), r.end());
    }
    const Row s = study_stl_features(z, periods, startx);
    out.insert(out.end(), s.begin(), s.end());
    return out;
}

struct Table {
    Vec periods;
    std::vector<Row> rows;
    std::vector<std::string> columns;
    bool finished = false;

    // features.tsfeatures_table: the columns of the first series that every series with features has
    void finish() {
        if (finished) return;
        columns.clear();
        const Row* first = nullptr;
        for (const Row& r : rows)
            if (!r.empty()) { first = &r; break; }
        if (first) {
            for (const auto& kv : *first) {
                bool everywhere = true;
                for (const Row& r : rows) {
                    if (r.empty()) continue;
                    bool found = false;
                    for (const auto& c : r)
                        if (c.first == kv.first) { found = true; break; }
                    if (!found) { everywhere = false; break; }
                }
                if (everywhere) columns.push_back(kv.first);
            }
        }
        finished = true;
    }
};

}  // namespace

}  // namespace tsfeat
}  // namespace tsf

// ---------------------------------------------------------------------------------------------------------- C API
// void*   tsf_tsfeat_new(periods, n_periods)        a table for a dataset with these seasonal periods (the study's)
// int32   tsf_tsfeat_add(h, y, n, startx)           adds one series (missing values are removed; fewer than 4 values or a
//                                                   failure give a row without features); returns its number of features
// int32   tsf_tsfeat_columns(h, buf, len)           the features every series has, '\n'-separated; returns their number,
//                                                   or -(bytes needed) when buf is too small
// int32   tsf_tsfeat_table(h, out)                  out: n_series x n_columns, row-major, NaN where a series has no value
// void    tsf_tsfeat_free(h)
// int32   tsf_tsfeat_startx(year, month, day, periods, n_periods)   tsfeat.study_startx for a series starting that day
#if defined(_WIN32)
#define TSFEAT_API extern "C" __declspec(dllexport)
#else
#define TSFEAT_API extern "C" __attribute__((visibility("default")))
#endif

using tsf::tsfeat::Table;

TSFEAT_API void* tsf_tsfeat_new(const double* periods, int32_t n_periods) {
    if (!periods || n_periods < 1) return nullptr;
    Table* t = new Table();
    t->periods.assign(periods, periods + n_periods);
    return t;
}

TSFEAT_API int32_t tsf_tsfeat_add(void* h, const double* y, int64_t n, int32_t startx) {
    if (!h) return -1;
    Table* t = static_cast<Table*>(h);
    tsf::rs::X87Extended fpu_guard;              // R runs with 64-bit x87 precision
    tsf::Vec x;
    x.reserve(static_cast<size_t>(std::max<int64_t>(n, 0)));
    for (int64_t i = 0; i < n; ++i)
        if (std::isfinite(y[i])) x.push_back(y[i]);
    tsf::tsfeat::Row row;
    if (x.size() >= 4) {
        try {
            row = tsf::tsfeat::study_features(x, t->periods, startx);
        } catch (...) {
            row.clear();
        }
    }
    t->rows.push_back(row);
    t->finished = false;
    return static_cast<int32_t>(row.size());
}

TSFEAT_API int32_t tsf_tsfeat_columns(void* h, char* buf, int32_t len) {
    if (!h) return -1;
    Table* t = static_cast<Table*>(h);
    t->finish();
    std::string s;
    for (size_t i = 0; i < t->columns.size(); ++i) s += (i ? "\n" : "") + t->columns[i];
    if (!buf || len < static_cast<int32_t>(s.size()) + 1) return -static_cast<int32_t>(s.size() + 1);
    std::memcpy(buf, s.c_str(), s.size() + 1);
    return static_cast<int32_t>(t->columns.size());
}

TSFEAT_API int32_t tsf_tsfeat_table(void* h, double* out) {
    if (!h || !out) return -1;
    Table* t = static_cast<Table*>(h);
    t->finish();
    const size_t p = t->columns.size();
    for (size_t i = 0; i < t->rows.size(); ++i) {
        for (size_t j = 0; j < p; ++j) {
            double v = std::numeric_limits<double>::quiet_NaN();
            for (const auto& kv : t->rows[i])
                if (kv.first == t->columns[j]) { v = kv.second; break; }
            out[i * p + j] = v;
        }
    }
    return static_cast<int32_t>(t->rows.size());
}

TSFEAT_API void tsf_tsfeat_free(void* h) { delete static_cast<Table*>(h); }

TSFEAT_API int32_t tsf_tsfeat_startx(int32_t year, int32_t month, int32_t day, const double* periods, int32_t n_periods) {
    if (!periods || n_periods < 1 || month < 1 || month > 12) return 0;
    static const int cum[12] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
    const bool leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    const double days = cum[month - 1] + (leap && month > 2 ? 1 : 0) + (day - 1);
    const double s = year + (days * 86400.0) / ((leap ? 366 : 365) * 86400.0);
    const double frac = s - std::floor(s);
    const double maxs = *std::max_element(periods, periods + n_periods);
    const double v = frac * maxs;
    if (std::fabs(v - std::nearbyint(v)) < 1e-5) return static_cast<int32_t>(std::floor(v + 0.001));   // a whole cycle
    const double b = std::floor(v);
    const int32_t freq = std::max<int32_t>(1, static_cast<int32_t>(std::floor(maxs)));
    return b >= 1 ? static_cast<int32_t>(b) - 1 : freq - 1;
}
