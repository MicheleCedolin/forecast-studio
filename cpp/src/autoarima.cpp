// Port of forecast::auto.arima (forecast 9.0.2) with the pieces it uses: is.constant, ur.kpss / ndiffs,
// stats::stl (netlib Fortran, Cleveland et al. 1990) / mstl / seas.heuristic / nsdiffs, myarima, the stepwise
// search and the approximation refit. See autoarima.hpp.
#include "autoarima.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace tsf {

namespace {

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

Vec diff_lag(const Vec& v, int lag) {
    if (static_cast<int>(v.size()) <= lag) return {};
    Vec o(v.size() - lag);
    for (size_t i = 0; i < o.size(); ++i) o[i] = v[i + lag] - v[i];
    return o;
}

double r_mean(const Vec& v) {                       // R's mean(): long double sum plus one refinement pass
    long double s = 0;
    for (double x : v) s += x;
    s /= static_cast<long double>(v.size());
    if (std::isfinite(static_cast<double>(s))) {
        long double t = 0;
        for (double x : v) t += (x - s);
        s += t / static_cast<long double>(v.size());
    }
    return static_cast<double>(s);
}

double r_var(const Vec& v) {                        // var(x), n - 1 denominator
    const size_t n = v.size();
    if (n < 2) return NaN;
    const double m = r_mean(v);
    long double s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return static_cast<double>(s / static_cast<long double>(n - 1));
}

// ------------------------------------------------------------------------------------------------ stl.f
// 1-based helpers: y[i - 1] is Fortran's y(i).
void stlest(const double* y, int n, int len, int ideg, double xs, double& ys, int nleft, int nright, double* w,
            bool userw, const double* rw, bool& ok) {
    const double range = static_cast<double>(n) - 1.0;
    double h = std::max(xs - nleft, nright - xs);
    if (len > n) h += static_cast<double>((len - n) / 2);
    const double h9 = 0.999 * h, h1 = 0.001 * h;
    double a = 0.0;
    for (int j = nleft; j <= nright; ++j) {
        w[j - 1] = 0.0;
        const double r = std::fabs(j - xs);
        if (r <= h9) {
            if (r <= h1) w[j - 1] = 1.0;
            else { const double u = r / h; const double v = 1.0 - u * u * u; w[j - 1] = v * v * v; }
            if (userw) w[j - 1] *= rw[j - 1];
            a += w[j - 1];
        }
    }
    if (a <= 0.0) { ok = false; return; }
    ok = true;
    for (int j = nleft; j <= nright; ++j) w[j - 1] /= a;
    if (h > 0.0 && ideg > 0) {
        a = 0.0;
        for (int j = nleft; j <= nright; ++j) a += w[j - 1] * j;
        double b = xs - a, c = 0.0;
        for (int j = nleft; j <= nright; ++j) c += w[j - 1] * (j - a) * (j - a);
        if (std::sqrt(c) > 0.001 * range) {
            b /= c;
            for (int j = nleft; j <= nright; ++j) w[j - 1] *= (b * (j - a) + 1.0);
        }
    }
    double s = 0.0;
    for (int j = nleft; j <= nright; ++j) s += w[j - 1] * y[j - 1];
    ys = s;
}

void stless(const double* y, int n, int len, int ideg, int njump, bool userw, const double* rw, double* ys, double* res) {
    if (n < 2) { ys[0] = y[0]; return; }
    const int newnj = std::min(njump, n - 1);
    int nleft = 1, nright = n;
    bool ok;
    if (len >= n) {
        nleft = 1; nright = n;
        for (int i = 1; i <= n; i += newnj) {
            stlest(y, n, len, ideg, i, ys[i - 1], nleft, nright, res, userw, rw, ok);
            if (!ok) ys[i - 1] = y[i - 1];
        }
    } else if (newnj == 1) {
        const int nsh = (len + 1) / 2;
        nleft = 1; nright = len;
        for (int i = 1; i <= n; ++i) {
            if (i > nsh && nright != n) { ++nleft; ++nright; }
            stlest(y, n, len, ideg, i, ys[i - 1], nleft, nright, res, userw, rw, ok);
            if (!ok) ys[i - 1] = y[i - 1];
        }
    } else {
        const int nsh = (len + 1) / 2;
        for (int i = 1; i <= n; i += newnj) {
            if (i < nsh) { nleft = 1; nright = len; }
            else if (i >= n - nsh + 1) { nleft = n - len + 1; nright = n; }
            else { nleft = i - nsh + 1; nright = len + i - nsh; }
            stlest(y, n, len, ideg, i, ys[i - 1], nleft, nright, res, userw, rw, ok);
            if (!ok) ys[i - 1] = y[i - 1];
        }
    }
    if (newnj != 1) {
        for (int i = 1; i <= n - newnj; i += newnj) {
            const double delta = (ys[i + newnj - 1] - ys[i - 1]) / newnj;
            for (int j = i + 1; j <= i + newnj - 1; ++j) ys[j - 1] = ys[i - 1] + delta * (j - i);
        }
        const int k = ((n - 1) / newnj) * newnj + 1;
        if (k != n) {
            stlest(y, n, len, ideg, n, ys[n - 1], nleft, nright, res, userw, rw, ok);
            if (!ok) ys[n - 1] = y[n - 1];
            if (k != n - 1) {
                const double delta = (ys[n - 1] - ys[k - 1]) / (n - k);
                for (int j = k + 1; j <= n - 1; ++j) ys[j - 1] = ys[k - 1] + delta * (j - k);
            }
        }
    }
}

void stlma(const double* x, int n, int len, double* ave) {
    const int newn = n - len + 1;
    const double flen = len;
    double v = 0.0;
    for (int i = 0; i < len; ++i) v += x[i];
    ave[0] = v / flen;
    if (newn > 1) {
        int k = len, m = 0;
        for (int j = 2; j <= newn; ++j) {
            ++k; ++m;
            v = v - x[m - 1] + x[k - 1];
            ave[j - 1] = v / flen;
        }
    }
}

void stlfts(const double* x, int n, int np, double* trend, double* work) {
    stlma(x, n, np, trend);
    stlma(trend, n - np + 1, np, work);
    stlma(work, n - 2 * np + 2, 3, trend);
}

void stlss(const double* y, int n, int np, int ns, int isdeg, int nsjump, bool userw, const double* rw, double* season,
           double* work1, double* work2, double* work3, double* work4) {
    bool ok;
    for (int j = 1; j <= np; ++j) {
        const int k = (n - j) / np + 1;
        for (int i = 1; i <= k; ++i) work1[i - 1] = y[(i - 1) * np + j - 1];
        if (userw) for (int i = 1; i <= k; ++i) work3[i - 1] = rw[(i - 1) * np + j - 1];
        stless(work1, k, ns, isdeg, nsjump, userw, work3, work2 + 1, work4);
        const int nright = std::min(ns, k);
        stlest(work1, k, ns, isdeg, 0.0, work2[0], 1, nright, work4, userw, work3, ok);
        if (!ok) work2[0] = work2[1];
        const int nleft = std::max(1, k - ns + 1);
        stlest(work1, k, ns, isdeg, k + 1.0, work2[k + 1], nleft, k, work4, userw, work3, ok);
        if (!ok) work2[k + 1] = work2[k];
        for (int m = 1; m <= k + 2; ++m) season[(m - 1) * np + j - 1] = work2[m - 1];
    }
}

}  // namespace

StlOut stl(const Vec& x, int period, int s_window) {
    const int n = static_cast<int>(x.size());
    if (period < 2 || n <= 2 * period) throw std::runtime_error("series is not periodic or has less than two periods");
    auto nextodd = [](double v) { int r = static_cast<int>(std::nearbyint(v)); return (r % 2 == 0) ? r + 1 : r; };
    auto win_check = [&](double k) { return std::max(3, nextodd(k)); };
    const int t_window = win_check(std::ceil(1.5 * period / (1.0 - 1.5 / s_window)));
    const int s_win = win_check(s_window), l_window = win_check(nextodd(period));
    const int isdeg = 0, itdeg = 1, ildeg = 1;
    const int nsjump = std::max(1, static_cast<int>(std::ceil(s_win / 10.0)));
    const int ntjump = std::max(1, static_cast<int>(std::ceil(t_window / 10.0)));
    const int nljump = std::max(1, static_cast<int>(std::ceil(l_window / 10.0)));
    return stl_core(x, period, s_win, t_window, l_window, isdeg, itdeg, ildeg, nsjump, ntjump, nljump, 2);
}

StlOut stl_core(const Vec& x, int period, int s_win, int t_window, int l_window, int isdeg, int itdeg, int ildeg,
                int nsjump, int ntjump, int nljump, int ni) {
    const int n = static_cast<int>(x.size());
    // subroutine stl: window sanity (odd and >= 3), np >= 2; robust = FALSE (no outer loop)
    int ns = std::max(3, s_win), nt = std::max(3, t_window), nl = std::max(3, l_window);
    if (ns % 2 == 0) ++ns;
    if (nt % 2 == 0) ++nt;
    if (nl % 2 == 0) ++nl;
    const int np = std::max(2, period);
    const int nw = n + 2 * np;
    Vec w1(nw), w2(nw), w3(nw), w4(nw), w5(nw), rw(n, 1.0);
    StlOut out;
    out.seasonal.assign(n, 0.0);
    out.trend.assign(n, 0.0);
    const bool userw = false;
    for (int j = 0; j < ni; ++j) {                      // stlstp
        for (int i = 0; i < n; ++i) w1[i] = x[i] - out.trend[i];
        stlss(w1.data(), n, np, ns, isdeg, nsjump, userw, rw.data(), w2.data(), w3.data(), w4.data(), w5.data(),
              out.seasonal.data());
        stlfts(w2.data(), nw, np, w3.data(), w1.data());
        stless(w3.data(), n, nl, ildeg, nljump, false, w4.data(), w1.data(), w5.data());
        for (int i = 0; i < n; ++i) out.seasonal[i] = w2[np + i] - w1[i];
        for (int i = 0; i < n; ++i) w1[i] = x[i] - out.seasonal[i];
        stless(w1.data(), n, nt, itdeg, ntjump, userw, rw.data(), out.trend.data(), w3.data());
    }
    return out;
}

double seas_heuristic(const Vec& x, int period) {
    const StlOut f = stl(x, period, 11);                // mstl(x) on a ts: one stl, s.window = 7 + 4 = 11
    const size_t n = x.size();
    Vec rem(n), rs(n);
    for (size_t i = 0; i < n; ++i) {
        const double deseas = x[i] - f.seasonal[i];
        rem[i] = deseas - f.trend[i];
        rs[i] = rem[i] + f.seasonal[i];
    }
    const double v = 1.0 - r_var(rem) / r_var(rs);
    if (std::isnan(v)) return NaN;                      // R: max(0, min(1, NaN)) is NaN
    return std::max(0.0, std::min(1.0, v));
}

int nsdiffs(const Vec& x, int period) {
    if (is_constant(x)) return 0;
    if (period >= static_cast<int>(x.size())) return 0;
    double s;
    try { s = seas_heuristic(x, period); } catch (const std::exception&) { return 0; }   // tryCatch -> 0
    if (std::isnan(s)) return 0;                        // stopifnot(diff %in% c(0, 1)) fails -> 0
    return s > 0.64 ? 1 : 0;                            // max.D = 1: one test only
}

bool is_constant(const Vec& x) {
    if (x.empty()) return true;
    const double x1 = x[0], tol = 1.5e-8;
    if (std::isnan(x1)) return false;
    long double sdiff = 0, sabs = 0;
    int N = 0;
    for (double v : x) {
        if (std::isnan(v)) return false;                // NA mismatch
        if (v == x1) continue;                          // countEQ = FALSE
        ++N;
        sdiff += std::fabs(v - x1);
        sabs += std::fabs(v);
    }
    if (N == 0) return true;
    double xy = static_cast<double>(sdiff / N);
    const double xn = static_cast<double>(sabs / N);
    if (std::isfinite(xn) && xn > tol) xy /= xn;
    return !(std::isnan(xy) || xy > tol);
}

double kpss_stat(const Vec& y) {
    const int n = static_cast<int>(y.size());
    const int lmax = static_cast<int>(3.0 * std::sqrt(static_cast<double>(n)) / 13.0);
    const double m = r_mean(y);
    Vec res(n);
    for (int i = 0; i < n; ++i) res[i] = y[i] - m;
    double S = 0, nom = 0, s2 = 0;
    for (int i = 0; i < n; ++i) { S += res[i]; nom += S * S; s2 += res[i] * res[i]; }
    nom /= static_cast<double>(n) * n;
    s2 /= n;
    double den = s2;
    if (lmax > 0) {
        double acc = 0;
        for (int k = 1; k <= lmax; ++k) {
            double cv = 0;
            for (int i = k; i < n; ++i) cv += res[i] * res[i - k];
            acc += (1.0 - static_cast<double>(k) / (lmax + 1)) * cv;
        }
        den = s2 + 2.0 / n * acc;
    }
    return nom / den;
}

namespace {

// urca_pval: approx(cval, c(.10, .05, .025, .01), xout = stat, rule = 2)$y; NaN for NaN.
double kpss_pval(double st) {
    static const double cx[4] = {0.347, 0.463, 0.574, 0.739}, cy[4] = {0.10, 0.05, 0.025, 0.01};
    if (std::isnan(st)) return NaN;
    if (st <= cx[0]) return cy[0];
    if (st >= cx[3]) return cy[3];
    int i = 0;
    while (st > cx[i + 1]) ++i;
    return cy[i] + (cy[i + 1] - cy[i]) * ((st - cx[i]) / (cx[i + 1] - cx[i]));
}

}  // namespace

int ndiffs(Vec x, int max_d) {
    if (is_constant(x)) return 0;
    auto test = [](const Vec& v, bool& na) {
        const double p = kpss_pval(kpss_stat(v));
        na = std::isnan(p);
        return !na && p < 0.05;
    };
    int d = 0;
    bool na = false;
    bool dodiff = test(x, na);
    if (na) return d;
    while (dodiff && d < max_d) {
        ++d;
        x = diff_lag(x, 1);
        if (is_constant(x)) return d;
        dodiff = test(x, na);
        if (na) return d - 1;
    }
    return d;
}

// ------------------------------------------------------------------------------------------------ auto.arima
namespace {

struct MyFit {
    bool ok = false;                // stats::arima succeeded (R's fit has $arma)
    double ic = INF;
    ArimaFit fit;
    int p = 0, d = 0, q = 0, P = 0, D = 0, Q = 0;
    bool constant = false;
    double aicc = NaN;
};

double min_root_modulus(const Vec& coefs, double sign) {   // polyroot(c(1, sign * coefs[1:last.nonzero]))
    int last = -1;
    for (int i = 0; i < static_cast<int>(coefs.size()); ++i) if (std::fabs(coefs[i]) > 1e-8) last = i;
    if (last < 0) return INF;
    Vec z(last + 2);
    z[0] = 1.0;
    for (int i = 0; i <= last; ++i) z[i + 1] = sign * coefs[i];
    double mn = INF;
    for (const auto& r : polyroot(z)) mn = std::min(mn, std::abs(r));
    return mn;
}

MyFit myarima(const Vec& x, int p, int d, int q, int P, int D, int Q, int freq, bool constant, bool approximation,
              double offset, bool use_aic, const std::vector<Vec>& xreg) {
    MyFit r;
    r.p = p; r.d = d; r.q = q; r.P = P; r.D = D; r.Q = Q; r.constant = constant;
    const int n = static_cast<int>(x.size());
    ArimaSpec s;
    s.p = p; s.d = d; s.q = q; s.P = P; s.D = D; s.Q = Q; s.period = freq;
    s.method = approximation ? ArimaMethod::CSS : ArimaMethod::CSS_ML;
    s.xreg = xreg;
    if (d + D == 1 && constant) { s.drift = true; s.include_mean = true; }
    else { s.drift = false; s.include_mean = constant; }
    try { r.fit = arima_fit(x, s); } catch (const std::exception&) { return r; }
    if (!r.fit.var_ok) return r;                        // solve() error inside stats::arima -> try-error
    r.ok = true;
    const int nstar = n - d - D * freq;
    const int npar = static_cast<int>(r.fit.coef.size()) + 1;
    double aic = r.fit.aic;
    if (approximation) aic = offset + nstar * std::log(r.fit.sigma2) + 2.0 * npar;
    double ic = INF;
    if (!std::isnan(aic)) {
        const double aicc = aic + 2.0 * npar * (npar + 1) / static_cast<double>(nstar - npar - 1);
        r.aicc = aicc;
        ic = use_aic ? aic : aicc;
        if (std::isnan(ic)) ic = INF;
    }
    double minroot = 2.0;
    if (p + P > 0) minroot = std::min(minroot, min_root_modulus(r.fit.model.phi, -1.0));
    if (q + Q > 0 && ic < INF) minroot = std::min(minroot, min_root_modulus(r.fit.model.theta, 1.0));
    bool nanvar = false;
    for (double v : r.fit.var_diag) if (std::isnan(v) || v < 0) nanvar = true;
    if (minroot < 1.01 || nanvar) ic = INF;
    r.ic = ic;
    // myarima's sigma2: sum(residuals^2, na.rm = TRUE) / (nstar - npar + 1)
    double ss = 0;
    for (double e : r.fit.residuals) if (!std::isnan(e)) ss += e * e;
    r.fit.sigma2 = ss / (nstar - npar + 1);
    return r;
}

// Forecast of a model with no ARMA part: differencing only, plus a known mean (reg = 1) or drift (reg = 2).
Vec differencing_forecast(const Vec& x, int d, int D, int m, double c, int reg, int h) {
    Vec Delta{1.0};
    auto conv = [](const Vec& a, const Vec& b) {
        Vec o(a.size() + b.size() - 1, 0.0);
        for (size_t i = 0; i < a.size(); ++i) for (size_t j = 0; j < b.size(); ++j) o[i + j] += a[i] * b[j];
        return o;
    };
    for (int i = 0; i < d; ++i) Delta = conv(Delta, {1.0, -1.0});
    for (int i = 0; i < D; ++i) { Vec sd(m + 1, 0.0); sd[0] = 1.0; sd[m] = -1.0; Delta = conv(Delta, sd); }
    const int n = static_cast<int>(x.size());
    auto regv = [&](int t) { return reg == 1 ? c : (reg == 2 ? c * t : 0.0); };   // t is 1-based
    Vec z(n + h);
    for (int t = 1; t <= n; ++t) z[t - 1] = x[t - 1] - regv(t);
    Vec f(h);
    for (int l = 1; l <= h; ++l) {
        double v = 0.0;
        for (size_t i = 1; i < Delta.size(); ++i) v += -Delta[i] * z[n + l - 1 - i];
        z[n + l - 1] = v;
        f[l - 1] = v + regv(n + l);
    }
    return f;
}

}  // namespace

AutoArimaResult auto_arima(const Vec& y, int freq, bool seasonal, bool log_transform, int h,
                           const std::vector<Vec>& xreg, const std::vector<Vec>& newxreg) {
    AutoArimaResult out;
    const int n = static_cast<int>(y.size());
    if (n == 0) throw std::runtime_error("All data are missing");
    for (double v : y) if (!std::isfinite(v)) throw std::runtime_error("missing or non-finite values are not supported");
    if (is_constant(y)) {                               // Arima(x, order = c(0, 0, 0), fixed = mean(x))
        out.forecast.assign(h, r_mean(y));
        out.mean = true;
        out.period = freq;
        return out;
    }
    int m = seasonal ? freq : 1;
    if (m < 1) m = 1;
    int max_p = std::min(5, n / 3), max_q = std::min(5, n / 3);
    int max_P = std::min(2, n / (3 * m)), max_Q = std::min(2, n / (3 * m));
    const bool use_aic = n <= 3;
    Vec x = y;
    if (log_transform) {
        // BoxCox(x, 0): zeros give -Inf and every model fit fails in R, so auto.arima stops; negative values would
        // become NA in R (not supported here). Either way this call of the chain fails.
        for (double v : y) if (!(v > 0)) throw std::runtime_error("lambda = 0 needs positive data");
        for (double& v : x) v = std::log(v);
    }
    // xreg: the differencing tests run on residuals(lm(x ~ xregg)), xregg = the non-constant columns
    for (const Vec& c : xreg) if (static_cast<int>(c.size()) != n) throw std::runtime_error("xreg length does not match");
    for (const Vec& c : newxreg) if (static_cast<int>(c.size()) < h) throw std::runtime_error("newxreg is shorter than h");
    if (newxreg.size() != xreg.size()) throw std::runtime_error("'xreg' and 'newxreg' have different numbers of columns");
    std::vector<Vec> xregg;
    for (const Vec& c : xreg) if (!is_constant(c)) xregg.push_back(c);
    Vec xx = x;
    if (!xregg.empty()) {
        std::vector<Vec> X1{Vec(n, 1.0)};
        X1.insert(X1.end(), xregg.begin(), xregg.end());
        Vec sv;
        svd_v(X1, &sv);
        double ssum = 0.0, smin = INF;
        for (double v : sv) { ssum += v; smin = std::min(smin, v); }
        if (smin / ssum < 2.220446049250313e-16) throw std::runtime_error("xreg is rank deficient");
        Vec cf, se;
        if (lm_qr(x, X1, cf, se) != static_cast<int>(X1.size())) throw std::runtime_error("xreg is rank deficient");
        for (int i = 0; i < n; ++i) {
            double fv = 0.0;
            for (size_t j = 0; j < X1.size(); ++j) fv += X1[j][i] * cf[j];
            xx[i] = x[i] - fv;
        }
    }
    auto any_constant = [](const std::vector<Vec>& cols) {
        for (const Vec& c : cols) if (is_constant(c)) return true;
        return false;
    };
    int D = 0;
    if (m == 1) { D = 0; max_P = max_Q = 0; }
    else if (n <= 2 * m) D = 0;
    else {
        D = nsdiffs(xx, freq);
        if (D > 0 && !xregg.empty()) {
            std::vector<Vec> dr;
            for (const Vec& c : xregg) dr.push_back(diff_lag(c, m));
            if (any_constant(dr)) --D;
        }
    }
    Vec dx = D > 0 ? diff_lag(xx, m) : xx;
    std::vector<Vec> diffxreg;
    for (const Vec& c : xregg) diffxreg.push_back(D > 0 ? diff_lag(c, m) : c);
    int d = ndiffs(dx, 2);
    if (d > 0 && !xregg.empty()) {
        for (Vec& c : diffxreg) for (int i = 0; i < d; ++i) c = diff_lag(c, 1);
        if (any_constant(diffxreg)) --d;
    }
    for (int i = 0; i < d; ++i) dx = diff_lag(dx, 1);
    if (dx.empty()) throw std::runtime_error("Not enough data to proceed");
    out.period = (D > 0) ? m : freq;
    if (is_constant(dx) && !xreg.empty()) {
        // Arima(x, order = c(0, d, 0), seasonal = c(0, D, 0), xreg = xreg): regression with differencing only (lambda
        // is not passed on in R, so no back-transform)
        ArimaSpec s;
        s.d = d; s.D = D; s.period = (D > 0) ? m : freq; s.include_mean = true; s.method = ArimaMethod::CSS_ML;
        s.xreg = xreg;
        const ArimaFit f = arima_fit(x, s);
        out.d = d; out.D = D; out.mean = f.has_intercept;
        out.forecast = arima_forecast(f, h, &newxreg);
        return out;
    }
    if (is_constant(dx)) {
        // R fits these with Arima(x, ...) without passing lambda, so the forecasts stay on the transformed scale.
        out.d = d; out.D = D;
        if (D > 0 && d == 0) {
            Vec t(dx.size());
            for (size_t i = 0; i < dx.size(); ++i) t[i] = dx[i] / m;
            out.forecast = differencing_forecast(x, 0, D, m, r_mean(t), 2, h);
            out.drift = true;
        } else if (D > 0 && d > 0) {
            out.forecast = differencing_forecast(x, d, D, m, 0.0, 0, h);
        } else if (d == 2) {
            out.forecast = differencing_forecast(x, d, 0, m, 0.0, 0, h);
        } else {
            out.forecast = differencing_forecast(x, d, 0, m, r_mean(dx), d == 0 ? 1 : 2, h);
            (d == 0 ? out.mean : out.drift) = true;
        }
        return out;
    }
    if (m > 1) {
        if (max_P > 0) max_p = std::min(max_p, m - 1);
        if (max_Q > 0) max_q = std::min(max_q, m - 1);
    }
    const bool approximation = n > 150 || freq > 12;
    double offset = 0.0;
    if (approximation) {
        ArimaSpec s;
        s.d = d; s.D = D; s.period = (D > 0) ? m : freq;
        s.include_mean = true;
        s.method = ArimaMethod::CSS_ML;
        s.xreg = xreg;
        try {
            const ArimaFit f = arima_fit(x, s);
            if (f.var_ok) offset = -2.0 * f.loglik - n * std::log(f.sigma2);
        } catch (const std::exception&) { offset = 0.0; }
    }
    const bool allowdrift = (d + D) == 1, allowmean = (d + D) == 0;
    bool constant = allowdrift || allowmean;
    int start_p = 2, start_q = 2, start_P = 1, start_Q = 1;
    if (n < 10) { start_p = std::min(start_p, 1); start_q = std::min(start_q, 1); start_P = 0; start_Q = 0; }
    int p = std::min(start_p, max_p), q = std::min(start_q, max_q);
    int P = std::min(start_P, max_P), Q = std::min(start_Q, max_Q);
    const int nmodels = 94;
    struct Row { int p, q, P, Q; bool c; double ic; };
    std::vector<Row> results;
    auto fitm = [&](int p_, int q_, int P_, int Q_, bool c_) {
        return myarima(x, p_, d, q_, P_, D, Q_, freq, c_, approximation, offset, use_aic, xreg);
    };
    auto newmodel = [&](int p_, int q_, int P_, int Q_, bool c_) {
        for (const Row& r : results) if (r.p == p_ && r.q == q_ && r.P == P_ && r.Q == Q_ && r.c == c_) return false;
        return true;
    };
    MyFit best = fitm(p, q, P, Q, constant);
    results.push_back({p, q, P, Q, constant, best.ic});
    MyFit f = fitm(0, 0, 0, 0, constant);
    results.push_back({0, 0, 0, 0, constant, f.ic});
    if (f.ic < best.ic) { best = f; p = q = P = Q = 0; }
    int k = 2;
    if (max_p > 0 || max_P > 0) {
        const int pp = max_p > 0, PP = (m > 1) && (max_P > 0);
        f = fitm(pp, 0, PP, 0, constant);
        results.push_back({pp, 0, PP, 0, constant, f.ic});
        if (f.ic < best.ic) { best = f; p = pp; P = PP; q = Q = 0; }
        ++k;
    }
    if (max_q > 0 || max_Q > 0) {
        const int qq = max_q > 0, QQ = (m > 1) && (max_Q > 0);
        f = fitm(0, qq, 0, QQ, constant);
        results.push_back({0, qq, 0, QQ, constant, f.ic});
        if (f.ic < best.ic) { best = f; p = P = 0; Q = QQ; q = qq; }
        ++k;
    }
    if (constant) {
        f = fitm(0, 0, 0, 0, false);
        results.push_back({0, 0, 0, 0, false, f.ic});
        if (f.ic < best.ic) { best = f; p = q = P = Q = 0; }   // R leaves `constant` unchanged here
        ++k;
    }
    struct Move { int dp, dq, dP, dQ; };
    static const Move moves[] = {
        {0, 0, -1, 0}, {0, 0, 0, -1}, {0, 0, 1, 0}, {0, 0, 0, 1}, {0, 0, -1, -1}, {0, 0, -1, 1}, {0, 0, 1, -1},
        {0, 0, 1, 1}, {-1, 0, 0, 0}, {0, -1, 0, 0}, {1, 0, 0, 0}, {0, 1, 0, 0}, {-1, -1, 0, 0}, {-1, 1, 0, 0},
        {1, -1, 0, 0}, {1, 1, 0, 0}};
    int startk = 0;
    while (startk < k && k < nmodels) {
        startk = k;
        bool jumped = false;                            // R's `next`
        for (const Move& mv : moves) {
            const int np_ = p + mv.dp, nq = q + mv.dq, nP = P + mv.dP, nQ = Q + mv.dQ;
            if (np_ < 0 || np_ > max_p || nq < 0 || nq > max_q || nP < 0 || nP > max_P || nQ < 0 || nQ > max_Q) continue;
            if (!newmodel(np_, nq, nP, nQ, constant)) continue;
            ++k;
            if (k > nmodels) { jumped = true; break; }
            f = fitm(np_, nq, nP, nQ, constant);
            results.push_back({np_, nq, nP, nQ, constant, f.ic});
            if (f.ic < best.ic) { best = f; p = np_; q = nq; P = nP; Q = nQ; jumped = true; break; }
        }
        if (jumped) continue;
        if (allowdrift || allowmean) {
            if (newmodel(p, q, P, Q, !constant)) {
                ++k;
                if (k > nmodels) continue;
                f = fitm(p, q, P, Q, !constant);
                results.push_back({p, q, P, Q, !constant, f.ic});
                if (f.ic < best.ic) { best = f; constant = !constant; }
            }
        }
    }
    if (approximation && best.ok) {                     // re-fit without approximation, best IC first
        std::vector<size_t> order(results.size());
        std::iota(order.begin(), order.end(), size_t{0});
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return results[a].ic < results[b].ic; });
        for (size_t i : order) {
            const Row& r = results[i];
            MyFit g = myarima(x, r.p, d, r.q, r.P, D, r.Q, freq, r.c, false, 0.0, use_aic, xreg);
            if (g.ic < INF) { best = g; break; }
        }
    }
    if (!(best.ic < INF)) throw std::runtime_error("No suitable ARIMA model found");
    out.forecast = arima_forecast(best.fit, h, &newxreg);
    if (log_transform) for (double& v : out.forecast) v = std::exp(v);
    out.p = best.p; out.d = best.d; out.q = best.q; out.P = best.P; out.D = best.D; out.Q = best.Q;
    out.drift = best.fit.has_drift; out.mean = best.fit.has_intercept;
    out.loglik = best.fit.loglik; out.aicc = best.aicc; out.sigma2 = best.fit.sigma2;
    return out;
}

namespace {

double r_sinpi(double x) {                              // R's sinpi / cospi (nmath/cospi.c)
    x = std::fmod(x, 2.0);
    if (x <= -1) x += 2.0; else if (x > 1.0) x -= 2.0;
    if (x == 0.0 || x == 1.0) return 0.0;
    if (x == 0.5) return 1.0;
    if (x == -0.5) return -1.0;
    return std::sin(3.14159265358979323846 * x);
}

double r_cospi(double x) {
    x = std::fmod(std::fabs(x), 2.0);
    if (std::fmod(x, 1.0) == 0.5) return 0.0;
    if (x == 1.0) return -1.0;
    if (x == 0.0) return 1.0;
    return std::cos(3.14159265358979323846 * x);
}

}  // namespace

std::vector<Vec> fourier_terms(double period, int K, int t0, int len) {
    if (2.0 * K > period) throw std::runtime_error("K must be not be greater than period/2");
    std::vector<Vec> X;
    for (int j = 1; j <= K; ++j) {
        const double p = j / period;
        const bool with_sin = std::fabs(2 * p - std::nearbyint(2 * p)) > 2.220446049250313e-16;
        Vec s(len), c(len);
        for (int t = 0; t < len; ++t) {
            const double arg = 2 * p * (t0 + t + 1);
            s[t] = r_sinpi(arg);
            c[t] = r_cospi(arg);
        }
        if (with_sin) X.push_back(std::move(s));
        X.push_back(std::move(c));
    }
    return X;
}

AutoArimaResult dhr_arima(const Vec& y, int freq, int n_periods, int K, int h) {
    if (n_periods > 1) throw std::runtime_error("Number of periods does not match number of orders");
    const int n = static_cast<int>(y.size());
    const std::vector<Vec> X = fourier_terms(freq, K, 0, n), Xf = fourier_terms(freq, K, n, h);
    return auto_arima(y, freq, false, false, h, X, Xf);
}

AutoArimaResult auto_arima_study(const Vec& y, int freq, int h) {
    try { AutoArimaResult r = auto_arima(y, freq, true, true, h); r.step = 1; return r; }
    catch (const std::exception&) {}
    try { AutoArimaResult r = auto_arima(y, freq, true, false, h); r.step = 2; return r; }
    catch (const std::exception&) {}
    AutoArimaResult r = auto_arima(y, freq, false, false, h);
    r.step = 3;
    return r;
}

}  // namespace tsf
