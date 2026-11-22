#include "util.hpp"

#include <algorithm>
#include <numeric>

namespace tsf {

int r_frequency(const double* periods, int n_periods) {
    double mx = 1.0;
    for (int i = 0; i < n_periods; ++i) mx = std::max(mx, periods[i]);
    return std::max(1, static_cast<int>(std::floor(mx)));
}

double ptp(const Vec& x) {
    auto [lo, hi] = std::minmax_element(x.begin(), x.end());
    return *hi - *lo;
}

double mean(const Vec& x) { return std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size()); }

Vec acf(const Vec& x, int nlags) {
    const int n = static_cast<int>(x.size());
    const double mu = mean(x);
    Vec d(n);
    for (int i = 0; i < n; ++i) d[i] = x[i] - mu;
    double c0 = 0.0;
    for (double v : d) c0 += v * v;
    Vec r(nlags);
    for (int k = 1; k <= nlags; ++k) {
        double s = 0.0;
        for (int i = 0; i < n - k; ++i) s += d[i] * d[i + k];
        r[k - 1] = s / c0;
    }
    return r;
}

Vec decompose_multiplicative(const Vec& x, int m) {
    const int n = static_cast<int>(x.size());
    Vec filt;
    if (m % 2 == 0) {
        filt.assign(m + 1, 1.0 / m);
        filt.front() = filt.back() = 0.5 / m;
    } else {
        filt.assign(m, 1.0 / m);
    }
    const int k = static_cast<int>(filt.size()), half = k / 2;
    Vec trend(n, std::nan(""));
    for (int j = 0; j + k <= n; ++j) {                 // np.convolve(x, filt[::-1], "valid"); filt is symmetric
        double s = 0.0;
        for (int i = 0; i < k; ++i) s += x[j + i] * filt[k - 1 - i];
        trend[half + j] = s;
    }
    Vec figure(m);
    for (int i = 0; i < m; ++i) {
        double s = 0.0;
        int c = 0;
        for (int t = i; t < n; t += m) {
            const double v = x[t] / trend[t];
            if (!std::isnan(v)) { s += v; ++c; }
        }
        figure[i] = c ? s / c : std::nan("");
    }
    double fs = 0.0;
    int fc = 0;
    for (double v : figure) if (!std::isnan(v)) { fs += v; ++fc; }
    if (fc == 0) return {};
    const double fm = fs / fc;
    Vec seas(n);
    for (int t = 0; t < n; ++t) seas[t] = figure[t % m] / fm;
    return seas;
}

Vec linspace(double start, double stop, int num) {
    Vec v(num);
    if (num == 1) { v[0] = start; return v; }
    const double step = (stop - start) / (num - 1);
    for (int i = 0; i < num; ++i) v[i] = start + i * step;
    v[num - 1] = stop;
    return v;
}

double fminbound(const std::function<double(double)>& func, double x1, double x2, double xatol, int maxiter) {
    const double sqrt_eps = std::sqrt(2.2e-16);
    const double golden_mean = 0.5 * (3.0 - std::sqrt(5.0));
    double a = x1, b = x2;
    double fulc = a + golden_mean * (b - a);
    double nfc = fulc, xf = fulc;
    double rat = 0.0, e = 0.0;
    double x = xf;
    double fx = func(x);
    int num = 1;
    double ffulc = fx, fnfc = fx;
    double xm = 0.5 * (a + b);
    double tol1 = sqrt_eps * std::fabs(xf) + xatol / 3.0;
    double tol2 = 2.0 * tol1;
    auto sign = [](double v) { return (v > 0) - (v < 0); };
    while (std::fabs(xf - xm) > (tol2 - 0.5 * (b - a))) {
        bool golden = true;
        if (std::fabs(e) > tol1) {
            golden = false;
            double r = (xf - nfc) * (fx - ffulc);
            double q = (xf - fulc) * (fx - fnfc);
            double p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if (q > 0.0) p = -p;
            q = std::fabs(q);
            r = e;
            e = rat;
            if ((std::fabs(p) < std::fabs(0.5 * q * r)) && (p > q * (a - xf)) && (p < q * (b - xf))) {
                rat = (p + 0.0) / q;
                x = xf + rat;
                if (((x - a) < tol2) || ((b - x) < tol2)) {
                    const double si = sign(xm - xf) + ((xm - xf) == 0);
                    rat = tol1 * si;
                }
            } else {
                golden = true;
            }
        }
        if (golden) {
            e = (xf >= xm) ? a - xf : b - xf;
            rat = golden_mean * e;
        }
        const double si = sign(rat) + (rat == 0);
        x = xf + si * std::max(std::fabs(rat), tol1);
        const double fu = func(x);
        ++num;
        if (fu <= fx) {
            if (x >= xf) a = xf; else b = xf;
            fulc = nfc; ffulc = fnfc;
            nfc = xf; fnfc = fx;
            xf = x; fx = fu;
        } else {
            if (x < xf) a = x; else b = x;
            if ((fu <= fnfc) || (nfc == xf)) {
                fulc = nfc; ffulc = fnfc;
                nfc = x; fnfc = fu;
            } else if ((fu <= ffulc) || (fulc == xf) || (fulc == nfc)) {
                fulc = x; ffulc = fu;
            }
        }
        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * std::fabs(xf) + xatol / 3.0;
        tol2 = 2.0 * tol1;
        if (num >= maxiter) break;
    }
    return xf;
}

Vec nelder_mead(const std::function<double(const Vec&)>& f, Vec x0, const Vec& step, double& fbest, int maxiter,
                double ftol, double xtol) {
    const int n = static_cast<int>(x0.size());
    std::vector<Vec> s(n + 1, x0);
    Vec fv(n + 1);
    for (int i = 0; i < n; ++i) s[i + 1][i] += step[i];
    for (int i = 0; i <= n; ++i) fv[i] = f(s[i]);
    std::vector<int> idx(n + 1);
    for (int it = 0; it < maxiter; ++it) {
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return fv[a] < fv[b]; });
        const int best = idx[0], worst = idx[n], second = idx[n - 1];
        double xspread = 0.0;
        for (int i = 1; i <= n; ++i)
            for (int j = 0; j < n; ++j) xspread = std::max(xspread, std::fabs(s[idx[i]][j] - s[best][j]));
        if (std::fabs(fv[worst] - fv[best]) <= ftol * (std::fabs(fv[best]) + 1e-300) && xspread <= xtol) break;
        Vec c(n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) c[j] += s[idx[i]][j] / n;
        auto along = [&](double t) { Vec p(n); for (int j = 0; j < n; ++j) p[j] = c[j] + t * (s[worst][j] - c[j]); return p; };
        Vec xr = along(-1.0);
        const double fr = f(xr);
        if (fr < fv[best]) {
            Vec xe = along(-2.0);
            const double fe = f(xe);
            if (fe < fr) { s[worst] = xe; fv[worst] = fe; } else { s[worst] = xr; fv[worst] = fr; }
        } else if (fr < fv[second]) {
            s[worst] = xr; fv[worst] = fr;
        } else {
            const bool outside = fr < fv[worst];
            Vec xc = along(outside ? -0.5 : 0.5);
            const double fc = f(xc);
            if (fc < (outside ? fr : fv[worst])) {
                s[worst] = xc; fv[worst] = fc;
            } else {
                for (int i = 1; i <= n; ++i) {
                    const int k = idx[i];
                    for (int j = 0; j < n; ++j) s[k][j] = s[best][j] + 0.5 * (s[k][j] - s[best][j]);
                    fv[k] = f(s[k]);
                }
            }
        }
    }
    int b = 0;
    for (int i = 1; i <= n; ++i) if (fv[i] < fv[b]) b = i;
    fbest = fv[b];
    return s[b];
}

double ls_slope(const Vec& x) {
    const int n = static_cast<int>(x.size());
    if (n < 2) return 0.0;
    const double tm = 0.5 * (n - 1), xm = mean(x);
    double sxy = 0.0, sxx = 0.0;
    for (int t = 0; t < n; ++t) { sxy += (t - tm) * (x[t] - xm); sxx += (t - tm) * (t - tm); }
    return sxy / sxx;
}

}  // namespace tsf
