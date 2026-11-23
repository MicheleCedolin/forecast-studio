// R's least squares as lm() / lsfit() compute it: LINPACK dqrdc2 (Householder QR with R's limited column pivoting,
// tol 1e-7) and dqrsl (job 1110: Q'y and back-substituted coefficients) of R's src/appl, with the reference BLAS
// (ddot, daxpy, dscal and the LAPACK 3.10+ dnrm2 of R's src/extra/blas). Header-only.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "util.hpp"

namespace tsf {
namespace rl {

inline double ddot(int n, const double* x, const double* y) {
    double t = 0.0;
    for (int i = 0; i < n; ++i) t = t + x[i] * y[i];
    return t;
}
inline void daxpy(int n, double a, const double* x, double* y) {
    if (n <= 0 || a == 0.0) return;
    for (int i = 0; i < n; ++i) y[i] = y[i] + a * x[i];
}
inline void dscal(int n, double a, double* x) { for (int i = 0; i < n; ++i) x[i] = a * x[i]; }
inline double dnrm2(int n, const double* x) {
    const double tsml = std::ldexp(1.0, -511), tbig = std::ldexp(1.0, 486);
    const double ssml = std::ldexp(1.0, 537), sbig = std::ldexp(1.0, -538);
    const double maxN = 1.7976931348623157e308;
    if (n <= 0) return 0.0;
    double asml = 0.0, amed = 0.0, abig = 0.0;
    bool notbig = true;
    for (int i = 0; i < n; ++i) {
        const double ax = std::fabs(x[i]);
        if (ax > tbig) { const double t = ax * sbig; abig = abig + t * t; notbig = false; }
        else if (ax < tsml) { if (notbig) { const double t = ax * ssml; asml = asml + t * t; } }
        else amed = amed + ax * ax;
    }
    double scl, sumsq;
    if (abig > 0.0) {
        if (amed > 0.0 || amed > maxN || amed != amed) abig = abig + (amed * sbig) * sbig;
        scl = 1.0 / sbig; sumsq = abig;
    } else if (asml > 0.0) {
        if (amed > 0.0 || amed > maxN || amed != amed) {
            amed = std::sqrt(amed);
            asml = std::sqrt(asml) / ssml;
            const double ymin = (asml > amed) ? amed : asml, ymax = (asml > amed) ? asml : amed;
            const double r = ymin / ymax;
            scl = 1.0; sumsq = (ymax * ymax) * (1.0 + r * r);
        } else {
            scl = 1.0 / ssml; sumsq = asml;
        }
    } else {
        scl = 1.0; sumsq = amed;
    }
    return scl * std::sqrt(sumsq);
}

// Coefficients of y on the columns cols (each of length n) as lm.fit(): dqrls(x, n, p, y, 1, tol = 1e-7, ...).
// Aliased (rank-deficient) columns get NaN, as R's NA coefficients.
inline Vec dqrls_coef(const std::vector<Vec>& cols, const Vec& yin, double tol = 1e-7) {
    const int n = static_cast<int>(yin.size()), p = static_cast<int>(cols.size());
    std::vector<double> X(static_cast<size_t>(n) * p);
    for (int j = 0; j < p; ++j) for (int i = 0; i < n; ++i) X[static_cast<size_t>(j) * n + i] = cols[j][i];
    auto x = [&](int i, int j) -> double& { return X[static_cast<size_t>(j - 1) * n + (i - 1)]; };
    std::vector<double> qraux(p + 1), w1(p + 1), w2(p + 1);
    std::vector<int> jpvt(p + 1);
    for (int j = 1; j <= p; ++j) jpvt[j] = j;
    // dqrdc2
    if (n > 0)
        for (int j = 1; j <= p; ++j) {
            qraux[j] = dnrm2(n, &x(1, j));
            w1[j] = qraux[j];
            w2[j] = qraux[j];
            if (w2[j] == 0.0) w2[j] = 1.0;
        }
    const int lup = std::min(n, p);
    int k = p + 1;
    for (int l = 1; l <= lup; ++l) {
        while (!(l >= k || qraux[l] >= w2[l] * tol)) {
            for (int i = 1; i <= n; ++i) {
                const double t = x(i, l);
                for (int j = l + 1; j <= p; ++j) x(i, j - 1) = x(i, j);
                x(i, p) = t;
            }
            const int ii = jpvt[l];
            const double t = qraux[l], tt = w1[l], ttt = w2[l];
            for (int j = l + 1; j <= p; ++j) {
                jpvt[j - 1] = jpvt[j]; qraux[j - 1] = qraux[j]; w1[j - 1] = w1[j]; w2[j - 1] = w2[j];
            }
            jpvt[p] = ii; qraux[p] = t; w1[p] = tt; w2[p] = ttt;
            k = k - 1;
        }
        if (l != n) {
            double nrmxl = dnrm2(n - l + 1, &x(l, l));
            if (nrmxl != 0.0) {
                if (x(l, l) != 0.0) nrmxl = std::copysign(std::fabs(nrmxl), x(l, l));
                dscal(n - l + 1, 1.0 / nrmxl, &x(l, l));
                x(l, l) = 1.0 + x(l, l);
                for (int j = l + 1; j <= p; ++j) {
                    const double t = -ddot(n - l + 1, &x(l, l), &x(l, j)) / x(l, l);
                    daxpy(n - l + 1, t, &x(l, l), &x(l, j));
                    if (qraux[j] != 0.0) {
                        const double r = std::fabs(x(l, j)) / qraux[j];
                        double tt = 1.0 - r * r;
                        tt = std::max(tt, 0.0);
                        if (std::fabs(tt) >= 1e-6) qraux[j] = qraux[j] * std::sqrt(tt);
                        else { qraux[j] = dnrm2(n - l, &x(l + 1, j)); w1[j] = qraux[j]; }
                    }
                }
                qraux[l] = x(l, l);
                x(l, l) = -nrmxl;
            }
        }
    }
    k = std::min(k - 1, n);
    // dqrsl(job = 1110) for b: qty = Q'y, b = R^{-1} qty[1..k]
    std::vector<double> qty(yin.begin(), yin.end());
    const int ju = std::min(k, n - 1);
    for (int j = 1; j <= ju; ++j) {
        if (qraux[j] == 0.0) continue;
        const double temp = x(j, j);
        x(j, j) = qraux[j];
        const double t = -ddot(n - j + 1, &x(j, j), &qty[j - 1]) / x(j, j);
        daxpy(n - j + 1, t, &x(j, j), &qty[j - 1]);
        x(j, j) = temp;
    }
    Vec b(qty.begin(), qty.begin() + k);
    for (int jj = 1; jj <= k; ++jj) {
        const int j = k - jj + 1;
        b[j - 1] = b[j - 1] / x(j, j);
        if (j != 1) daxpy(j - 1, -b[j - 1], &x(1, j), b.data());
    }
    Vec coef(p, std::nan(""));
    for (int j = 1; j <= k; ++j) coef[jpvt[j] - 1] = b[j - 1];
    return coef;
}

}  // namespace rl
}  // namespace tsf
