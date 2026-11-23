// stats::supsmu(x, y) with its defaults (span = "cv", bass = 0, periodic = FALSE, unit weights): Friedman's super
// smoother, subroutines supsmu() and smooth() of R's src/library/stats/src/ppr.f (GPL-2+), loop order kept.
// forecast::mstl() uses supsmu(seq_len(n), x) as the trend of a non-seasonal series.
#include <cmath>
#include <vector>

#include "supsmu.hpp"

namespace tsf {
namespace {

// block data bksupsmu: spans (0.05_8, 0.2_8, 0.5), big, sml, eps (the last three single precision literals)
const double SPANS[3] = {0.05, 0.2, 0.5};
const double BIG = static_cast<double>(1.0e20f);
const double EPS = static_cast<double>(1.0e-3f);

// 1-based arrays: x[1..n]
void smooth(int n, const double* x, const double* y, const double* w, double span, int iper, double vsmlsq,
            double* smo, double* acvr) {
    double xm = 0., ym = 0., var = 0., cvar = 0., fbw = 0., fbo, xti, xto, tmp, a, h, sy, wt;
    const int jper = std::abs(iper);
    int ibw = static_cast<int>(0.5 * span * n + 0.5);
    if (ibw < 2) ibw = 2;
    int it = 2 * ibw + 1;
    if (it > n) it = n;
    for (int i = 1; i <= it; ++i) {
        int j = i;
        if (jper == 2) j = i - ibw - 1;
        if (j >= 1) xti = x[j];
        else { j = n + j; xti = x[j] - 1.; }
        wt = w[j];
        fbo = fbw;
        fbw = fbw + wt;
        if (fbw > 0.) xm = (fbo * xm + wt * xti) / fbw;
        if (fbw > 0.) ym = (fbo * ym + wt * y[j]) / fbw;
        tmp = 0.;
        if (fbo > 0.) tmp = fbw * wt * (xti - xm) / fbo;
        var = var + tmp * (xti - xm);
        cvar = cvar + tmp * (y[j] - ym);
    }
    for (int j = 1; j <= n; ++j) {
        int out = j - ibw - 1, in = j + ibw;
        if (!((jper != 2) && (out < 1 || in > n))) {
            if (out < 1) { out = n + out; xto = x[out] - 1.; xti = x[in]; }
            else if (in > n) { in = in - n; xti = x[in] + 1.; xto = x[out]; }
            else { xto = x[out]; xti = x[in]; }
            wt = w[out];
            fbo = fbw;
            fbw = fbw - wt;
            tmp = 0.;
            if (fbw > 0.) tmp = fbo * wt * (xto - xm) / fbw;
            var = var - tmp * (xto - xm);
            cvar = cvar - tmp * (y[out] - ym);
            if (fbw > 0.) xm = (fbo * xm - wt * xto) / fbw;
            if (fbw > 0.) ym = (fbo * ym - wt * y[out]) / fbw;
            wt = w[in];
            fbo = fbw;
            fbw = fbw + wt;
            if (fbw > 0.) xm = (fbo * xm + wt * xti) / fbw;
            if (fbw > 0.) ym = (fbo * ym + wt * y[in]) / fbw;
            tmp = 0.;
            if (fbo > 0.) tmp = fbw * wt * (xti - xm) / fbo;
            var = var + tmp * (xti - xm);
            cvar = cvar + tmp * (y[in] - ym);
        }
        a = 0.;
        if (var > vsmlsq) a = cvar / var;
        smo[j] = a * (x[j] - xm) + ym;
        if (iper > 0) {
            h = 0.;
            if (fbw > 0.) h = 1. / fbw;
            if (var > vsmlsq) h = h + (x[j] - xm) * (x[j] - xm) / var;
            a = 1. - w[j] * h;
            if (a > 0.) acvr[j] = std::fabs(y[j] - smo[j]) / a;
            else if (j > 1) acvr[j] = acvr[j - 1];
            else acvr[j] = 0.;
        }
    }
    int j = 1;
    while (j <= n) {                                    // average smo over tied x
        const int j0 = j;
        sy = smo[j] * w[j];
        fbw = w[j];
        while (j < n && x[j + 1] <= x[j]) {
            ++j;
            sy = sy + w[j] * smo[j];
            fbw = fbw + w[j];
        }
        if (j > j0) {
            a = 0.;
            if (fbw > 0.) a = sy / fbw;
            for (int i = j0; i <= j; ++i) smo[i] = a;
        }
        ++j;
    }
}

}  // namespace

std::vector<double> supsmu(const std::vector<double>& xv, const std::vector<double>& yv) {
    const int n = static_cast<int>(xv.size());
    std::vector<double> x(n + 1), y(n + 1), w(n + 1, 1.0), smo(n + 1), h(n + 1);
    std::vector<std::vector<double>> sc(8, std::vector<double>(n + 1));   // sc[1..7][1..n]
    for (int i = 0; i < n; ++i) { x[i + 1] = xv[i]; y[i + 1] = yv[i]; }
    if (!(x[n] > x[1])) {
        double sy = 0., sw = 0.;
        for (int j = 1; j <= n; ++j) { sy += w[j] * y[j]; sw += w[j]; }
        const double a = sw > 0. ? sy / sw : 0.;
        return std::vector<double>(n, a);
    }
    int i = n / 4, j = 3 * i;
    double scale = x[j] - x[i];
    while (!(scale > 0.)) {
        if (j < n) ++j;
        if (i > 1) --i;
        scale = x[j] - x[i];
    }
    const double vsmlsq = (EPS * scale) * (EPS * scale);
    const int jper = 1;
    for (int k = 1; k <= 3; ++k) {
        smooth(n, x.data(), y.data(), w.data(), SPANS[k - 1], jper, vsmlsq, sc[2 * k - 1].data(), sc[7].data());
        smooth(n, x.data(), sc[7].data(), w.data(), SPANS[1], -jper, vsmlsq, sc[2 * k].data(), h.data());
    }
    for (int jj = 1; jj <= n; ++jj) {
        double resmin = BIG;
        for (int k = 1; k <= 3; ++k) {
            if (sc[2 * k][jj] >= resmin) continue;
            resmin = sc[2 * k][jj];
            sc[7][jj] = SPANS[k - 1];
        }
        // alpha (bass) = 0: no bass adjustment
    }
    smooth(n, x.data(), sc[7].data(), w.data(), SPANS[1], -jper, vsmlsq, sc[2].data(), h.data());
    for (int jj = 1; jj <= n; ++jj) {
        if (sc[2][jj] <= SPANS[0]) sc[2][jj] = SPANS[0];
        if (sc[2][jj] >= SPANS[2]) sc[2][jj] = SPANS[2];
        double f = sc[2][jj] - SPANS[1];
        if (f >= 0.) {
            f = f / (SPANS[2] - SPANS[1]);
            sc[4][jj] = (1. - f) * sc[3][jj] + f * sc[5][jj];
        } else {
            f = -f / (SPANS[1] - SPANS[0]);
            sc[4][jj] = (1. - f) * sc[3][jj] + f * sc[1][jj];
        }
    }
    smooth(n, x.data(), sc[4].data(), w.data(), SPANS[0], -jper, vsmlsq, smo.data(), h.data());
    return std::vector<double>(smo.begin() + 1, smo.end());
}

}  // namespace tsf
