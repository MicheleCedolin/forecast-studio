// fracdiff::fracdiff(x, nar = 0, nma = 0)$d (R package fracdiff 1.5, GPL-2+; Fraley, Leisch, Maechler, Reisen,
// Lemonte): the C routines fracdf() / dopt() / pqopt() / fdfilt() of src/fdcore.c for the case without ARMA terms
// (pq = 0), with drange = c(0, 0.5), dtol = .Machine$double.eps^0.25 and M = 100. Loop order kept as in the C code.
// tsfeatures::hurst() is this d + 0.5.
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "fracdiff.hpp"

namespace tsf {
namespace {

struct Filt {
    int n = 0, m = 0;
};

// fdfilt() for pq = 0: fills y (standardised innovations) and returns slogvk; false on a gamma-function failure.
bool fdfilt(const std::vector<double>& xv, double d, const Filt& D, std::vector<double>& yv, double& slogvk) {
    const int n = D.n;
    const int mcap = std::min(D.m, n);
    // 1-based arrays as in the C code
    std::vector<double> x(n + 1), y(n + 1), amk(n + 1), ak(n + 1), vk(std::max(mcap, 2) + 1), phi(D.m + 2), pi(D.m + 2);
    for (int i = 0; i < n; ++i) x[i + 1] = xv[i];
    amk[1] = 0.;
    ak[1] = 1.;
    double z = d / (1. - d);
    if (n >= 2) {
        amk[2] = z * x[1];
        ak[2] = 1. - z;
    }
    phi[1] = z;
    const double g1 = std::tgamma(1. - d);
    const double g2 = std::tgamma(1. - d * 2.);
    if (!std::isfinite(g1) || g1 == 0. || !std::isfinite(g2)) return false;
    const double t0 = 1. / g1;                                     // dgamr(1 - d)
    const double g0 = g2 * (t0 * t0);
    vk[1] = g0;
    if (mcap >= 2) vk[2] = g0 * (1. - z * z);
    double t, u, v, s, r;
    for (int k = 3; k <= mcap; ++k) {
        const int km = k - 1;
        t = (double)km;
        u = t - d;
        for (int j = 1; j <= km - 1; ++j) {
            s = t - (double)j;
            phi[j] *= t * (s - d) / (u * s);
        }
        v = d / u;
        phi[km] = v;
        vk[k] = vk[km] * (1. - v * v);
        u = 0.;
        v = 1.;
        for (int j = 1; j <= km; ++j) {
            t = phi[j];
            u += t * x[k - j];
            v -= t;
        }
        amk[k] = u;
        ak[k] = v;
    }
    if (D.m < n) {
        pi[1] = d;
        s = d;
        for (int j = 2; j <= mcap; ++j) {
            u = (double)j;
            t = pi[j - 1] * ((u - 1. - d) / u);
            s += t;
            pi[j] = t;
        }
        s = 1. - s;
        r = 0.;
        u = (double)mcap;
        t = u * pi[mcap];
        for (int k = mcap + 1; k <= n; ++k) {
            const int km = k - mcap;
            z = 0.;
            for (int j = 1; j <= mcap; ++j) z += pi[j] * x[k - j];
            if (r == 0.) {
                amk[k] = z;
                ak[k] = s;
            } else {
                v = t * (1. - std::pow(u / k, d)) / d;
                amk[k] = z + v * r / ((double)km - 1.);
                ak[k] = s - v;
            }
            r += x[km];
        }
    }
    r = 0.;
    s = 0.;
    for (int k = 1; k <= n; ++k) {
        t = ak[k];
        u = (x[k] - amk[k]) * t;
        v = t * t;
        if (k <= mcap) {
            z = vk[k];
            u /= z;
            v /= z;
        }
        r += u;
        s += v;
    }
    const double hatmu = r / s;
    s = 0.;
    for (int k = 1; k <= mcap; ++k) s += std::log(vk[k]);
    slogvk = s;
    yv.assign(n, 0.);
    for (int k = 1; k <= n; ++k) {
        t = x[k] - amk[k] - hatmu * ak[k];
        if (k <= mcap) t /= std::sqrt(vk[k]);
        yv[k - 1] = t;
    }
    return true;
}

// pqopt() for pq = 0: minus the log likelihood (up to constants) at d.
double pqopt(const std::vector<double>& x, double d, const Filt& D, bool& bad) {
    std::vector<double> y;
    double slogvk = 0.;
    if (!fdfilt(x, d, D, y, slogvk)) {
        bad = true;
        return 1.7976931348623157e308;
    }
    const double t = (double)D.n;
    double dot = 0.;
    for (double v : y) dot += v * v;                               // BLAS ddot (reference: sequential)
    const double wnv = dot / t;
    const double u = t * (std::log(wnv) + 2.8378) + slogvk;
    return u / 2;
}

}  // namespace

double fracdiff_d(const std::vector<double>& x, int M) {
    const int n = static_cast<int>(x.size());
    if (n < 2) throw std::runtime_error("fracdiff: series too short");
    Filt D;
    D.n = n;
    D.m = M > 0 ? M : 100;
    const double epsmin = 1.1102230246251565e-16, epsmax = 2.220446049250313e-16;
    double dtol = std::pow(epsmax, 0.25);
    if (dtol > .1) dtol = .1;
    const double tol_d = std::max(dtol, std::sqrt(epsmin));
    const int maxopt = 100;
    // dopt(): Brent's fmin on [0, 0.5] started at a + c (b - a) (dinit = 0 is not inside the open range)
    const double cc = .38196601125011;
    double aa = 0., bb = 0.5;
    double eps = std::sqrt(epsmax);
    double vv = aa + cc * (bb - aa), ww = vv, xx = vv, uu = xx, dd = 0., ee = 0.;
    double rr, ss, tt, hh, tol1, tol2, fu = 0.;
    int nopt = 1;
    bool bad = false;
    double fx = pqopt(x, xx, D, bad);
    double fv = fx, fw = fx;
    const double tol = std::max(tol_d, 0.);
    const double tol3 = tol / 3.;
    for (;;) {
        if (bad) throw std::runtime_error("fracdiff: gamma function error");
        hh = (aa + bb) * .5;
        tol1 = eps * (std::fabs(xx) + 1.) + tol3;
        tol2 = tol1 * 2.;
        const double delta = std::fabs(xx - hh) + (bb - aa) * .5;
        if (delta <= tol2) break;
        if (nopt >= maxopt) break;
        rr = 0.;
        ss = 0.;
        tt = 0.;
        if (std::fabs(ee) > tol1) {
            rr = (xx - ww) * (fx - fv);
            ss = (xx - vv) * (fx - fw);
            tt = (xx - vv) * ss - (xx - ww) * rr;
            ss = (ss - rr) * 2.;
            if (ss <= 0.) ss = -ss;
            else tt = -tt;
            rr = ee;
            ee = dd;
        }
        if (std::fabs(tt) >= std::fabs(ss * .5 * rr) || tt <= ss * (aa - xx) || tt >= ss * (bb - xx)) {
            ee = (xx >= hh) ? aa - xx : bb - xx;
            dd = cc * ee;
        } else {
            dd = tt / ss;
            uu = xx + dd;
            if (uu - aa < tol2 || bb - uu < tol2) {
                dd = tol1;
                if (xx >= hh) dd = -dd;
            }
        }
        if (std::fabs(dd) >= tol1) uu = xx + dd;
        else uu = (dd <= 0.) ? xx - tol1 : xx + tol1;
        ++nopt;
        fu = pqopt(x, uu, D, bad);
        if (fx >= fu) {
            if (uu >= xx) aa = xx;
            else bb = xx;
            vv = ww;
            fv = fw;
            ww = xx;
            fw = fx;
            xx = uu;
            fx = fu;
        } else {
            if (uu >= xx) bb = uu;
            else aa = uu;
            if (fu > fw && ww != xx) {
                if (fu <= fv || vv == xx || vv == ww) {
                    vv = uu;
                    fv = fu;
                }
            } else {
                vv = ww;
                fv = fw;
                ww = uu;
                fw = fu;
            }
        }
    }
    return xx;
}

}  // namespace tsf
