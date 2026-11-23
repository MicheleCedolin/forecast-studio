// See cpoly.hpp. Port of forecast's src/etspolyroot.c (= R's R_cpolyroot of src/main/complex.c). The C file- and
// function-level statics live in one thread_local struct, looked up once per call (thread_local access is costly with
// MinGW's emulated TLS) and passed by reference.
#include "cpoly.hpp"

#include <cfloat>
#include <cmath>
#include <vector>

namespace tsf {

namespace {

const double eta = DBL_EPSILON;
const double are = DBL_EPSILON;
const double mre = 2. * 1.41421356237309504880 * DBL_EPSILON;
const double infin = DBL_MAX;

struct St {
    int nn;
    double *pr, *pi, *hr, *hi, *qpr, *qpi, *qhr, *qhi, *shr, *shi;
    double sr, si, tr, ti, pvr, pvi;
    // vrshft statics
    int v_i, v_j;
    double v_r1, v_r2, v_mp, v_ms, v_tp, v_relstp, v_omp;
    // fxshft statics
    double f_svsi, f_svsr, f_oti, f_otr;
    int f_i, f_j, f_n;
    // R_cpolyroot statics
    int c_d_n, c_i, c_i1, c_i2;
    double c_zi, c_zr, c_xx, c_yy, c_bnd, c_xxx;
};
thread_local St g_state;                                 // trivially constructible, no TLS destructor

double powdi(double x, int n) {                         // R_pow_di
    double xn = 1.0;
    if (n != 0) {
        const bool neg = n < 0;
        if (neg) n = -n;
        for (;;) {
            if (n & 01) xn *= x;
            if (n >>= 1) x *= x; else break;
        }
        if (neg) xn = 1. / xn;
    }
    return xn;
}

void cdivid(double ar, double ai, double br, double bi, double* cr, double* ci) {
    double d, r;
    if (br == 0. && bi == 0.) {
        *cr = *ci = HUGE_VAL;
    } else if (std::fabs(br) >= std::fabs(bi)) {
        r = bi / br;
        d = br + r * bi;
        *cr = (ar + ai * r) / d;
        *ci = (ai - ar * r) / d;
    } else {
        r = br / bi;
        d = bi + r * br;
        *cr = (ar * r + ai) / d;
        *ci = (ai * r - ar) / d;
    }
}

void polyev(int n, double s_r, double s_i, double* p_r, double* p_i, double* q_r, double* q_i, double* v_r, double* v_i) {
    q_r[0] = p_r[0];
    q_i[0] = p_i[0];
    *v_r = q_r[0];
    *v_i = q_i[0];
    for (int i = 1; i < n; i++) {
        const double t = *v_r * s_r - *v_i * s_i + p_r[i];
        q_i[i] = *v_i = *v_r * s_i + *v_i * s_r + p_i[i];
        q_r[i] = *v_r = t;
    }
}

double errev(int n, double* qr, double* qi, double ms, double mp, double a_re, double m_re) {
    double e = std::hypot(qr[0], qi[0]) * m_re / (a_re + m_re);
    for (int i = 0; i < n; i++) e = e * ms + std::hypot(qr[i], qi[i]);
    return e * (a_re + m_re) - mp * m_re;
}

double cpoly_cauchy(int n, double* pot, double* q) {
    double f, x, delf, dx, xm;
    const int n1 = n - 1;
    pot[n1] = -pot[n1];
    x = std::exp((std::log(-pot[n1]) - std::log(pot[0])) / static_cast<double>(n1));
    if (pot[n1 - 1] != 0.) {
        xm = -pot[n1] / pot[n1 - 1];
        if (xm < x) x = xm;
    }
    for (;;) {
        xm = x * 0.1;
        f = pot[0];
        for (int i = 1; i < n; i++) f = f * xm + pot[i];
        if (f <= 0.0) break;
        x = xm;
    }
    dx = x;
    while (std::fabs(dx / x) > 0.005) {
        q[0] = pot[0];
        for (int i = 1; i < n; i++) q[i] = q[i - 1] * x + pot[i];
        f = q[n1];
        delf = q[0];
        for (int i = 1; i < n1; i++) delf = delf * x + q[i];
        dx = f / delf;
        x -= dx;
    }
    return x;
}

double cpoly_scale(int n, double* pot, double eps, double BIG, double small, double base) {
    double x, sc, min_, max_;
    const double high = std::sqrt(BIG), lo = small / eps;
    max_ = 0.;
    min_ = BIG;
    for (int i = 0; i < n; i++) {
        x = pot[i];
        if (x > max_) max_ = x;
        if (x != 0. && x < min_) min_ = x;
    }
    if (min_ < lo || max_ > high) {
        x = lo / min_;
        if (x <= 1.) {
            sc = 1. / (std::sqrt(max_) * std::sqrt(min_));
        } else {
            sc = x;
            if (BIG / sc > max_) sc = 1.0;
        }
        const int ell = static_cast<int>(std::log(sc) / std::log(base) + 0.5);
        return powdi(base, ell);
    }
    return 1.0;
}

void calct(St& S, bool* h_s_0) {
    const int n = S.nn - 1;
    double hvi, hvr;
    polyev(n, S.sr, S.si, S.hr, S.hi, S.qhr, S.qhi, &hvr, &hvi);
    *h_s_0 = std::hypot(hvr, hvi) <= are * 10. * std::hypot(S.hr[n - 1], S.hi[n - 1]);
    if (!*h_s_0) cdivid(-S.pvr, -S.pvi, hvr, hvi, &S.tr, &S.ti);
    else { S.tr = 0.; S.ti = 0.; }
}

void nexth(St& S, bool h_s_0) {
    const int n = S.nn - 1;
    if (!h_s_0) {
        for (int j = 1; j < n; j++) {
            const double t1 = S.qhr[j - 1], t2 = S.qhi[j - 1];
            S.hr[j] = S.tr * t1 - S.ti * t2 + S.qpr[j];
            S.hi[j] = S.tr * t2 + S.ti * t1 + S.qpi[j];
        }
        S.hr[0] = S.qpr[0];
        S.hi[0] = S.qpi[0];
    } else {
        for (int j = 1; j < n; j++) { S.hr[j] = S.qhr[j - 1]; S.hi[j] = S.qhi[j - 1]; }
        S.hr[0] = 0.;
        S.hi[0] = 0.;
    }
}

void noshft(St& S, int l1) {
    const int nn = S.nn, n = nn - 1, nm1 = n - 1;
    double *pr = S.pr, *pi = S.pi, *hr = S.hr, *hi = S.hi;
    for (int i = 0; i < n; i++) {
        const double xni = static_cast<double>(nn - i - 1);
        hr[i] = xni * pr[i] / n;
        hi[i] = xni * pi[i] / n;
    }
    for (int jj = 1; jj <= l1; jj++) {
        if (std::hypot(hr[n - 1], hi[n - 1]) <= eta * 10.0 * std::hypot(pr[n - 1], pi[n - 1])) {
            for (int i = 1; i <= nm1; i++) {
                const int j = nn - i;
                hr[j - 1] = hr[j - 2];
                hi[j - 1] = hi[j - 2];
            }
            hr[0] = 0.;
            hi[0] = 0.;
        } else {
            cdivid(-pr[nn - 1], -pi[nn - 1], hr[n - 1], hi[n - 1], &S.tr, &S.ti);
            for (int i = 1; i <= nm1; i++) {
                const int j = nn - i;
                const double t1 = hr[j - 2], t2 = hi[j - 2];
                hr[j - 1] = S.tr * t1 - S.ti * t2 + pr[j - 1];
                hi[j - 1] = S.tr * t2 + S.ti * t1 + pi[j - 1];
            }
            hr[0] = pr[0];
            hi[0] = pi[0];
        }
    }
}

bool vrshft(St& S, int l3, double* zr, double* zi) {
    bool h_s_0, b = false;
    S.sr = *zr;
    S.si = *zi;
    for (S.v_i = 1; S.v_i <= l3; S.v_i++) {
        polyev(S.nn, S.sr, S.si, S.pr, S.pi, S.qpr, S.qpi, &S.pvr, &S.pvi);
        S.v_mp = std::hypot(S.pvr, S.pvi);
        S.v_ms = std::hypot(S.sr, S.si);
        if (S.v_mp <= 20. * errev(S.nn, S.qpr, S.qpi, S.v_ms, S.v_mp, eta, mre)) {
            *zr = S.sr;
            *zi = S.si;
            return true;
        }
        bool skip_omp = false;
        if (S.v_i != 1) {
            if (!b && S.v_mp >= S.v_omp && S.v_relstp < .05) {
                S.v_tp = S.v_relstp;
                b = true;
                if (S.v_relstp < eta) S.v_tp = eta;
                S.v_r1 = std::sqrt(S.v_tp);
                S.v_r2 = S.sr * (S.v_r1 + 1.) - S.si * S.v_r1;
                S.si = S.sr * S.v_r1 + S.si * (S.v_r1 + 1.);
                S.sr = S.v_r2;
                polyev(S.nn, S.sr, S.si, S.pr, S.pi, S.qpr, S.qpi, &S.pvr, &S.pvi);
                for (S.v_j = 1; S.v_j <= 5; ++S.v_j) {
                    calct(S, &h_s_0);
                    nexth(S, h_s_0);
                }
                S.v_omp = infin;
                skip_omp = true;                         // goto L10
            } else {
                if (S.v_mp * .1 > S.v_omp) return false;
            }
        }
        if (!skip_omp) S.v_omp = S.v_mp;
        calct(S, &h_s_0);
        nexth(S, h_s_0);
        calct(S, &h_s_0);
        if (!h_s_0) {
            S.v_relstp = std::hypot(S.tr, S.ti) / std::hypot(S.sr, S.si);
            S.sr += S.tr;
            S.si += S.ti;
        }
    }
    return false;
}

bool fxshft(St& S, int l2, double* zr, double* zi) {
    bool pasd, h_s_0, test;
    S.f_n = S.nn - 1;
    polyev(S.nn, S.sr, S.si, S.pr, S.pi, S.qpr, S.qpi, &S.pvr, &S.pvi);
    test = true;
    pasd = false;
    calct(S, &h_s_0);
    for (S.f_j = 1; S.f_j <= l2; S.f_j++) {
        S.f_otr = S.tr;
        S.f_oti = S.ti;
        nexth(S, h_s_0);
        calct(S, &h_s_0);
        *zr = S.sr + S.tr;
        *zi = S.si + S.ti;
        if (!h_s_0 && test && S.f_j != l2) {
            if (std::hypot(S.tr - S.f_otr, S.ti - S.f_oti) >= std::hypot(*zr, *zi) * 0.5) {
                pasd = false;
            } else if (!pasd) {
                pasd = true;
            } else {
                for (S.f_i = 0; S.f_i < S.f_n; S.f_i++) { S.shr[S.f_i] = S.hr[S.f_i]; S.shi[S.f_i] = S.hi[S.f_i]; }
                S.f_svsr = S.sr;
                S.f_svsi = S.si;
                if (vrshft(S, 10, zr, zi)) return true;
                test = false;
                for (S.f_i = 1; S.f_i <= S.f_n; S.f_i++) { S.hr[S.f_i - 1] = S.shr[S.f_i - 1]; S.hi[S.f_i - 1] = S.shi[S.f_i - 1]; }
                S.sr = S.f_svsr;
                S.si = S.f_svsi;
                polyev(S.nn, S.sr, S.si, S.pr, S.pi, S.qpr, S.qpi, &S.pvr, &S.pvi);
                calct(S, &h_s_0);
            }
        }
    }
    return vrshft(S, 10, zr, zi);
}

}  // namespace

void r_cpolyroot(const double* opr, const double* opi, int degree, double* zeror, double* zeroi, bool* fail) {
    St& S = g_state;
    const double smalno = DBL_MIN, base = static_cast<double>(FLT_RADIX);
    const double cosr = -0.06975647374412529990, sinr = 0.99756405025982424767;
    S.c_xx = 0.70710678118654752440;                     // M_SQRT1_2
    S.c_yy = -S.c_xx;
    *fail = false;
    S.nn = degree;
    const int d1 = S.nn - 1;
    if (opr[0] == 0. && opi[0] == 0.) { *fail = true; return; }
    while (opr[S.nn] == 0. && opi[S.nn] == 0.) {
        S.c_d_n = d1 - S.nn + 1;
        zeror[S.c_d_n] = 0.;
        zeroi[S.c_d_n] = 0.;
        S.nn--;
    }
    S.nn++;
    const int nn0 = S.nn;
    if (nn0 == 1) return;
    double stackbuf[10 * 32];                            // R_alloc(10 * nn): on the stack for small degrees
    std::vector<double> heapbuf;
    double* buf = stackbuf;
    if (nn0 > 32) { heapbuf.resize(static_cast<size_t>(10 * nn0)); buf = heapbuf.data(); }
    S.pr = buf; S.pi = buf + nn0; S.hr = buf + 2 * nn0; S.hi = buf + 3 * nn0; S.qpr = buf + 4 * nn0;
    S.qpi = buf + 5 * nn0; S.qhr = buf + 6 * nn0; S.qhi = buf + 7 * nn0; S.shr = buf + 8 * nn0; S.shi = buf + 9 * nn0;
    for (S.c_i = 0; S.c_i < S.nn; S.c_i++) {
        S.pr[S.c_i] = opr[S.c_i];
        S.pi[S.c_i] = opi[S.c_i];
        S.shr[S.c_i] = std::hypot(S.pr[S.c_i], S.pi[S.c_i]);
    }
    S.c_bnd = cpoly_scale(S.nn, S.shr, eta, infin, smalno, base);
    if (S.c_bnd != 1.) for (S.c_i = 0; S.c_i < S.nn; S.c_i++) { S.pr[S.c_i] *= S.c_bnd; S.pi[S.c_i] *= S.c_bnd; }
    while (S.nn > 2) {
        for (S.c_i = 0; S.c_i < S.nn; S.c_i++) S.shr[S.c_i] = std::hypot(S.pr[S.c_i], S.pi[S.c_i]);
        S.c_bnd = cpoly_cauchy(S.nn, S.shr, S.shi);
        bool conv = false;
        for (S.c_i1 = 1; S.c_i1 <= 2 && !conv; S.c_i1++) {
            noshft(S, 5);
            for (S.c_i2 = 1; S.c_i2 <= 9; S.c_i2++) {
                S.c_xxx = cosr * S.c_xx - sinr * S.c_yy;
                S.c_yy = sinr * S.c_xx + cosr * S.c_yy;
                S.c_xx = S.c_xxx;
                S.sr = S.c_bnd * S.c_xx;
                S.si = S.c_bnd * S.c_yy;
                conv = fxshft(S, S.c_i2 * 10, &S.c_zr, &S.c_zi);
                if (conv) break;
            }
        }
        if (!conv) { *fail = true; return; }
        S.c_d_n = d1 + 2 - S.nn;
        zeror[S.c_d_n] = S.c_zr;
        zeroi[S.c_d_n] = S.c_zi;
        --S.nn;
        for (S.c_i = 0; S.c_i < S.nn; S.c_i++) { S.pr[S.c_i] = S.qpr[S.c_i]; S.pi[S.c_i] = S.qpi[S.c_i]; }
    }
    cdivid(-S.pr[1], -S.pi[1], S.pr[0], S.pi[0], &zeror[d1], &zeroi[d1]);
}

}  // namespace tsf
