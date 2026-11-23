// Port of the LAPACK 3.12.1 / reference BLAS routines on R's svd() path for a tall matrix (see lapack_svd.hpp):
// dgesdd (JOBZ = 'S', paths 3 and 5s) -> dgeqr2, dgebd2 (dlarfg, dlarf1f, dlapy2, dnrm2), dbdsdc (N <= SMLSIZ = 25:
// dlasdq -> dbdsqr with dlartg, dlas2, dlasv2, dlasr, drot), dormbr('P', 'R', 'T') -> dorml2 and the BLAS dgemv, dger,
// daxpy, dscal, dswap. Only the right singular vectors are formed (U never feeds back into D, E or VT). Arrays are
// column-major as in Fortran; comments give the Fortran names. Reference LAPACK is BSD-licensed (Univ. of Tennessee,
// Univ. of California Berkeley, Univ. of Colorado Denver, NAG Ltd.).
#include "lapack_svd.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tsf {

namespace {

bool g_new_dnrm2 = true;

const double EPS_E = std::numeric_limits<double>::epsilon() * 0.5;   // DLAMCH('E')
const double SAFMIN = std::numeric_limits<double>::min();            // DLAMCH('S'), la_constants dsafmin
const double SAFMAX = 1.0 / SAFMIN;
const double HUGEV = std::numeric_limits<double>::max();             // DLAMCH('O')

inline double fsign(double a, double b) { return std::signbit(b) ? -std::fabs(a) : std::fabs(a); }   // Fortran SIGN

// ------------------------------------------------------------------------------------------------ BLAS
double dnrm2(int n, const double* x, int incx) {
    if (n <= 0) return 0.0;
    if (!g_new_dnrm2) {                                 // classic reference dnrm2 (scaled sum of squares)
        if (incx < 1) return 0.0;
        if (n == 1) return std::fabs(x[0]);
        double scale = 0.0, ssq = 1.0;
        for (int i = 0; i < n; ++i) {
            const double xi = x[i * incx];
            if (xi != 0.0) {
                const double a = std::fabs(xi);
                if (scale < a) { const double r = scale / a; ssq = 1.0 + ssq * (r * r); scale = a; }
                else { const double r = a / scale; ssq = ssq + r * r; }
            }
        }
        return scale * std::sqrt(ssq);
    }
    const double tsml = std::ldexp(1.0, -511), tbig = std::ldexp(1.0, 486);
    const double ssml = std::ldexp(1.0, 537), sbig = std::ldexp(1.0, -538);
    bool notbig = true;
    double asml = 0.0, amed = 0.0, abig = 0.0;
    int ix = incx < 0 ? -(n - 1) * incx : 0;
    for (int i = 0; i < n; ++i) {
        const double ax = std::fabs(x[ix]);
        if (ax > tbig) { const double t = ax * sbig; abig = abig + t * t; notbig = false; }
        else if (ax < tsml) { if (notbig) { const double t = ax * ssml; asml = asml + t * t; } }
        else amed = amed + ax * ax;
        ix += incx;
    }
    double scl, sumsq;
    if (abig > 0.0) {
        if (amed > 0.0 || amed > HUGEV || amed != amed) abig = abig + (amed * sbig) * sbig;
        scl = 1.0 / sbig;
        sumsq = abig;
    } else if (asml > 0.0) {
        if (amed > 0.0 || amed > HUGEV || amed != amed) {
            amed = std::sqrt(amed);
            asml = std::sqrt(asml) / ssml;
            const double ymin = asml > amed ? amed : asml, ymax = asml > amed ? asml : amed;
            const double r = ymin / ymax;
            scl = 1.0;
            sumsq = (ymax * ymax) * (1.0 + r * r);
        } else {
            scl = 1.0 / ssml;
            sumsq = asml;
        }
    } else {
        scl = 1.0;
        sumsq = amed;
    }
    return scl * std::sqrt(sumsq);
}

void dscal(int n, double da, double* x, int incx) {
    if (n <= 0 || incx <= 0 || da == 1.0) return;
    for (int i = 0; i < n; ++i) x[i * incx] = da * x[i * incx];
}

void daxpy(int n, double da, const double* x, int incx, double* y, int incy) {
    if (n <= 0 || da == 0.0) return;
    int ix = incx < 0 ? (-n + 1) * incx : 0, iy = incy < 0 ? (-n + 1) * incy : 0;
    for (int i = 0; i < n; ++i, ix += incx, iy += incy) y[iy] = y[iy] + da * x[ix];
}

void dswap(int n, double* x, int incx, double* y, int incy) {
    for (int i = 0; i < n; ++i) std::swap(x[i * incx], y[i * incy]);   // incx, incy > 0 here
}

void drot(int n, double* x, int incx, double* y, int incy, double c, double s) {
    for (int i = 0; i < n; ++i) {
        double& xi = x[i * incx];
        double& yi = y[i * incy];
        const double t = c * xi + s * yi;
        yi = c * yi - s * xi;
        xi = t;
    }
}

// y := A^T x (alpha = 1, beta = 0), A m x n
void dgemv_t(int m, int n, const double* A, int lda, const double* x, int incx, double* y) {
    if (m == 0 || n == 0) return;
    for (int j = 0; j < n; ++j) y[j] = 0.0;
    for (int j = 0; j < n; ++j) {
        double temp = 0.0;
        for (int i = 0; i < m; ++i) temp = temp + A[i + j * lda] * x[i * incx];
        y[j] = y[j] + 1.0 * temp;
    }
}

// y := A x (alpha = 1, beta = 0), A m x n
void dgemv_n(int m, int n, const double* A, int lda, const double* x, int incx, double* y) {
    if (m == 0 || n == 0) return;
    for (int i = 0; i < m; ++i) y[i] = 0.0;
    for (int j = 0; j < n; ++j) {
        const double temp = 1.0 * x[j * incx];
        for (int i = 0; i < m; ++i) y[i] = y[i] + temp * A[i + j * lda];
    }
}

// A := A + alpha x y^T, A m x n
void dger(int m, int n, double alpha, const double* x, int incx, const double* y, int incy, double* A, int lda) {
    if (m == 0 || n == 0 || alpha == 0.0) return;
    for (int j = 0; j < n; ++j) {
        const double yj = y[j * incy];
        if (yj != 0.0) {
            const double temp = alpha * yj;
            for (int i = 0; i < m; ++i) A[i + j * lda] = A[i + j * lda] + x[i * incx] * temp;
        }
    }
}

// ------------------------------------------------------------------------------------------------ LAPACK auxiliaries
double dlapy2(double x, double y) {
    if (std::isnan(x)) return x;
    if (std::isnan(y)) return y;
    const double xa = std::fabs(x), ya = std::fabs(y);
    const double w = std::max(xa, ya), z = std::min(xa, ya);
    if (z == 0.0 || w > HUGEV) return w;
    const double q = z / w;
    return w * std::sqrt(1.0 + q * q);
}

void dlarfg(int n, double& alpha, double* x, int incx, double& tau) {
    if (n <= 1) { tau = 0.0; return; }
    double xnorm = dnrm2(n - 1, x, incx);
    if (xnorm == 0.0) { tau = 0.0; return; }
    double beta = -fsign(dlapy2(alpha, xnorm), alpha);
    const double safmin = SAFMIN / EPS_E;
    int knt = 0;
    if (std::fabs(beta) < safmin) {
        const double rsafmn = 1.0 / safmin;
        do {
            ++knt;
            dscal(n - 1, rsafmn, x, incx);
            beta = beta * rsafmn;
            alpha = alpha * rsafmn;
        } while (std::fabs(beta) < safmin && knt < 20);
        xnorm = dnrm2(n - 1, x, incx);
        beta = -fsign(dlapy2(alpha, xnorm), alpha);
    }
    tau = (beta - alpha) / beta;
    dscal(n - 1, 1.0 / (alpha - beta), x, incx);
    for (int j = 0; j < knt; ++j) beta = beta * safmin;
    alpha = beta;
}

int iladlc(int m, int n, const double* A, int lda) {
    if (n == 0) return 0;
    if (A[(n - 1) * lda] != 0.0 || A[(m - 1) + (n - 1) * lda] != 0.0) return n;
    for (int c = n; c >= 1; --c)
        for (int i = 0; i < m; ++i)
            if (A[i + (c - 1) * lda] != 0.0) return c;
    return 0;
}

int iladlr(int m, int n, const double* A, int lda) {
    if (m == 0) return 0;
    if (A[m - 1] != 0.0 || A[(m - 1) + (n - 1) * lda] != 0.0) return m;
    int r = 0;
    for (int j = 0; j < n; ++j) {
        int i = m;
        while (i >= 1 && A[(std::max(i, 1) - 1) + j * lda] == 0.0) --i;
        r = std::max(r, i);
    }
    return r;
}

// H = I - tau v v^T with v(1) = 1 not stored (v points at v(1)); side 'L': C := H C (m x n), 'R': C := C H.
void dlarf1f(bool left, int m, int n, const double* v, int incv, double tau, double* C, int ldc, double* work) {
    int lastv = 1, lastc = 0;
    if (tau != 0.0) {
        lastv = left ? m : n;
        int i = incv > 0 ? (lastv - 1) * incv : 0;
        while (lastv > 1 && v[i] == 0.0) { --lastv; i -= incv; }
        lastc = left ? iladlc(lastv, n, C, ldc) : iladlr(m, lastv, C, ldc);
    }
    if (lastc == 0) return;
    if (left) {
        if (lastv == 1) { dscal(lastc, 1.0 - tau, C, ldc); return; }
        dgemv_t(lastv - 1, lastc, C + 1, ldc, v + incv, incv, work);
        daxpy(lastc, 1.0, C, ldc, work, 1);
        daxpy(lastc, -tau, work, 1, C, ldc);
        dger(lastv - 1, lastc, -tau, v + incv, incv, work, 1, C + 1, ldc);
    } else {
        if (lastv == 1) { dscal(lastc, 1.0 - tau, C, 1); return; }
        dgemv_n(lastc, lastv - 1, C + ldc, ldc, v + incv, incv, work);
        daxpy(lastc, 1.0, C, 1, work, 1);
        daxpy(lastc, -tau, work, 1, C, 1);
        dger(lastc, lastv - 1, -tau, work, 1, v + incv, incv, C + ldc, ldc);
    }
}

void dgeqr2(int m, int n, double* A, int lda, double* tau, double* work) {
    const int k = std::min(m, n);
    for (int i = 0; i < k; ++i) {
        dlarfg(m - i, A[i + i * lda], &A[std::min(i + 1, m - 1) + i * lda], 1, tau[i]);
        if (i < n - 1) dlarf1f(true, m - i, n - i - 1, &A[i + i * lda], 1, tau[i], &A[i + (i + 1) * lda], lda, work);
    }
}

// m >= n: upper bidiagonal d (n), e (n - 1)
void dgebd2(int m, int n, double* A, int lda, double* d, double* e, double* tauq, double* taup, double* work) {
    for (int i = 0; i < n; ++i) {
        dlarfg(m - i, A[i + i * lda], &A[std::min(i + 1, m - 1) + i * lda], 1, tauq[i]);
        d[i] = A[i + i * lda];
        if (i < n - 1) dlarf1f(true, m - i, n - i - 1, &A[i + i * lda], 1, tauq[i], &A[i + (i + 1) * lda], lda, work);
        if (i < n - 1) {
            dlarfg(n - i - 1, A[i + (i + 1) * lda], &A[i + std::min(i + 2, n - 1) * lda], lda, taup[i]);
            e[i] = A[i + (i + 1) * lda];
            dlarf1f(false, m - i - 1, n - i - 1, &A[i + (i + 1) * lda], lda, taup[i], &A[(i + 1) + (i + 1) * lda], lda, work);
        } else {
            taup[i] = 0.0;
        }
    }
}

void dlartg(double f, double g, double& c, double& s, double& r) {
    const double rtmin = std::sqrt(SAFMIN), rtmax = std::sqrt(SAFMAX / 2);
    const double f1 = std::fabs(f), g1 = std::fabs(g);
    if (g == 0.0) { c = 1.0; s = 0.0; r = f; }
    else if (f == 0.0) { c = 0.0; s = fsign(1.0, g); r = g1; }
    else if (f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax) {
        const double d = std::sqrt(f * f + g * g);
        c = f1 / d;
        r = fsign(d, f);
        s = g / r;
    } else {
        const double u = std::min(SAFMAX, std::max(SAFMIN, std::max(f1, g1)));
        const double fs = f / u, gs = g / u;
        const double d = std::sqrt(fs * fs + gs * gs);
        c = std::fabs(fs) / d;
        r = fsign(d, f);
        s = gs / r;
        r = r * u;
    }
}

void dlas2(double f, double g, double h, double& ssmin, double& ssmax) {
    const double fa = std::fabs(f), ga = std::fabs(g), ha = std::fabs(h);
    const double fhmn = std::min(fa, ha), fhmx = std::max(fa, ha);
    if (fhmn == 0.0) {
        ssmin = 0.0;
        if (fhmx == 0.0) ssmax = ga;
        else {
            const double q = std::min(fhmx, ga) / std::max(fhmx, ga);
            ssmax = std::max(fhmx, ga) * std::sqrt(1.0 + q * q);
        }
    } else if (ga < fhmx) {
        const double as = 1.0 + fhmn / fhmx, at = (fhmx - fhmn) / fhmx;
        const double q = ga / fhmx, au = q * q;
        const double c = 2.0 / (std::sqrt(as * as + au) + std::sqrt(at * at + au));
        ssmin = fhmn * c;
        ssmax = fhmx / c;
    } else {
        const double au = fhmx / ga;
        if (au == 0.0) {
            ssmin = (fhmn * fhmx) / ga;
            ssmax = ga;
        } else {
            const double as = 1.0 + fhmn / fhmx, at = (fhmx - fhmn) / fhmx;
            const double p = as * au, q = at * au;
            const double c = 1.0 / (std::sqrt(1.0 + p * p) + std::sqrt(1.0 + q * q));
            ssmin = (fhmn * c) * au;
            ssmin = ssmin + ssmin;
            ssmax = ga / (c + c);
        }
    }
}

void dlasv2(double f, double g, double h, double& ssmin, double& ssmax, double& snr, double& csr, double& snl,
            double& csl) {
    double ft = f, fa = std::fabs(ft), ht = h, ha = std::fabs(h);
    int pmax = 1;
    const bool swap = ha > fa;
    if (swap) { pmax = 3; std::swap(ft, ht); std::swap(fa, ha); }
    const double gt = g, ga = std::fabs(gt);
    double clt, crt, slt, srt;
    if (ga == 0.0) {
        ssmin = ha; ssmax = fa;
        clt = 1.0; crt = 1.0; slt = 0.0; srt = 0.0;
    } else {
        bool gasmal = true;
        if (ga > fa) {
            pmax = 2;
            if ((fa / ga) < EPS_E) {
                gasmal = false;
                ssmax = ga;
                if (ha > 1.0) ssmin = fa / (ga / ha);
                else ssmin = (fa / ga) * ha;
                clt = 1.0;
                slt = ht / gt;
                srt = 1.0;
                crt = ft / gt;
            }
        }
        if (gasmal) {
            const double dd = fa - ha;
            double l = (dd == fa) ? 1.0 : dd / fa;
            const double m = gt / ft;
            double t = 2.0 - l;
            const double mm = m * m, tt = t * t;
            const double s = std::sqrt(tt + mm);
            const double r = (l == 0.0) ? std::fabs(m) : std::sqrt(l * l + mm);
            const double a = 0.5 * (s + r);
            ssmin = ha / a;
            ssmax = fa * a;
            if (mm == 0.0) {
                if (l == 0.0) t = fsign(2.0, ft) * fsign(1.0, gt);
                else t = gt / fsign(dd, ft) + m / t;
            } else {
                t = (m / (s + t) + m / (r + l)) * (1.0 + a);
            }
            l = std::sqrt(t * t + 4.0);
            crt = 2.0 / l;
            srt = t / l;
            clt = (crt + srt * m) / a;
            slt = (ht / ft) * srt / a;
        }
    }
    if (swap) { csl = srt; snl = crt; csr = slt; snr = clt; }
    else { csl = clt; snl = slt; csr = crt; snr = srt; }
    double tsign = 1.0;
    if (pmax == 1) tsign = fsign(1.0, csr) * fsign(1.0, csl) * fsign(1.0, f);
    if (pmax == 2) tsign = fsign(1.0, snr) * fsign(1.0, csl) * fsign(1.0, g);
    if (pmax == 3) tsign = fsign(1.0, snr) * fsign(1.0, snl) * fsign(1.0, h);
    ssmax = fsign(ssmax, tsign);
    ssmin = fsign(ssmin, tsign * fsign(1.0, f) * fsign(1.0, h));
}

// dlasr('L', 'V', direct, m, n, c, s, A, lda): plane rotations on rows j, j + 1 of A (m x n)
void dlasr_lv(bool forward, int m, int n, const double* c, const double* s, double* A, int lda) {
    auto rot = [&](int j) {
        const double ct = c[j], st = s[j];
        if (ct != 1.0 || st != 0.0)
            for (int i = 0; i < n; ++i) {
                const double temp = A[(j + 1) + i * lda];
                A[(j + 1) + i * lda] = ct * temp - st * A[j + i * lda];
                A[j + i * lda] = st * temp + ct * A[j + i * lda];
            }
    };
    if (forward) for (int j = 0; j < m - 1; ++j) rot(j);
    else for (int j = m - 2; j >= 0; --j) rot(j);
}

// dbdsqr('U', n, ncvt, nru = 0, ncc = 0): upper bidiagonal SVD, rotations applied to VT (n x ncvt, ldvt)
int dbdsqr(int n, int ncvt, double* dd, double* ee, double* vt, int ldvt, double* work) {
    if (n == 0) return 0;
    auto D = [&](int i) -> double& { return dd[i - 1]; };
    auto E = [&](int i) -> double& { return ee[i - 1]; };
    auto W = [&](int i) -> double& { return work[i - 1]; };
    auto VT = [&](int i, int j) -> double* { return &vt[(i - 1) + (j - 1) * ldvt]; };
    const int MAXITR = 6;
    const double TEN = 10.0, HNDRD = 100.0, HNDRTH = 0.01, MEIGTH = -0.125;
    int info = 0;
    int nm1 = 0, nm12 = 0, nm13 = 0, idir = 0, iter = 0, iterdivn = 0, maxitdivn = 0, oldll = 0, oldm = 0, m = 0, ll = 0;
    double eps = 0, unfl = 0, tolmul = 0, tol = 0, smax = 0, smin = 0, sminoa = 0, mu = 0, thresh = 0, abss = 0, abse = 0;
    double sigmn = 0, sigmx = 0, sinr = 0, cosr = 0, sinl = 0, cosl = 0, shift = 0, r = 0, sll = 0, cs = 0, sn = 0;
    double oldcs = 0, oldsn = 0, h = 0, f = 0, g = 0;
    if (n == 1) goto L160;
    nm1 = n - 1;
    nm12 = nm1 + nm1;
    nm13 = nm12 + nm1;
    idir = 0;
    eps = EPS_E;
    unfl = SAFMIN;
    {
        volatile double ev = eps;                       // runtime pow, as Fortran's EPS**MEIGTH
        tolmul = std::max(TEN, std::min(HNDRD, std::pow(static_cast<double>(ev), MEIGTH)));
    }
    tol = tolmul * eps;
    smax = 0.0;
    for (int i = 1; i <= n; ++i) smax = std::max(smax, std::fabs(D(i)));
    for (int i = 1; i <= n - 1; ++i) smax = std::max(smax, std::fabs(E(i)));
    smin = 0.0;
    if (tol >= 0.0) {
        sminoa = std::fabs(D(1));
        if (sminoa != 0.0) {
            mu = sminoa;
            for (int i = 2; i <= n; ++i) {
                mu = std::fabs(D(i)) * (mu / (mu + std::fabs(E(i - 1))));
                sminoa = std::min(sminoa, mu);
                if (sminoa == 0.0) break;
            }
        }
        sminoa = sminoa / std::sqrt(static_cast<double>(n));
        thresh = std::max(tol * sminoa, MAXITR * (n * (n * unfl)));
    } else {
        thresh = std::max(std::fabs(tol) * smax, MAXITR * (n * (n * unfl)));
    }
    maxitdivn = MAXITR * n;
    iterdivn = 0;
    iter = -1;
    oldll = -1;
    oldm = -1;
    m = n;
L60:
    if (m <= 1) goto L160;
    if (iter >= n) {
        iter = iter - n;
        iterdivn = iterdivn + 1;
        if (iterdivn >= maxitdivn) goto L200;
    }
    if (tol < 0.0 && std::fabs(D(m)) <= thresh) D(m) = 0.0;
    smax = std::fabs(D(m));
    for (int lll = 1; lll <= m - 1; ++lll) {
        ll = m - lll;
        abss = std::fabs(D(ll));
        abse = std::fabs(E(ll));
        if (tol < 0.0 && abss <= thresh) D(ll) = 0.0;
        if (abse <= thresh) goto L80;
        smax = std::max(smax, std::max(abss, abse));
    }
    ll = 0;
    goto L90;
L80:
    E(ll) = 0.0;
    if (ll == m - 1) { m = m - 1; goto L60; }
L90:
    ll = ll + 1;
    if (ll == m - 1) {
        dlasv2(D(m - 1), E(m - 1), D(m), sigmn, sigmx, sinr, cosr, sinl, cosl);
        D(m - 1) = sigmx;
        E(m - 1) = 0.0;
        D(m) = sigmn;
        if (ncvt > 0) drot(ncvt, VT(m - 1, 1), ldvt, VT(m, 1), ldvt, cosr, sinr);
        m = m - 2;
        goto L60;
    }
    if (ll > oldm || m < oldll) {
        if (std::fabs(D(ll)) >= std::fabs(D(m))) idir = 1;
        else idir = 2;
    }
    if (idir == 1) {
        if (std::fabs(E(m - 1)) <= std::fabs(tol) * std::fabs(D(m)) || (tol < 0.0 && std::fabs(E(m - 1)) <= thresh)) {
            E(m - 1) = 0.0;
            goto L60;
        }
        if (tol >= 0.0) {
            mu = std::fabs(D(ll));
            smin = mu;
            for (int lll = ll; lll <= m - 1; ++lll) {
                if (std::fabs(E(lll)) <= tol * mu) { E(lll) = 0.0; goto L60; }
                mu = std::fabs(D(lll + 1)) * (mu / (mu + std::fabs(E(lll))));
                smin = std::min(smin, mu);
            }
        }
    } else {
        if (std::fabs(E(ll)) <= std::fabs(tol) * std::fabs(D(ll)) || (tol < 0.0 && std::fabs(E(ll)) <= thresh)) {
            E(ll) = 0.0;
            goto L60;
        }
        if (tol >= 0.0) {
            mu = std::fabs(D(m));
            smin = mu;
            for (int lll = m - 1; lll >= ll; --lll) {
                if (std::fabs(E(lll)) <= tol * mu) { E(lll) = 0.0; goto L60; }
                mu = std::fabs(D(lll)) * (mu / (mu + std::fabs(E(lll))));
                smin = std::min(smin, mu);
            }
        }
    }
    oldll = ll;
    oldm = m;
    if (tol >= 0.0 && n * tol * (smin / smax) <= std::max(eps, HNDRTH * tol)) {
        shift = 0.0;
    } else {
        if (idir == 1) {
            sll = std::fabs(D(ll));
            dlas2(D(m - 1), E(m - 1), D(m), shift, r);
        } else {
            sll = std::fabs(D(m));
            dlas2(D(ll), E(ll), D(ll + 1), shift, r);
        }
        if (sll > 0.0) {
            const double q = shift / sll;
            if (q * q < eps) shift = 0.0;
        }
    }
    iter = iter + m - ll;
    if (shift == 0.0) {
        if (idir == 1) {
            cs = 1.0;
            oldcs = 1.0;
            for (int i = ll; i <= m - 1; ++i) {
                dlartg(D(i) * cs, E(i), cs, sn, r);
                if (i > ll) E(i - 1) = oldsn * r;
                dlartg(oldcs * r, D(i + 1) * sn, oldcs, oldsn, D(i));
                W(i - ll + 1) = cs;
                W(i - ll + 1 + nm1) = sn;
                W(i - ll + 1 + nm12) = oldcs;
                W(i - ll + 1 + nm13) = oldsn;
            }
            h = D(m) * cs;
            D(m) = h * oldcs;
            E(m - 1) = h * oldsn;
            if (ncvt > 0) dlasr_lv(true, m - ll + 1, ncvt, &W(1), &W(n), VT(ll, 1), ldvt);
            if (std::fabs(E(m - 1)) <= thresh) E(m - 1) = 0.0;
        } else {
            cs = 1.0;
            oldcs = 1.0;
            for (int i = m; i >= ll + 1; --i) {
                dlartg(D(i) * cs, E(i - 1), cs, sn, r);
                if (i < m) E(i) = oldsn * r;
                dlartg(oldcs * r, D(i - 1) * sn, oldcs, oldsn, D(i));
                W(i - ll) = cs;
                W(i - ll + nm1) = -sn;
                W(i - ll + nm12) = oldcs;
                W(i - ll + nm13) = -oldsn;
            }
            h = D(ll) * cs;
            D(ll) = h * oldcs;
            E(ll) = h * oldsn;
            if (ncvt > 0) dlasr_lv(false, m - ll + 1, ncvt, &W(nm12 + 1), &W(nm13 + 1), VT(ll, 1), ldvt);
            if (std::fabs(E(ll)) <= thresh) E(ll) = 0.0;
        }
    } else {
        if (idir == 1) {
            f = (std::fabs(D(ll)) - shift) * (fsign(1.0, D(ll)) + shift / D(ll));
            g = E(ll);
            for (int i = ll; i <= m - 1; ++i) {
                dlartg(f, g, cosr, sinr, r);
                if (i > ll) E(i - 1) = r;
                f = cosr * D(i) + sinr * E(i);
                E(i) = cosr * E(i) - sinr * D(i);
                g = sinr * D(i + 1);
                D(i + 1) = cosr * D(i + 1);
                dlartg(f, g, cosl, sinl, r);
                D(i) = r;
                f = cosl * E(i) + sinl * D(i + 1);
                D(i + 1) = cosl * D(i + 1) - sinl * E(i);
                if (i < m - 1) {
                    g = sinl * E(i + 1);
                    E(i + 1) = cosl * E(i + 1);
                }
                W(i - ll + 1) = cosr;
                W(i - ll + 1 + nm1) = sinr;
                W(i - ll + 1 + nm12) = cosl;
                W(i - ll + 1 + nm13) = sinl;
            }
            E(m - 1) = f;
            if (ncvt > 0) dlasr_lv(true, m - ll + 1, ncvt, &W(1), &W(n), VT(ll, 1), ldvt);
            if (std::fabs(E(m - 1)) <= thresh) E(m - 1) = 0.0;
        } else {
            f = (std::fabs(D(m)) - shift) * (fsign(1.0, D(m)) + shift / D(m));
            g = E(m - 1);
            for (int i = m; i >= ll + 1; --i) {
                dlartg(f, g, cosr, sinr, r);
                if (i < m) E(i) = r;
                f = cosr * D(i) + sinr * E(i - 1);
                E(i - 1) = cosr * E(i - 1) - sinr * D(i);
                g = sinr * D(i - 1);
                D(i - 1) = cosr * D(i - 1);
                dlartg(f, g, cosl, sinl, r);
                D(i) = r;
                f = cosl * E(i - 1) + sinl * D(i - 1);
                D(i - 1) = cosl * D(i - 1) - sinl * E(i - 1);
                if (i > ll + 1) {
                    g = sinl * E(i - 2);
                    E(i - 2) = cosl * E(i - 2);
                }
                W(i - ll) = cosr;
                W(i - ll + nm1) = -sinr;
                W(i - ll + nm12) = cosl;
                W(i - ll + nm13) = -sinl;
            }
            E(ll) = f;
            if (std::fabs(E(ll)) <= thresh) E(ll) = 0.0;
            if (ncvt > 0) dlasr_lv(false, m - ll + 1, ncvt, &W(nm12 + 1), &W(nm13 + 1), VT(ll, 1), ldvt);
        }
    }
    goto L60;
L160:
    for (int i = 1; i <= n; ++i) {
        if (D(i) == 0.0) D(i) = 0.0;
        if (D(i) < 0.0) {
            D(i) = -D(i);
            if (ncvt > 0) dscal(ncvt, -1.0, VT(i, 1), ldvt);
        }
    }
    for (int i = 1; i <= n - 1; ++i) {
        int isub = 1;
        double sm = D(1);
        for (int j = 2; j <= n + 1 - i; ++j)
            if (D(j) <= sm) { isub = j; sm = D(j); }
        if (isub != n + 1 - i) {
            D(isub) = D(n + 1 - i);
            D(n + 1 - i) = sm;
            if (ncvt > 0) dswap(ncvt, VT(isub, 1), ldvt, VT(n + 1 - i, 1), ldvt);
        }
    }
    return info;
L200:
    info = 0;
    for (int i = 1; i <= n - 1; ++i)
        if (E(i) != 0.0) info = info + 1;
    return info;
}

}  // namespace

void r_svd_set_new_dnrm2(bool on) { g_new_dnrm2 = on; }

bool r_svd_v(const std::vector<Vec>& cols, std::vector<Vec>& V, Vec* dout) {
    const int N = static_cast<int>(cols.size());
    if (N == 0) return false;
    const int M = static_cast<int>(cols[0].size());
    const int SMLSIZ = 25;                              // ILAENV(9, 'DBDSDC')
    if (N > SMLSIZ || M < N) return false;
    std::vector<double> A(static_cast<size_t>(M) * N);
    double anrm = 0.0;                                  // DLANGE('M')
    for (int j = 0; j < N; ++j) {
        if (static_cast<int>(cols[j].size()) != M) return false;
        for (int i = 0; i < M; ++i) {
            const double v = cols[j][i];
            if (!std::isfinite(v)) return false;
            A[i + static_cast<size_t>(j) * M] = v;
            anrm = std::max(anrm, std::fabs(v));
        }
    }
    const double smlnum = std::sqrt(SAFMIN) / (2.0 * EPS_E), bignum = 1.0 / smlnum;
    if ((anrm > 0.0 && anrm < smlnum) || anrm > bignum) return false;   // dgesdd would rescale A first
    const int mnthr = static_cast<int>(N * 11.0 / 6.0);
    std::vector<double> d(N), e(std::max(N - 1, 1)), tauq(N), taup(N), work(std::max(M, N) + 1), tau(N);
    std::vector<double> B;                              // the matrix that is bidiagonalised (R or A)
    int ldb;
    if (M >= mnthr) {                                   // path 3: A = QR, bidiagonalise R
        dgeqr2(M, N, A.data(), M, tau.data(), work.data());
        B.assign(static_cast<size_t>(N) * N, 0.0);
        for (int j = 0; j < N; ++j)
            for (int i = 0; i <= j; ++i) B[i + static_cast<size_t>(j) * N] = A[i + static_cast<size_t>(j) * M];
        ldb = N;
        dgebd2(N, N, B.data(), ldb, d.data(), e.data(), tauq.data(), taup.data(), work.data());
    } else {                                            // path 5s: bidiagonalise A
        B = A;
        ldb = M;
        dgebd2(M, N, B.data(), ldb, d.data(), e.data(), tauq.data(), taup.data(), work.data());
    }
    // dbdsdc('U', 'I'): VT = I, then dlasdq -> dbdsqr (N <= SMLSIZ); N = 1: VT = 1, d = |d|
    std::vector<double> vt(static_cast<size_t>(N) * N, 0.0);
    for (int i = 0; i < N; ++i) vt[i + static_cast<size_t>(i) * N] = 1.0;
    if (N == 1) {
        d[0] = std::fabs(d[0]);
    } else {
        std::vector<double> w(4 * N);
        if (dbdsqr(N, N, d.data(), e.data(), vt.data(), N, w.data()) != 0) return false;
        for (int i = 0; i < N; ++i) {                   // dlasdq: ascending order
            int isub = i;
            double sm = d[i];
            for (int j = i + 1; j < N; ++j)
                if (d[j] < sm) { isub = j; sm = d[j]; }
            if (isub != i) {
                d[isub] = d[i];
                d[i] = sm;
                dswap(N, &vt[isub], N, &vt[i], N);
            }
        }
        for (int ii = 1; ii < N; ++ii) {                // dbdsdc: decreasing order (selection sort)
            const int i = ii - 1;
            int kk = i;
            double p = d[i];
            for (int j = ii; j < N; ++j)
                if (d[j] > p) { kk = j; p = d[j]; }
            if (kk != i) {
                d[kk] = d[i];
                d[i] = p;
                dswap(N, &vt[i], N, &vt[kk], N);
            }
        }
    }
    // dormbr('P', 'R', 'T', N, N, N, B, ldb, taup, VT): dormlq('R', 'N', N, N - 1, N - 1, B(1, 2), ..., VT(1, 2)) ->
    // dorml2: i = N - 1 .. 1, dlarf1f('R', N, N - i, B(i, i + 1), ldb, taup(i), VT(1, i + 1), N)
    for (int i = N - 1; i >= 1; --i)
        dlarf1f(false, N, N - 1 - i + 1, &B[(i - 1) + static_cast<size_t>(i) * ldb], ldb, taup[i - 1],
                &vt[static_cast<size_t>(i) * N], N, work.data());
    V.assign(N, Vec(N));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) V[i][j] = vt[j + static_cast<size_t>(i) * N];   // v = t(vt)
    if (dout) dout->assign(d.begin(), d.end());
    return true;
}

}  // namespace tsf
