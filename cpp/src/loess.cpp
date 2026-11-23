// See loess.hpp. Port of R 4.6's stats/src/loessc.c + loessf.f (lowesd, lowesb -> ehg131 -> ehg126, ehg124, ehg125,
// ehg129, ehg106, ehg139 -> ehg127; lowese -> ehg133 -> ehg128) for d = 1, degree = 1, family = "gaussian", with
// LINPACK dqrdc (job 0), dqrsl (job 1000) and dsvdc (job 21) from src/appl and the reference BLAS of src/extra/blas.
// Fortran 1-based indexing is kept: arrays are allocated one longer and element 0 is unused.
#include "loess.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace tsf {

namespace {

// ---------------------------------------------------------------- reference BLAS (src/extra/blas)
// strided views: p[(i - 1) * inc] for i = 1..n
double ddot(int n, const double* dx, int incx, const double* dy, int incy) {
    double dtemp = 0.0;
    if (n <= 0) return 0.0;
    if (incx == 1 && incy == 1) {
        const int m = n % 5;
        for (int i = 0; i < m; ++i) dtemp = dtemp + dx[i] * dy[i];
        for (int i = m; i < n; i += 5)
            dtemp = dtemp + dx[i] * dy[i] + dx[i + 1] * dy[i + 1] + dx[i + 2] * dy[i + 2] + dx[i + 3] * dy[i + 3] +
                    dx[i + 4] * dy[i + 4];
        return dtemp;
    }
    for (int i = 0; i < n; ++i) dtemp = dtemp + dx[i * incx] * dy[i * incy];
    return dtemp;
}

void daxpy(int n, double da, const double* dx, double* dy) {
    if (n <= 0 || da == 0.0) return;
    for (int i = 0; i < n; ++i) dy[i] = dy[i] + da * dx[i];
}

void dscal(int n, double da, double* dx) { for (int i = 0; i < n; ++i) dx[i] = da * dx[i]; }

void drot(int n, double* dx, double* dy, double c, double s) {
    for (int i = 0; i < n; ++i) {
        const double dtemp = c * dx[i] + s * dy[i];
        dy[i] = c * dy[i] - s * dx[i];
        dx[i] = dtemp;
    }
}

void dswap(int n, double* dx, double* dy) { for (int i = 0; i < n; ++i) std::swap(dx[i], dy[i]); }

double dnrm2(int n, const double* x) {                   // LAPACK 3.10+ (blas2.f90), incx = 1
    const double tsml = std::ldexp(1.0, -511), tbig = std::ldexp(1.0, 486);
    const double ssml = std::ldexp(1.0, 537), sbig = std::ldexp(1.0, -538);
    const double maxN = 1.7976931348623157e308;
    if (n <= 0) return 0.0;
    double scl = 1.0, sumsq = 0.0, asml = 0.0, amed = 0.0, abig = 0.0;
    bool notbig = true;
    for (int i = 0; i < n; ++i) {
        const double ax = std::fabs(x[i]);
        if (ax > tbig) { const double t = ax * sbig; abig = abig + t * t; notbig = false; }
        else if (ax < tsml) { if (notbig) { const double t = ax * ssml; asml = asml + t * t; } }
        else amed = amed + ax * ax;
    }
    if (abig > 0.0) {
        if (amed > 0.0 || amed > maxN || amed != amed) abig = abig + (amed * sbig) * sbig;
        scl = 1.0 / sbig;
        sumsq = abig;
    } else if (asml > 0.0) {
        if (amed > 0.0 || amed > maxN || amed != amed) {
            amed = std::sqrt(amed);
            asml = std::sqrt(asml) / ssml;
            double ymin, ymax;
            if (asml > amed) { ymin = amed; ymax = asml; } else { ymin = asml; ymax = amed; }
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

void drotg(double& a, double& b, double& c, double& s) {  // LAPACK 3.10+ (blas2.f90)
    const double safmin = std::ldexp(1.0, -1022), safmax = std::ldexp(1.0, 1023);
    const double anorm = std::fabs(a), bnorm = std::fabs(b);
    if (bnorm == 0.0) { c = 1.0; s = 0.0; b = 0.0; }
    else if (anorm == 0.0) { c = 0.0; s = 1.0; a = b; b = 1.0; }
    else {
        const double scl = std::min(safmax, std::max(safmin, std::max(anorm, bnorm)));
        const double sigma = (anorm > bnorm) ? std::copysign(1.0, a) : std::copysign(1.0, b);
        const double as = a / scl, bs = b / scl;
        const double r = sigma * (scl * std::sqrt(as * as + bs * bs));
        c = a / r;
        s = b / r;
        double z;
        if (anorm > bnorm) z = s;
        else if (c != 0.0) z = 1.0 / c;
        else z = 1.0;
        a = r;
        b = z;
    }
}

double dsign(double a, double b) { return (b >= 0.0) ? std::fabs(a) : -std::fabs(a); }

// ---------------------------------------------------------------- LINPACK (src/appl), column-major, 1-based
struct Mat {                                            // x(i, j), 1 <= i <= ld, 1 <= j
    int ld;
    std::vector<double> a;
    Mat(int ld_, int ncol) : ld(ld_), a(static_cast<size_t>(ld_) * (ncol + 1) + 1, 0.0) {}
    double& operator()(int i, int j) { return a[static_cast<size_t>(j - 1) * ld + i]; }
    double* p(int i, int j) { return &a[static_cast<size_t>(j - 1) * ld + i]; }
};

// dqrdc(x, ldx, n, p, qraux, jpvt, work, job = 0): Householder QR without pivoting
void dqrdc0(Mat& x, int n, int p, double* qraux) {
    const int lup = std::min(n, p);
    for (int l = 1; l <= lup; ++l) {
        qraux[l] = 0.0;
        if (l == n) continue;
        double nrmxl = dnrm2(n - l + 1, x.p(l, l));
        if (nrmxl == 0.0) continue;
        if (x(l, l) != 0.0) nrmxl = dsign(nrmxl, x(l, l));
        dscal(n - l + 1, 1.0 / nrmxl, x.p(l, l));
        x(l, l) = 1.0 + x(l, l);
        for (int j = l + 1; j <= p; ++j) {
            const double t = -ddot(n - l + 1, x.p(l, l), 1, x.p(l, j), 1) / x(l, l);
            daxpy(n - l + 1, t, x.p(l, l), x.p(l, j));
        }
        qraux[l] = x(l, l);
        x(l, l) = -nrmxl;
    }
}

// dqrsl(..., job = 1000): qty = Q^T y in place (y is overwritten, as ehg127 passes eta for both)
void dqrsl_qty(Mat& x, int n, int k, const double* qraux, double* y /* 1-based */) {
    const int ju = std::min(k, n - 1);
    if (ju == 0) return;                                // qty(1) = y(1)
    for (int j = 1; j <= ju; ++j) {
        if (qraux[j] == 0.0) continue;
        const double temp = x(j, j);
        x(j, j) = qraux[j];
        const double t = -ddot(n - j + 1, x.p(j, j), 1, y + j, 1) / x(j, j);
        daxpy(n - j + 1, t, x.p(j, j), y + j);
        x(j, j) = temp;
    }
}

// dsvdc(x, ldx, n, p, s, e, u, ldu, v, ldv, work, job = 21, info) with u aliasing x (as ehg127 calls it)
int dsvdc21(Mat& x, int n, int p, double* s, double* e, Mat& v, double* work) {
    Mat& u = x;
    int info = 0;
    const int maxit = 30;
    const bool wantu = true, wantv = true;
    const int ncu = std::min(n, p);
    const int nct = std::min(n - 1, p), nrt = std::max(0, std::min(p - 2, n));
    const int lu = std::max(nct, nrt);
    int l = 0, ls = 0;
    for (l = 1; l <= lu; ++l) {
        const int lp1 = l + 1;
        if (l <= nct) {
            s[l] = dnrm2(n - l + 1, x.p(l, l));
            if (s[l] != 0.0) {
                if (x(l, l) != 0.0) s[l] = dsign(s[l], x(l, l));
                dscal(n - l + 1, 1.0 / s[l], x.p(l, l));
                x(l, l) = 1.0 + x(l, l);
            }
            s[l] = -s[l];
        }
        for (int j = lp1; j <= p; ++j) {
            if (l <= nct && s[l] != 0.0) {
                const double t = -ddot(n - l + 1, x.p(l, l), 1, x.p(l, j), 1) / x(l, l);
                daxpy(n - l + 1, t, x.p(l, l), x.p(l, j));
            }
            e[j] = x(l, j);
        }
        if (wantu && l <= nct)
            for (int i = l; i <= n; ++i) u(i, l) = x(i, l);
        if (l <= nrt) {
            e[l] = dnrm2(p - l, e + lp1);
            if (e[l] != 0.0) {
                if (e[lp1] != 0.0) e[l] = dsign(e[l], e[lp1]);
                dscal(p - l, 1.0 / e[l], e + lp1);
                e[lp1] = 1.0 + e[lp1];
            }
            e[l] = -e[l];
            if (!(lp1 > n || e[l] == 0.0)) {
                for (int i = lp1; i <= n; ++i) work[i] = 0.0;
                for (int j = lp1; j <= p; ++j) daxpy(n - l, e[j], x.p(lp1, j), work + lp1);
                for (int j = lp1; j <= p; ++j) daxpy(n - l, -e[j] / e[lp1], work + lp1, x.p(lp1, j));
            }
            if (wantv)
                for (int i = lp1; i <= p; ++i) v(i, l) = e[i];
        }
    }
    int m = std::min(p, n + 1);
    const int nctp1 = nct + 1, nrtp1 = nrt + 1;
    if (nct < p) s[nctp1] = x(nctp1, nctp1);
    if (n < m) s[m] = 0.0;
    if (nrtp1 < m) e[nrtp1] = x(nrtp1, m);
    e[m] = 0.0;
    if (wantu) {
        for (int j = nctp1; j <= ncu; ++j) {
            for (int i = 1; i <= n; ++i) u(i, j) = 0.0;
            u(j, j) = 1.0;
        }
        for (int ll = 1; ll <= nct; ++ll) {
            l = nct - ll + 1;
            if (s[l] != 0.0) {
                const int lp1 = l + 1;
                for (int j = lp1; j <= ncu; ++j) {
                    const double t = -ddot(n - l + 1, u.p(l, l), 1, u.p(l, j), 1) / u(l, l);
                    daxpy(n - l + 1, t, u.p(l, l), u.p(l, j));
                }
                dscal(n - l + 1, -1.0, u.p(l, l));
                u(l, l) = 1.0 + u(l, l);
                for (int i = 1; i <= l - 1; ++i) u(i, l) = 0.0;
            } else {
                for (int i = 1; i <= n; ++i) u(i, l) = 0.0;
                u(l, l) = 1.0;
            }
        }
    }
    if (wantv) {
        for (int ll = 1; ll <= p; ++ll) {
            l = p - ll + 1;
            const int lp1 = l + 1;
            if (l <= nrt && e[l] != 0.0)
                for (int j = lp1; j <= p; ++j) {
                    const double t = -ddot(p - l, v.p(lp1, l), 1, v.p(lp1, j), 1) / v(lp1, l);
                    daxpy(p - l, t, v.p(lp1, l), v.p(lp1, j));
                }
            for (int i = 1; i <= p; ++i) v(i, l) = 0.0;
            v(l, l) = 1.0;
        }
    }
    const int mm = m;
    int iter = 0;
    for (;;) {
        if (m == 0) break;
        if (iter >= maxit) { info = m; break; }
        int ll;
        for (ll = 1; ll <= m; ++ll) {
            l = m - ll;
            if (l == 0) break;
            const double test = std::fabs(s[l]) + std::fabs(s[l + 1]);
            const double ztest = test + std::fabs(e[l]);
            const double acc = std::fabs(test - ztest) / (1.0e-100 + test);
            if (!(acc > 1.e-15)) { e[l] = 0.0; break; }
        }
        int kase;
        if (l == m - 1) {
            kase = 4;
        } else {
            const int lp1 = l + 1, mp1 = m + 1;
            int lls;
            for (lls = lp1; lls <= mp1; ++lls) {
                ls = m - lls + lp1;
                if (ls == l) break;
                double test = 0.0;
                if (ls != m) test = test + std::fabs(e[ls]);
                if (ls != l + 1) test = test + std::fabs(e[ls - 1]);
                const double ztest = test + std::fabs(s[ls]);
                const double acc = std::fabs(test - ztest) / (1.0e-100 + test);
                if (!(acc > 1.e-15)) { s[ls] = 0.0; break; }
            }
            if (ls == l) kase = 3;
            else if (ls == m) kase = 1;
            else { kase = 2; l = ls; }
        }
        l = l + 1;
        if (kase == 1) {
            const int mm1 = m - 1;
            double f = e[m - 1];
            e[m - 1] = 0.0;
            for (int kk = l; kk <= mm1; ++kk) {
                const int k = mm1 - kk + l;
                double t1 = s[k], cs, sn;
                drotg(t1, f, cs, sn);
                s[k] = t1;
                if (k != l) { f = -sn * e[k - 1]; e[k - 1] = cs * e[k - 1]; }
                if (wantv) drot(p, v.p(1, k), v.p(1, m), cs, sn);
            }
        } else if (kase == 2) {
            double f = e[l - 1];
            e[l - 1] = 0.0;
            for (int k = l; k <= m; ++k) {
                double t1 = s[k], cs, sn;
                drotg(t1, f, cs, sn);
                s[k] = t1;
                f = -sn * e[k];
                e[k] = cs * e[k];
                if (wantu) drot(n, u.p(1, k), u.p(1, l - 1), cs, sn);
            }
        } else if (kase == 3) {
            const double scale = std::max({std::fabs(s[m]), std::fabs(s[m - 1]), std::fabs(e[m - 1]), std::fabs(s[l]),
                                           std::fabs(e[l])});
            const double sm = s[m] / scale, smm1 = s[m - 1] / scale, emm1 = e[m - 1] / scale;
            const double sl = s[l] / scale, el = e[l] / scale;
            const double b = ((smm1 + sm) * (smm1 - sm) + emm1 * emm1) / 2.0;
            const double smemm1 = sm * emm1;
            const double c = smemm1 * smemm1;
            double shift = 0.0;
            if (!(b == 0.0 && c == 0.0)) {
                shift = std::sqrt(b * b + c);
                if (b < 0.0) shift = -shift;
                shift = c / (b + shift);
            }
            double f = (sl + sm) * (sl - sm) + shift;
            double g = sl * el;
            const int mm1 = m - 1;
            for (int k = l; k <= mm1; ++k) {
                double cs, sn;
                drotg(f, g, cs, sn);
                if (k != l) e[k - 1] = f;
                f = cs * s[k] + sn * e[k];
                e[k] = cs * e[k] - sn * s[k];
                g = sn * s[k + 1];
                s[k + 1] = cs * s[k + 1];
                if (wantv) drot(p, v.p(1, k), v.p(1, k + 1), cs, sn);
                drotg(f, g, cs, sn);
                s[k] = f;
                f = cs * e[k] + sn * s[k + 1];
                s[k + 1] = -sn * e[k] + cs * s[k + 1];
                g = sn * e[k + 1];
                e[k + 1] = cs * e[k + 1];
                if (wantu && k < n) drot(n, u.p(1, k), u.p(1, k + 1), cs, sn);
            }
            e[m - 1] = f;
            iter = iter + 1;
        } else {                                        // kase 4: convergence
            if (s[l] < 0.0) {
                s[l] = -s[l];
                if (wantv) dscal(p, -1.0, v.p(1, l));
            }
            while (l != mm && !(s[l] >= s[l + 1])) {
                std::swap(s[l], s[l + 1]);
                if (wantv && l < p) dswap(p, v.p(1, l), v.p(1, l + 1));
                if (wantu && l < n) dswap(n, u.p(1, l), u.p(1, l + 1));
                l = l + 1;
            }
            iter = 0;
            m = m - 1;
        }
    }
    return info;
}

// ---------------------------------------------------------------- loessf.f
// ehg106: partial sort of p(pi(il..ir)) so that p(pi(k)) is the k-th smallest (Floyd & Rivest)
void ehg106(int il, int ir, int k, const double* p /* 1-based */, int* pi /* 1-based */) {
    int l = il, r = ir;
    while (l < r) {
        const double t = p[pi[k]];
        int i = l, j = r;
        std::swap(pi[l], pi[k]);
        if (t < p[pi[r]]) std::swap(pi[l], pi[r]);
        while (i < j) {
            std::swap(pi[i], pi[j]);
            ++i;
            --j;
            while (p[pi[i]] < t) ++i;
            while (t < p[pi[j]]) --j;
        }
        if (p[pi[l]] == t) std::swap(pi[l], pi[j]);
        else { ++j; std::swap(pi[r], pi[j]); }
        if (j <= k) l = j + 1;
        if (k <= j) r = j - 1;
    }
}

struct KdTree {                                         // d = 1, vc = 2
    int nv = 0, nc = 0;
    std::vector<double> v;                              // v(i): vertex coordinate
    std::vector<int> a, c1, c2, lo, hi;                 // cell split dimension (0 = leaf), vertex ids, sons / ranges
    std::vector<double> xi;
    std::vector<double> vval0, vval1;                   // value and slope at each vertex
};

}  // namespace

Vec loess_fit_1d(const Vec& xin, const Vec& y, double span, double cell_in) {
    const int n = static_cast<int>(xin.size());
    if (n < 2 || static_cast<int>(y.size()) != n) throw std::runtime_error("loess: need at least two points");
    // loess_workspace / lowesd
    const int nvmax = std::max(200, n), ncmax = nvmax;
    const int nf = std::min(n, static_cast<int>(std::floor(n * span + 1e-5)));
    if (nf <= 0) throw std::runtime_error("span is too small");
    const int k = 2;                                    // degree 1, d = 1: k = d + 1
    if (k > nf - 1) throw std::runtime_error("loess: span too small for degree 1");   // loesswarn(104)
    const double cell = span * cell_in;                 // simpleLoess passes span * cell
    const int fc = static_cast<int>(std::floor(n * cell));   // ifloor(iv(3) * wv(2))
    const double fd = 0.0;
    std::vector<double> x(n + 1), yy(n + 1);
    for (int i = 1; i <= n; ++i) { x[i] = xin[i - 1]; yy[i] = y[i - 1]; }

    KdTree T;
    T.v.assign(nvmax + 2, 0.0);
    T.a.assign(ncmax + 2, 0); T.c1.assign(ncmax + 2, 0); T.c2.assign(ncmax + 2, 0);
    T.lo.assign(ncmax + 2, 0); T.hi.assign(ncmax + 2, 0); T.xi.assign(ncmax + 2, 0.0);
    std::vector<int> vhit(nvmax + 2, 0);
    // ehg126: bounding box, expanded a little
    {
        double alpha = 1.7976931348623157e308, beta = -1.7976931348623157e308;
        for (int i = 1; i <= n; ++i) { alpha = std::min(alpha, x[i]); beta = std::max(beta, x[i]); }
        const double mu = 0.005 * std::max(beta - alpha, 1.e-10 * std::max(std::fabs(alpha), std::fabs(beta)) + 1.e-30);
        T.v[1] = alpha - mu;
        T.v[2] = beta + mu;
    }
    T.nv = 2;
    T.nc = 1;
    T.c1[1] = 1; T.c2[1] = 2;
    vhit[1] = vhit[2] = 0;
    std::vector<int> pi(n + 1);
    for (int i = 1; i <= n; ++i) pi[i] = i;
    // ehg124: build the tree
    {
        int p = 1, l = 1, u = n;
        T.lo[p] = l; T.hi[p] = u;
        while (p <= T.nc) {
            const double diag = T.v[T.c2[p]] - T.v[T.c1[p]];
            double diam = 0;
            diam = diam + diag * diag;
            diam = std::sqrt(diam);
            bool leaf;
            if ((u - l) + 1 <= fc || diam <= fd) leaf = true;
            else leaf = (ncmax < T.nc + 2) || (nvmax < T.nv + 2.0 / 2.0);
            int m = 0;
            if (!leaf) {
                m = static_cast<int>(static_cast<double>(l + u) / 2.0);
                ehg106(l, u, m, x.data(), pi.data());
                int offset = 0;
                while (!((m + offset) >= u || (m + offset) < l)) {
                    int lower, check, upper;
                    if (offset < 0) { lower = l; check = m + offset; upper = check; }
                    else { lower = m + offset + 1; check = lower; upper = u; }
                    ehg106(lower, upper, check, x.data(), pi.data());
                    if (x[pi[m + offset]] == x[pi[m + offset + 1]]) {
                        offset = -offset;
                        if (offset >= 0) offset = offset + 1;
                    } else {
                        m = m + offset;
                        break;
                    }
                }
                if (T.v[T.c1[p]] == x[pi[m]]) leaf = true;
                else leaf = (T.v[T.c2[p]] == x[pi[m]]);
            }
            if (leaf) {
                T.a[p] = 0;
            } else {
                T.a[p] = 1;
                T.xi[p] = x[pi[m]];
                T.nc = T.nc + 1;
                T.lo[p] = T.nc; T.lo[T.nc] = l; T.hi[T.nc] = m;
                T.nc = T.nc + 1;
                T.hi[p] = T.nc; T.lo[T.nc] = m + 1; T.hi[T.nc] = u;
                // ehg125 (r = s = 1, k = d = 1): new vertex at xi(p) unless it already exists
                int h = T.nv + 1;
                T.v[h] = T.v[T.c1[p]];
                T.v[h] = T.xi[p];
                bool match = false;
                int mv = 1;
                while (!match && mv <= T.nv) { match = (T.v[mv] == T.v[h]); ++mv; }
                mv = mv - 1;
                if (match) h = h - 1;
                else { mv = h; if (vhit[1] >= 0) vhit[mv] = p; }
                T.c1[T.lo[p]] = T.c1[p]; T.c2[T.lo[p]] = mv;
                T.c1[T.hi[p]] = mv;     T.c2[T.hi[p]] = T.c2[p];
                T.nv = h;
                if (T.nv > nvmax) throw std::runtime_error("loess: k-d tree vertex overflow");
            }
            p = p + 1;
            l = T.lo[p];
            u = T.hi[p];
        }
    }
    // ehg139 -> ehg127 at each vertex (psi carried over between vertices, as in the Fortran)
    T.vval0.assign(T.nv + 1, 0.0);
    T.vval1.assign(T.nv + 1, 0.0);
    std::vector<int> psi(n + 1);
    for (int i = 1; i <= n; ++i) psi[i] = i;
    std::vector<double> dist(n + 1), w(nf + 1), eta(nf + 1);
    const double machep = 2.220446049250313e-16;        // d1mach(4)
    for (int lv = 1; lv <= T.nv; ++lv) {
        const double q = T.v[lv];
        for (int i = 1; i <= n; ++i) dist[i] = 0;
        for (int i = 1; i <= n; ++i) { const double dd = x[i] - q; dist[i] = dist[i] + dd * dd; }
        ehg106(1, n, nf, dist.data(), psi.data());
        const double rho = dist[psi[nf]] * std::max(1.0, span);
        if (rho <= 0) throw std::runtime_error("loess: span too small (zero neighbourhood radius)");
        for (int i = 1; i <= nf; ++i) w[i] = std::sqrt(dist[psi[i]] / rho);
        for (int i = 1; i <= nf; ++i) {
            const double w3 = w[i] * w[i] * w[i];
            const double t = 1 - w3;
            w[i] = std::sqrt(1.0 * (t * t * t));         // rw = 1
        }
        Mat b(nf, k);
        for (int i = 1; i <= nf; ++i) b(i, 1) = w[i];
        for (int i = 1; i <= nf; ++i) b(i, 2) = w[i] * (x[psi[i]] - q);
        for (int i = 1; i <= nf; ++i) eta[i] = w[i] * yy[psi[i]];
        double colnor[3];
        for (int j = 1; j <= k; ++j) {
            double scal = 0;
            for (int i = 1; i <= nf; ++i) scal = scal + b(i, j) * b(i, j);
            scal = std::sqrt(scal);
            if (0 < scal) {
                for (int i = 1; i <= nf; ++i) b(i, j) = b(i, j) / scal;
                colnor[j] = scal;
            } else {
                colnor[j] = 1;
            }
        }
        double qraux[16] = {0};
        dqrdc0(b, nf, k, qraux);
        dqrsl_qty(b, nf, k, qraux, eta.data());
        Mat u(15, 15), e(15, 15);
        for (int i = 1; i <= k; ++i)
            for (int j = i; j <= k; ++j) u(i, j) = b(i, j);
        double sigma[17] = {0}, g[17] = {0}, work[17] = {0};
        if (dsvdc21(u, k, k, sigma, g, e, work) != 0) throw std::runtime_error("loess: dsvdc failed");
        const double tol = sigma[1] * (100 * machep);
        for (int j = 1; j <= k; ++j)
            for (int i = 1; i <= k; ++i) e(j, i) = e(j, i) / colnor[j];
        double dgamma[3];
        for (int j = 1; j <= k; ++j)
            dgamma[j] = (tol < sigma[j]) ? ddot(k, u.p(1, j), 1, eta.data() + 1, 1) / sigma[j] : 0.0;
        T.vval0[lv] = ddot(k, e.p(1, 1), 15, dgamma + 1, 1);
        T.vval1[lv] = ddot(k, e.p(2, 1), 15, dgamma + 1, 1);
    }
    // lowese -> ehg128 at the data points
    Vec out(n);
    for (int i = 1; i <= n; ++i) {
        const double z = x[i];
        int j = 1;
        while (T.a[j] != 0) j = (z <= T.xi[j]) ? T.lo[j] : T.hi[j];
        const int ll = T.c1[j], ur = T.c2[j];
        const double h = (z - T.v[ll]) / (T.v[ur] - T.v[ll]);
        const double omh = 1 - h;
        const double phi0 = (omh * omh) * (1 + 2 * h);
        const double phi1 = (h * h) * (3 - 2 * h);
        const double psi0 = h * (omh * omh);
        const double psi1 = (h * h) * (h - 1);
        out[i - 1] = phi0 * T.vval0[ll] + phi1 * T.vval0[ur] + (psi0 * T.vval1[ll] + psi1 * T.vval1[ur]) * (T.v[ur] - T.v[ll]);
    }
    return out;
}

}  // namespace tsf
