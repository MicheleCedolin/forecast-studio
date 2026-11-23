// See lbfgsb.hpp. The f2c style of R's lbfgsb.c (1-based pointer offsets, column-major 2-D arrays) is kept so the
// arithmetic follows R's line by line; R's static variables of mainlb() and dcsrch() are thread_local here. Printing
// (iprint < 0 in optim's default trace = 0) is omitted.
#include "lbfgsb.hpp"

#include <cfloat>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace tsf {

namespace {

// ---------------------------------------------------------------- BLAS / LINPACK (unit strides)
double ddot(int n, const double* dx, const double* dy) {
    double t = 0.0;
    for (int i = 0; i < n; ++i) t = t + dx[i] * dy[i];
    return t;
}
void dcopy(int n, const double* dx, double* dy) { for (int i = 0; i < n; ++i) dy[i] = dx[i]; }
void dscal(int n, double da, double* dx) { for (int i = 0; i < n; ++i) dx[i] = da * dx[i]; }
void daxpy(int n, double da, const double* dx, double* dy) {
    if (n <= 0 || da == 0.0) return;
    for (int i = 0; i < n; ++i) dy[i] = dy[i] + da * dx[i];
}

// dpofa(a, lda, n, info): Cholesky factor (upper) in place; a points at a(1,1)
void dpofa(double* a0, int lda, int n, int* info) {
    auto A = [&](int i, int j) -> double& { return a0[(i - 1) + static_cast<size_t>(j - 1) * lda]; };
    const double eps = 1.e-14;
    for (int j = 1; j <= n; ++j) {
        *info = j;
        double s = 0.0;
        for (int k = 1; k <= j - 1; ++k) {
            double t = A(k, j) - ddot(k - 1, &A(1, k), &A(1, j));
            t = t / A(k, k);
            A(k, j) = t;
            s = s + t * t;
        }
        s = A(j, j) - s;
        if (s <= eps * std::fabs(A(j, j))) return;
        A(j, j) = std::sqrt(s);
    }
    *info = 0;
}

// dtrsl(t, ldt, n, b, job, info); t points at t(1,1), b at b(1)
void dtrsl(double* t0, int ldt, int n, double* b0, int job, int* info) {
    auto T = [&](int i, int j) -> double& { return t0[(i - 1) + static_cast<size_t>(j - 1) * ldt]; };
    double* b = b0 - 1;
    for (*info = 1; *info <= n; ++*info)
        if (T(*info, *info) == 0.0) return;
    *info = 0;
    int kase = 1;
    if (job % 10 != 0) kase = 2;
    if ((job % 100) / 10 != 0) kase += 2;
    switch (kase) {
        case 1:
            b[1] = b[1] / T(1, 1);
            for (int j = 2; j <= n; ++j) {
                const double temp = -b[j - 1];
                daxpy(n - j + 1, temp, &T(j, j - 1), &b[j]);
                b[j] = b[j] / T(j, j);
            }
            break;
        case 2:
            b[n] = b[n] / T(n, n);
            for (int jj = 2; jj <= n; ++jj) {
                const int j = n - jj + 1;
                const double temp = -b[j + 1];
                daxpy(j, temp, &T(1, j + 1), &b[1]);
                b[j] = b[j] / T(j, j);
            }
            break;
        case 3:
            b[n] = b[n] / T(n, n);
            for (int jj = 2; jj <= n; ++jj) {
                const int j = n - jj + 1;
                b[j] = b[j] - ddot(jj - 1, &T(j + 1, j), &b[j + 1]);
                b[j] = b[j] / T(j, j);
            }
            break;
        default:
            b[1] = b[1] / T(1, 1);
            for (int j = 2; j <= n; ++j) {
                b[j] = b[j] - ddot(j - 1, &T(1, j), &b[1]);
                b[j] = b[j] / T(j, j);
            }
            break;
    }
}

inline double dmax(double a, double b) { return (a < b) ? b : a; }
inline double dmin(double a, double b) { return (a > b) ? b : a; }
inline int imin(int a, int b) { return (a > b) ? b : a; }

// ---------------------------------------------------------------- lbfgsb.c routines
void active(int n, double* l, double* u, int* nbd, double* x, int* iwhere, int* prjctd, int* cnstnd, int* boxed) {
    --iwhere; --x; --nbd; --u; --l;
    *prjctd = 0; *cnstnd = 0; *boxed = 1;
    for (int i = 1; i <= n; ++i) {
        if (nbd[i] > 0) {
            if (nbd[i] <= 2 && x[i] <= l[i]) {
                if (x[i] < l[i]) { *prjctd = 1; x[i] = l[i]; }
            } else if (nbd[i] >= 2 && x[i] >= u[i]) {
                if (x[i] > u[i]) { *prjctd = 1; x[i] = u[i]; }
            }
        }
    }
    for (int i = 1; i <= n; ++i) {
        if (nbd[i] != 2) *boxed = 0;
        if (nbd[i] == 0) {
            iwhere[i] = -1;
        } else {
            *cnstnd = 1;
            if (nbd[i] == 2 && u[i] - l[i] <= 0.) iwhere[i] = 3;
            else iwhere[i] = 0;
        }
    }
}

void bmv(int m, double* sy, double* wt, int* col, double* v, double* p, int* info) {
    const int wt_dim1 = m, wt_offset = 1 + wt_dim1;
    wt -= wt_offset;
    const int sy_dim1 = m, sy_offset = 1 + sy_dim1;
    sy -= sy_offset;
    --p; --v;
    if (*col == 0) return;
    const int Col = *col;
    p[*col + 1] = v[*col + 1];
    for (int i = 2; i <= Col; ++i) {
        const int i2 = *col + i;
        double sum = 0.;
        for (int k = 1; k <= i - 1; ++k) sum += sy[i + k * sy_dim1] * v[k] / sy[k + k * sy_dim1];
        p[i2] = v[i2] + sum;
    }
    dtrsl(&wt[wt_offset], m, *col, &p[*col + 1], 11, info);
    if (*info != 0) return;
    for (int i = 1; i <= Col; ++i) p[i] = v[i] / std::sqrt(sy[i + i * sy_dim1]);
    dtrsl(&wt[wt_offset], m, *col, &p[*col + 1], 1, info);
    if (*info != 0) return;
    for (int i = 1; i <= Col; ++i) p[i] = -p[i] / std::sqrt(sy[i + i * sy_dim1]);
    for (int i = 1; i <= Col; ++i) {
        double sum = 0.;
        for (int k = i + 1; k <= Col; ++k) sum += sy[k + i * sy_dim1] * p[*col + k] / sy[i + i * sy_dim1];
        p[i] += sum;
    }
}

void hpsolb(int n, double* t, int* iorder, int iheap) {
    --iorder; --t;
    double ddum, out;
    int i, j, indxin, indxou;
    if (iheap == 0) {
        for (int k = 2; k <= n; ++k) {
            ddum = t[k];
            indxin = iorder[k];
            i = k;
            while (i > 1) {
                j = i / 2;
                if (ddum < t[j]) { t[i] = t[j]; iorder[i] = iorder[j]; i = j; }
                else break;
            }
            t[i] = ddum;
            iorder[i] = indxin;
        }
    }
    if (n > 1) {
        i = 1;
        out = t[1];
        indxou = iorder[1];
        ddum = t[n];
        indxin = iorder[n];
        for (;;) {
            j = i + i;
            if (j <= n - 1) {
                if (t[j + 1] < t[j]) ++j;
                if (t[j] < ddum) { t[i] = t[j]; iorder[i] = iorder[j]; i = j; continue; }
            }
            break;
        }
        t[i] = ddum;
        iorder[i] = indxin;
        t[n] = out;
        iorder[n] = indxou;
    }
}

void cauchy(int n, double* x, double* l, double* u, int* nbd, double* g, int* iorder, int* iwhere, double* t,
            double* d, double* xcp, int m, double* wy, double* ws, double* sy, double* wt, double* theta, int* col,
            int* head, double* p, double* c, double* wbp, double* v, int* nint, double* sbgnrm, int* info,
            double* epsmch) {
    double bkmin, dibp, dibp2, zibp, neggi, tsum;
    double f1, f2, f2_org__, dt, tj, tj0, tl = 0.0, tu = 0.0, dtm, wmc, wmp, wmw;
    int ibp, iter, bnded, nfree, nleft, nbreak, ibkmin, pointr;
    int xlower, xupper, col2;
    --xcp; --d; --t; --iwhere; --iorder; --g; --nbd; --u; --l; --x; --v; --wbp; --c; --p;
    const int wt_dim1 = m, wt_offset = 1 + wt_dim1; wt -= wt_offset;
    const int sy_dim1 = m, sy_offset = 1 + sy_dim1; sy -= sy_offset;
    const int ws_dim1 = n, ws_offset = 1 + ws_dim1; ws -= ws_offset;
    const int wy_dim1 = n, wy_offset = 1 + wy_dim1; wy -= wy_offset;
    if (*sbgnrm <= 0.) { dcopy(n, &x[1], &xcp[1]); return; }
    bnded = 1;
    nfree = n + 1;
    nbreak = 0;
    ibkmin = 0;
    bkmin = 0.;
    col2 = *col << 1;
    f1 = 0.;
    for (int i = 1; i <= col2; ++i) p[i] = 0.;
    for (int i = 1; i <= n; ++i) {
        neggi = -g[i];
        if (iwhere[i] != 3 && iwhere[i] != -1) {
            if (nbd[i] <= 2) tl = x[i] - l[i];
            if (nbd[i] >= 2) tu = u[i] - x[i];
            xlower = nbd[i] <= 2 && tl <= 0.;
            xupper = nbd[i] >= 2 && tu <= 0.;
            iwhere[i] = 0;
            if (xlower) { if (neggi <= 0.) iwhere[i] = 1; }
            else if (xupper) { if (neggi >= 0.) iwhere[i] = 2; }
            else { if (std::fabs(neggi) <= 0.) iwhere[i] = -3; }
        }
        pointr = *head;
        if (iwhere[i] != 0 && iwhere[i] != -1) {
            d[i] = 0.;
        } else {
            d[i] = neggi;
            f1 -= neggi * neggi;
            for (int j = 1; j <= *col; ++j) {
                p[j] += wy[i + pointr * wy_dim1] * neggi;
                p[*col + j] += ws[i + pointr * ws_dim1] * neggi;
                pointr = pointr % m + 1;
            }
            if (nbd[i] <= 2 && nbd[i] != 0 && neggi < 0.) {
                ++nbreak;
                iorder[nbreak] = i;
                t[nbreak] = tl / (-neggi);
                if (nbreak == 1 || t[nbreak] < bkmin) { bkmin = t[nbreak]; ibkmin = nbreak; }
            } else if (nbd[i] >= 2 && neggi > 0.) {
                ++nbreak;
                iorder[nbreak] = i;
                t[nbreak] = tu / neggi;
                if (nbreak == 1 || t[nbreak] < bkmin) { bkmin = t[nbreak]; ibkmin = nbreak; }
            } else {
                --nfree;
                iorder[nfree] = i;
                if (std::fabs(neggi) > 0.) bnded = 0;
            }
        }
    }
    if (*theta != 1.) dscal(*col, *theta, &p[*col + 1]);
    dcopy(n, &x[1], &xcp[1]);
    if (nbreak == 0 && nfree == n + 1) return;
    for (int j = 1; j <= col2; ++j) c[j] = 0.;
    f2 = -(*theta) * f1;
    f2_org__ = f2;
    if (*col > 0) {
        bmv(m, &sy[sy_offset], &wt[wt_offset], col, &p[1], &v[1], info);
        if (*info != 0) return;
        f2 -= ddot(col2, &v[1], &p[1]);
    }
    dtm = -f1 / f2;
    tsum = 0.;
    *nint = 1;
    if (nbreak == 0) goto L888;
    nleft = nbreak;
    iter = 1;
    tj = 0.;
L777:
    tj0 = tj;
    if (iter == 1) {
        tj = bkmin;
        ibp = iorder[ibkmin];
    } else {
        if (iter == 2) {
            if (ibkmin != nbreak) { t[ibkmin] = t[nbreak]; iorder[ibkmin] = iorder[nbreak]; }
        }
        hpsolb(nleft, &t[1], &iorder[1], iter - 2);
        tj = t[nleft];
        ibp = iorder[nleft];
    }
    dt = tj - tj0;
    if (dtm < dt) goto L888;
    tsum += dt;
    --nleft;
    ++iter;
    dibp = d[ibp];
    d[ibp] = 0.;
    if (dibp > 0.) { zibp = u[ibp] - x[ibp]; xcp[ibp] = u[ibp]; iwhere[ibp] = 2; }
    else { zibp = l[ibp] - x[ibp]; xcp[ibp] = l[ibp]; iwhere[ibp] = 1; }
    if (nleft == 0 && nbreak == n) { dtm = dt; goto L999; }
    ++(*nint);
    dibp2 = dibp * dibp;
    f1 += dt * f2 + dibp2 - *theta * dibp * zibp;
    f2 -= *theta * dibp2;
    if (*col > 0) {
        daxpy(col2, dt, &p[1], &c[1]);
        pointr = *head;
        for (int j = 1; j <= *col; ++j) {
            wbp[j] = wy[ibp + pointr * wy_dim1];
            wbp[*col + j] = *theta * ws[ibp + pointr * ws_dim1];
            pointr = pointr % m + 1;
        }
        bmv(m, &sy[sy_offset], &wt[wt_offset], col, &wbp[1], &v[1], info);
        if (*info != 0) return;
        wmc = ddot(col2, &c[1], &v[1]);
        wmp = ddot(col2, &p[1], &v[1]);
        wmw = ddot(col2, &wbp[1], &v[1]);
        daxpy(col2, -dibp, &wbp[1], &p[1]);
        f1 += dibp * wmc;
        f2 += (2. * dibp * wmp - dibp2 * wmw);
    }
    {
        const double d1 = *epsmch * f2_org__;
        if (f2 < d1) f2 = d1;
    }
    if (nleft > 0) {
        dtm = -f1 / f2;
        goto L777;
    } else if (bnded) {
        f1 = 0.; f2 = 0.; dtm = 0.;
    } else {
        dtm = -f1 / f2;
    }
L888:
    if (dtm <= 0.) dtm = 0.;
    tsum += dtm;
    daxpy(n, tsum, &d[1], &xcp[1]);
L999:
    if (*col > 0) daxpy(col2, dtm, &p[1], &c[1]);
}

void cmprlb(int n, int m, double* x, double* g, double* ws, double* wy, double* sy, double* wt, double* z, double* r,
            double* wa, int* indx, double* theta, int* col, int* head, int* nfree, int* cnstnd, int* info) {
    --indx; --r; --z; --g; --x; --wa;
    const int wt_dim1 = m, wt_offset = 1 + wt_dim1; wt -= wt_offset;
    const int sy_dim1 = m, sy_offset = 1 + sy_dim1; sy -= sy_offset;
    const int wy_dim1 = n, wy_offset = 1 + wy_dim1; wy -= wy_offset;
    const int ws_dim1 = n, ws_offset = 1 + ws_dim1; ws -= ws_offset;
    const int Col = *col;
    if (!(*cnstnd) && Col > 0) {
        for (int i = 1; i <= n; ++i) r[i] = -g[i];
    } else {
        const int n_f = *nfree;
        for (int i = 1; i <= n_f; ++i) { const int k = indx[i]; r[i] = -(*theta) * (z[k] - x[k]) - g[k]; }
        bmv(m, &sy[sy_offset], &wt[wt_offset], col, &wa[(m << 1) + 1], &wa[1], info);
        if (*info != 0) { *info = -8; return; }
        int pointr = *head;
        for (int j = 1; j <= Col; ++j) {
            const double a1 = wa[j], a2 = *theta * wa[Col + j];
            for (int i = 1; i <= n_f; ++i) {
                const int k = indx[i];
                r[i] += wy[k + pointr * wy_dim1] * a1 + ws[k + pointr * ws_dim1] * a2;
            }
            pointr = pointr % m + 1;
        }
    }
}

void errclb(int n, int m, double factr, double* l, double* u, int* nbd, char* task, int* info, int* k) {
    --nbd; --u; --l;
    if (n <= 0) std::strcpy(task, "ERROR: N .LE. 0");
    if (m <= 0) std::strcpy(task, "ERROR: M .LE. 0");
    if (factr < 0.) std::strcpy(task, "ERROR: FACTR .LT. 0");
    for (int i = 1; i <= n; ++i) {
        if (nbd[i] < 0 || nbd[i] > 3) { std::strcpy(task, "ERROR: INVALID NBD"); *info = -6; *k = i; }
        if (nbd[i] == 2 && l[i] > u[i]) { std::strcpy(task, "ERROR: NO FEASIBLE SOLUTION"); *info = -7; *k = i; }
    }
}

void formk(int n, int* nsub, int* ind, int* nenter, int* ileave, int* indx2, int* iupdat, int* updatd, double* wn,
           double* wn1, int m, double* ws, double* wy, double* sy, double* theta, int* col, int* head, int* info) {
    int dend, pend, upcl;
    double temp1, temp2, temp3, temp4;
    int ipntr, jpntr, k1, m2, dbegin, is, js, iy, pbegin, is1, js1, col2;
    --indx2; --ind;
    const int sy_dim1 = m, sy_offset = 1 + sy_dim1; sy -= sy_offset;
    const int wy_dim1 = n, wy_offset = 1 + wy_dim1; wy -= wy_offset;
    const int ws_dim1 = n, ws_offset = 1 + ws_dim1; ws -= ws_offset;
    const int wn1_dim1 = 2 * m, wn1_offset = 1 + wn1_dim1; wn1 -= wn1_offset;
    const int wn_dim1 = 2 * m, wn_offset = 1 + wn_dim1; wn -= wn_offset;
    if (*updatd) {
        if (*iupdat > m) {
            for (int jy = 1; jy <= m - 1; ++jy) {
                js = m + jy;
                dcopy(m - jy, &wn1[jy + 1 + (jy + 1) * wn1_dim1], &wn1[jy + jy * wn1_dim1]);
                dcopy(m - jy, &wn1[js + 1 + (js + 1) * wn1_dim1], &wn1[js + js * wn1_dim1]);
                dcopy(m - 1, &wn1[m + 2 + (jy + 1) * wn1_dim1], &wn1[m + 1 + jy * wn1_dim1]);
            }
        }
        pbegin = 1;
        pend = *nsub;
        dbegin = *nsub + 1;
        dend = n;
        iy = *col;
        is = m + *col;
        ipntr = *head + *col - 1;
        if (ipntr > m) ipntr -= m;
        jpntr = *head;
        for (int jy = 1; jy <= *col; ++jy) {
            js = m + jy;
            temp1 = 0.; temp2 = 0.; temp3 = 0.;
            for (int k = pbegin; k <= pend; ++k) {
                k1 = ind[k];
                temp1 += wy[k1 + ipntr * wy_dim1] * wy[k1 + jpntr * wy_dim1];
            }
            for (int k = dbegin; k <= dend; ++k) {
                k1 = ind[k];
                temp2 += ws[k1 + ipntr * ws_dim1] * ws[k1 + jpntr * ws_dim1];
                temp3 += ws[k1 + ipntr * ws_dim1] * wy[k1 + jpntr * wy_dim1];
            }
            wn1[iy + jy * wn1_dim1] = temp1;
            wn1[is + js * wn1_dim1] = temp2;
            wn1[is + jy * wn1_dim1] = temp3;
            jpntr = jpntr % m + 1;
        }
        const int jy = *col;
        jpntr = *head + *col - 1;
        if (jpntr > m) jpntr -= m;
        ipntr = *head;
        for (int i = 1; i <= *col; ++i) {
            is = m + i;
            temp3 = 0.;
            for (int k = pbegin; k <= pend; ++k) {
                k1 = ind[k];
                temp3 += ws[k1 + ipntr * ws_dim1] * wy[k1 + jpntr * wy_dim1];
            }
            ipntr = ipntr % m + 1;
            wn1[is + jy * wn1_dim1] = temp3;
        }
        upcl = *col - 1;
    } else {
        upcl = *col;
    }
    ipntr = *head;
    for (int iy2 = 1; iy2 <= upcl; ++iy2) {
        is = m + iy2;
        jpntr = *head;
        for (int jy = 1; jy <= iy2; ++jy) {
            js = m + jy;
            temp1 = 0.; temp2 = 0.; temp3 = 0.; temp4 = 0.;
            for (int k = 1; k <= *nenter; ++k) {
                k1 = indx2[k];
                temp1 += wy[k1 + ipntr * wy_dim1] * wy[k1 + jpntr * wy_dim1];
                temp2 += ws[k1 + ipntr * ws_dim1] * ws[k1 + jpntr * ws_dim1];
            }
            for (int k = *ileave; k <= n; ++k) {
                k1 = indx2[k];
                temp3 += wy[k1 + ipntr * wy_dim1] * wy[k1 + jpntr * wy_dim1];
                temp4 += ws[k1 + ipntr * ws_dim1] * ws[k1 + jpntr * ws_dim1];
            }
            wn1[iy2 + jy * wn1_dim1] = wn1[iy2 + jy * wn1_dim1] + temp1 - temp3;
            wn1[is + js * wn1_dim1] = wn1[is + js * wn1_dim1] - temp2 + temp4;
            jpntr = jpntr % m + 1;
        }
        ipntr = ipntr % m + 1;
    }
    ipntr = *head;
    for (int is2 = m + 1; is2 <= m + upcl; ++is2) {
        jpntr = *head;
        for (int jy = 1; jy <= upcl; ++jy) {
            temp1 = 0.; temp3 = 0.;
            for (int k = 1; k <= *nenter; ++k) {
                k1 = indx2[k];
                temp1 += ws[k1 + ipntr * ws_dim1] * wy[k1 + jpntr * wy_dim1];
            }
            for (int k = *ileave; k <= n; ++k) {
                k1 = indx2[k];
                temp3 += ws[k1 + ipntr * ws_dim1] * wy[k1 + jpntr * wy_dim1];
            }
            if (is2 <= jy + m) wn1[is2 + jy * wn1_dim1] += temp1 - temp3;
            else wn1[is2 + jy * wn1_dim1] += -temp1 + temp3;
            jpntr = jpntr % m + 1;
        }
        ipntr = ipntr % m + 1;
    }
    m2 = m << 1;
    for (int iy2 = 1; iy2 <= *col; ++iy2) {
        is = *col + iy2;
        is1 = m + iy2;
        for (int jy = 1; jy <= iy2; ++jy) {
            js = *col + jy;
            js1 = m + jy;
            wn[jy + iy2 * wn_dim1] = wn1[iy2 + jy * wn1_dim1] / *theta;
            wn[js + is * wn_dim1] = wn1[is1 + js1 * wn1_dim1] * *theta;
        }
        for (int jy = 1; jy <= iy2 - 1; ++jy) wn[jy + is * wn_dim1] = -wn1[is1 + jy * wn1_dim1];
        for (int jy = iy2; jy <= *col; ++jy) wn[jy + is * wn_dim1] = wn1[is1 + jy * wn1_dim1];
        wn[iy2 + iy2 * wn_dim1] += sy[iy2 + iy2 * sy_dim1];
    }
    dpofa(&wn[wn_offset], m2, *col, info);
    if (*info != 0) { *info = -1; return; }
    col2 = *col << 1;
    for (int js2 = *col + 1; js2 <= col2; ++js2) dtrsl(&wn[wn_offset], m2, *col, &wn[js2 * wn_dim1 + 1], 11, info);
    for (int is2 = *col + 1; is2 <= col2; ++is2)
        for (int js2 = is2; js2 <= col2; ++js2)
            wn[is2 + js2 * wn_dim1] += ddot(*col, &wn[is2 * wn_dim1 + 1], &wn[js2 * wn_dim1 + 1]);
    dpofa(&wn[*col + 1 + (*col + 1) * wn_dim1], m2, *col, info);
    if (*info != 0) { *info = -2; return; }
}

void formt(int m, double* wt, double* sy, double* ss, int* col, double* theta, int* info) {
    const int ss_dim1 = m, ss_offset = 1 + ss_dim1; ss -= ss_offset;
    const int sy_dim1 = m, sy_offset = 1 + sy_dim1; sy -= sy_offset;
    const int wt_dim1 = m, wt_offset = 1 + wt_dim1; wt -= wt_offset;
    for (int j = 1; j <= *col; ++j) wt[j * wt_dim1 + 1] = *theta * ss[j * ss_dim1 + 1];
    for (int i = 2; i <= *col; ++i) {
        for (int j = i; j <= *col; ++j) {
            const int k1 = imin(i, j) - 1;
            double ddum = 0.;
            for (int k = 1; k <= k1; ++k) ddum += sy[i + k * sy_dim1] * sy[j + k * sy_dim1] / sy[k + k * sy_dim1];
            wt[i + j * wt_dim1] = ddum + *theta * ss[i + j * ss_dim1];
        }
    }
    dpofa(&wt[wt_offset], m, *col, info);
    if (*info != 0) *info = -3;
}

void freev(int n, int* nfree, int* indx, int* nenter, int* ileave, int* indx2, int* iwhere, int* wrk, int* updatd,
           int* cnstnd, int* iter) {
    --iwhere; --indx2; --indx;
    *nenter = 0;
    *ileave = n + 1;
    if (*iter > 0 && *cnstnd) {
        for (int i = 1; i <= *nfree; ++i) {
            const int k = indx[i];
            if (iwhere[k] > 0) { --(*ileave); indx2[*ileave] = k; }
        }
        for (int i = *nfree + 1; i <= n; ++i) {
            const int k = indx[i];
            if (iwhere[k] <= 0) { ++(*nenter); indx2[*nenter] = k; }
        }
    }
    *wrk = *ileave < n + 1 || *nenter > 0 || *updatd;
    *nfree = 0;
    int iact = n + 1;
    for (int i = 1; i <= n; ++i) {
        if (iwhere[i] <= 0) { ++(*nfree); indx[*nfree] = i; }
        else { --iact; indx[iact] = i; }
    }
}

void dcstep(double* stx, double* fx, double* dx, double* sty, double* fy, double* dy, double* stp, double* fp,
            double* dp, int* brackt, double* stpmin, double* stpmax) {
    double d__1, d__2, sgnd, stpc, stpf, stpq, p, q, gamm, r__, s, theta;
    sgnd = *dp * (*dx / std::fabs(*dx));
    if (*fp > *fx) {
        theta = (*fx - *fp) * 3. / (*stp - *stx) + *dx + *dp;
        d__1 = std::fabs(theta); d__2 = std::fabs(*dx); d__1 = dmax(d__1, d__2); d__2 = std::fabs(*dp);
        s = dmax(d__1, d__2);
        d__1 = theta / s;
        gamm = s * std::sqrt(d__1 * d__1 - *dx / s * (*dp / s));
        if (*stp < *stx) gamm = -gamm;
        p = gamm - *dx + theta;
        q = gamm - *dx + gamm + *dp;
        r__ = p / q;
        stpc = *stx + r__ * (*stp - *stx);
        stpq = *stx + *dx / ((*fx - *fp) / (*stp - *stx) + *dx) / 2. * (*stp - *stx);
        if (std::fabs(stpc - *stx) < std::fabs(stpq - *stx)) stpf = stpc;
        else stpf = stpc + (stpq - stpc) / 2.;
        *brackt = 1;
    } else if (sgnd < 0.) {
        theta = (*fx - *fp) * 3. / (*stp - *stx) + *dx + *dp;
        d__1 = std::fabs(theta); d__2 = std::fabs(*dx); d__1 = dmax(d__1, d__2); d__2 = std::fabs(*dp);
        s = dmax(d__1, d__2);
        d__1 = theta / s;
        gamm = s * std::sqrt(d__1 * d__1 - *dx / s * (*dp / s));
        if (*stp > *stx) gamm = -gamm;
        p = gamm - *dp + theta;
        q = gamm - *dp + gamm + *dx;
        r__ = p / q;
        stpc = *stp + r__ * (*stx - *stp);
        stpq = *stp + *dp / (*dp - *dx) * (*stx - *stp);
        if (std::fabs(stpc - *stp) > std::fabs(stpq - *stp)) stpf = stpc;
        else stpf = stpq;
        *brackt = 1;
    } else if (std::fabs(*dp) < std::fabs(*dx)) {
        theta = (*fx - *fp) * 3. / (*stp - *stx) + *dx + *dp;
        d__1 = std::fabs(theta); d__2 = std::fabs(*dx); d__1 = dmax(d__1, d__2); d__2 = std::fabs(*dp);
        s = dmax(d__1, d__2);
        d__1 = theta / s;
        d__1 = d__1 * d__1 - *dx / s * (*dp / s);
        gamm = d__1 < 0 ? 0. : s * std::sqrt(d__1);
        if (*stp > *stx) gamm = -gamm;
        p = gamm - *dp + theta;
        q = gamm + (*dx - *dp) + gamm;
        r__ = p / q;
        if (r__ < 0. && gamm != 0.) stpc = *stp + r__ * (*stx - *stp);
        else if (*stp > *stx) stpc = *stpmax;
        else stpc = *stpmin;
        stpq = *stp + *dp / (*dp - *dx) * (*stx - *stp);
        if (*brackt) {
            if (std::fabs(stpc - *stp) < std::fabs(stpq - *stp)) stpf = stpc;
            else stpf = stpq;
            d__1 = *stp + (*sty - *stp) * .66;
            if (*stp > *stx) stpf = dmin(d__1, stpf);
            else stpf = dmax(d__1, stpf);
        } else {
            if (std::fabs(stpc - *stp) > std::fabs(stpq - *stp)) stpf = stpc;
            else stpf = stpq;
            stpf = dmin(*stpmax, stpf);
            stpf = dmax(*stpmin, stpf);
        }
    } else {
        if (*brackt) {
            theta = (*fp - *fy) * 3. / (*sty - *stp) + *dy + *dp;
            d__1 = std::fabs(theta); d__2 = std::fabs(*dy); d__1 = dmax(d__1, d__2); d__2 = std::fabs(*dp);
            s = dmax(d__1, d__2);
            d__1 = theta / s;
            gamm = s * std::sqrt(d__1 * d__1 - *dy / s * (*dp / s));
            if (*stp > *sty) gamm = -gamm;
            p = gamm - *dp + theta;
            q = gamm - *dp + gamm + *dy;
            r__ = p / q;
            stpc = *stp + r__ * (*sty - *stp);
            stpf = stpc;
        } else if (*stp > *stx) {
            stpf = *stpmax;
        } else {
            stpf = *stpmin;
        }
    }
    if (*fp > *fx) {
        *sty = *stp; *fy = *fp; *dy = *dp;
    } else {
        if (sgnd < 0.) { *sty = *stx; *fy = *fx; *dy = *dx; }
        *stx = *stp; *fx = *fp; *dx = *dp;
    }
    *stp = stpf;
}

struct SearchState {                                    // dcsrch()'s statics
    int stage = 0, brackt = 0;
    double ginit = 0, gtest = 0, gx = 0, gy = 0, finit = 0, fx = 0, fy = 0, stx = 0, sty = 0, stmin = 0, stmax = 0,
           width = 0, width1 = 0;
};
thread_local SearchState S_;

void dcsrch(double* f, double* g, double* stp, double ftol, double gtol, double xtol, double stpmin, double stpmax,
            char* task) {
    SearchState& S = S_;
    double ftest, fm, gm, fxm, fym, gxm, gym;
    if (std::strncmp(task, "START", 5) == 0) {
        if (*stp < stpmin) std::strcpy(task, "ERROR: STP .LT. STPMIN");
        if (*stp > stpmax) std::strcpy(task, "ERROR: STP .GT. STPMAX");
        if (*g >= 0.) std::strcpy(task, "ERROR: INITIAL G .GE. ZERO");
        if (ftol < 0.) std::strcpy(task, "ERROR: FTOL .LT. ZERO");
        if (gtol < 0.) std::strcpy(task, "ERROR: GTOL .LT. ZERO");
        if (xtol < 0.) std::strcpy(task, "ERROR: XTOL .LT. ZERO");
        if (stpmin < 0.) std::strcpy(task, "ERROR: STPMIN .LT. ZERO");
        if (stpmax < stpmin) std::strcpy(task, "ERROR: STPMAX .LT. STPMIN");
        if (std::strncmp(task, "ERROR", 5) == 0) return;
        S.brackt = 0;
        S.stage = 1;
        S.finit = *f;
        S.ginit = *g;
        S.gtest = ftol * S.ginit;
        S.width = stpmax - stpmin;
        S.width1 = S.width / .5;
        S.stx = 0.; S.fx = S.finit; S.gx = S.ginit;
        S.sty = 0.; S.fy = S.finit; S.gy = S.ginit;
        S.stmin = 0.;
        S.stmax = *stp + *stp * 4.;
        std::strcpy(task, "FG");
        return;
    }
    ftest = S.finit + *stp * S.gtest;
    if (S.stage == 1 && *f <= ftest && *g >= 0.) S.stage = 2;
    if (S.brackt && (*stp <= S.stmin || *stp >= S.stmax)) std::strcpy(task, "WARNING: ROUNDING ERRORS PREVENT PROGRESS");
    if (S.brackt && S.stmax - S.stmin <= xtol * S.stmax) std::strcpy(task, "WARNING: XTOL TEST SATISFIED");
    if (*stp == stpmax && *f <= ftest && *g <= S.gtest) std::strcpy(task, "WARNING: STP = STPMAX");
    if (*stp == stpmin && (*f > ftest || *g >= S.gtest)) std::strcpy(task, "WARNING: STP = STPMIN");
    if (*f <= ftest && std::fabs(*g) <= gtol * (-S.ginit)) std::strcpy(task, "CONVERGENCE");
    if (std::strncmp(task, "WARN", 4) == 0 || std::strncmp(task, "CONV", 4) == 0) return;
    if (S.stage == 1 && *f <= S.fx && *f > ftest) {
        fm = *f - *stp * S.gtest;
        fxm = S.fx - S.stx * S.gtest;
        fym = S.fy - S.sty * S.gtest;
        gm = *g - S.gtest;
        gxm = S.gx - S.gtest;
        gym = S.gy - S.gtest;
        dcstep(&S.stx, &fxm, &gxm, &S.sty, &fym, &gym, stp, &fm, &gm, &S.brackt, &S.stmin, &S.stmax);
        S.fx = fxm + S.stx * S.gtest;
        S.fy = fym + S.sty * S.gtest;
        S.gx = gxm + S.gtest;
        S.gy = gym + S.gtest;
    } else {
        dcstep(&S.stx, &S.fx, &S.gx, &S.sty, &S.fy, &S.gy, stp, f, g, &S.brackt, &S.stmin, &S.stmax);
    }
    if (S.brackt) {
        if (std::fabs(S.sty - S.stx) >= S.width1 * .66) *stp = S.stx + (S.sty - S.stx) * .5;
        S.width1 = S.width;
        S.width = std::fabs(S.sty - S.stx);
    }
    if (S.brackt) {
        S.stmin = dmin(S.stx, S.sty);
        S.stmax = dmax(S.stx, S.sty);
    } else {
        S.stmin = *stp + (*stp - S.stx) * 1.1;
        S.stmax = *stp + (*stp - S.stx) * 4.;
    }
    if (*stp < stpmin) *stp = stpmin;
    if (*stp > stpmax) *stp = stpmax;
    if ((S.brackt && (*stp <= S.stmin || *stp >= S.stmax)) || (S.brackt && (S.stmax - S.stmin <= xtol * S.stmax)))
        *stp = S.stx;
    std::strcpy(task, "FG");
}

void lnsrlb(int n, double* l, double* u, int* nbd, double* x, double* f, double* fold, double* gd, double* gdold,
            double* g, double* d, double* r, double* t, double* z, double* stp, double* dnorm, double* dtd,
            double* xstep, double* stpmx, int* iter, int* ifun, int* iback, int* nfgv, int* info, char* task,
            int* boxed, int* cnstnd, char* csave) {
    const double stpmin = 0., ftol = .001, gtol = .9, xtol = .1;
    --z; --t; --r; --d; --g; --x; --nbd; --u; --l;
    if (std::strncmp(task, "FG_LN", 5) == 0) goto L556;
    *dtd = ddot(n, &d[1], &d[1]);
    *dnorm = std::sqrt(*dtd);
    *stpmx = 1e10;
    if (*cnstnd) {
        if (*iter == 0) {
            *stpmx = 1.;
        } else {
            for (int i = 1; i <= n; ++i) {
                const double a1 = d[i];
                if (nbd[i] != 0) {
                    if (a1 < 0. && nbd[i] <= 2) {
                        const double a2 = l[i] - x[i];
                        if (a2 >= 0.) *stpmx = 0.;
                        else if (a1 * *stpmx < a2) *stpmx = a2 / a1;
                    } else if (a1 > 0. && nbd[i] >= 2) {
                        const double a2 = u[i] - x[i];
                        if (a2 <= 0.) *stpmx = 0.;
                        else if (a1 * *stpmx > a2) *stpmx = a2 / a1;
                    }
                }
            }
        }
    }
    if (*iter == 0 && !(*boxed)) {
        const double d1 = 1. / *dnorm;
        *stp = dmin(d1, *stpmx);
    } else {
        *stp = 1.;
    }
    dcopy(n, &x[1], &t[1]);
    dcopy(n, &g[1], &r[1]);
    *fold = *f;
    *ifun = 0;
    *iback = 0;
    std::strcpy(csave, "START");
L556:
    *gd = ddot(n, &g[1], &d[1]);
    if (*ifun == 0) {
        *gdold = *gd;
        if (*gd >= 0.) { *info = -4; return; }
    }
    dcsrch(f, gd, stp, ftol, gtol, xtol, stpmin, *stpmx, csave);
    *xstep = *stp * *dnorm;
    if (std::strncmp(csave, "CONV", 4) != 0 && std::strncmp(csave, "WARN", 4) != 0) {
        std::strcpy(task, "FG_LNSRCH");
        ++(*ifun);
        ++(*nfgv);
        *iback = *ifun - 1;
        if (*stp == 1.) dcopy(n, &z[1], &x[1]);
        else for (int i = 1; i <= n; ++i) x[i] = *stp * d[i] + t[i];
    } else {
        std::strcpy(task, "NEW_X");
    }
}

void matupd(int n, int m, double* ws, double* wy, double* sy, double* ss, double* d, double* r, int* itail,
            int* iupdat, int* col, int* head, double* theta, double* rr, double* dr, double* stp, double* dtd) {
    --r; --d;
    const int ss_dim1 = m, ss_offset = 1 + ss_dim1; ss -= ss_offset;
    const int sy_dim1 = m, sy_offset = 1 + sy_dim1; sy -= sy_offset;
    const int wy_dim1 = n, wy_offset = 1 + wy_dim1; wy -= wy_offset;
    const int ws_dim1 = n, ws_offset = 1 + ws_dim1; ws -= ws_offset;
    if (*iupdat <= m) {
        *col = *iupdat;
        *itail = (*head + *iupdat - 2) % m + 1;
    } else {
        *itail = *itail % m + 1;
        *head = *head % m + 1;
    }
    dcopy(n, &d[1], &ws[*itail * ws_dim1 + 1]);
    dcopy(n, &r[1], &wy[*itail * wy_dim1 + 1]);
    *theta = *rr / *dr;
    if (*iupdat > m) {
        for (int j = 1; j <= *col - 1; ++j) {
            dcopy(j, &ss[(j + 1) * ss_dim1 + 2], &ss[j * ss_dim1 + 1]);
            dcopy(*col - j, &sy[j + 1 + (j + 1) * sy_dim1], &sy[j + j * sy_dim1]);
        }
    }
    int pointr = *head;
    for (int j = 1; j <= *col - 1; ++j) {
        sy[*col + j * sy_dim1] = ddot(n, &d[1], &wy[pointr * wy_dim1 + 1]);
        ss[j + *col * ss_dim1] = ddot(n, &ws[pointr * ws_dim1 + 1], &d[1]);
        pointr = pointr % m + 1;
    }
    if (*stp == 1.) ss[*col + *col * ss_dim1] = *dtd;
    else ss[*col + *col * ss_dim1] = *stp * *stp * *dtd;
    sy[*col + *col * sy_dim1] = *dr;
}

void projgr(int n, double* l, double* u, int* nbd, double* x, double* g, double* sbgnrm) {
    *sbgnrm = 0.;
    for (int i = 0; i < n; ++i) {
        double gi = g[i], d__1;
        if (nbd[i] != 0) {
            if (gi < 0.) {
                if (nbd[i] >= 2) { if (gi < (d__1 = x[i] - u[i])) gi = d__1; }
            } else {
                if (nbd[i] <= 2) { if (gi > (d__1 = x[i] - l[i])) gi = d__1; }
            }
        }
        if (*sbgnrm < (d__1 = std::fabs(gi))) *sbgnrm = d__1;
    }
}

void subsm(int n, int m, int* nsub, int* ind, double* l, double* u, int* nbd, double* x, double* d, double* ws,
           double* wy, double* theta, int* col, int* head, int* iword, double* wv, double* wn, int* info) {
    double alpha, dk, temp1, temp2;
    int k, m2, js, pointr, ibd = 0, col2, ns;
    --d; --u; --l; --x; --ind; --nbd; --wv;
    const int wn_dim1 = 2 * m, wn_offset = 1 + wn_dim1;
    wn -= wn_offset;
    const int ws_offset = 1 + n;
    ws -= ws_offset;
    wy -= ws_offset;
    ns = *nsub;
    if (ns <= 0) return;
    pointr = *head;
    for (int i = 1; i <= *col; ++i) {
        temp1 = 0.; temp2 = 0.;
        for (int j = 1; j <= ns; ++j) {
            k = ind[j];
            temp1 += wy[k + pointr * n] * d[j];
            temp2 += ws[k + pointr * n] * d[j];
        }
        wv[i] = temp1;
        wv[*col + i] = *theta * temp2;
        pointr = pointr % m + 1;
    }
    m2 = m << 1;
    col2 = *col << 1;
    dtrsl(&wn[wn_offset], m2, col2, &wv[1], 11, info);
    if (*info != 0) return;
    for (int i = 1; i <= *col; ++i) wv[i] = -wv[i];
    dtrsl(&wn[wn_offset], m2, col2, &wv[1], 1, info);
    if (*info != 0) return;
    pointr = *head;
    for (int jy = 1; jy <= *col; ++jy) {
        js = *col + jy;
        for (int i = 1; i <= ns; ++i) {
            k = ind[i];
            d[i] += (wy[k + pointr * n] * wv[jy] / *theta + ws[k + pointr * n] * wv[js]);
        }
        pointr = pointr % m + 1;
    }
    for (int i = 1; i <= ns; ++i) d[i] /= *theta;
    alpha = 1.;
    temp1 = alpha;
    for (int i = 1; i <= ns; ++i) {
        k = ind[i];
        dk = d[i];
        if (nbd[k] != 0) {
            if (dk < 0. && nbd[k] <= 2) {
                temp2 = l[k] - x[k];
                if (temp2 >= 0.) temp1 = 0.;
                else if (dk * alpha < temp2) temp1 = temp2 / dk;
            } else if (dk > 0. && nbd[k] >= 2) {
                temp2 = u[k] - x[k];
                if (temp2 <= 0.) temp1 = 0.;
                else if (dk * alpha > temp2) temp1 = temp2 / dk;
            }
            if (temp1 < alpha) { alpha = temp1; ibd = i; }
        }
    }
    if (alpha < 1.) {
        dk = d[ibd];
        k = ind[ibd];
        if (dk > 0.) { x[k] = u[k]; d[ibd] = 0.; }
        else if (dk < 0.) { x[k] = l[k]; d[ibd] = 0.; }
    }
    for (int i = 1; i <= ns; ++i) x[ind[i]] += alpha * d[i];
    *iword = (alpha < 1.) ? 1 : 0;
}

struct MainState {                                      // mainlb()'s statics
    int prjctd = 0, cnstnd = 0, boxed = 0, updatd = 0;
    int nintol = 0, iback = 0, nskip = 0, head = 0, col = 0, itail = 0, iter = 0, iupdat = 0, nint = 0, nfgv = 0,
        info = 0, ifun = 0, iword = 0, nfree = 0, nact = 0, ileave = 0, nenter = 0;
    double theta = 0, fold = 0, tol = 0, dnorm = 0, epsmch = 0, gd = 0, stpmx = 0, sbgnrm = 0, stp = 0, gdold = 0,
           dtd = 0;
};
thread_local MainState M_;

void mainlb(int n, int m, double* x, double* l, double* u, int* nbd, double* f, double* g, double factr,
            double* pgtol, double* ws, double* wy, double* sy, double* ss, double* wt, double* wn, double* snd,
            double* z, double* r, double* d, double* t, double* wa, int* indx, int* iwhere, int* indx2, char* task,
            char* csave, int* isave) {
    MainState& S = M_;
    double ddum;
    int k = 0;
    double xstep = 0.0;
    double dr, rr;
    int wrk = 0;
    --indx2; --iwhere; --indx; --t; --d; --r; --z; --g; --nbd; --u; --l; --x; --wa;
    if (std::strncmp(task, "START", 5) == 0) {
        S.epsmch = DBL_EPSILON;
        S.fold = 0.; S.dnorm = 0.; S.gd = 0.; S.sbgnrm = 0.; S.stp = 0.; xstep = 0.; S.stpmx = 0.; S.gdold = 0.;
        S.dtd = 0.;
        S.col = 0; S.head = 1; S.theta = 1.; S.iupdat = 0; S.updatd = 0; S.iback = 0; S.itail = 0; S.ifun = 0;
        S.iword = 0; S.nact = 0; S.ileave = 0; S.nenter = 0; S.iter = 0; S.nfgv = 0; S.nint = 0; S.nintol = 0;
        S.nskip = 0; S.nfree = n;
        S.tol = factr * S.epsmch;
        S.info = 0;
        errclb(n, m, factr, &l[1], &u[1], &nbd[1], task, &S.info, &k);
        if (std::strncmp(task, "ERROR", 5) == 0) return;
        active(n, &l[1], &u[1], &nbd[1], &x[1], &iwhere[1], &S.prjctd, &S.cnstnd, &S.boxed);
    } else {
        if (std::strncmp(task, "FG_LN", 5) == 0) goto L666;
        if (std::strncmp(task, "NEW_X", 5) == 0) goto L777;
        if (std::strncmp(task, "FG_ST", 5) == 0) goto L111;
        if (std::strncmp(task, "STOP", 4) == 0) {
            if (std::strncmp(task + 6, "CPU", 3) == 0) {
                dcopy(n, &t[1], &x[1]);
                dcopy(n, &r[1], &g[1]);
                *f = S.fold;
            }
            goto L999;
        }
    }
    std::strcpy(task, "FG_START");
    goto L1000;
L111:
    S.nfgv = 1;
    projgr(n, &l[1], &u[1], &nbd[1], &x[1], &g[1], &S.sbgnrm);
    if (S.sbgnrm <= *pgtol) {
        std::strcpy(task, "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL");
        goto L999;
    }
L222:
    S.iword = -1;
    if (!S.cnstnd && S.col > 0) {
        dcopy(n, &x[1], &z[1]);
        wrk = S.updatd;
        S.nint = 0;
        goto L333;
    }
    cauchy(n, &x[1], &l[1], &u[1], &nbd[1], &g[1], &indx2[1], &iwhere[1], &t[1], &d[1], &z[1], m, wy, ws, sy, wt,
           &S.theta, &S.col, &S.head, &wa[1], &wa[(m << 1) + 1], &wa[(m << 2) + 1], &wa[m * 6 + 1], &S.nint,
           &S.sbgnrm, &S.info, &S.epsmch);
    if (S.info != 0) {
        S.info = 0; S.col = 0; S.head = 1; S.theta = 1.; S.iupdat = 0; S.updatd = 0;
        goto L222;
    }
    S.nintol += S.nint;
    freev(n, &S.nfree, &indx[1], &S.nenter, &S.ileave, &indx2[1], &iwhere[1], &wrk, &S.updatd, &S.cnstnd, &S.iter);
    S.nact = n - S.nfree;
L333:
    if (S.nfree == 0 || S.col == 0) goto L555;
    if (wrk)
        formk(n, &S.nfree, &indx[1], &S.nenter, &S.ileave, &indx2[1], &S.iupdat, &S.updatd, wn, snd, m, ws, wy, sy,
              &S.theta, &S.col, &S.head, &S.info);
    if (S.info != 0) {
        S.info = 0; S.col = 0; S.head = 1; S.theta = 1.; S.iupdat = 0; S.updatd = 0;
        goto L222;
    }
    cmprlb(n, m, &x[1], &g[1], ws, wy, sy, wt, &z[1], &r[1], &wa[1], &indx[1], &S.theta, &S.col, &S.head, &S.nfree,
           &S.cnstnd, &S.info);
    if (S.info != 0) goto L444;
    subsm(n, m, &S.nfree, &indx[1], &l[1], &u[1], &nbd[1], &z[1], &r[1], ws, wy, &S.theta, &S.col, &S.head, &S.iword,
          &wa[1], wn, &S.info);
L444:
    if (S.info != 0) {
        S.info = 0; S.col = 0; S.head = 1; S.theta = 1.; S.iupdat = 0; S.updatd = 0;
        goto L222;
    }
L555:
    for (int i = 1; i <= n; ++i) d[i] = z[i] - x[i];
L666:
    lnsrlb(n, &l[1], &u[1], &nbd[1], &x[1], f, &S.fold, &S.gd, &S.gdold, &g[1], &d[1], &r[1], &t[1], &z[1], &S.stp,
           &S.dnorm, &S.dtd, &xstep, &S.stpmx, &S.iter, &S.ifun, &S.iback, &S.nfgv, &S.info, task, &S.boxed,
           &S.cnstnd, csave);
    if (S.info != 0 || S.iback >= 20) {
        dcopy(n, &t[1], &x[1]);
        dcopy(n, &r[1], &g[1]);
        *f = S.fold;
        if (S.col == 0) {
            if (S.info == 0) { S.info = -9; --S.nfgv; --S.ifun; --S.iback; }
            std::strcpy(task, "ERROR: ABNORMAL_TERMINATION_IN_LNSRCH");
            ++S.iter;
            goto L999;
        } else {
            if (S.info == 0) --S.nfgv;
            S.info = 0; S.col = 0; S.head = 1; S.theta = 1.; S.iupdat = 0; S.updatd = 0;
            std::strcpy(task, "RESTART_FROM_LNSRCH");
            goto L222;
        }
    } else if (std::strncmp(task, "FG_LN", 5) == 0) {
        goto L1000;
    } else {
        ++S.iter;
        projgr(n, &l[1], &u[1], &nbd[1], &x[1], &g[1], &S.sbgnrm);
        goto L1000;
    }
L777:
    if (S.sbgnrm <= *pgtol) {
        std::strcpy(task, "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL");
        goto L999;
    }
    {
        double d__1 = std::fabs(S.fold), d__2 = std::fabs(*f);
        d__1 = dmax(d__1, d__2);
        ddum = dmax(d__1, 1.);
    }
    if (S.fold - *f <= S.tol * ddum) {
        std::strcpy(task, "CONVERGENCE: REL_REDUCTION_OF_F <= FACTR*EPSMCH");
        if (S.iback >= 10) S.info = -5;
        goto L999;
    }
    for (int i = 1; i <= n; ++i) r[i] = g[i] - r[i];
    rr = ddot(n, &r[1], &r[1]);
    if (S.stp == 1.) {
        dr = S.gd - S.gdold;
        ddum = -S.gdold;
    } else {
        dr = (S.gd - S.gdold) * S.stp;
        dscal(n, S.stp, &d[1]);
        ddum = -S.gdold * S.stp;
    }
    if (dr <= S.epsmch * ddum) {
        ++S.nskip;
        S.updatd = 0;
        goto L888;
    }
    S.updatd = 1;
    ++S.iupdat;
    matupd(n, m, ws, wy, sy, ss, &d[1], &r[1], &S.itail, &S.iupdat, &S.col, &S.head, &S.theta, &rr, &dr, &S.stp,
           &S.dtd);
    formt(m, wt, sy, ss, &S.col, &S.theta, &S.info);
    if (S.info != 0) {
        S.info = 0; S.col = 0; S.head = 1; S.theta = 1.; S.iupdat = 0; S.updatd = 0;
        goto L222;
    }
L888:
    goto L222;
L999:
L1000:
    isave[13 - 1] = S.nfgv;
}

void setulb(int n, int m, double* x, double* l, double* u, int* nbd, double* f, double* g, double factr,
            double* pgtol, double* wa, int* iwa, char* task, int* isave, char* csave) {
    // the workspace offsets (setulb's statics) depend only on n and m
    --wa;
    const int lws = 1;
    const int lwy = lws + m * n;
    const int lsy = lwy + m * n;
    const int lss = lsy + m * m;
    const int lwt = lss + m * m;
    const int lwn = lwt + m * m;
    const int lsnd = lwn + (m * m << 2);
    const int lz = lsnd + (m * m << 2);
    const int lr = lz + n;
    const int ld = lr + n;
    const int lt = ld + n;
    const int lwa = lt + n;
    mainlb(n, m, x, l, u, nbd, f, g, factr, pgtol, &wa[lws], &wa[lwy], &wa[lsy], &wa[lss], &wa[lwt], &wa[lwn],
           &wa[lsnd], &wa[lz], &wa[lr], &wa[ld], &wa[lt], &wa[lwa], iwa, &iwa[n], &iwa[n << 1], task, csave, isave);
}

}  // namespace

LbfgsbResult r_optim_lbfgsb(const std::function<double(const Vec&)>& fn, const Vec& par, const Vec& lower,
                            const Vec& upper, int maxit, int lmm, double factr, double pgtol, double ndeps) {
    const int n = static_cast<int>(par.size());
    LbfgsbResult res;
    Vec x = par, lo = lower, up = upper;
    std::vector<int> nbd(n);
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(lo[i])) nbd[i] = std::isfinite(up[i]) ? 3 : 0;
        else nbd[i] = std::isfinite(up[i]) ? 2 : 1;
    }
    if (n == 0) { res.value = fn(x); res.msg = "NOTHING TO DO"; return res; }
    auto fminfn = [&](const Vec& p) {
        for (double v : p) if (!std::isfinite(v)) throw std::runtime_error("non-finite value supplied by optim");
        return fn(p);
    };
    // stats/src/optim.c fmingr with usebounds = 1 (parscale = fnscale = 1)
    auto fmingr = [&](const Vec& p, double* df) {
        Vec xx = p;
        for (int i = 0; i < n; ++i) {
            double eps = ndeps, epsused = ndeps;
            double tmp = p[i] + eps;
            if (tmp > up[i]) { tmp = up[i]; epsused = tmp - p[i]; }
            xx[i] = tmp;
            const double val1 = fn(xx);
            tmp = p[i] - eps;
            if (tmp < lo[i]) { tmp = lo[i]; eps = p[i] - tmp; }
            xx[i] = tmp;
            const double val2 = fn(xx);
            df[i] = (val1 - val2) / (epsused + eps);
            if (!std::isfinite(df[i])) throw std::runtime_error("non-finite finite-difference value");
            xx[i] = p[i];
        }
    };
    char task[60], csave[60];
    double f = 0.0;
    Vec g(n, 0.0);
    std::vector<double> wa(static_cast<size_t>(2 * lmm * n + 4 * n + 11 * lmm * lmm + 8 * lmm), 0.0);
    std::vector<int> iwa(static_cast<size_t>(3 * n), 0);
    int isave[21] = {0};
    int iter = 0;
    M_ = MainState();
    S_ = SearchState();
    std::strcpy(task, "START");
    for (;;) {
        setulb(n, lmm, x.data(), lo.data(), up.data(), nbd.data(), &f, g.data(), factr, &pgtol, wa.data(), iwa.data(),
               task, isave, csave);
        if (std::strncmp(task, "FG", 2) == 0) {
            f = fminfn(x);
            if (!std::isfinite(f)) throw std::runtime_error("L-BFGS-B needs finite values of 'fn'");
            fmingr(x, g.data());
        } else if (std::strncmp(task, "NEW_X", 5) == 0) {
            ++iter;
            if (iter > maxit) { res.fail = 1; break; }
        } else if (std::strncmp(task, "WARN", 4) == 0) {
            res.fail = 51; break;
        } else if (std::strncmp(task, "CONV", 4) == 0) {
            break;
        } else {
            res.fail = 52; break;
        }
    }
    res.value = f;
    res.par = x;
    res.msg = task;
    return res;
}

}  // namespace tsf
