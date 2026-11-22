// Port of R stats::arima (src/library/stats/R/arima.R, src/library/stats/src/arima.c), R's optim BFGS (vmmin,
// src/appl/optim.c) with numerical derivatives (optim.c fmingr / optimhess), and KalmanFore.
#include "arima.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace tsf {

using C = std::complex<double>;

// ------------------------------------------------------------------------------------------- polynomials
CVec polyroot(const Vec& z) {
    int n = static_cast<int>(z.size()) - 1;
    while (n > 0 && z[n] == 0.0) --n;
    if (n < 1) return {};
    std::vector<C> c(n + 1);                          // monic, highest degree first
    for (int i = 0; i <= n; ++i) c[i] = z[n - i] / z[n];
    double rad = 0.0;
    for (int i = 1; i <= n; ++i) rad = std::max(rad, std::abs(c[i]));
    rad = 1.0 + rad;
    CVec r(n);
    const C seed(0.4, 0.9);
    for (int i = 0; i < n; ++i) r[i] = std::pow(seed, i) * (0.5 * rad);
    auto peval = [&](C x) { C v = c[0]; for (int i = 1; i <= n; ++i) v = v * x + c[i]; return v; };
    for (int it = 0; it < 2000; ++it) {               // Durand-Kerner
        double delta = 0.0;
        for (int i = 0; i < n; ++i) {
            C den = 1.0;
            for (int j = 0; j < n; ++j) if (j != i) den *= (r[i] - r[j]);
            if (std::abs(den) < 1e-300) den = 1e-300;
            const C dz = peval(r[i]) / den;
            r[i] -= dz;
            delta = std::max(delta, std::abs(dz) / std::max(1.0, std::abs(r[i])));
        }
        if (delta < 1e-15) break;
    }
    for (C& x : r) {                                  // Newton polish
        for (int k = 0; k < 3; ++k) {
            C v = c[0], dv = 0.0;
            for (int i = 1; i <= n; ++i) { dv = dv * x + v; v = v * x + c[i]; }
            if (std::abs(dv) > 0) x -= v / dv;
        }
    }
    return r;
}

namespace {

Vec conv(const Vec& a, const Vec& b) {               // C_TSconv
    Vec r(a.size() + b.size() - 1, 0.0);
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j) r[i + j] += a[i] * b[j];
    return r;
}

bool ar_check(const Vec& ar) {                       // arCheck
    int p = 0;
    for (int i = 0; i < static_cast<int>(ar.size()); ++i) if (ar[i] != 0.0) p = i + 1;
    if (!p) return true;
    Vec z(p + 1);
    z[0] = 1;
    for (int i = 0; i < p; ++i) z[i + 1] = -ar[i];
    for (const C& r : polyroot(z)) if (!(std::abs(r) > 1)) return false;
    return true;
}

Vec ma_invert(const Vec& ma) {                       // maInvert
    const int q = static_cast<int>(ma.size());
    int q0 = 0;
    for (int i = 0; i < q; ++i) if (ma[i] != 0.0) q0 = i + 1;
    if (!q0) return ma;
    Vec z(q0 + 1);
    z[0] = 1;
    for (int i = 0; i < q0; ++i) z[i + 1] = ma[i];
    CVec roots = polyroot(z);
    bool any = false;
    for (const C& r : roots) if (std::abs(r) < 1) any = true;
    if (!any) return ma;
    if (q0 == 1) { Vec out(q, 0.0); out[0] = 1 / ma[0]; return out; }
    for (C& r : roots) if (std::abs(r) < 1) r = 1.0 / r;
    std::vector<C> x{1.0};
    for (const C& r : roots) {                        // x <- c(x, 0) - c(0, x)/r
        std::vector<C> nx(x.size() + 1, 0.0);
        for (size_t i = 0; i < x.size(); ++i) nx[i] += x[i];
        for (size_t i = 0; i < x.size(); ++i) nx[i + 1] -= x[i] / r;
        x = nx;
    }
    Vec out(q, 0.0);
    for (int i = 0; i < q0; ++i) out[i] = x[i + 1].real();
    return out;
}

void partrans(int p, const double* raw, double* nw) {
    double work[100];
    for (int j = 0; j < p; ++j) work[j] = nw[j] = std::tanh(raw[j]);
    for (int j = 1; j < p; ++j) {
        const double a = nw[j];
        for (int k = 0; k < j; ++k) work[k] -= a * nw[j - k - 1];
        for (int k = 0; k < j; ++k) nw[k] = work[k];
    }
}

void invpartrans(int p, const double* phi, double* nw) {
    double work[100];
    for (int j = 0; j < p; ++j) work[j] = nw[j] = phi[j];
    for (int j = p - 1; j > 0; --j) {
        const double a = nw[j];
        for (int k = 0; k < j; ++k) work[k] = (nw[k] + a * nw[j - k - 1]) / (1 - a * a);
        for (int k = 0; k < j; ++k) nw[k] = work[k];
    }
    for (int j = 0; j < p; ++j) nw[j] = std::atanh(nw[j]);
}

struct Arma { int mp, mq, msp, msq, ns, d, D; };

Vec undo_pars(const Vec& in, const Arma& a) {        // ARIMA_undoPars
    Vec out = in;
    if (a.mp > 0) partrans(a.mp, in.data(), out.data());
    const int v = a.mp + a.mq;
    if (a.msp > 0) partrans(a.msp, in.data() + v, out.data() + v);
    return out;
}

Vec inv_trans(const Vec& in, const Arma& a) {        // ARIMA_Invtrans
    Vec out = in;
    if (a.mp > 0) invpartrans(a.mp, in.data(), out.data());
    const int v = a.mp + a.mq;
    if (a.msp > 0) invpartrans(a.msp, in.data() + v, out.data() + v);
    return out;
}

void trans_pars(const Vec& in, const Arma& a, bool trans, Vec& phi, Vec& theta) {   // ARIMA_transPars
    const int p = a.mp + a.ns * a.msp, q = a.mq + a.ns * a.msq;
    phi.assign(p, 0.0);
    theta.assign(q, 0.0);
    Vec params = in;
    if (trans) {
        if (a.mp > 0) partrans(a.mp, in.data(), params.data());
        const int v = a.mp + a.mq;
        if (a.msp > 0) partrans(a.msp, in.data() + v, params.data() + v);
    }
    if (a.ns > 0) {
        for (int i = 0; i < a.mp; ++i) phi[i] = params[i];
        for (int i = 0; i < a.mq; ++i) theta[i] = params[i + a.mp];
        for (int j = 0; j < a.msp; ++j) {
            phi[(j + 1) * a.ns - 1] += params[j + a.mp + a.mq];
            for (int i = 0; i < a.mp; ++i) phi[(j + 1) * a.ns + i] -= params[i] * params[j + a.mp + a.mq];
        }
        for (int j = 0; j < a.msq; ++j) {
            theta[(j + 1) * a.ns - 1] += params[j + a.mp + a.mq + a.msp];
            for (int i = 0; i < a.mq; ++i) theta[(j + 1) * a.ns + i] += params[i + a.mp] * params[j + a.mp + a.mq + a.msp];
        }
    } else {
        for (int i = 0; i < a.mp; ++i) phi[i] = params[i];
        for (int i = 0; i < a.mq; ++i) theta[i] = params[i + a.mp];
    }
}

std::vector<Vec> grad_trans(const Vec& raw, const Arma& a) {   // ARIMA_Gradtrans: A[i][j]
    const int n = static_cast<int>(raw.size());
    std::vector<Vec> A(n, Vec(n, 0.0));
    for (int i = 0; i < n; ++i) A[i][i] = 1.0;
    const double eps = 1e-3;
    double w1[100], w2[100], w3[100];
    if (a.mp > 0) {
        for (int i = 0; i < a.mp; ++i) w1[i] = raw[i];
        partrans(a.mp, w1, w2);
        for (int i = 0; i < a.mp; ++i) {
            w1[i] += eps;
            partrans(a.mp, w1, w3);
            for (int j = 0; j < a.mp; ++j) A[i][j] = (w3[j] - w2[j]) / eps;
            w1[i] -= eps;
        }
    }
    if (a.msp > 0) {
        const int v = a.mp + a.mq;
        for (int i = 0; i < a.msp; ++i) w1[i] = raw[i + v];
        partrans(a.msp, w1, w2);
        for (int i = 0; i < a.msp; ++i) {
            w1[i] += eps;
            partrans(a.msp, w1, w3);
            for (int j = 0; j < a.msp; ++j) A[i + v][j + v] = (w3[j] - w2[j]) / eps;
            w1[i] -= eps;
        }
    }
    return A;
}

// ------------------------------------------------------------------------------------------- getQ0 (Gardner 1980)
void inclu2(int np, const double* xnext, double* xrow, double ynext, double* d, double* rbar, double* thetab) {
    for (int i = 0; i < np; ++i) xrow[i] = xnext[i];
    for (int ithisr = 0, i = 0; i < np; ++i) {
        if (xrow[i] != 0.0) {
            const double xi = xrow[i], di = d[i], dpi = di + xi * xi;
            d[i] = dpi;
            const double cbar = di / dpi, sbar = xi / dpi;
            for (int k = i + 1; k < np; ++k) {
                const double xk = xrow[k], rbthis = rbar[ithisr];
                xrow[k] = xk - xi * rbthis;
                rbar[ithisr++] = cbar * rbthis + sbar * xk;
            }
            const double xk = ynext;
            ynext = xk - xi * thetab[i];
            thetab[i] = cbar * thetab[i] + sbar * xk;
            if (di == 0.0) return;
        } else {
            ithisr = ithisr + np - i - 1;
        }
    }
}

Vec getQ0(const Vec& phi, const Vec& theta) {
    const int p = static_cast<int>(phi.size()), q = static_cast<int>(theta.size());
    const int r = std::max(p, q + 1);
    const int np = r * (r + 1) / 2, nrbar = np * (np - 1) / 2;
    Vec V(np), P(static_cast<size_t>(r) * r, 0.0);
    for (int ind = 0, j = 0; j < r; ++j) {
        const double vj = (j == 0) ? 1.0 : (j - 1 < q ? theta[j - 1] : 0.0);
        for (int i = j; i < r; ++i) {
            const double vi = (i == 0) ? 1.0 : (i - 1 < q ? theta[i - 1] : 0.0);
            V[ind++] = vi * vj;
        }
    }
    if (r == 1) {
        P[0] = (p == 0) ? 1.0 : 1.0 / (1.0 - phi[0] * phi[0]);
        return P;
    }
    if (p > 0) {
        Vec rbar(std::max(nrbar, 1), 0.0), thetab(np, 0.0), xnext(np, 0.0), xrow(np, 0.0);
        for (int i = 0; i < np; ++i) P[i] = 0.0;
        int ind = 0, ind1 = -1;
        const int npr = np - r, npr1 = npr + 1;
        int indj = npr, ind2 = npr - 1;
        for (int j = 0; j < r; ++j) {
            const double phij = (j < p) ? phi[j] : 0.0;
            xnext[indj++] = 0.0;
            int indi = npr1 + j;
            for (int i = j; i < r; ++i) {
                const double ynext = V[ind++];
                const double phii = (i < p) ? phi[i] : 0.0;
                if (j != r - 1) {
                    xnext[indj] = -phii;
                    if (i != r - 1) {
                        xnext[indi] -= phij;
                        xnext[++ind1] = -1.0;
                    }
                }
                xnext[npr] = -phii * phij;
                if (++ind2 >= np) ind2 = 0;
                xnext[ind2] += 1.0;
                inclu2(np, xnext.data(), xrow.data(), ynext, P.data(), rbar.data(), thetab.data());
                xnext[ind2] = 0.0;
                if (i != r - 1) {
                    xnext[indi++] = 0.0;
                    xnext[ind1] = 0.0;
                }
            }
        }
        int ithisr = nrbar - 1, im = np - 1;
        for (int i = 0; i < np; ++i) {
            double bi = thetab[im];
            for (int jm = np - 1, j = 0; j < i; ++j) bi -= rbar[ithisr--] * P[jm--];
            P[im--] = bi;
        }
        ind = npr;
        for (int i = 0; i < r; ++i) xnext[i] = P[ind++];
        ind = np - 1;
        ind1 = npr - 1;
        for (int i = 0; i < npr; ++i) P[ind--] = P[ind1--];
        for (int i = 0; i < r; ++i) P[i] = xnext[i];
    } else {
        int indn = np, ind = np;
        for (int i = 0; i < r; ++i)
            for (int j = 0; j <= i; ++j) {
                --ind;
                P[ind] = V[ind];
                if (j != 0) P[ind] += P[--indn];
            }
    }
    for (int i = r - 1, ind = np; i > 0; --i)
        for (int j = r - 1; j >= i; --j) P[r * i + j] = P[--ind];
    for (int i = 0; i < r - 1; ++i)
        for (int j = i + 1; j < r; ++j) P[i + r * j] = P[j + r * i];
    return P;
}

// ------------------------------------------------------------------------------------------- state space
ArimaModel make_arima(const Vec& phi, Vec theta, const Vec& Delta, double kappa) {
    ArimaModel m;
    const int p = static_cast<int>(phi.size()), q = static_cast<int>(theta.size());
    m.r = std::max(p, q + 1);
    m.d = static_cast<int>(Delta.size());
    m.rd = m.r + m.d;
    if (q < m.r - 1) theta.resize(m.r - 1, 0.0);
    m.phi = phi;
    m.theta = theta;
    m.Delta = Delta;
    m.a.assign(m.rd, 0.0);
    m.P.assign(static_cast<size_t>(m.rd) * m.rd, 0.0);
    m.Pn.assign(static_cast<size_t>(m.rd) * m.rd, 0.0);
    if (m.r > 1) {
        const Vec Q0 = getQ0(phi, theta);
        for (int j = 0; j < m.r; ++j) for (int i = 0; i < m.r; ++i) m.Pn[i + m.rd * j] = Q0[i + m.r * j];
    } else {
        m.Pn[0] = (p > 0) ? 1.0 / (1.0 - phi[0] * phi[0]) : 1.0;
    }
    for (int i = 0; i < m.d; ++i) m.Pn[(m.r + i) + m.rd * (m.r + i)] = kappa;
    return m;
}

void up_arima(ArimaModel& m, const Vec& phi, const Vec& theta) {   // upARIMA (theta not padded)
    const int p = static_cast<int>(phi.size()), q = static_cast<int>(theta.size());
    m.phi = phi;
    m.theta = theta;
    const int r = std::max(p, q + 1);
    if (r > 1) {
        const Vec Q0 = getQ0(phi, theta);
        for (int j = 0; j < r; ++j) for (int i = 0; i < r; ++i) m.Pn[i + m.rd * j] = Q0[i + r * j];
    } else {
        m.Pn[0] = (p > 0) ? 1.0 / (1.0 - phi[0] * phi[0]) : 1.0;
    }
    std::fill(m.a.begin(), m.a.end(), 0.0);
}

struct LikeOut { double ssq, sumlog; int nu; };

// ARIMA_Like(y, mod, 0, giveResid): Kalman filter; updates m.a, m.P, m.Pn in place (as R does).
LikeOut arima_like(const Vec& y, ArimaModel& m, Vec* resid) {
    const int n = static_cast<int>(y.size()), rd = m.rd, r = m.r, d = m.d;
    const int p = static_cast<int>(m.phi.size()), q = static_cast<int>(m.theta.size());
    const double *phi = m.phi.data(), *theta = m.theta.data(), *delta = m.Delta.data();
    double *a = m.a.data(), *P = m.P.data(), *Pnew = m.Pn.data();
    Vec anew(rd), M(rd), mm(d > 0 ? static_cast<size_t>(rd) * rd : 0);
    double ssq = 0.0, sumlog = 0.0;
    int nu = 0;
    if (resid) resid->assign(n, 0.0);
    for (int l = 0; l < n; ++l) {
        for (int i = 0; i < r; ++i) {
            double tmp = (i < r - 1) ? a[i + 1] : 0.0;
            if (i < p) tmp += phi[i] * a[0];
            anew[i] = tmp;
        }
        if (d > 0) {
            for (int i = r + 1; i < rd; ++i) anew[i] = a[i - 1];
            double tmp = a[0];
            for (int i = 0; i < d; ++i) tmp += delta[i] * a[r + i];
            anew[r] = tmp;
        }
        if (l > 0) {
            if (d == 0) {
                for (int i = 0; i < r; ++i) {
                    const double vi = (i == 0) ? 1.0 : (i - 1 < q ? theta[i - 1] : 0.0);
                    for (int j = 0; j < r; ++j) {
                        double tmp = 0.0;
                        if (j == 0) tmp = vi; else if (j - 1 < q) tmp = vi * theta[j - 1];
                        if (i < p && j < p) tmp += phi[i] * phi[j] * P[0];
                        if (i < r - 1 && j < r - 1) tmp += P[i + 1 + r * (j + 1)];
                        if (i < p && j < r - 1) tmp += phi[i] * P[j + 1];
                        if (j < p && i < r - 1) tmp += phi[j] * P[i + 1];
                        Pnew[i + r * j] = tmp;
                    }
                }
            } else {
                for (int i = 0; i < r; ++i)                      // mm = TP
                    for (int j = 0; j < rd; ++j) {
                        double tmp = 0.0;
                        if (i < p) tmp += phi[i] * P[rd * j];
                        if (i < r - 1) tmp += P[i + 1 + rd * j];
                        mm[i + rd * j] = tmp;
                    }
                for (int j = 0; j < rd; ++j) {
                    double tmp = P[rd * j];
                    for (int k = 0; k < d; ++k) tmp += delta[k] * P[r + k + rd * j];
                    mm[r + rd * j] = tmp;
                }
                for (int i = 1; i < d; ++i)
                    for (int j = 0; j < rd; ++j) mm[r + i + rd * j] = P[r + i - 1 + rd * j];
                for (int i = 0; i < r; ++i)                      // Pnew = mm T'
                    for (int j = 0; j < rd; ++j) {
                        double tmp = 0.0;
                        if (i < p) tmp += phi[i] * mm[j];
                        if (i < r - 1) tmp += mm[rd * (i + 1) + j];
                        Pnew[j + rd * i] = tmp;
                    }
                for (int j = 0; j < rd; ++j) {
                    double tmp = mm[j];
                    for (int k = 0; k < d; ++k) tmp += delta[k] * mm[rd * (r + k) + j];
                    Pnew[j + rd * r] = tmp;
                }
                for (int i = 1; i < d; ++i)
                    for (int j = 0; j < rd; ++j) Pnew[j + rd * (r + i)] = mm[rd * (r + i - 1) + j];
                for (int i = 0; i <= q; ++i) {                   // + (1 theta) %o% (1 theta)
                    const double vi = (i == 0) ? 1. : theta[i - 1];
                    for (int j = 0; j <= q; ++j) Pnew[i + rd * j] += vi * ((j == 0) ? 1. : theta[j - 1]);
                }
            }
        }
        if (!std::isnan(y[l])) {
            double resid0 = y[l] - anew[0];
            for (int i = 0; i < d; ++i) resid0 -= delta[i] * anew[r + i];
            double gain = Pnew[0];
            for (int i = 0; i < d; ++i) gain += 2 * delta[i] * Pnew[r + i];
            for (int i = 0; i < d; ++i)
                for (int j = 0; j < d; ++j) gain += delta[i] * delta[j] * Pnew[r + i + rd * (r + j)];
            for (int i = 0; i < rd; ++i) {
                double tmp = Pnew[i];
                for (int k = 0; k < d; ++k) tmp += Pnew[i + rd * (r + k)] * delta[k];
                M[i] = tmp;
            }
            if (gain < 1e4) {
                ++nu;
                ssq += resid0 * resid0 / gain;
                sumlog += std::log(gain);
            }
            if (resid) (*resid)[l] = resid0 / std::sqrt(gain);
            for (int i = 0; i < rd; ++i) a[i] = anew[i] + M[i] * resid0 / gain;
            for (int i = 0; i < rd; ++i)
                for (int j = 0; j < rd; ++j) P[i + rd * j] = Pnew[i + rd * j] - M[i] * M[j] / gain;
        } else {
            for (int i = 0; i < rd; ++i) a[i] = anew[i];
            for (int i = 0; i < rd * rd; ++i) P[i] = Pnew[i];
            if (resid) (*resid)[l] = std::nan("");
        }
    }
    return {ssq, sumlog, nu};
}

// ARIMA_CSS: conditional sum of squares / nu; residuals optional.
double arima_css(const Vec& y, const Arma& arma, const Vec& phi, const Vec& theta, int ncond, Vec* resid_out) {
    const int n = static_cast<int>(y.size()), p = static_cast<int>(phi.size()), q = static_cast<int>(theta.size());
    Vec w = y;
    for (int i = 0; i < arma.d; ++i)
        for (int l = n - 1; l > 0; --l) w[l] -= w[l - 1];
    const int ns = arma.ns;
    for (int i = 0; i < arma.D; ++i)
        for (int l = n - 1; l >= ns; --l) w[l] -= w[l - ns];
    Vec resid(n, 0.0);
    double ssq = 0.0;
    int nu = 0;
    for (int l = ncond; l < n; ++l) {
        double tmp = w[l];
        for (int j = 0; j < p; ++j) tmp -= phi[j] * w[l - j - 1];
        const int jmax = std::min(l - ncond, q);
        for (int j = 0; j < jmax; ++j) tmp -= theta[j] * resid[l - j - 1];
        resid[l] = tmp;
        if (!std::isnan(tmp)) { ++nu; ssq += tmp * tmp; }
    }
    if (resid_out) *resid_out = resid;
    return ssq / static_cast<double>(nu);
}

// ------------------------------------------------------------------------------------------- optim (BFGS)
struct Optim {
    std::function<double(const Vec&)> fn;          // on the original scale
    Vec parscale;
    double ndeps = 1e-3;
    int fncount = 0;
    double fminfn(const Vec& p) const {             // scaled
        Vec x(p.size());
        for (size_t i = 0; i < p.size(); ++i) x[i] = p[i] * parscale[i];
        return fn(x);
    }
    void fmingr(const Vec& p, Vec& df) const {
        const size_t n = p.size();
        Vec x(n);
        for (size_t i = 0; i < n; ++i) x[i] = p[i] * parscale[i];
        df.assign(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            const double eps = ndeps;
            x[i] = (p[i] + eps) * parscale[i];
            const double val1 = fn(x);
            x[i] = (p[i] - eps) * parscale[i];
            const double val2 = fn(x);
            df[i] = (val1 - val2) / (2 * eps);
            if (!std::isfinite(df[i])) throw std::runtime_error("non-finite finite-difference value");
            x[i] = p[i] * parscale[i];
        }
    }
};

// vmmin: returns the scaled optimum in b; Fmin; fail code (0 ok, 1 maxit).
int vmmin(const Optim& os, Vec& b, double& Fmin, int maxit) {
    const double stepredn = 0.2, acctol = 0.0001, reltest = 10.0;
    const double abstol = -INF, reltol = 1.490116119384765625e-08;
    const int n = static_cast<int>(b.size());
    if (maxit <= 0) { Fmin = os.fminfn(b); return 0; }
    Vec g(n), t(n), X(n), c(n);
    std::vector<Vec> B(n, Vec(n, 0.0));
    double f = os.fminfn(b);
    if (!std::isfinite(f)) throw std::runtime_error("initial value in 'vmmin' is not finite");
    Fmin = f;
    int funcount = 1, gradcount = 1;
    os.fmingr(b, g);
    int iter = 1, ilast = gradcount, count;
    do {
        if (ilast == gradcount)
            for (int i = 0; i < n; ++i) { for (int j = 0; j < i; ++j) B[i][j] = 0.0; B[i][i] = 1.0; }
        for (int i = 0; i < n; ++i) { X[i] = b[i]; c[i] = g[i]; }
        double gradproj = 0.0;
        for (int i = 0; i < n; ++i) {
            double s = 0.0;
            for (int j = 0; j <= i; ++j) s -= B[i][j] * g[j];
            for (int j = i + 1; j < n; ++j) s -= B[j][i] * g[j];
            t[i] = s;
            gradproj += s * g[i];
        }
        if (gradproj < 0.0) {
            double steplength = 1.0;
            bool accpoint = false;
            do {
                count = 0;
                for (int i = 0; i < n; ++i) {
                    b[i] = X[i] + steplength * t[i];
                    if (reltest + X[i] == reltest + b[i]) ++count;
                }
                if (count < n) {
                    f = os.fminfn(b);
                    ++funcount;
                    accpoint = std::isfinite(f) && (f <= Fmin + gradproj * steplength * acctol);
                    if (!accpoint) steplength *= stepredn;
                }
            } while (!(count == n || accpoint));
            const bool enough = (f > abstol) && std::fabs(f - Fmin) > reltol * (std::fabs(Fmin) + reltol);
            if (!enough) { count = n; Fmin = f; }
            if (count < n) {
                Fmin = f;
                os.fmingr(b, g);
                ++gradcount;
                ++iter;
                double D1 = 0.0;
                for (int i = 0; i < n; ++i) { t[i] = steplength * t[i]; c[i] = g[i] - c[i]; D1 += t[i] * c[i]; }
                if (D1 > 0) {
                    double D2 = 0.0;
                    for (int i = 0; i < n; ++i) {
                        double s = 0.0;
                        for (int j = 0; j <= i; ++j) s += B[i][j] * c[j];
                        for (int j = i + 1; j < n; ++j) s += B[j][i] * c[j];
                        X[i] = s;
                        D2 += s * c[i];
                    }
                    D2 = 1.0 + D2 / D1;
                    for (int i = 0; i < n; ++i)
                        for (int j = 0; j <= i; ++j) B[i][j] += (D2 * t[i] * t[j] - X[i] * t[j] - t[i] * X[j]) / D1;
                } else {
                    ilast = gradcount;
                }
            } else {
                if (ilast < gradcount) { count = 0; ilast = gradcount; }
            }
        } else {
            count = 0;
            if (ilast == gradcount) count = n; else ilast = gradcount;
        }
        if (iter >= maxit) break;
        if (gradcount - ilast > 2 * n) ilast = gradcount;
    } while (count != n || ilast != gradcount);
    (void)funcount;
    return (iter < maxit) ? 0 : 1;
}

// optimhess at the (unscaled) point par: returns H (n x n) on the original scale.
std::vector<Vec> optimhess(const Optim& os, const Vec& par) {
    const size_t n = par.size();
    Vec dpar(n), df1, df2;
    for (size_t i = 0; i < n; ++i) dpar[i] = par[i] / os.parscale[i];
    std::vector<Vec> H(n, Vec(n));
    for (size_t i = 0; i < n; ++i) {
        const double eps = os.ndeps / os.parscale[i];
        dpar[i] += eps;
        os.fmingr(dpar, df1);
        dpar[i] -= 2 * eps;
        os.fmingr(dpar, df2);
        for (size_t j = 0; j < n; ++j) H[i][j] = (df1[j] - df2[j]) / (2 * eps * os.parscale[i] * os.parscale[j]);
        dpar[i] += eps;
    }
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < i; ++j) { const double tmp = 0.5 * (H[i][j] + H[j][i]); H[i][j] = H[j][i] = tmp; }
    return H;
}

// solve(H) via Gauss-Jordan with partial pivoting; returns false if singular (R would error).
bool invert(std::vector<Vec> A, std::vector<Vec>& inv) {
    const int n = static_cast<int>(A.size());
    inv.assign(n, Vec(n, 0.0));
    for (int i = 0; i < n; ++i) inv[i][i] = 1.0;
    double scale = 0.0;
    for (const Vec& r : A) for (double v : r) scale = std::max(scale, std::fabs(v));
    for (int c = 0; c < n; ++c) {
        int piv = c;
        for (int r = c + 1; r < n; ++r) if (std::fabs(A[r][c]) > std::fabs(A[piv][c])) piv = r;
        if (!(std::fabs(A[piv][c]) > 1e-14 * std::max(scale, 1e-300))) return false;   // LAPACK-like rcond guard
        std::swap(A[c], A[piv]);
        std::swap(inv[c], inv[piv]);
        const double d = A[c][c];
        for (int k = 0; k < n; ++k) { A[c][k] /= d; inv[c][k] /= d; }
        for (int r = 0; r < n; ++r) {
            if (r == c) continue;
            const double f = A[r][c];
            if (f == 0.0) continue;
            for (int k = 0; k < n; ++k) { A[r][k] -= f * A[c][k]; inv[r][k] -= f * inv[c][k]; }
        }
    }
    return true;
}

}  // namespace

// ------------------------------------------------------------------------------------------- stats::arima
ArimaFit arima_fit(const Vec& x0, const ArimaSpec& s) {
    Vec x = x0;
    const int n = static_cast<int>(x.size());
    const Arma arma{s.p, s.q, s.P, s.Q, s.period, s.d, s.D};
    const int narma = s.p + s.q + s.P + s.Q;
    Vec Delta{1.0};
    for (int i = 0; i < s.d; ++i) Delta = conv(Delta, {1.0, -1.0});
    for (int i = 0; i < s.D; ++i) { Vec sd(s.period + 1, 0.0); sd[0] = 1.0; sd[s.period] = -1.0; Delta = conv(Delta, sd); }
    Vec Dl(Delta.begin() + 1, Delta.end());
    for (double& v : Dl) v = -v;
    const int nd = s.d + s.D;
    int n_used = n - static_cast<int>(Dl.size());
    // regressor: drift (1..n) or intercept (include.mean and no differencing); at most one column here
    Vec xreg;
    bool has_intercept = false, has_drift = false;
    if (s.drift) { xreg.resize(n); for (int i = 0; i < n; ++i) xreg[i] = i + 1; has_drift = true; }
    else if (s.include_mean && nd == 0) { xreg.assign(n, 1.0); has_intercept = true; }
    const int ncxreg = xreg.empty() ? 0 : 1;
    ArimaMethod method = s.method;
    int ncond = 0;
    if (method == ArimaMethod::CSS || method == ArimaMethod::CSS_ML)
        ncond = s.d + s.D * s.period + (s.p + s.period * s.P);
    Vec init0(narma, 0.0), parscale(narma, 1.0);
    if (ncxreg) {
        Vec dx = x, dxr = xreg;
        auto diffv = [](Vec v, int lag, int times) {
            for (int k = 0; k < times; ++k) {
                Vec o(v.size() > static_cast<size_t>(lag) ? v.size() - lag : 0);
                for (size_t i = 0; i < o.size(); ++i) o[i] = v[i + lag] - v[i];
                v = o;
            }
            return v;
        };
        if (s.d > 0) { dx = diffv(dx, 1, s.d); dxr = diffv(dxr, 1, s.d); }
        if (s.period > 1 && s.D > 0) { dx = diffv(dx, s.period, s.D); dxr = diffv(dxr, s.period, s.D); }
        auto lm1 = [](const Vec& yv, const Vec& xv, double& coef, double& se) {   // lm(y ~ x - 1)
            double sxy = 0, sxx = 0;
            for (size_t i = 0; i < yv.size(); ++i) { sxy += xv[i] * yv[i]; sxx += xv[i] * xv[i]; }
            if (!(sxx > 0)) return false;
            coef = sxy / sxx;
            double rss = 0;
            for (size_t i = 0; i < yv.size(); ++i) { const double e = yv[i] - coef * xv[i]; rss += e * e; }
            const double df = static_cast<double>(yv.size()) - 1;
            se = df > 0 ? std::sqrt(rss / df / sxx) : std::nan("");
            return true;
        };
        double coef = 0, se = 0;
        bool okfit = dx.size() > 1 && lm1(dx, dxr, coef, se);
        if (!okfit) lm1(x, xreg, coef, se);
        init0.push_back(coef);
        parscale.push_back(10 * se);
    }
    if (n_used <= 0) throw std::runtime_error("too few non-missing observations");
    Vec init = init0;
    const int npar = narma + ncxreg;
    Vec coef(npar, 0.0);
    Vec xr = x;                                         // x minus regression (set per evaluation)
    auto resid_x = [&](const Vec& par) {
        if (!ncxreg) return x;
        Vec o(n);
        for (int i = 0; i < n; ++i) o[i] = x[i] - xreg[i] * par[narma];
        return o;
    };
    auto armaCSS = [&](const Vec& par) {
        Vec phi, theta;
        trans_pars(par, arma, false, phi, theta);
        const double res = arima_css(resid_x(par), arma, phi, theta, ncond, nullptr);
        return 0.5 * std::log(res);
    };
    ArimaFit fit;
    fit.narma = narma; fit.ncxreg = ncxreg; fit.has_intercept = has_intercept; fit.has_drift = has_drift; fit.n = n;
    Optim os;
    os.parscale = parscale;
    std::vector<Vec> var;
    double value;
    ArimaModel mod;
    Vec phi, theta;
    if (method == ArimaMethod::CSS) {
        os.fn = armaCSS;
        Vec b(npar);
        for (int i = 0; i < npar; ++i) b[i] = init[i] / parscale[i];
        double fmin;
        fit.convergence = vmmin(os, b, fmin, 100);
        for (int i = 0; i < npar; ++i) coef[i] = b[i] * parscale[i];
        value = fmin;
        trans_pars(coef, arma, false, phi, theta);
        mod = make_arima(phi, theta, Dl, 1e6);
        const Vec xx = resid_x(coef);
        arima_like(xx, mod, nullptr);                   // arimaSS(x, mod): final state for forecasting
        Vec res;
        fit.sigma2 = arima_css(xx, arma, phi, theta, ncond, &res);
        fit.residuals = res;
        const std::vector<Vec> H = optimhess(os, coef);
        std::vector<Vec> Hn = H;
        for (auto& r : Hn) for (double& v : r) v *= n_used;
        fit.var_ok = invert(Hn, var);
    } else {
        if (method == ArimaMethod::CSS_ML) {
            os.fn = armaCSS;
            Vec b(npar);
            for (int i = 0; i < npar; ++i) b[i] = init[i] / parscale[i];
            double fmin;
            const int conv = vmmin(os, b, fmin, 100);
            if (conv == 0) for (int i = 0; i < npar; ++i) init[i] = b[i] * parscale[i];
            if (s.p > 0 && !ar_check(Vec(init.begin(), init.begin() + s.p))) throw std::runtime_error("non-stationary AR part from CSS");
            if (s.P > 0 && !ar_check(Vec(init.begin() + s.p + s.q, init.begin() + s.p + s.q + s.P)))
                throw std::runtime_error("non-stationary seasonal AR part from CSS");
            ncond = 0;
        }
        // transform.pars = TRUE
        init = inv_trans(init, arma);
        if (s.q > 0) { Vec v(init.begin() + s.p, init.begin() + s.p + s.q); v = ma_invert(v); std::copy(v.begin(), v.end(), init.begin() + s.p); }
        if (s.Q > 0) {
            const int o = s.p + s.q + s.P;
            Vec v(init.begin() + o, init.begin() + o + s.Q);
            v = ma_invert(v);
            std::copy(v.begin(), v.end(), init.begin() + o);
        }
        trans_pars(init, arma, true, phi, theta);
        mod = make_arima(phi, theta, Dl, 1e6);
        auto armafn = [&](const Vec& par) {
            Vec ph, th;
            trans_pars(par, arma, true, ph, th);
            ArimaModel Z = mod;
            try { up_arima(Z, ph, th); } catch (...) { return 1.7976931348623157e308; }
            const LikeOut lo = arima_like(resid_x(par), Z, nullptr);
            const double s2 = lo.ssq / lo.nu;
            return 0.5 * (std::log(s2) + lo.sumlog / lo.nu);
        };
        os.fn = armafn;
        Vec b(npar);
        for (int i = 0; i < npar; ++i) b[i] = init[i] / parscale[i];
        double fmin;
        fit.convergence = vmmin(os, b, fmin, 100);
        for (int i = 0; i < npar; ++i) coef[i] = b[i] * parscale[i];
        value = fmin;
        const Vec respar = coef;
        if (s.q > 0) { Vec v(coef.begin() + s.p, coef.begin() + s.p + s.q); v = ma_invert(v); std::copy(v.begin(), v.end(), coef.begin() + s.p); }
        if (s.Q > 0) {
            const int o = s.p + s.q + s.P;
            Vec v(coef.begin() + o, coef.begin() + o + s.Q);
            v = ma_invert(v);
            std::copy(v.begin(), v.end(), coef.begin() + o);
        }
        if (coef != respar) value = armafn(coef);       // optim(maxit = 0): value at the new point
        const std::vector<Vec> H = optimhess(os, coef);
        const std::vector<Vec> A = grad_trans(coef, arma);
        std::vector<Vec> Hn = H, Hinv;
        for (auto& r : Hn) for (double& v : r) v *= n_used;
        fit.var_ok = invert(Hn, Hinv);
        if (fit.var_ok) {                               // var = A' solve(H n) A
            var.assign(npar, Vec(npar, 0.0));
            for (int i = 0; i < npar; ++i)
                for (int j = 0; j < npar; ++j) {
                    double sacc = 0;
                    for (int k = 0; k < npar; ++k)
                        for (int l = 0; l < npar; ++l) sacc += A[k][i] * Hinv[k][l] * A[l][j];
                    var[i][j] = sacc;
                }
        }
        coef = undo_pars(coef, arma);
        trans_pars(coef, arma, false, phi, theta);
        mod = make_arima(phi, theta, Dl, 1e6);
        Vec res;
        const LikeOut lo = arima_like(resid_x(coef), mod, &res);
        fit.sigma2 = lo.ssq / n_used;
        fit.residuals = res;
    }
    const double val = 2 * n_used * value + n_used + n_used * std::log(2 * 3.14159265358979323846);
    fit.loglik = -0.5 * val;
    fit.aic = (method != ArimaMethod::CSS) ? val + 2 * npar + 2 : std::nan("");
    fit.coef = coef;
    fit.n_used = n_used;
    fit.model = mod;
    fit.var_diag.assign(npar, std::nan(""));
    if (fit.var_ok) for (int i = 0; i < npar; ++i) fit.var_diag[i] = var[i][i];
    fit.ok = true;
    return fit;
}

Vec arima_forecast(const ArimaFit& fit, int h) {
    const ArimaModel& m = fit.model;
    const int rd = m.rd, r = m.r, d = m.d, p = static_cast<int>(m.phi.size());
    // T and Z as makeARIMA builds them
    std::vector<Vec> T(rd, Vec(rd, 0.0));
    for (int i = 0; i < p; ++i) T[i][0] = m.phi[i];
    for (int i = 1; i < r; ++i) T[i - 1][i] = 1.0;
    Vec Z(rd, 0.0);
    Z[0] = 1.0;
    for (int i = 0; i < d; ++i) Z[r + i] = m.Delta[i];
    if (d > 0) {
        T[r] = Z;
        for (int i = 1; i < d; ++i) T[r + i][r + i - 1] = 1.0;
    }
    Vec a = m.a, an(rd);
    Vec f(h);
    for (int l = 0; l < h; ++l) {
        double fc = 0.0;
        for (int i = 0; i < rd; ++i) {
            double t = 0.0;
            for (int k = 0; k < rd; ++k) t += T[i][k] * a[k];
            an[i] = t;
            fc += t * Z[i];
        }
        a = an;
        f[l] = fc;
    }
    if (fit.has_intercept) for (double& v : f) v += fit.coef[fit.narma];
    if (fit.has_drift) for (int l = 0; l < h; ++l) f[l] += fit.coef[fit.narma] * (fit.n + l + 1);
    return f;
}

}  // namespace tsf
