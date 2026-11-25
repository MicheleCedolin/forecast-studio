// Port of R forecast 9.0.2's tbats() / bats() (R and Rcpp code, GPL-3, Rob J Hyndman and contributors). See tbats.hpp.
#include "tbats.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "arima.hpp"
#include "autoarima.hpp"
#include "r_optim.hpp"

namespace tsf {

namespace {

constexpr double PI = 3.141592653589793238462643383280;
constexpr double NA = std::numeric_limits<double>::quiet_NaN();

// ------------------------------------------------------------------------------------------------ R helpers
double r_sum(const Vec& v) { long double s = 0; for (double x : v) s += x; return static_cast<double>(s); }
double r_mean(const Vec& v) {
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
double r_sd(const Vec& v) {
    const size_t n = v.size();
    if (n < 2) return NA;
    const double m = r_mean(v);
    long double s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(static_cast<double>(s / static_cast<long double>(n - 1)));
}

double box_cox(double x, double lam) {
    if (lam < 0 && x < 0) return NA;
    if (lam == 0) return std::log(x);
    const double s = (x > 0) ? 1.0 : (x < 0 ? -1.0 : 0.0);
    return (s * std::pow(std::fabs(x), lam) - 1) / lam;
}
double inv_box_cox(double x, double lam) {
    if (lam < 0 && x > -1 / lam) return NA;
    if (lam == 0) return std::exp(x);
    const double xx = x * lam + 1;
    const double s = (xx > 0) ? 1.0 : (xx < 0 ? -1.0 : 0.0);
    return s * std::pow(std::fabs(xx), 1 / lam);
}

// R's optimize(): Brent_fmin (src/library/stats/src/optimize.c), tol = .Machine$double.eps^0.25
template <class F>
double brent_fmin(double ax, double bx, F f, double tol) {
    const double c = (3. - std::sqrt(5.)) * .5;
    double eps = std::sqrt(2.220446049250313e-16);
    double a = ax, b = bx, v = a + c * (b - a), w = v, x = v, d = 0., e = 0.;
    double fx = f(x), fv = fx, fw = fx;
    const double tol3 = tol / 3.;
    for (;;) {
        const double xm = (a + b) * .5, tol1 = eps * std::fabs(x) + tol3, t2 = tol1 * 2.;
        if (std::fabs(x - xm) <= t2 - (b - a) * .5) break;
        double p = 0., q = 0., r = 0.;
        if (std::fabs(e) > tol1) {
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = (q - r) * 2.;
            if (q > 0.) p = -p; else q = -q;
            r = e;
            e = d;
        }
        double u;
        if (std::fabs(p) >= std::fabs(q * .5 * r) || p <= q * (a - x) || p >= q * (b - x)) {
            e = (x < xm) ? b - x : a - x;
            d = c * e;
        } else {
            d = p / q;
            u = x + d;
            if (u - a < t2 || b - u < t2) { d = tol1; if (x >= xm) d = -d; }
        }
        if (std::fabs(d) >= tol1) u = x + d;
        else if (d > 0.) u = x + tol1;
        else u = x - tol1;
        const double fu = f(u);
        if (fu <= fx) {
            if (u < x) b = x; else a = x;
            v = w; w = x; x = u;
            fv = fw; fw = fx; fx = fu;
        } else {
            if (u < x) a = u; else b = u;
            if (fu <= fw || w == x) { v = w; fv = fw; w = u; fw = fu; }
            else if (fu <= fv || v == x || v == w) { v = u; fv = fu; }
        }
    }
    return x;
}

// forecast::BoxCox.lambda(x, method = "guerrero", lower, upper) for a series of frequency freq
double box_cox_lambda(const Vec& x, double lower, double upper, double freq) {
    for (double v : x) if (v <= 0) { lower = std::max(lower, 0.0); break; }
    const int n = static_cast<int>(x.size());
    if (n <= 2 * freq) return 1.0;
    const int period = static_cast<int>(std::nearbyint(std::max(2.0, freq)));
    const int nyr = n / period, nobst = nyr * period;
    auto cv = [&](double lam) {
        Vec rat(nyr);
        for (int j = 0; j < nyr; ++j) {
            Vec col(x.begin() + (n - nobst) + j * period, x.begin() + (n - nobst) + (j + 1) * period);
            rat[j] = r_sd(col) / std::pow(r_mean(col), 1 - lam);
        }
        return r_sd(rat) / r_mean(rat);
    };
    return brent_fmin(lower, upper, cv, std::pow(2.220446049250313e-16, 0.25));
}

// moduli of the eigenvalues of a real square matrix (elimination to Hessenberg form, then the shifted QR iteration of
// Numerical Recipes' hqr); used only for the admissibility test max |lambda| < 1.01 (R: eigen(D)$values)
//
// Before that, eigenvalues that a permutation isolates are split off (the permutation step of EISPACK's balanc, which
// LAPACK's dgebal applies inside R's eigen()): a row or a column of the active block whose off-diagonal elements are
// all zero gives the eigenvalue on its diagonal. Without this step the QR iteration does not converge on the matrix
// D of a model whose AR coefficients are still at their starting value 0: the shift rows of the AR states form a
// nilpotent (defective) block, of size 5 for ARMA(5, 1) errors (NN5 daily T35, the candidate R selects).
bool eigen_moduli_below(std::vector<Vec> a, double bound) {
    {
        int lo = 0, hi = static_cast<int>(a.size()) - 1;
        auto exchange = [&](int j, int k) {
            if (j == k) return;
            std::swap(a[j], a[k]);
            for (auto& row : a) std::swap(row[j], row[k]);
        };
        for (bool found = true; found && lo <= hi;) {
            found = false;
            for (int j = hi; j >= lo && !found; --j) {  // a row of zeros: its eigenvalue goes to the bottom
                bool zero = true;
                for (int i = lo; i <= hi && zero; ++i) if (i != j && a[j][i] != 0.0) zero = false;
                if (zero) {
                    if (!(std::fabs(a[j][j]) < bound)) return false;
                    exchange(j, hi);
                    --hi;
                    found = true;
                }
            }
            for (int j = lo; j <= hi && !found; ++j) {  // a column of zeros: its eigenvalue goes to the top
                bool zero = true;
                for (int i = lo; i <= hi && zero; ++i) if (i != j && a[i][j] != 0.0) zero = false;
                if (zero) {
                    if (!(std::fabs(a[j][j]) < bound)) return false;
                    exchange(j, lo);
                    ++lo;
                    found = true;
                }
            }
        }
        if (lo > 0 || hi < static_cast<int>(a.size()) - 1) {            // keep the block that is left
            std::vector<Vec> b;
            for (int i = lo; i <= hi; ++i) b.emplace_back(a[i].begin() + lo, a[i].begin() + hi + 1);
            a.swap(b);
        }
    }
    const int n = static_cast<int>(a.size());
    if (n == 0) return true;
    for (int m = 1; m < n - 1; ++m) {                  // elmhes
        double x = 0.0;
        int i = m;
        for (int j = m; j < n; ++j) if (std::fabs(a[j][m - 1]) > std::fabs(x)) { x = a[j][m - 1]; i = j; }
        if (i != m) {
            for (int j = m - 1; j < n; ++j) std::swap(a[i][j], a[m][j]);
            for (int j = 0; j < n; ++j) std::swap(a[j][i], a[j][m]);
        }
        if (x != 0.0)
            for (i = m + 1; i < n; ++i) {
                double y = a[i][m - 1];
                if (y != 0.0) {
                    y /= x;
                    a[i][m - 1] = y;
                    for (int j = m; j < n; ++j) a[i][j] -= y * a[m][j];
                    for (int j = 0; j < n; ++j) a[j][m] += y * a[j][i];
                }
            }
    }
    for (int i = 2; i < n; ++i) for (int j = 0; j < i - 1; ++j) a[i][j] = 0.0;
    Vec wr(n, 0.0), wi(n, 0.0);
    double anorm = 0.0;
    for (int i = 0; i < n; ++i) for (int j = std::max(i - 1, 0); j < n; ++j) anorm += std::fabs(a[i][j]);
    int nn = n - 1;
    double t = 0.0;
    auto sgn = [](double aa, double bb) { return bb >= 0 ? std::fabs(aa) : -std::fabs(aa); };
    while (nn >= 0) {
        int its = 0, l;
        do {
            for (l = nn; l > 0; --l) {
                double s = std::fabs(a[l - 1][l - 1]) + std::fabs(a[l][l]);
                if (s == 0.0) s = anorm;
                if (std::fabs(a[l][l - 1]) + s == s) { a[l][l - 1] = 0.0; break; }
            }
            double x = a[nn][nn];
            if (l == nn) {
                wr[nn] = x + t; wi[nn] = 0.0; --nn;
            } else {
                double y = a[nn - 1][nn - 1], w = a[nn][nn - 1] * a[nn - 1][nn];
                if (l == nn - 1) {
                    const double p = 0.5 * (y - x), q = p * p + w;
                    double z = std::sqrt(std::fabs(q));
                    x += t;
                    if (q >= 0.0) {
                        z = p + sgn(z, p);
                        wr[nn - 1] = wr[nn] = x + z;
                        if (z != 0.0) wr[nn] = x - w / z;
                        wi[nn - 1] = wi[nn] = 0.0;
                    } else {
                        wr[nn - 1] = wr[nn] = x + p;
                        wi[nn - 1] = -(wi[nn] = z);
                    }
                    nn -= 2;
                } else {
                    if (its == 60) throw std::runtime_error("eigen: no convergence");
                    if (its == 10 || its == 20) {
                        t += x;
                        for (int i = 0; i <= nn; ++i) a[i][i] -= x;
                        const double s = std::fabs(a[nn][nn - 1]) + std::fabs(a[nn - 1][nn - 2]);
                        y = x = 0.75 * s;
                        w = -0.4375 * s * s;
                    }
                    ++its;
                    int m;
                    double p = 0, q = 0, r = 0, z;
                    for (m = nn - 2; m >= l; --m) {
                        z = a[m][m];
                        r = x - z;
                        double s = y - z;
                        p = (r * s - w) / a[m + 1][m] + a[m][m + 1];
                        q = a[m + 1][m + 1] - z - r - s;
                        r = a[m + 2][m + 1];
                        s = std::fabs(p) + std::fabs(q) + std::fabs(r);
                        p /= s; q /= s; r /= s;
                        if (m == l) break;
                        const double u = std::fabs(a[m][m - 1]) * (std::fabs(q) + std::fabs(r));
                        const double v = std::fabs(p) * (std::fabs(a[m - 1][m - 1]) + std::fabs(z) + std::fabs(a[m + 1][m + 1]));
                        if (u + v == v) break;
                    }
                    for (int i = m + 2; i <= nn; ++i) {
                        a[i][i - 2] = 0.0;
                        if (i != m + 2) a[i][i - 3] = 0.0;
                    }
                    for (int k = m; k <= nn - 1; ++k) {
                        if (k != m) {
                            p = a[k][k - 1];
                            q = a[k + 1][k - 1];
                            r = 0.0;
                            if (k != nn - 1) r = a[k + 2][k - 1];
                            if ((x = std::fabs(p) + std::fabs(q) + std::fabs(r)) != 0.0) { p /= x; q /= x; r /= x; }
                        }
                        const double s = sgn(std::sqrt(p * p + q * q + r * r), p);
                        if (s != 0.0) {
                            if (k == m) { if (l != m) a[k][k - 1] = -a[k][k - 1]; }
                            else a[k][k - 1] = -s * x;
                            p += s;
                            x = p / s; y = q / s; z = r / s;
                            q /= p; r /= p;
                            for (int j = k; j <= nn; ++j) {
                                p = a[k][j] + q * a[k + 1][j];
                                if (k != nn - 1) { p += r * a[k + 2][j]; a[k + 2][j] -= p * z; }
                                a[k + 1][j] -= p * y;
                                a[k][j] -= p * x;
                            }
                            const int mmin = nn < k + 3 ? nn : k + 3;
                            for (int i = l; i <= mmin; ++i) {
                                p = x * a[i][k] + y * a[i][k + 1];
                                if (k != nn - 1) { p += z * a[i][k + 2]; a[i][k + 2] -= p * r; }
                                a[i][k + 1] -= p * q;
                                a[i][k] -= p;
                            }
                        }
                    }
                }
            }
        } while (nn >= 0 && l < nn - 1);
    }
    for (int i = 0; i < n; ++i) if (!(std::hypot(wr[i], wi[i]) < bound)) return false;
    return true;
}

double min_root(const Vec& coefs, double sign) {       // min(Mod(polyroot(c(1, sign * coefs[1:last])))) over |c| > 1e-8
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

// ------------------------------------------------------------------------------------------------ model
struct Params {
    bool boxcox = false; double lambda = 0;
    double alpha = 0;
    bool has_beta = false; double beta = 0;       // trend
    bool has_phi = false; double phi = 1;         // damping parameter present (== has_beta); phi = 1 if not damped
    Vec g1, g2;                                   // per period (TBATS)
    Vec ar, ma;
};

struct Layout { std::vector<int> k; Vec periods; int tau = 0, p = 0, q = 0, hb = 0, N = 0; };

Layout make_layout(const std::vector<int>& k, const Vec& periods, int p, int q, bool beta) {
    Layout L;
    L.k = k; L.periods = periods; L.p = p; L.q = q; L.hb = beta ? 1 : 0;
    for (int v : k) L.tau += 2 * v;
    L.N = 1 + L.hb + L.tau + p + q;
    return L;
}

struct System { Vec w, g; std::vector<Vec> F; };

System build(const Layout& L, const Params& P) {
    System S;
    const int N = L.N, hb = L.hb, tau = L.tau, p = L.p, q = L.q;
    S.w.assign(N, 0.0);
    S.w[0] = 1.0;
    int pos = hb;
    for (size_t s = 0; s < L.k.size(); ++s) {
        for (int j = pos + 1; j <= pos + L.k[s]; ++j) S.w[j] = 1.0;
        pos += 2 * L.k[s];
    }
    if (hb) S.w[1] = P.phi;
    for (int i = 1; i <= p; ++i) S.w[hb + tau + i] = P.ar[i - 1];
    for (int i = 1; i <= q; ++i) S.w[hb + tau + p + i] = P.ma[i - 1];
    Vec gb(tau, 0.0);                                   // updateTBATSGammaBold
    int endPos = 0;
    for (size_t i = 0; i < L.k.size(); ++i) {
        for (int j = endPos; j < L.k[i] + endPos; ++j) gb[j] = P.g1[i];
        for (int j = L.k[i] + endPos; j < 2 * L.k[i] + endPos; ++j) gb[j] = P.g2[i];
        endPos += 2 * L.k[i];
    }
    S.g.assign(N, 0.0);
    if (p) S.g[1 + hb + tau] = 1.0;
    if (q) S.g[1 + hb + tau + p] = 1.0;
    S.g[0] = P.alpha;
    if (hb) S.g[1] = P.beta;
    for (int j = 0; j < tau; ++j) S.g[hb + 1 + j] = gb[j];
    S.F.assign(N, Vec(N, 0.0));
    auto& F = S.F;
    const int a0 = hb + tau + 1, m0 = hb + tau + p + 1;
    F[0][0] = 1.0;
    if (hb) F[0][1] = P.phi;
    for (int j = 0; j < p; ++j) F[0][a0 + j] = P.alpha * P.ar[j];
    for (int j = 0; j < q; ++j) F[0][m0 + j] = P.alpha * P.ma[j];
    if (hb) {
        F[1][1] = P.phi;
        for (int j = 0; j < p; ++j) F[1][a0 + j] = P.beta * P.ar[j];
        for (int j = 0; j < q; ++j) F[1][m0 + j] = P.beta * P.ma[j];
    }
    int last = 0;
    for (size_t i = 0; i < L.k.size(); ++i) {           // block-diagonal A, seasonal rows
        const int kk = L.k[i];
        const double m = L.periods[i];
        const int o = hb + 1 + last;
        for (int j = 1; j <= kk; ++j) {
            const double c = (m != 2) ? std::cos((2 * PI * j) / m) : 0.0;
            const double s = std::sin((2 * PI * j) / m);
            F[o + j - 1][o + j - 1] = c;
            F[o + j - 1][o + kk + j - 1] = s;
            F[o + kk + j - 1][o + j - 1] = -s;
            F[o + kk + j - 1][o + kk + j - 1] = c;
        }
        last += 2 * kk;
    }
    for (int r = 0; r < tau; ++r) {
        for (int j = 0; j < p; ++j) F[hb + 1 + r][a0 + j] = gb[r] * P.ar[j];
        for (int j = 0; j < q; ++j) F[hb + 1 + r][m0 + j] = gb[r] * P.ma[j];
    }
    if (p) {
        for (int j = 0; j < p; ++j) F[a0][a0 + j] = P.ar[j];
        for (int i = 1; i < p; ++i) F[a0 + i][a0 + i - 1] = 1.0;
        for (int j = 0; j < q; ++j) F[a0][m0 + j] = P.ma[j];
    }
    for (int i = 1; i < q; ++i) F[m0 + i][m0 + i - 1] = 1.0;
    return S;
}

struct Sparse { std::vector<std::vector<std::pair<int, double>>> rows; };
Sparse sparse_of(const std::vector<Vec>& F) {
    Sparse s;
    s.rows.resize(F.size());
    for (size_t i = 0; i < F.size(); ++i)
        for (size_t j = 0; j < F[i].size(); ++j) if (F[i][j] != 0.0) s.rows[i].push_back({static_cast<int>(j), F[i][j]});
    return s;
}

// calcTBATSFaster / calcModel: e (and the last state)
void filter(const System& S, const Sparse& Fs, const Vec& y, const Vec& x0, Vec& e, Vec& xlast) {
    const size_t n = y.size(), N = x0.size();
    std::vector<std::pair<int, double>> wnz;
    for (size_t j = 0; j < N; ++j) if (S.w[j] != 0.0) wnz.push_back({static_cast<int>(j), S.w[j]});
    e.assign(n, 0.0);
    Vec x = x0, xn(N);
    for (size_t t = 0; t < n; ++t) {
        double yh = 0.0;
        for (const auto& [j, v] : wnz) yh += v * x[j];
        e[t] = y[t] - yh;
        for (size_t i = 0; i < N; ++i) {
            double s = 0.0;
            for (const auto& [j, v] : Fs.rows[i]) s += v * x[j];
            xn[i] = s + S.g[i] * e[t];
        }
        x.swap(xn);
    }
    xlast = x;
}

bool admissible(const System& S, const Params& P, bool check_lambda) {
    if (check_lambda && (P.lambda <= 0 || P.lambda >= 1)) return false;
    if (P.has_phi && (P.phi < 0.8 || P.phi > 1)) return false;
    if (!P.ar.empty() && min_root(P.ar, -1.0) < 1.01) return false;
    if (!P.ma.empty() && min_root(P.ma, 1.0) < 1.01) return false;
    const size_t N = S.w.size();
    std::vector<Vec> D(N, Vec(N));
    for (size_t i = 0; i < N; ++i) for (size_t j = 0; j < N; ++j) D[i][j] = S.F[i][j] - S.g[i] * S.w[j];
    return eigen_moduli_below(D, 1.01);
}

struct Fitted {
    double aic = INF;
    bool ok = false;
    Params P;
    Layout L;
    Vec xlast, errors;
};

// parameterise() / unParameteriseTBATS(): [lambda], alpha, [phi if damped], [beta], gamma1..., gamma2..., ar..., ma...
Vec pack(const Params& P, bool damped) {
    Vec v;
    if (P.boxcox) v.push_back(P.lambda);
    v.push_back(P.alpha);
    if (P.has_beta) { if (damped) v.push_back(P.phi); v.push_back(P.beta); }
    for (double g : P.g1) v.push_back(g);
    for (double g : P.g2) v.push_back(g);
    for (double a : P.ar) v.push_back(a);
    for (double m : P.ma) v.push_back(m);
    return v;
}
Params unpack(const Vec& v, const Params& tmpl, bool damped) {
    Params P = tmpl;
    size_t j = 0;
    if (P.boxcox) P.lambda = v[j++];
    P.alpha = v[j++];
    if (P.has_beta) { P.phi = damped ? v[j++] : 1.0; P.beta = v[j++]; }
    for (double& g : P.g1) g = v[j++];
    for (double& g : P.g2) g = v[j++];
    for (double& a : P.ar) a = v[j++];
    for (double& m : P.ma) m = v[j++];
    return P;
}

// fitSpecificTBATS (bats = false) / fitSpecificBATS without seasonal periods (bats = true)
Fitted fit_specific(const Vec& y, bool use_bc, bool use_beta, bool use_damp, const std::vector<int>& k, const Vec& periods,
                    int p, int q, double init_lambda, bool bats) {
    const int n = static_cast<int>(y.size());
    Params P;
    P.alpha = 0.09;
    if (use_beta) { P.has_beta = true; P.has_phi = true; P.beta = 0.05; P.phi = use_damp ? 0.999 : 1.0; }
    else use_damp = false;
    P.g1.assign(k.size(), 0.0);
    P.g2.assign(k.size(), 0.0);
    P.ar.assign(p, 0.0);
    P.ma.assign(q, 0.0);
    if (use_bc) {
        P.boxcox = true;
        P.lambda = std::isnan(init_lambda) ? box_cox_lambda(y, 0, 1.5, 1) : init_lambda;
    }
    const bool damped = P.has_beta && P.phi != 1;       // parameterise(): small.phi != 1
    const Layout L = make_layout(k, periods, p, q, P.has_beta);
    Vec par = pack(P, damped);
    Vec scale;                                          // makeParscale / makeParscaleBATS
    if (use_bc) { scale.push_back(0.001); scale.push_back(bats ? 0.1 : 0.01); }
    else scale.push_back(bats ? 0.1 : 0.01);
    if (P.has_beta) { if (damped) scale.push_back(0.01); scale.push_back(0.01); }
    for (size_t i = 0; i < 2 * k.size(); ++i) scale.push_back(bats ? 0.01 : 1e-05);
    for (int i = 0; i < p + q; ++i) scale.push_back(0.1);
    System S = build(L, P);
    Vec ytr = y;
    if (use_bc) for (double& v : ytr) v = box_cox(v, P.lambda);
    Vec e, xl;
    filter(S, sparse_of(S.F), ytr, Vec(L.N, 0.0), e, xl);
    // w.tilda: row t = row t-1 * D; drop the ARMA columns; x0 = lm(e ~ w.tilda - 1)
    std::vector<Vec> D(L.N, Vec(L.N));
    for (int i = 0; i < L.N; ++i) for (int j = 0; j < L.N; ++j) D[i][j] = S.F[i][j] - S.g[i] * S.w[j];
    const int nc = L.N - p - q;
    std::vector<Vec> cols(nc, Vec(n));
    Vec row = S.w, nrow(L.N);
    for (int t = 0; t < n; ++t) {
        for (int j = 0; j < nc; ++j) cols[j][t] = row[j];
        for (int j = 0; j < L.N; ++j) {
            double s = 0.0;
            for (int i = 0; i < L.N; ++i) s += row[i] * D[i][j];
            nrow[j] = s;
        }
        row.swap(nrow);
    }
    Vec coef, se;
    if (lm_qr(e, cols, coef, se) < nc) throw std::runtime_error("tbats: singular seed-state regression");
    Vec x0(L.N, 0.0);
    for (int j = 0; j < nc; ++j) x0[j] = coef[j];
    double ylog = 0;
    if (use_bc) { Vec lg(n); for (int t = 0; t < n; ++t) lg[t] = std::log(y[t]); ylog = r_sum(lg); }
    Vec x0u = x0;
    if (use_bc) for (double& v : x0u) v = inv_box_cox(v, P.lambda);
    const double bad = bats ? 1e20 : INF;
    auto lik = [&](const Vec& v) {
        const Params Q = unpack(v, P, damped);
        const System Sq = build(L, Q);
        Vec yy = y, xx0 = x0;
        if (use_bc) {
            for (double& t : yy) t = box_cox(t, Q.lambda);
            for (int i = 0; i < L.N; ++i) xx0[i] = box_cox(x0u[i], Q.lambda);
        }
        Vec ee, xx;
        filter(Sq, sparse_of(Sq.F), yy, xx0, ee, xx);
        Vec e2(ee.size());
        for (size_t t = 0; t < ee.size(); ++t) e2[t] = ee[t] * ee[t];
        double ll = n * std::log(r_sum(e2));
        if (use_bc) ll -= 2 * (Q.lambda - 1) * ylog;
        if (!bats && std::isnan(ll)) return INF;
        return admissible(Sq, Q, use_bc) ? ll : bad;
    };
    double value;
    int fail = 0;
    const int len = static_cast<int>(par.size());
    if (use_bc || len > 1) {
        auto fminfn = [&](const Vec& b) { Vec v(b.size()); for (size_t i = 0; i < b.size(); ++i) v[i] = b[i] * scale[i]; return lik(v); };
        Vec b(len);
        for (int i = 0; i < len; ++i) b[i] = par[i] / scale[i];
        b = nmmin(fminfn, b, value, 100 * len * len, fail);
        for (int i = 0; i < len; ++i) par[i] = b[i] * scale[i];
    } else {
        r_optim_bfgs(lik, scale, par, value, 100);
    }
    Fitted out;
    out.P = unpack(par, P, damped);
    out.L = L;
    Vec yy = y, xx0 = x0;
    if (use_bc) {
        for (int i = 0; i < L.N; ++i) xx0[i] = box_cox(x0u[i], out.P.lambda);
        for (double& t : yy) t = box_cox(t, out.P.lambda);
    }
    const System Sf = build(L, out.P);
    filter(Sf, sparse_of(Sf.F), yy, xx0, out.errors, out.xlast);
    out.aic = value + 2 * (len + L.N);
    out.ok = true;
    return out;
}

AutoArimaResult arma_of(const Vec& errors) { return auto_arima(errors, 1, true, false, 1, {}, {}, 0); }

// filterTBATSSpecifics
Fitted filter_tbats(const Vec& y, bool bc, bool trend, bool damp, const std::vector<int>& k, const Vec& periods,
                    double init_lambda, const Fitted* aux) {
    Fitted first;
    if (aux) first = *aux;
    else { try { first = fit_specific(y, bc, trend, damp, k, periods, 0, 0, init_lambda, false); } catch (const std::exception&) { first = Fitted{}; } }
    if (!first.ok) return first;                        // auto.arima(NULL errors) fails -> first.model
    AutoArimaResult a;
    try { a = arma_of(first.errors); } catch (const std::exception&) { return first; }
    if (a.p == 0 && a.q == 0) return first;
    Fitted second;
    try { second = fit_specific(y, bc, trend, damp, k, periods, a.p, a.q, init_lambda, false); } catch (const std::exception&) { second = Fitted{}; }
    return (second.aic < first.aic) ? second : first;
}

// bats(y) without seasonal periods (filterSpecifics errors propagate to bats()'s try)
Fitted bats_fit(const Vec& y, const std::vector<bool>& bc_opts) {
    bool any_bc = false;
    for (bool b : bc_opts) any_bc |= b;
    const double init_lambda = any_bc ? box_cox_lambda(y, 0, 1, 1) : NA;
    Fitted best;
    double best_aic = INF;
    bool found = false;
    for (bool bc : bc_opts)
        for (bool trend : {false, true})
            for (bool damp : {false, true}) {
                try {
                    Fitted m;
                    if (!trend && damp) { m.aic = INF; m.ok = false; }
                    else {
                        const Fitted first = fit_specific(y, bc, trend, damp, {}, {}, 0, 0, init_lambda, true);
                        const AutoArimaResult a = arma_of(first.errors);
                        m = first;
                        if (a.p != 0 || a.q != 0) {
                            const Fitted second = fit_specific(y, bc, trend, damp, {}, {}, a.p, a.q, init_lambda, true);
                            if (second.aic < first.aic) m = second;
                        }
                    }
                    if (m.aic < best_aic) { best_aic = m.aic; best = m; found = true; }
                } catch (const std::exception&) {}
            }
    if (!found) throw std::runtime_error("Unable to fit a model");
    return best;
}

Vec forecast_of(const Fitted& f, int h) {
    const System S = build(f.L, f.P);
    const Sparse Fs = sparse_of(S.F);
    Vec x = f.xlast, xn(x.size()), out(h);
    for (int t = 0; t < h; ++t) {
        double yh = 0.0;
        for (size_t j = 0; j < x.size(); ++j) if (S.w[j] != 0.0) yh += S.w[j] * x[j];
        out[t] = yh;
        for (size_t i = 0; i < x.size(); ++i) {
            double s = 0.0;
            for (const auto& [j, v] : Fs.rows[i]) s += v * x[j];
            xn[i] = s;
        }
        x.swap(xn);
    }
    if (f.P.boxcox) for (double& v : out) v = inv_box_cox(v, f.P.lambda);
    return out;
}

}  // namespace

Vec tbats_forecast(const Vec& y, const Vec& periods_in, int h) {
    const int n = static_cast<int>(y.size());
    for (double v : y) if (!std::isfinite(v)) throw std::runtime_error("tbats: missing values are not supported");
    Vec sp;
    double freq = 1;
    if (periods_in.size() == 1) { freq = std::floor(periods_in[0]); sp = {freq}; }
    else if (!periods_in.empty()) {
        sp = periods_in;
        std::sort(sp.begin(), sp.end());
        freq = std::floor(sp.back());
    }
    for (double& v : sp) v = std::max(v, 1.0);
    sp.erase(std::unique(sp.begin(), sp.end()), sp.end());
    bool all_one = true;
    for (double v : sp) if (v != 1) all_one = false;
    if (all_one) sp.clear();
    if (is_constant(y)) return Vec(h, y[n - 1]);
    bool pos = true;
    for (double v : y) if (v <= 0) pos = false;
    const std::vector<bool> bc_opts = pos ? std::vector<bool>{false, true} : std::vector<bool>{false};
    const Fitted non_seasonal = bats_fit(y, bc_opts);
    if (sp.empty()) return forecast_of(non_seasonal, h);
    Vec per;
    for (double v : sp) if (v != 1) per.push_back(v);
    const double init_lambda = pos ? box_cox_lambda(y, 0, 1, freq) : NA;
    const bool mp_bc = pos, mp_trend = true, mp_damp = true;
    std::vector<int> k(per.size(), 1);
    auto fit = [&](const std::vector<int>& kv) {
        try { return fit_specific(y, mp_bc, mp_trend, mp_damp, kv, per, 0, 0, init_lambda, false); }
        catch (const std::exception&) { return Fitted{}; }
    };
    Fitted best = fit(k);
    for (size_t i = 0; i < per.size(); ++i) {
        if (per[i] == 2) continue;
        int max_k = static_cast<int>(std::floor((per[i] - 1) / 2));
        if (i != 0) {
            int cur = 2;
            while (cur <= max_k) {
                if (std::fmod(per[i], cur) != 0) { ++cur; continue; }
                const double latter = per[i] / cur;
                bool any = false;
                for (size_t j = 0; j < i; ++j) if (std::fmod(per[j], latter) == 0) any = true;
                if (any) { max_k = cur - 1; break; }
                ++cur;
            }
        }
        if (max_k == 1) continue;
        if (max_k <= 6) {
            k[i] = max_k;
            best.aic = INF;
            for (;;) {
                Fitted nm = fit(k);
                if (nm.aic > best.aic) { k[i] += 1; break; }
                if (k[i] == 1) break;
                k[i] -= 1;
                best = nm;
            }
            continue;
        }
        std::vector<int> up = k, down = k;
        up[i] = 7; down[i] = 5; k[i] = 6;
        Fitted upm = fit(up), lev = fit(k), dn = fit(down);
        const double mn = std::min({upm.aic, lev.aic, dn.aic});
        if (mn == dn.aic) {
            best = dn;
            k[i] = 5;
            for (;;) {
                k[i] -= 1;
                dn = fit(k);
                if (dn.aic > best.aic) { k[i] += 1; break; }
                best = dn;
                if (k[i] == 1) break;
            }
        } else if (mn == lev.aic) {
            best = lev;
            continue;
        } else {
            best = upm;
            k[i] = 7;
            for (;;) {
                k[i] += 1;
                upm = fit(k);
                if (upm.aic > best.aic) { k[i] -= 1; break; }
                best = upm;
                if (k[i] == max_k) break;
            }
        }
    }
    const Fitted aux = best;
    Fitted chosen = (non_seasonal.aic < best.aic) ? non_seasonal : best;
    Fitted nm;
    for (bool bc : bc_opts)
        for (bool trend : {false, true})
            for (bool damp : {false, true}) {
                if (bc == mp_bc && trend == mp_trend && damp == mp_damp) nm = filter_tbats(y, bc, trend, damp, k, per, init_lambda, &aux);
                else if (trend || !damp) nm = filter_tbats(y, bc, trend, damp, k, per, init_lambda, nullptr);
                if (nm.aic < chosen.aic) chosen = nm;
            }
    if (!chosen.ok) throw std::runtime_error("tbats: no model able to be fitted");
    return forecast_of(chosen, h);
}

}  // namespace tsf
