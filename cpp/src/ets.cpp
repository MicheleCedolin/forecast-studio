// Port of R forecast::ets() (forecast 9.x): R/ets.R (ets, etsmodel, initparam, initstate, check.param),
// src/etscalc.c (etscalc, forecast, update), src/etsTargetFunction.cpp (bounds, admissibility) and R's optim
// Nelder-Mead (nmmin, src/appl/optim.c) with the settings ets uses. Variable names follow the R/C sources.
#include "ets.hpp"

#include <algorithm>
#include <array>
#include <complex>
#include <stdexcept>

namespace tsf {
namespace {

constexpr int NONE = 0, ADD = 1, MULT = 2;
constexpr double TOL = 1.0e-10, HUGEN = 1.0e10, NA = -99999.0;
const std::array<double, 4> LOWER{1e-4, 1e-4, 1e-4, 0.8}, UPPER{0.9999, 0.9999, 0.9999, 0.98};

// ------------------------------------------------------------------------------------------- etscalc.c
void ets_forecast_step(double l, double b, const double* s, int m, int trend, int season, double phi, double* f, int h) {
    double phistar = phi;
    for (int i = 0; i < h; ++i) {
        if (trend == NONE) f[i] = l;
        else if (trend == ADD) f[i] = l + phistar * b;
        else if (b < 0) f[i] = NA;
        else f[i] = l * std::pow(b, phistar);
        int j = m - 1 - i;
        while (j < 0) j += m;
        if (season == ADD) f[i] = f[i] + s[j];
        else if (season == MULT) f[i] = f[i] * s[j];
        if (i < h - 1) {
            if (std::fabs(phi - 1.0) < TOL) phistar = phistar + 1.0;
            else phistar = phistar + std::pow(phi, static_cast<double>(i + 1));
        }
    }
}

void ets_update(double oldl, double& l, double oldb, double& b, const double* olds, double* s, int m, int trend, int season,
                double alpha, double beta, double gamma, double phi, double y) {
    double q, phib;
    if (trend == NONE) { q = oldl; phib = 0; }
    else if (trend == ADD) { phib = phi * oldb; q = oldl + phib; }
    else if (std::fabs(phi - 1.0) < TOL) { phib = oldb; q = oldl * oldb; }
    else { phib = std::pow(oldb, phi); q = oldl * phib; }
    double p;
    if (season == NONE) p = y;
    else if (season == ADD) p = y - olds[m - 1];
    else p = (std::fabs(olds[m - 1]) < TOL) ? HUGEN : y / olds[m - 1];
    l = q + alpha * (p - q);
    if (trend > NONE) {
        double r;
        if (trend == ADD) r = l - oldl;
        else r = (std::fabs(oldl) < TOL) ? HUGEN : l / oldl;
        b = phib + (beta / alpha) * (r - phib);
    }
    if (season > NONE) {
        double t;
        if (season == ADD) t = y - q;
        else t = (std::fabs(q) < TOL) ? HUGEN : y / q;
        s[0] = olds[m - 1] + gamma * (t - olds[m - 1]);
        for (int j = 1; j < m; ++j) s[j] = olds[j - 1];
    }
}

// Likelihood term n*log(sum e^2) (+2 sum log|f|) and the final state (l, b, s[0..m-1]).
double etscalc(const Vec& y, const Vec& x0, int m, int error, int trend, int season, double alpha, double beta,
               double gamma, double phi, Vec* final_state) {
    if (m < 1) m = 1;
    double l = x0[0], b = 0.0;
    if (trend > NONE) b = x0[1];
    double s[24] = {0}, olds[24] = {0};
    if (season > NONE)
        for (int j = 0; j < m; ++j) s[j] = x0[(trend > NONE) + j + 1];
    double lik = 0.0, lik2 = 0.0, f[1];
    for (double yi : y) {
        const double oldl = l, oldb = b;
        if (season > NONE) std::copy(s, s + m, olds);
        ets_forecast_step(oldl, oldb, olds, m, trend, season, phi, f, 1);
        if (std::fabs(f[0] - NA) < TOL) return NA;
        const double e = (error == ADD) ? yi - f[0] : (yi - f[0]) / f[0];
        ets_update(oldl, l, oldb, b, olds, s, m, trend, season, alpha, beta, gamma, phi, yi);
        lik += e * e;
        lik2 += std::log(std::fabs(f[0]));
    }
    lik = static_cast<double>(y.size()) * std::log(lik);
    if (error == MULT) lik += 2 * lik2;
    if (final_state) {
        final_state->assign({l});
        if (trend > NONE) final_state->push_back(b);
        if (season > NONE) final_state->insert(final_state->end(), s, s + m);
    }
    return lik;
}

// ------------------------------------------------------------------------------------------- admissibility
// Largest modulus of the roots of the monic polynomial with coefficients c (highest degree first, c[0] == 1).
// Durand-Kerner iteration; used only for the stability test |root| <= 1 + 1e-10.
double max_root_modulus(const Vec& c) {
    const int d = static_cast<int>(c.size()) - 1;
    if (d < 1) return 0.0;
    using C = std::complex<double>;
    double rad = 0.0;
    for (int i = 1; i <= d; ++i) rad = std::max(rad, std::fabs(c[i]));
    rad = 1.0 + rad;                                       // Cauchy bound
    std::vector<C> z(d);
    const C seed(0.4, 0.9);
    for (int i = 0; i < d; ++i) z[i] = std::pow(seed, i) * (0.5 * rad);
    auto peval = [&](C x) { C v = 1.0; for (int i = 1; i <= d; ++i) v = v * x + c[i]; return v; };
    for (int it = 0; it < 1000; ++it) {
        double delta = 0.0;
        for (int i = 0; i < d; ++i) {
            C den = 1.0;
            for (int j = 0; j < d; ++j) if (j != i) den *= (z[i] - z[j]);
            if (std::abs(den) < 1e-300) den = 1e-300;
            const C dz = peval(z[i]) / den;
            z[i] -= dz;
            delta = std::max(delta, std::abs(dz));
        }
        if (delta < 1e-14) break;
    }
    double mx = 0.0;
    for (const C& v : z) mx = std::max(mx, std::abs(v));
    return mx;
}

struct Target {
    const Vec* y;
    int m, error, trend, season, nstate;
    bool optAlpha, optBeta, optGamma, optPhi;
    double alpha, beta, gamma, phi;

    bool admissible() const {
        if (phi < 0 || phi > 1 + 1e-8) return false;
        if (!optGamma) {                                   // non-seasonal (gamma not optimised, not given)
            if (alpha < 1 - 1 / phi || alpha > 1 + 1 / phi) return false;
            if (optBeta && (beta < alpha * (phi - 1) || beta > (1 + phi) * (2 - alpha))) return false;
        } else if (m > 1) {
            const double bt = optBeta ? beta : 0.0;
            const double d = 1 - 1 / phi - alpha;
            if (gamma < (d > 0 ? d : 0) || gamma > 1 + 1 / phi - alpha) return false;
            if (alpha < 1 - 1 / phi - gamma * (1 - m + phi + phi * m) / (2 * phi * m)) return false;
            if (bt < -(1 - phi) * (gamma / m + alpha)) return false;
            Vec opr{1.0, alpha + bt - phi};
            for (int i = 0; i < m - 2; ++i) opr.push_back(alpha + bt - alpha * phi);
            opr.push_back(alpha + bt - alpha * phi + gamma - 1);
            opr.push_back(phi * (1 - alpha - gamma));
            if (max_root_modulus(opr) > 1 + 1e-10) return false;
        }
        return true;
    }

    bool check_params() const {                            // bounds = "both": usual, then admissible
        if (optAlpha && (alpha < LOWER[0] || alpha > UPPER[0])) return false;
        if (optBeta && (beta < LOWER[1] || beta > alpha || beta > UPPER[1])) return false;
        if (optPhi && (phi < LOWER[3] || phi > UPPER[3])) return false;
        if (optGamma && (gamma < LOWER[2] || gamma > 1 - alpha || gamma > UPPER[2])) return false;
        return admissible();
    }

    // Full initial state (l, [b], s_0..s_{m-1}) from the optimised part (the last seasonal state is implied).
    Vec full_state(const Vec& par) const {
        Vec st(par.end() - nstate, par.end());
        if (season != NONE) {
            double sum = 0;
            for (int i = 1 + (trend != NONE); i < nstate; ++i) sum += st[i];
            st.push_back(m * (season == MULT ? 1 : 0) - sum);
        }
        return st;
    }

    double operator()(const Vec& par) {
        int j = 0;
        if (optAlpha) alpha = par[j++];
        if (optBeta) beta = par[j++];
        if (optGamma) gamma = par[j++];
        if (optPhi) phi = par[j++];
        if (!check_params()) return INF;
        const Vec st = full_state(par);
        if (season == MULT) {
            double mn = INF;
            for (size_t i = (trend != NONE) ? 2 : 1; i < st.size(); ++i) mn = std::min(mn, st[i]);
            if (mn < 0) return INF;
        }
        double lik = etscalc(*y, st, m, error, trend, season, alpha, beta, gamma, phi, nullptr);
        if (lik < -1e10) lik = -1e10;                     // avoid perfect fits
        if (std::isnan(lik)) lik = INF;
        if (std::fabs(lik + 99999) < 1e-7) lik = INF;
        return lik;
    }
};

// ------------------------------------------------------------------------------------------- R optim Nelder-Mead
// Port of nmmin() (R src/appl/optim.c) with ets's call: abstol = -Inf, intol = sqrt(eps), alpha 1, beta 0.5, gamma 2.
template <class F>
Vec nmmin(F& fminfn, Vec Bvec, double& Fmin, int maxit, int& fail) {
    const double big = 1.0e+35, abstol = -INF, intol = std::sqrt(2.220446049250313e-16);
    const double alpha = 1.0, bet = 0.5, gamm = 2.0;
    const int n = static_cast<int>(Bvec.size());
    fail = 0;
    double f = fminfn(Bvec);
    if (!std::isfinite(f)) throw std::runtime_error("function cannot be evaluated at initial parameters");
    int funcount = 1;
    const double convtol = intol * (std::fabs(f) + intol);
    const int n1 = n + 1, C = n + 2;
    std::vector<Vec> P(n1, Vec(n1 + 1));                  // P[i][j]: i = 0..n (row n = function value), j = 0..n+1
    P[n1 - 1][0] = f;
    for (int i = 0; i < n; ++i) P[i][0] = Bvec[i];
    int L = 1;
    double size = 0.0, step = 0.0;
    for (int i = 0; i < n; ++i) if (0.1 * std::fabs(Bvec[i]) > step) step = 0.1 * std::fabs(Bvec[i]);
    if (step == 0.0) step = 0.1;
    for (int j = 2; j <= n1; ++j) {
        for (int i = 0; i < n; ++i) P[i][j - 1] = Bvec[i];
        double trystep = step;
        while (P[j - 2][j - 1] == Bvec[j - 2]) {
            P[j - 2][j - 1] = Bvec[j - 2] + trystep;
            trystep *= 10;
        }
        size += trystep;
    }
    double oldsize = size;
    bool calcvert = true;
    int H = 1;
    double VH, VL, VR;
    do {
        if (calcvert) {
            for (int j = 0; j < n1; ++j) {
                if (j + 1 != L) {
                    for (int i = 0; i < n; ++i) Bvec[i] = P[i][j];
                    f = fminfn(Bvec);
                    if (!std::isfinite(f)) f = big;
                    ++funcount;
                    P[n1 - 1][j] = f;
                }
            }
            calcvert = false;
        }
        VL = P[n1 - 1][L - 1];
        VH = VL;
        H = L;
        for (int j = 1; j <= n1; ++j) {
            if (j != L) {
                f = P[n1 - 1][j - 1];
                if (f < VL) { L = j; VL = f; }
                if (f > VH) { H = j; VH = f; }
            }
        }
        if (VH <= VL + convtol || VL <= abstol) break;
        for (int i = 0; i < n; ++i) {
            double temp = -P[i][H - 1];
            for (int j = 0; j < n1; ++j) temp += P[i][j];
            P[i][C - 1] = temp / n;
        }
        for (int i = 0; i < n; ++i) Bvec[i] = (1.0 + alpha) * P[i][C - 1] - alpha * P[i][H - 1];
        f = fminfn(Bvec);
        if (!std::isfinite(f)) f = big;
        ++funcount;
        VR = f;
        if (VR < VL) {
            P[n1 - 1][C - 1] = f;
            for (int i = 0; i < n; ++i) {
                f = gamm * Bvec[i] + (1 - gamm) * P[i][C - 1];
                P[i][C - 1] = Bvec[i];
                Bvec[i] = f;
            }
            f = fminfn(Bvec);
            if (!std::isfinite(f)) f = big;
            ++funcount;
            if (f < VR) {
                for (int i = 0; i < n; ++i) P[i][H - 1] = Bvec[i];
                P[n1 - 1][H - 1] = f;
            } else {
                for (int i = 0; i < n; ++i) P[i][H - 1] = P[i][C - 1];
                P[n1 - 1][H - 1] = VR;
            }
        } else {
            if (VR < VH) {
                for (int i = 0; i < n; ++i) P[i][H - 1] = Bvec[i];
                P[n1 - 1][H - 1] = VR;
            }
            for (int i = 0; i < n; ++i) Bvec[i] = (1 - bet) * P[i][H - 1] + bet * P[i][C - 1];
            f = fminfn(Bvec);
            if (!std::isfinite(f)) f = big;
            ++funcount;
            if (f < P[n1 - 1][H - 1]) {
                for (int i = 0; i < n; ++i) P[i][H - 1] = Bvec[i];
                P[n1 - 1][H - 1] = f;
            } else if (VR >= VH) {
                calcvert = true;
                size = 0.0;
                for (int j = 0; j < n1; ++j) {
                    if (j + 1 != L) {
                        for (int i = 0; i < n; ++i) {
                            P[i][j] = bet * (P[i][j] - P[i][L - 1]) + P[i][L - 1];
                            size += std::fabs(P[i][j] - P[i][L - 1]);
                        }
                    }
                }
                if (size < oldsize) {
                    oldsize = size;
                } else {
                    fail = 10;
                    break;
                }
            }
        }
    } while (funcount <= maxit);
    Fmin = P[n1 - 1][L - 1];
    Vec X(n);
    for (int i = 0; i < n; ++i) X[i] = P[i][L - 1];
    if (funcount > maxit) fail = 1;
    return X;
}

// ------------------------------------------------------------------------------------------- initial state / params
// R decompose(ts(x, frequency = m), type)$seasonal on the given points (additive or multiplicative), recycled.
Vec decompose_seasonal(const Vec& x, int m, bool mult) {
    const int n = static_cast<int>(x.size());
    Vec filt;
    if (m % 2 == 0) { filt.assign(m + 1, 1.0 / m); filt.front() = filt.back() = 0.5 / m; }
    else filt.assign(m, 1.0 / m);
    const int k = static_cast<int>(filt.size()), half = k / 2;
    Vec trend(n, std::nan(""));
    for (int j = 0; j + k <= n; ++j) {
        double s = 0.0;
        for (int i = 0; i < k; ++i) s += x[j + i] * filt[k - 1 - i];
        trend[half + j] = s;
    }
    Vec figure(m);
    for (int i = 0; i < m; ++i) {
        double s = 0.0;
        int c = 0;
        for (int t = i; t < n; t += m) {
            const double v = mult ? x[t] / trend[t] : x[t] - trend[t];
            if (!std::isnan(v)) { s += v; ++c; }
        }
        figure[i] = c ? s / c : std::nan("");
    }
    double fs = 0.0;
    for (double v : figure) fs += v;
    const double fm = fs / m;
    Vec seas(n);
    for (int t = 0; t < n; ++t) seas[t] = mult ? figure[t % m] / fm : figure[t % m] - fm;
    return seas;
}

// Solve a small dense least-squares problem (normal equations, Gaussian elimination with partial pivoting).
Vec lstsq_small(const std::vector<Vec>& X, const Vec& y) {
    const int p = static_cast<int>(X.size()), n = static_cast<int>(y.size());
    std::vector<Vec> A(p, Vec(p + 1, 0.0));
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < p; ++j) { double s = 0; for (int t = 0; t < n; ++t) s += X[i][t] * X[j][t]; A[i][j] = s; }
        double s = 0; for (int t = 0; t < n; ++t) s += X[i][t] * y[t]; A[i][p] = s;
    }
    for (int c = 0; c < p; ++c) {
        int piv = c;
        for (int r = c + 1; r < p; ++r) if (std::fabs(A[r][c]) > std::fabs(A[piv][c])) piv = r;
        std::swap(A[c], A[piv]);
        if (std::fabs(A[c][c]) < 1e-300) throw std::runtime_error("singular regression in initstate");
        for (int r = 0; r < p; ++r) {
            if (r == c) continue;
            const double fct = A[r][c] / A[c][c];
            for (int k = c; k <= p; ++k) A[r][k] -= fct * A[c][k];
        }
    }
    Vec b(p);
    for (int i = 0; i < p; ++i) b[i] = A[i][p] / A[i][i];
    return b;
}

// R initstate(y, trendtype, seasontype): (l0, [b0], s_0..s_{m-2}).
Vec initstate(const Vec& y, int trend, int season, int m) {
    const int n = static_cast<int>(y.size());
    Vec init_seas, y_sa;
    if (season != NONE) {
        if (n < 4) throw std::runtime_error("not enough data");
        Vec seasonal;                                      // y.d$seasonal (length 3m, or n in the Fourier branch)
        if (n < 3 * m) {                                   // simple Fourier model: y ~ trend + fourier(y, K = 1)
            std::vector<Vec> X(1, Vec(n, 1.0));
            Vec tt(n), sn(n), cs(n);
            for (int t = 0; t < n; ++t) {
                tt[t] = t + 1;
                sn[t] = std::sin(2 * 3.14159265358979323846 * (t + 1) / m);
                cs[t] = std::cos(2 * 3.14159265358979323846 * (t + 1) / m);
            }
            X.push_back(tt);
            if (m > 2) X.push_back(sn);                     // fourier() drops sin when 2K == m
            X.push_back(cs);
            const Vec coef = lstsq_small(X, y);
            seasonal.resize(n);
            for (int t = 0; t < n; ++t) {
                const double fit = coef[0] + coef[1] * (t + 1);
                seasonal[t] = (season == ADD) ? y[t] - fit : y[t] / fit;
            }
        } else {
            // forecast >= 8.x decomposes the whole series (older versions used only the first 3m points)
            seasonal = decompose_seasonal(y, m, season == MULT);
        }
        for (int i = m - 1; i >= 1; --i) init_seas.push_back(seasonal[i]);   // rev(seasonal[2:m])
        const int ns = static_cast<int>(seasonal.size());
        y_sa.resize(n);
        if (season == ADD) {
            for (int t = 0; t < n; ++t) y_sa[t] = y[t] - seasonal[t % ns];
        } else {
            double sum = 0;
            for (double& v : init_seas) { v = std::max(v, 1e-2); sum += v; }
            if (sum > m) {
                double s2 = 0;
                for (double v : init_seas) s2 += v + 1e-2;
                for (double& v : init_seas) v /= s2;
            }
            for (int t = 0; t < n; ++t) y_sa[t] = y[t] / std::max(seasonal[t % ns], 1e-2);
        }
    } else {
        m = 1;
        y_sa = y;
    }
    const int maxn = std::min(std::max(10, 2 * m), n);
    Vec out;
    if (trend == NONE) {
        double s = 0;
        for (int t = 0; t < maxn; ++t) s += y_sa[t];
        out.push_back(s / maxn);
    } else {                                               // lsfit(1:maxn, y.sa[1:maxn])
        double xm = 0, ym = 0;
        for (int t = 0; t < maxn; ++t) { xm += t + 1; ym += y_sa[t]; }
        xm /= maxn; ym /= maxn;
        double sxy = 0, sxx = 0;
        for (int t = 0; t < maxn; ++t) { sxy += (t + 1 - xm) * (y_sa[t] - ym); sxx += (t + 1 - xm) * (t + 1 - xm); }
        const double c1 = sxy / sxx, c0 = ym - c1 * xm;
        double l0 = c0, b0 = c1;
        if (trend == ADD) {
            if (std::fabs(l0 + b0) < 1e-8) { l0 *= 1 + 1e-3; b0 *= 1 - 1e-3; }
        } else {
            l0 = c0 + c1;
            if (std::fabs(l0) < 1e-8) l0 = 1e-7;
            b0 = (c0 + 2 * c1) / l0;
            l0 = l0 / b0;
            if (std::fabs(b0) > 1e10) b0 = (b0 > 0 ? 1 : -1) * 1e10;
            if (l0 < 1e-8 || b0 < 1e-8) { l0 = std::max(y_sa[0], 1e-3); b0 = std::max(y_sa[1] / y_sa[0], 1e-3); }
        }
        out.push_back(l0);
        out.push_back(b0);
    }
    out.insert(out.end(), init_seas.begin(), init_seas.end());
    return out;
}

struct Fit { double aicc = INF, lik = INF, alpha = 0, beta = 0, gamma = 0, phi = 1; Vec final_state; };

// R etsmodel() for one model with all parameters estimated.
Fit etsmodel(const Vec& y, int error, int trend, int season, bool damped, int m_freq) {
    const int m = (season != NONE) ? m_freq : 1;
    // initparam
    double alpha = LOWER[0] + 0.2 * (UPPER[0] - LOWER[0]) / m;
    if (alpha > 1 || alpha < 0) alpha = LOWER[0] + 2e-3;
    Vec par{alpha};
    double beta = 0, gamma = 0, phi = 1;
    if (trend != NONE) {
        const double up = std::min(UPPER[1], alpha);
        beta = LOWER[1] + 0.1 * (up - LOWER[1]);
        if (beta < 0 || beta > alpha) beta = alpha - 1e-3;
        par.push_back(beta);
    }
    if (season != NONE) {
        const double up = std::min(UPPER[2], 1 - alpha);
        gamma = LOWER[2] + 0.05 * (up - LOWER[2]);
        if (gamma < 0 || gamma > 1 - alpha) gamma = 1 - alpha - 1e-3;
        par.push_back(gamma);
    }
    if (damped) {
        phi = LOWER[3] + 0.99 * (UPPER[3] - LOWER[3]);
        if (phi < 0 || phi > 1) phi = UPPER[3] - 1e-3;
        par.push_back(phi);
    }
    Target tg{&y, m, error, trend, season, 0, true, trend != NONE, season != NONE, damped, alpha, beta, gamma, phi};
    if (!tg.check_params()) throw std::runtime_error("Parameters out of range");
    const Vec st0 = initstate(y, trend, season, m);
    tg.nstate = static_cast<int>(st0.size());
    par.insert(par.end(), st0.begin(), st0.end());
    const int np0 = static_cast<int>(par.size());
    Fit out;
    if (np0 >= static_cast<int>(y.size()) - 1) return out;     // not enough data: aicc = Inf
    double fmin;
    int fail;
    const Vec fp = nmmin(tg, par, fmin, 2000, fail);
    tg(fp);                                                    // set alpha.. from the optimum
    const Vec st = tg.full_state(fp);
    Vec fin;
    const double lik = etscalc(y, st, m, error, trend, season, tg.alpha, tg.beta, tg.gamma, tg.phi, &fin);
    if (std::fabs(lik - NA) < 1e-7 || !std::isfinite(lik)) return out;
    const double np = np0 + 1, ny = static_cast<double>(y.size());
    const double aic = lik + 2 * np;
    out.aicc = aic + 2 * np * (np + 1) / (ny - np - 1);
    out.lik = lik;
    out.alpha = tg.alpha; out.beta = tg.beta; out.gamma = tg.gamma; out.phi = tg.phi;
    out.final_state = fin;
    return out;
}

}  // namespace

// Diagnostic: initial state and fitted parameter vector (as R's fit$par, before the implied seasonal state) of one model.
double ets_debug(const Vec& y, int m, int error, int trend, int season, bool damped, Vec& init, Vec& fitted) {
    if (season == NONE) m = 1;
    init = initstate(y, trend, season, m);
    const Fit f = etsmodel(y, error, trend, season, damped, m);
    fitted = {f.alpha};
    if (trend != NONE) fitted.push_back(f.beta);
    if (season != NONE) fitted.push_back(f.gamma);
    if (damped) fitted.push_back(f.phi);
    return f.lik;
}

EtsResult ets_auto(const Vec& y, int m, int h) {
    const int n = static_cast<int>(y.size());
    if (m > 24 || m < 1) m = 1;                                // R: "Seasonality will be ignored" for m > 24
    double ymin = INF;
    for (double v : y) ymin = std::min(ymin, v);
    const bool data_positive = ymin > 0;
    if (n <= 2 + 4) throw std::runtime_error("ets: series too short (R would use HoltWinters)");
    const int errs[2] = {ADD, MULT}, trends[2] = {NONE, ADD};
    const int seasons[3] = {NONE, ADD, MULT};
    const bool dampeds[2] = {true, false};
    EtsResult best;
    int bE = 0, bT = 0, bS = 0;
    bool bD = false;
    for (int e : errs)
        for (int t : trends)
            for (int s : seasons)
                for (bool d : dampeds) {
                    if (t == NONE && d) continue;
                    if (e == ADD && (t == MULT || s == MULT)) continue;
                    if (e == MULT && t == MULT && s == ADD) continue;
                    if (!data_positive && e == MULT) continue;
                    if (s != NONE && m == 1) continue;
                    Fit f;
                    try { f = etsmodel(y, e, t, s, d, m); } catch (const std::exception&) { continue; }
                    if (std::isfinite(f.aicc) && f.aicc < best.aicc) {
                        best.aicc = f.aicc; best.lik = f.lik;
                        best.alpha = f.alpha; best.beta = f.beta; best.gamma = f.gamma; best.phi = f.phi;
                        best.forecast = f.final_state;           // temporarily: final state
                        bE = e; bT = t; bS = s; bD = d;
                    }
                }
    if (!std::isfinite(best.aicc)) throw std::runtime_error("ets: no model able to be fitted");
    const Vec x = best.forecast;
    const int ms = (bS != NONE) ? m : 1;
    double l = x[0], b = (bT != NONE) ? x[1] : 0.0, s[24] = {0};
    if (bS != NONE) for (int j = 0; j < ms; ++j) s[j] = x[(bT != NONE) + j + 1];
    best.forecast.assign(h, 0.0);
    ets_forecast_step(l, b, s, ms, bT, bS, bD ? best.phi : 1.0, best.forecast.data(), h);
    const char* E = bE == ADD ? "A" : "M";
    const char* T = bT == NONE ? "N" : (bD ? "Ad" : "A");
    const char* S = bS == NONE ? "N" : (bS == ADD ? "A" : "M");
    best.model = std::string("ETS(") + E + "," + T + "," + S + ")";
    if (!bD) best.phi = 1.0;
    return best;
}

}  // namespace tsf
