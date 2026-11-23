// Port of R forecast::ets() (forecast 9.x): R/ets.R (ets, etsmodel, initparam, initstate with decompose / lm /
// lsfit, check.param, the small-sample HoltWintersZZ branch), src/etscalc.c (etscalc, forecast, update),
// src/etsTargetFunction.cpp (bounds, admissibility with etspolyroot.c's R_cpolyroot), forecast.ets()'s class3 mean
// forecasts and R's optim Nelder-Mead (nmmin) / L-BFGS-B. Variable names follow the R/C sources.
#include "ets.hpp"
#include "autoarima.hpp"
#include "cpoly.hpp"
#include "lbfgsb.hpp"
#include "r_optim.hpp"
#include "rlinpack.hpp"
#include "rstats.hpp"

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
            else phistar = phistar + std::pow(phi, static_cast<double>(i + 2));   // phi + phi^2 + ... (etscalc.c)
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
               double gamma, double phi, Vec* final_state, long double* sum_e2 = nullptr) {
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
        if (sum_e2) *sum_e2 += e * e;                   // sum(model$residuals^2) (long double, as R's sum)
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
            // R_cpolyroot (forecast's etspolyroot.c), max sqrt(re^2 + im^2) <= 1 + 1e-10
            const int degree = static_cast<int>(opr.size()) - 1;
            double opi[32] = {0}, zr[32] = {0}, zi[32] = {0};          // degree = m + 1 <= 25
            bool fail = false;
            r_cpolyroot(opr.data(), opi, degree, zr, zi, &fail);
            if (fail) return false;
            double mx = 0;
            for (int i = 0; i < degree; ++i) mx = std::max(mx, std::sqrt(zr[i] * zr[i] + zi[i] * zi[i]));
            if (mx > 1 + 1e-10) return false;
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

// ------------------------------------------------------------------------------------------- initial state / params
// R decompose(ts(x, frequency = m), type)$seasonal on the given points (additive or multiplicative), recycled.
Vec decompose_seasonal(const Vec& x, int m, bool mult) {
    const int n = static_cast<int>(x.size());
    Vec filt;
    if (m % 2 == 0) { filt.assign(m + 1, 1.0 / m); filt.front() = filt.back() = 0.5 / m; }
    else filt.assign(m, 1.0 / m);
    const int k = static_cast<int>(filt.size()), half = k / 2;
    Vec trend(n, std::nan(""));
    for (int j = 0; j + k <= n; ++j) {                   // stats::filter (cfilter): z += filter[jj] * x[i + nshift - jj]
        double s = 0.0;
        for (int jj = 0; jj < k; ++jj) s += filt[jj] * x[j + k - 1 - jj];
        trend[half + j] = s;
    }
    Vec figure(m);
    for (int i = 0; i < m; ++i) {                        // mean(season[index + i], na.rm = TRUE)
        Vec v;
        for (int t = i; t < n; t += m) {
            const double d = mult ? x[t] / trend[t] : x[t] - trend[t];
            if (!std::isnan(d)) v.push_back(d);
        }
        figure[i] = v.empty() ? std::nan("") : rs::mean(v.data(), v.size());
    }
    const double fm = rs::mean(figure.data(), figure.size());
    Vec seas(n);
    for (int t = 0; t < n; ++t) seas[t] = mult ? figure[t % m] / fm : figure[t % m] - fm;
    return seas;
}

// R initstate(y, trendtype, seasontype): (l0, [b0], s_0..s_{m-2}).
Vec initstate(const Vec& y, int trend, int season, int m) {
    const int n = static_cast<int>(y.size());
    Vec init_seas, y_sa;
    if (season != NONE) {
        if (n < 4) throw std::runtime_error("not enough data");
        Vec seasonal;                                      // y.d$seasonal (length 3m, or n in the Fourier branch)
        if (n < 3 * m) {                                   // tslm(y ~ trend + fourier(y, 1)) (lm: dqrls)
            std::vector<Vec> X(1, Vec(n, 1.0));
            Vec tt(n);
            for (int t = 0; t < n; ++t) tt[t] = t + 1;
            X.push_back(tt);
            for (const Vec& col : fourier_terms(static_cast<double>(m), 1, 0, n)) X.push_back(col);
            const Vec coef = rl::dqrls_coef(X, y);
            seasonal.resize(n);
            for (int t = 0; t < n; ++t) {
                if (season == ADD) seasonal[t] = y[t] - coef[0] - coef[1] * (t + 1);
                else seasonal[t] = y[t] / (coef[0] + coef[1] * (t + 1));
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
            long double sum = 0;                           // sum(pmax(init.seas, 0.01)) > m
            for (double& v : init_seas) { v = std::max(v, 1e-2); sum += v; }
            if (static_cast<double>(sum) > m) {
                long double s2 = 0;                        // init.seas / sum(init.seas + 0.01)
                for (double v : init_seas) s2 += v + 1e-2;
                const double s2d = static_cast<double>(s2);
                for (double& v : init_seas) v /= s2d;
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
        out.push_back(rs::mean(y_sa.data(), maxn));        // mean(head(y.sa, maxn))
    } else {                                               // lsfit(seq(maxn), head(y.sa, maxn)) (dqrls)
        std::vector<Vec> X(1, Vec(maxn, 1.0));
        Vec tt(maxn);
        for (int t = 0; t < maxn; ++t) tt[t] = t + 1;
        X.push_back(tt);
        const Vec cf = rl::dqrls_coef(X, Vec(y_sa.begin(), y_sa.begin() + maxn));
        const double c1 = cf[1], c0 = cf[0];
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

struct Fit { double aicc = INF, lik = INF, alpha = 0, beta = 0, gamma = 0, phi = 1, sigma2 = NAN; Vec final_state, init_state; };

// forecast.ets() for "class 3" models (multiplicative error and season, trend N / A / Ad): the mean forecast of
// forecast:::class3, Mh <- F1 Mh F2' + G1 Mh G2' sigma2 and mu = H1 Mh H2', with the matrix products in the
// summation order of R's matprod (BLAS dgemm / dgemv, zero terms skipped where the reference BLAS skips them).
Vec ets_class3_mean(const Vec& st, int trend, bool damped, int m, double sigma2, double alpha, double beta, double gamma,
                    double phi, int h) {
    const int r = (trend != NONE) ? 2 : 1;
    std::vector<Vec> Mh(r, Vec(m));
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < m; ++j) Mh[i][j] = 0.0 + st[i] * st[r + j];
    const double ph = damped ? phi : 1.0;
    Vec mu(h);
    for (int k = 0; k < h; ++k) {
        double s = 0.0;                                 // H1 %*% Mh: column sums; then %*% t(H2): the last one
        s = s + Mh[0][m - 1];
        if (r == 2) s = s + Mh[1][m - 1];
        mu[k] = s;
        std::vector<Vec> T1(r, Vec(m)), U1(r, Vec(m));
        for (int j = 0; j < m; ++j) {
            if (r == 1) {
                T1[0][j] = 0.0 + Mh[0][j] * 1.0;
                U1[0][j] = 0.0 + Mh[0][j] * alpha;
            } else {
                double c1 = 0.0, c2 = 0.0, u1 = 0.0, u2 = 0.0;
                if (Mh[0][j] != 0.0) { c1 += Mh[0][j] * 1.0; c2 += Mh[0][j] * 0.0; u1 += Mh[0][j] * alpha; u2 += Mh[0][j] * beta; }
                if (Mh[1][j] != 0.0) { c1 += Mh[1][j] * 1.0; c2 += Mh[1][j] * ph; u1 += Mh[1][j] * alpha; u2 += Mh[1][j] * beta; }
                T1[0][j] = c1; T1[1][j] = c2; U1[0][j] = u1; U1[1][j] = u2;
            }
        }
        for (int i = 0; i < r; ++i) {
            Vec next(m);
            for (int j = 0; j < m; ++j) {
                const double t2 = (j == 0) ? T1[i][m - 1] : T1[i][j - 1];      // %*% t(F2): a cyclic shift
                const double u2 = (j == 0) ? (0.0 + U1[i][m - 1] * gamma) : 0.0; // %*% t(G2): gamma in column 1
                next[j] = t2 + u2 * sigma2;
            }
            Mh[i] = next;
        }
    }
    return mu;
}

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
    long double se2 = 0;
    const double lik = etscalc(y, st, m, error, trend, season, tg.alpha, tg.beta, tg.gamma, tg.phi, &fin, &se2);
    if (std::fabs(lik - NA) < 1e-7 || !std::isfinite(lik)) return out;
    out.sigma2 = static_cast<double>(se2) / (static_cast<double>(y.size()) - np0);   // sum(residuals^2) / (ny - np)
    const double np = np0 + 1, ny = static_cast<double>(y.size());
    const double aic = lik + 2 * np;
    out.aicc = aic + 2 * np * (np + 1) / (ny - np - 1);
    out.lik = lik;
    out.alpha = tg.alpha; out.beta = tg.beta; out.gamma = tg.gamma; out.phi = tg.phi;
    out.final_state = fin;
    out.init_state = st;
    return out;
}

// ------------------------------------------------------------------------------------------- small samples
// forecast:::HoltWintersZZ(x, alpha = NULL, beta = NULL or FALSE, gamma = NULL or FALSE, phi = NULL,
// exponential = FALSE, seasonal): zzhw() recursions, parameters by optim(method = "L-BFGS-B") on SSE in [0, 1].
struct HwzzFit { bool ok = false; double alpha = 0, beta = 0, gamma = 0, sigma2 = INF; int trend = NONE, season = NONE, m = 1; Vec init, fin; };

struct Zzhw { double sse = 0; Vec level, trend, season, res, xfit; };

// zzhw(); seas: 0 "none" (R's multiplicative formulas with season 1), 1 additive, 2 multiplicative
Zzhw zzhw(const Vec& x, double alpha, double beta, double gamma, int seas, int m, bool dotrend, bool doseasonal,
          double l_start, double b_start, const Vec& s_start_in) {
    const double phi = 1.0;
    const int n = static_cast<int>(x.size());
    if (!dotrend) { beta = 0; b_start = 0; }
    Vec s_start = s_start_in;
    if (!doseasonal) { gamma = 0; s_start.assign(1, seas == ADD ? 0.0 : 1.0); }
    Zzhw z;
    z.level.assign(n, 0.0); z.trend.assign(n, 0.0); z.season.assign(n, 0.0); z.xfit.assign(n, 0.0); z.res.assign(n, 0.0);
    double lastlevel = l_start, lasttrend = b_start;
    for (int i = 1; i <= n; ++i) {
        if (i > 1) { lastlevel = z.level[i - 2]; lasttrend = z.trend[i - 2]; }
        double lastseason = (i > m) ? z.season[i - m - 1] : (i - 1 < static_cast<int>(s_start.size()) ? s_start[i - 1] : NAN);
        if (std::isnan(lastseason)) lastseason = (seas == ADD) ? 0.0 : 1.0;
        double ladjust, sadjust;
        const double xi = x[i - 1];
        if (seas == ADD) {
            z.xfit[i - 1] = lastlevel + phi * lasttrend + lastseason;
            ladjust = sadjust = xi - z.xfit[i - 1];
        } else {
            z.xfit[i - 1] = (lastlevel + phi * lasttrend) * lastseason;
            ladjust = xi / lastseason - (lastlevel + phi * lasttrend);
            sadjust = xi / (lastlevel + phi * lasttrend) - lastseason;
        }
        z.level[i - 1] = lastlevel + phi * lasttrend + alpha * ladjust;
        z.trend[i - 1] = beta * (z.level[i - 1] - lastlevel) + (1 - beta) * phi * lasttrend;
        z.season[i - 1] = lastseason + gamma * sadjust;
    }
    long double sse = 0;                                // sum(res^2, na.rm = TRUE)
    for (int i = 0; i < n; ++i) {
        z.res[i] = x[i] - z.xfit[i];
        const double r2 = z.res[i] * z.res[i];
        if (!std::isnan(r2)) sse += r2;
    }
    z.sse = static_cast<double>(sse);
    return z;
}

HwzzFit holt_winters_zz(const Vec& x, int m_in, bool est_beta, bool est_gamma, bool mult_season) {
    HwzzFit out;
    int m = m_in;
    int seas = est_gamma ? (mult_season ? MULT : ADD) : ADD;  // match.arg(seasonal): "additive" unless given
    if (est_gamma && mult_season) for (double v : x) if (v <= 0) return out;   // "data must be positive"
    if (m <= 1) est_gamma = false;
    const int n = static_cast<int>(x.size());
    double l_start, b_start = 0;
    Vec s_start;
    if (!est_gamma) {
        seas = NONE;
        l_start = x[0];
        s_start.assign(1, 0.0);
        if (est_beta) {
            if (n < 2) return out;
            b_start = x[1] - x[0];
        }
    } else {
        if (n < 2 * m) return out;
        l_start = rs::mean(x.data(), m);
        b_start = (rs::mean(x.data() + m, m) - l_start) / m;
        s_start.resize(m);
        for (int i = 0; i < m; ++i) s_start[i] = (seas == ADD) ? x[i] - l_start : x[i] / l_start;
    }
    const int trendtype = est_beta ? ADD : NONE;
    const int seasontype = seas;
    if (seasontype == NONE) m = 1;
    // initparam(alpha = NULL, beta, gamma, phi = 1, ..., lower = 0, upper = 1, m, bounds = "usual")
    Vec par;
    double alpha = 0.0 + 0.2 * (1.0 - 0.0) / m;
    if (alpha > 1 || alpha < 0) alpha = 0.0 + 0.002;
    par.push_back(alpha);
    if (trendtype != NONE) {
        const double up = std::min(1.0, alpha);
        double beta = 0.0 + 0.1 * (up - 0.0);
        if (beta < 0 || beta > alpha) beta = alpha - 0.001;
        par.push_back(beta);
    }
    if (seasontype != NONE) {
        const double up = std::min(1.0, 1 - alpha);
        double gamma = 0.0 + 0.05 * (up - 0.0);
        if (gamma < 0 || gamma > 1 - alpha) gamma = 1 - alpha - 0.001;
        par.push_back(gamma);
    }
    const bool dotrend = est_beta, doseasonal = est_gamma;
    auto sse_of = [&](const Vec& p) {
        size_t k = 0;
        const double a = p[k++];
        const double b = est_beta ? p[k++] : 0.0;
        const double g = est_gamma ? p[k++] : 0.0;
        return zzhw(x, a, b, g, seas, m, dotrend, doseasonal, l_start, b_start, s_start).sse;
    };
    LbfgsbResult sol;
    try {
        sol = r_optim_lbfgsb(sse_of, par, Vec(par.size(), 0.0), Vec(par.size(), 1.0));
    } catch (const std::exception&) {
        return out;
    }
    bool outside = false;
    for (double v : sol.par) if (v < 0 || v > 1) outside = true;
    if ((sol.fail != 0 || outside) && !(sol.fail > 50)) return out;   // stop("optimization failure")
    size_t k = 0;
    out.alpha = sol.par[k++];
    out.beta = est_beta ? sol.par[k++] : 0.0;
    out.gamma = est_gamma ? sol.par[k++] : 0.0;
    const Zzhw z = zzhw(x, out.alpha, out.beta, out.gamma, seas, m, dotrend, doseasonal, l_start, b_start, s_start);
    out.trend = trendtype;
    out.season = seasontype;
    out.m = m;
    // states: l, [b], s1..sm with s_i = season_all[(m - i) + row] (season_all = c(s.start, season))
    Vec season_all = doseasonal ? s_start : Vec{};
    if (doseasonal) season_all.insert(season_all.end(), z.season.begin(), z.season.end());
    out.init = {l_start};
    out.fin = {z.level[n - 1]};
    if (trendtype != NONE) { out.init.push_back(dotrend ? b_start : 0.0); out.fin.push_back(z.trend[n - 1]); }
    if (seasontype != NONE)
        for (int i = 1; i <= m; ++i) {
            out.init.push_back(season_all[(m - i) + 1 - 1]);
            out.fin.push_back(season_all[(m - i) + (n + 1) - 1]);
        }
    // sigma2 = mean(res^2) (additive / none) or mean((res / fitted)^2) (multiplicative)
    Vec q(n);
    for (int i = 0; i < n; ++i) {
        const double r = (seasontype == MULT) ? z.res[i] / z.xfit[i] : z.res[i];
        q[i] = r * r;
    }
    out.sigma2 = rs::mean(q.data(), q.size());
    out.ok = true;
    return out;
}

// The small-sample branch of ets() (n <= npars + 4); trendT / seasS: requested types (-1 = "Z").
EtsResult ets_small(const Vec& y, int m, int h, int trendT, int seasS) {
    auto finish = [&](const HwzzFit& f, int err) {
        EtsResult r;
        r.hw = true;
        r.err = err; r.trend = f.trend; r.season = f.season; r.m = f.m; r.damped = false;
        r.alpha = f.alpha; r.beta = f.beta; r.gamma = f.gamma; r.phi = 1.0;
        r.init = f.init; r.fin = f.fin;
        const double l = f.fin[0], b = (f.trend != NONE) ? f.fin[1] : 0.0;
        double s[24] = {0};
        if (f.season != NONE) for (int j = 0; j < f.m; ++j) s[j] = f.fin[(f.trend != NONE) + 1 + j];
        r.forecast.assign(h, 0.0);
        r.sigma2 = f.sigma2;
        if (err == MULT && f.season == MULT)            // HoltWintersZZ multiplicative: components "M", ., "M" -> class3
            r.forecast = ets_class3_mean(f.fin, f.trend, false, f.m, f.sigma2, f.alpha, f.beta, f.gamma, 1.0, h);
        else
            ets_forecast_step(l, b, s, f.m, f.trend, f.season, 1.0, r.forecast.data(), h);
        const char* T = f.trend == NONE ? "N" : "A";
        const char* S = f.season == NONE ? "N" : (f.season == ADD ? "A" : "M");
        r.model = std::string("ETS(") + (err == ADD ? "A" : "M") + "," + T + "," + S + ")";
        return r;
    };
    if (trendT == MULT) throw std::runtime_error("ets: multiplicative trend not supported");
    if (seasS == ADD || seasS == MULT) {
        const HwzzFit f = holt_winters_zz(y, m, true, true, seasS == MULT);
        if (f.ok) return finish(f, seasS == MULT ? MULT : ADD);
    }
    if (trendT == ADD) {
        const HwzzFit f = holt_winters_zz(y, m, true, false, false);
        if (f.ok) return finish(f, ADD);
    }
    if (trendT == NONE && seasS == NONE) {
        const HwzzFit f = holt_winters_zz(y, m, false, false, false);
        if (f.ok) return finish(f, ADD);
    }
    const HwzzFit fit1 = holt_winters_zz(y, m, true, false, false);
    const HwzzFit fit2 = holt_winters_zz(y, m, false, false, false);
    if (!fit1.ok) {
        if (!fit2.ok) throw std::runtime_error("Unable to estimate a model.");
        return finish(fit2, ADD);
    }
    if (!fit2.ok) throw std::runtime_error("Unable to estimate a model.");   // fit2$sigma2 on a try-error
    return finish(fit1.sigma2 < fit2.sigma2 ? fit1 : fit2, ADD);
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

namespace {

// ets() model search; fixE / fixT / fixS restrict a component (-1 = any), as ets(y, model = "...") does.
EtsResult ets_select(const Vec& y, int m, int h, int fixE, int fixT, int fixS) {
    const int n = static_cast<int>(y.size());
    double ymin = INF;
    for (double v : y) ymin = std::min(ymin, v);
    const bool data_positive = ymin > 0;
    // ets(): m < 1 or n <= m -> seasontype "N"; m == 1 with a seasonal model -> "Nonseasonal data"
    if (m < 1 || n <= m) fixS = NONE;
    if (m <= 1) {
        if (fixS == ADD || fixS == MULT) throw std::runtime_error("Nonseasonal data");
        fixS = NONE;
        m = 1;
    }
    if (!data_positive && fixE == MULT) throw std::runtime_error("Inappropriate model for data with negative or zero values");
    int npars = 2;                                      // "Z" components do not count
    if (fixT == ADD || fixT == MULT) npars += 2;
    if (fixS == ADD || fixS == MULT) npars += m;
    if (n <= npars + 4) return ets_small(y, m, h, fixT, fixS);
    Vec best_init;
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
                    if ((fixE >= 0 && e != fixE) || (fixT >= 0 && t != fixT) || (fixS >= 0 && s != fixS)) continue;
                    Fit f;
                    try { f = etsmodel(y, e, t, s, d, m); } catch (const std::exception&) { continue; }
                    if (std::isfinite(f.aicc) && f.aicc < best.aicc) {
                        best.aicc = f.aicc; best.lik = f.lik;
                        best.alpha = f.alpha; best.beta = f.beta; best.gamma = f.gamma; best.phi = f.phi;
                        best.forecast = f.final_state;           // temporarily: final state
                        best_init = f.init_state;
                        best.sigma2 = f.sigma2;
                        bE = e; bT = t; bS = s; bD = d;
                    }
                }
    if (!std::isfinite(best.aicc)) throw std::runtime_error("ets: no model able to be fitted");
    const Vec x = best.forecast;
    const int ms = (bS != NONE) ? m : 1;
    best.err = bE; best.trend = bT; best.season = bS; best.damped = bD; best.m = ms;
    best.init = best_init;
    best.fin = x;
    double l = x[0], b = (bT != NONE) ? x[1] : 0.0, s[24] = {0};
    if (bS != NONE) for (int j = 0; j < ms; ++j) s[j] = x[(bT != NONE) + j + 1];
    best.forecast.assign(h, 0.0);
    if (bE == MULT && bS == MULT && bT != MULT)          // forecast.ets: class3 mean forecast
        best.forecast = ets_class3_mean(x, bT, bD, ms, best.sigma2, best.alpha, best.beta, best.gamma, best.phi, h);
    else
        ets_forecast_step(l, b, s, ms, bT, bS, bD ? best.phi : 1.0, best.forecast.data(), h);
    const char* E = bE == ADD ? "A" : "M";
    const char* T = bT == NONE ? "N" : (bD ? "Ad" : "A");
    const char* S = bS == NONE ? "N" : (bS == ADD ? "A" : "M");
    best.model = std::string("ETS(") + E + "," + T + "," + S + ")";
    if (!bD) best.phi = 1.0;
    return best;
}

}  // namespace

EtsResult ets_auto(const Vec& y, int m, int h) {
    if (m > 24 || m < 1) m = 1;                                // R: "Seasonality will be ignored" for m > 24
    return ets_select(y, m, h, -1, -1, -1);
}

EtsResult ets_holt(const Vec& y, int h, bool mult_error) { return ets_select(y, 1, h, mult_error ? MULT : ADD, ADD, NONE); }

EtsResult ets_zz(const Vec& y, int m, int h, bool no_season) {
    if (m > 24 || m < 1) m = 1;                                // seasontype "Z" -> "N" for m > 24
    return ets_select(y, m, h, -1, -1, no_season ? NONE : -1);
}

Vec ets_refit_final(const Vec& y, const EtsResult& fit) {
    // pegelsresid.C(y, m, initstate, ...): phi = 1 unless damped, beta = 0 without trend, gamma = 0 without season
    const double alpha = std::max(fit.alpha, 1e-10);
    const double beta = (fit.trend != NONE) ? fit.beta : 0.0;
    const double gamma = (fit.season != NONE) ? fit.gamma : 0.0;
    const double phi = fit.damped ? fit.phi : 1.0;
    Vec fin;
    etscalc(y, fit.init, fit.m, fit.err, fit.trend, fit.season, alpha, beta, gamma, phi, &fin);
    return fin;
}

EtsResult ets_holt_winters(const Vec& y, int m, int h, bool mult_error, int max_season) {
    const int n = static_cast<int>(y.size());
    if (m > max_season || n < 2 * m) m = 1;                    // the Python engine's usable_season(n, m, max_season)
    if (m <= 1) throw std::runtime_error("Nonseasonal data");  // ets(model = "AAA") on m = 1, as statsforecast / R
    const int e = mult_error ? MULT : ADD;                     // statsforecast HoltWinters: model = error + "A" + error
    return ets_select(y, m, h, e, ADD, e);
}

}  // namespace tsf
