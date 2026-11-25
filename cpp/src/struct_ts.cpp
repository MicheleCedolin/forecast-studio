// Port of R 4.6's stats::StructTS() (src/library/stats/R/StructTS.R) with the Kalman filter of
// src/library/stats/src/arima.c (KalmanLike, KalmanFore) and optim(method = "L-BFGS-B") (lbfgsb.cpp). See struct_ts.hpp.
//
//   state        level, slope [, m - 1 seasonal dummies]            (makeTrend / makeBSM)
//   variances    par * vx with vx = var(x) / 100, par = (level, slope [, seasonal], epsilon), start 1, bounds [0, Inf)
//   objective    0.5 * (sum(e^2 / F) / n + sum(log F) / n) from the filter started at a = (x[1], 0, ...), P = 1e6 * vx
//   forecast     Z' T^k a from the filtered state of the fitted model (KalmanRun, then KalmanForecast)
#include "struct_ts.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "lbfgsb.hpp"
#include "rstats.hpp"

namespace tsf {

namespace {

struct Model {
    int p = 0;
    Vec Z, a, P, T, V;                                  // matrices column-major, p x p
    double h = 1.0;
    std::vector<std::vector<int>> t_cols;               // non-zero columns of each row of T
    std::vector<int> z_idx;                             // non-zero elements of Z
};

// var(x): cov.c. The mean (two passes in long double) is stored as a double; the deviations from it and their
// squares are then formed and summed in long double. Forming the deviations in double changes the last bit of the
// result on about one series in five (checked against R 4.6.2 on M3 monthly and synthetic series).
double r_var(const Vec& x) {
    const size_t n = x.size();
    const long double xm = rs::mean(x.data(), n);
    long double ss = 0;
    for (double v : x) { const long double d = static_cast<long double>(v) - xm; ss += d * d; }
    return static_cast<double>(ss / static_cast<long double>(n - 1));
}

// log() as R's compiled code computes it. On Windows R is built with Rtools' mingw-w64, whose log() is the x87
// instruction (fyl2x) in extended precision rounded to double; the C runtime's log() that this library otherwise
// links differs from it in the last bit about once in 10^4 calls, enough to send L-BFGS-B down another path
// (M3 monthly: 9 of 200 series, forecasts up to 2e-4 apart). logl() here is that x87 routine; the caller holds the
// x87 precision at 64 bits (rs::X87Extended in capi.cpp). Elsewhere R and this library share the system's log().
inline double r_log(double x) {
#if defined(_WIN32) && (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    return static_cast<double>(std::log(static_cast<long double>(x)));
#else
    return std::log(x);
#endif
}

Model make_model(const Vec& x, int nf) {                // nf = 0: makeTrend, nf >= 2: makeBSM
    Model M;
    const int p = nf >= 2 ? nf + 1 : 2;
    M.p = p;
    M.T.assign(static_cast<size_t>(p) * p, 0.0);
    auto T = [&](int i, int j) -> double& { return M.T[i + static_cast<size_t>(p) * j]; };
    T(0, 0) = 1.0; T(0, 1) = 1.0; T(1, 1) = 1.0;
    M.Z.assign(p, 0.0);
    M.Z[0] = 1.0;
    M.V.assign(static_cast<size_t>(p) * p, 0.0);
    if (nf >= 2) {
        for (int j = 2; j < p; ++j) T(2, j) = -1.0;
        for (int ind = 3; ind <= nf; ++ind) T(ind, ind - 1) = 1.0;   // T[cbind(ind + 1, ind)] <- 1, 1-based ind = 3..nf
        M.Z[2] = 1.0;
    }
    M.a.assign(p, 0.0);
    M.a[0] = x[0];
    M.P.assign(static_cast<size_t>(p) * p, 0.0);
    M.t_cols.resize(p);
    for (int i = 0; i < p; ++i)
        for (int k = 0; k < p; ++k) if (T(i, k) != 0.0) M.t_cols[i].push_back(k);
    for (int i = 0; i < p; ++i) if (M.Z[i] != 0.0) M.z_idx.push_back(i);
    return M;
}

// KalmanLike(y, mod, nit = -1): returns (ssq / nu, sumlog / nu); a_out gets the last filtered state. The sums run in
// arima.c's order; terms with a zero element of T or Z are left out (they add an exact zero).
std::pair<double, double> kalman_like(const Vec& y, const Model& M, Vec* a_out) {
    const int n = static_cast<int>(y.size()), p = M.p;
    const size_t pp = static_cast<size_t>(p);
    Vec a = M.a, P = M.P, Pnew(pp * pp, 0.0), anew(p), Mv(p), mm(pp * pp);
    const double* T = M.T.data();
    const double* V = M.V.data();
    const double* Z = M.Z.data();
    double sumlog = 0.0, ssq = 0.0;
    int nu = 0;
    for (int l = 0; l < n; ++l) {
        for (int i = 0; i < p; ++i) {
            double tmp = 0.0;
            for (int k : M.t_cols[i]) tmp += T[i + pp * k] * a[k];
            anew[i] = tmp;
        }
        for (int i = 0; i < p; ++i)
            for (int j = 0; j < p; ++j) {
                double tmp = 0.0;
                for (int k : M.t_cols[i]) tmp += T[i + pp * k] * P[k + pp * j];
                mm[i + pp * j] = tmp;
            }
        for (int i = 0; i < p; ++i)
            for (int j = 0; j < p; ++j) {
                double tmp = V[i + pp * j];
                for (int k : M.t_cols[j]) tmp += mm[i + pp * k] * T[j + pp * k];
                Pnew[i + pp * j] = tmp;
            }
        ++nu;
        double resid0 = y[l];
        for (int i : M.z_idx) resid0 -= Z[i] * anew[i];
        double gain = M.h;
        for (int i = 0; i < p; ++i) {
            double tmp = 0.0;
            for (int k : M.z_idx) tmp += Pnew[i + k * pp] * Z[k];
            Mv[i] = tmp;
            if (Z[i] != 0.0) gain += Z[i] * Mv[i];
        }
        ssq += resid0 * resid0 / gain;
        sumlog += r_log(gain);
        for (int i = 0; i < p; ++i) a[i] = anew[i] + Mv[i] * resid0 / gain;
        for (int i = 0; i < p; ++i)
            for (int j = 0; j < p; ++j) P[i + j * pp] = Pnew[i + j * pp] - Mv[i] * Mv[j] / gain;
    }
    if (a_out) *a_out = a;
    return {ssq / nu, sumlog / nu};
}

}  // namespace

Vec struct_ts_forecast(const Vec& x, int h, int m) {
    const int n = static_cast<int>(x.size());
    if (n < 3) throw std::runtime_error("StructTS needs at least 3 observations");
    for (double v : x) if (!std::isfinite(v)) throw std::runtime_error("StructTS: missing values are not supported");
    const bool bsm = m > 1 && m <= STRUCT_TS_MAX_SEASON && n >= 2 * m;
    Model M = make_model(x, bsm ? m : 0);
    const int np = bsm ? 3 : 2, p = M.p;
    const double vx = r_var(x) / 100.0;
    for (double& v : M.P) v = 1e6 * vx;
    auto set_par = [&](const Vec& par) {
        for (int i = 0; i < np; ++i) M.V[i + static_cast<size_t>(p) * i] = par[i] * vx;
        M.h = par[np] * vx;
    };
    auto get_like = [&](const Vec& par) -> double {
        bool all0 = true;
        for (double v : par) if (v != 0.0) all0 = false;       // NaN != 0 is true here; R's all(p == 0) is then NA -> error
        if (all0) return 1000.0;
        set_par(par);
        const auto z = kalman_like(x, M, nullptr);
        const long double s = static_cast<long double>(z.first) + static_cast<long double>(z.second);   // sum(z)
        return 0.5 * static_cast<double>(s);
    };
    const Vec init(np + 1, 1.0), lower(np + 1, 0.0), upper(np + 1, INF);
    const LbfgsbResult res = r_optim_lbfgsb(get_like, init, lower, upper);
    set_par(res.par);
    Vec a;
    kalman_like(x, M, &a);                              // KalmanRun(y, Z, -1, update = TRUE)
    Vec fc(h), anew(p);
    const size_t pp = static_cast<size_t>(p);
    for (int l = 0; l < h; ++l) {                       // KalmanFore
        double f = 0.0;
        for (int i = 0; i < p; ++i) {
            double tmp = 0.0;
            for (int k : M.t_cols[i]) tmp += M.T[i + pp * k] * a[k];
            anew[i] = tmp;
            f += tmp * M.Z[i];
        }
        a = anew;
        fc[l] = f;
    }
    return fc;
}

}  // namespace tsf
