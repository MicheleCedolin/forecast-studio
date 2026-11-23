// See temporal.hpp. Sums and means use long double accumulation as R's colSums / colMeans / cumsum / sum do (the
// C API runs with 64-bit x87 precision, see rstats.hpp).
#include "temporal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "ets.hpp"
#include "rstats.hpp"

namespace tsf {

namespace {

constexpr int NONE = 0, ADD = 1, MULT = 2;

// Cholesky solve of the symmetric positive definite A x = b (A row-major n x n)
Vec chol_solve(std::vector<double> A, Vec b, int n) {
    for (int j = 0; j < n; ++j) {
        double s = A[j * n + j];
        for (int k = 0; k < j; ++k) s -= A[j * n + k] * A[j * n + k];
        if (!(s > 0)) throw std::runtime_error("thief: reconciliation matrix is not positive definite");
        const double d = std::sqrt(s);
        A[j * n + j] = d;
        for (int i = j + 1; i < n; ++i) {
            double t = A[i * n + j];
            for (int k = 0; k < j; ++k) t -= A[i * n + k] * A[j * n + k];
            A[i * n + j] = t / d;
        }
    }
    for (int i = 0; i < n; ++i) {
        double t = b[i];
        for (int k = 0; k < i; ++k) t -= A[i * n + k] * b[k];
        b[i] = t / A[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {
        double t = b[i];
        for (int k = i + 1; k < n; ++k) t -= A[k * n + i] * b[k];
        b[i] = t / A[i * n + i];
    }
    return b;
}

}  // namespace

Vec thief_ets(const Vec& y, const Vec& periods, int h) {
    const int n = static_cast<int>(y.size());
    const int m = r_frequency(periods.data(), static_cast<int>(periods.size()));
    if (m <= 1) throw std::runtime_error("Seasonal period (m) must be greater than 1");
    if (n < 2 * m) throw std::runtime_error("I need at least 2 periods of data");
    for (double v : y) if (std::isnan(v)) throw std::runtime_error("thief: missing values are not supported");
    // tsaggregates(y, m, align = "end")
    std::vector<int> mout;
    for (int k = 1; k <= m; ++k) if (m % k == 0 && k <= n) mout.push_back(k);
    const int K = static_cast<int>(mout.size());
    std::vector<Vec> agg(K);
    agg[0] = y;
    for (int i = 1; i < K; ++i) {
        const int a = mout[i], start = n % a, nk = n / a;
        agg[i].resize(nk);
        for (int c = 0; c < nk; ++c) {
            long double s = 0;
            for (int r = 0; r < a; ++r) s += y[start + c * a + r];
            agg[i][c] = static_cast<double>(s);
        }
    }
    // th.forecast: level i has frequency m / AL and H = (m / AL) * ceiling(h / m) forecasts
    const int nc = (h + m - 1) / m;
    std::vector<Vec> fc(K);
    for (int i = 0; i < K; ++i) {
        const int freq = m / mout[i];
        fc[i] = ets_auto(agg[i], freq, freq * nc).forecast;
    }
    // reconcilethief(comb = "struc"): rows top (AL = m) .. bottom (AL = 1); per year, bottom = f_b + S_agg' z with
    // (W_agg + S_agg S_agg') z = f_agg - S_agg f_b, W = diag(nsum) (hts LU with weights 1/nsum)
    std::vector<int> rowAL;                              // AL of each aggregate row, top first
    std::vector<int> rowIdx;                             // index of the row within its level
    for (int i = K - 1; i >= 1; --i)
        for (int j = 0; j < m / mout[i]; ++j) { rowAL.push_back(mout[i]); rowIdx.push_back(j); }
    const int nagg = static_cast<int>(rowAL.size());
    // hts LU: -1 * S[1:nagg, , drop = TRUE] turns a single aggregate row into a vector and cbind2() fails, so R's
    // thief stops whenever only the annual level exists (m prime, e.g. 7)
    if (nagg == 1) throw std::runtime_error("non-conformable arguments (thief with a single aggregate level)");
    // aggregate row r sums bottom positions rowIdx * AL .. rowIdx * AL + AL - 1
    std::vector<double> A(static_cast<size_t>(nagg) * nagg, 0.0);
    for (int r = 0; r < nagg; ++r) {
        for (int c = 0; c < nagg; ++c) {
            const int a0 = rowIdx[r] * rowAL[r], a1 = a0 + rowAL[r];
            const int b0 = rowIdx[c] * rowAL[c], b1 = b0 + rowAL[c];
            const int ov = std::max(0, std::min(a1, b1) - std::max(a0, b0));
            A[static_cast<size_t>(r) * nagg + c] = ov;
        }
        A[static_cast<size_t>(r) * nagg + r] += rowAL[r];
    }
    Vec out;
    out.reserve(static_cast<size_t>(nc) * m);
    for (int c = 0; c < nc; ++c) {
        Vec fb(m);
        for (int j = 0; j < m; ++j) fb[j] = fc[0][c * m + j];
        Vec rhs(nagg);
        int r = 0;
        for (int i = K - 1; i >= 1; --i) {
            const int freq = m / mout[i];
            for (int j = 0; j < freq; ++j, ++r) {
                double s = 0;
                for (int t = j * mout[i]; t < (j + 1) * mout[i]; ++t) s += fb[t];
                rhs[r] = fc[i][c * freq + j] - s;
            }
        }
        const Vec z = chol_solve(A, rhs, nagg);
        for (int t = 0; t < m; ++t) {
            double adj = 0;
            for (int rr = 0; rr < nagg; ++rr) {
                const int a0 = rowIdx[rr] * rowAL[rr];
                if (t >= a0 && t < a0 + rowAL[rr]) adj += z[rr];
            }
            out.push_back(fb[t] + adj);
        }
    }
    out.resize(h);
    return out;
}

Vec mapa_ets(const Vec& y, const Vec& periods, int h) {
    const int n = static_cast<int>(y.size());
    for (double v : y) if (std::isnan(v)) throw std::runtime_error("mapa: missing values are not supported");
    const int ppy = r_frequency(periods.data(), static_cast<int>(periods.size()));
    int maximumAL = ppy;
    if (maximumAL == 1) maximumAL = 2;
    const int fh = h;
    struct Level { int AL, q, ppyA; EtsResult fit; Vec yA; };
    std::vector<Level> used;
    for (int AL = 1; AL <= maximumAL; ++AL) {
        const int q0 = n / AL;
        int ppyA = ppy / AL;
        if (ppy % AL != 0) ppyA = 1;
        const int npars = 2;                             // model "ZZZ": damped NULL, no fixed trend / season
        const int q = (q0 <= npars) ? 1 : q0;
        if (q < 4) continue;
        Vec yA(q0);                                      // mapa.aggr: colMeans(matrix(tail(y, q * AL), nrow = AL))
        const int off = n - q0 * AL;
        for (int c = 0; c < q0; ++c) {
            long double s = 0;
            for (int r = 0; r < AL; ++r) s += y[off + c * AL + r];
            yA[c] = static_cast<double>(s / static_cast<long double>(AL));
        }
        const bool zzn = (ppyA == 2 && q <= 3 * ppyA);
        Level L{AL, q0, ppyA, ets_zz(yA, ppyA, 1, zzn), yA};
        used.push_back(std::move(L));
    }
    if (used.empty()) throw std::runtime_error("Sample is too short to estimate the requested model at any aggregation level.");
    const int nu = static_cast<int>(used.size());
    // statetranslate(): components [ETS, Level, Trend, Season] over fh steps
    std::vector<std::vector<Vec>> FC(nu, std::vector<Vec>(4, Vec(fh, 0.0)));
    for (int a = 0; a < nu; ++a) {
        const Level& L = used[a];
        const int AL = L.AL, fhA = (fh + AL - 1) / AL;
        const Vec st = ets_refit_final(L.yA, L.fit);    // ets(ats, AL.fit, use.initial.values = TRUE)$states[q + 1, ]
        const EtsResult& f = L.fit;
        const int b = (f.trend == NONE) ? 0 : 1;
        const double level = st[0];
        const double trend = b ? st[1] : 0.0;
        Vec season;
        if (f.season != NONE)
            for (int j = L.ppyA; j >= 1; --j) season.push_back(st[b + j]);   // rev(states[q + 1, (2 + b):(ppyA + 1 + b)])
        Vec trendk(fhA + 1, 0.0);
        if (f.trend == ADD && !f.damped) {
            for (int k = 1; k <= fhA; ++k) trendk[k] = trend * static_cast<double>(k);
        } else if (f.trend == ADD && f.damped) {
            long double cs = 0;                          // cumsum((trend / phi) * phi^(1:fhA))
            for (int k = 1; k <= fhA; ++k) {
                cs += (trend / f.phi) * rs::r_pow(f.phi, static_cast<double>(k));
                trendk[k] = static_cast<double>(cs);
            }
        } else if (f.trend == MULT) {
            throw std::runtime_error("mapa: multiplicative trend is not supported");
        }
        for (int t = 0; t < fh; ++t) {
            const int k = t / AL;                        // 0-based aggregated step
            const double Lv = level;
            const double Tv = trendk[k + 1];
            double Sv = 0.0;
            if (f.season == ADD) Sv = season[k % L.ppyA];
            else if (f.season == MULT) Sv = (season[k % L.ppyA] - 1) * (Lv + Tv);
            FC[a][1][t] = Lv;
            FC[a][2][t] = Tv;
            FC[a][3][t] = Sv;
            long double s = 0;                           // colSums(FCs_temp[2:5, ]) (xreg = 0)
            s += Lv; s += Tv; s += Sv; s += 0.0;
            FC[a][0][t] = static_cast<double>(s);
        }
    }
    // mapacomb(comb = "w.mean"): level / trend mean, season (ALs dividing ppy, AL < ppy) weighted by 1/AL
    std::vector<int> perm;
    for (int a = 0; a < nu; ++a) if (ppy % used[a].AL == 0 && used[a].AL < ppy) perm.push_back(a);
    Vec w(perm.size());
    {
        long double sw = 0;
        for (size_t i = 0; i < perm.size(); ++i) { w[i] = 1.0 / used[perm[i]].AL; sw += w[i]; }
        const double swd = static_cast<double>(sw);
        for (double& v : w) v = v / swd;
    }
    Vec out(fh);
    for (int t = 0; t < fh; ++t) {
        long double sl = 0, st2 = 0, ss = 0;
        for (int a = 0; a < nu; ++a) { sl += FC[a][1][t]; st2 += FC[a][2][t]; }
        const double lev = static_cast<double>(sl / static_cast<long double>(nu));
        const double tre = static_cast<double>(st2 / static_cast<long double>(nu));
        for (size_t i = 0; i < perm.size(); ++i) ss += FC[perm[i]][3][t] * w[i];
        const double sea = perm.empty() ? 0.0 : static_cast<double>(ss);
        long double tot = 0;                             // colSums(combined): level, trend, season, xreg (0)
        tot += lev; tot += tre; tot += sea; tot += 0.0;
        const double comb = static_cast<double>(tot);
        out[t] = (FC[0][0][t] + comb) / 2;               // hybrid = TRUE
    }
    return out;
}

}  // namespace tsf
