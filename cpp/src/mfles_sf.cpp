// Port of statsforecast 2.1.1's MFLES / AutoMFLES (Apache-2.0, Nixtla) with scikit-learn 1.9's Lasso coordinate descent
// (BSD-3-Clause). See mfles_sf.hpp.
#include "mfles_sf.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>

#include "sf_common.hpp"

namespace tsf {

namespace {

constexpr double PI = 3.141592653589793;

double npmean(const Vec& v) { return np_mean(v.data(), v.size()); }
double npsum(const Vec& v) { return np_sum(v.data(), v.size()); }
double npstd(const Vec& v) {                            // population std (ddof 0)
    const double m = npmean(v);
    Vec d(v.size());
    for (size_t i = 0; i < v.size(); ++i) { const double t = std::fabs(v[i] - m); d[i] = t * t; }
    return std::sqrt(npmean(d));
}
double npmedian(Vec v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}
double calc_mse(const Vec& y, const Vec& f) {
    Vec d(y.size());
    for (size_t i = 0; i < y.size(); ++i) { const double e = y[i] - f[i]; d[i] = e * e; }
    return npmean(d);
}
double calc_smape(const Vec& y, const Vec& p) {
    Vec d(y.size());
    for (size_t i = 0; i < y.size(); ++i) d[i] = 2 * std::fabs(y[i] - p[i]) / std::fabs(y[i] + p[i] + 1e-6);
    return npmean(d);
}

// pinv(A) for a symmetric positive semi-definite A (cyclic Jacobi eigendecomposition; singular values = eigenvalues;
// cutoff 1e-15 * largest, numpy's default rcond)
std::vector<Vec> sym_pinv(std::vector<Vec> A) {
    const int k = static_cast<int>(A.size());
    std::vector<Vec> V(k, Vec(k, 0.0));
    for (int i = 0; i < k; ++i) V[i][i] = 1.0;
    double fro2 = 0.0;
    for (int p = 0; p < k; ++p) for (int q = 0; q < k; ++q) fro2 += A[p][q] * A[p][q];
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < k; ++p) for (int q = p + 1; q < k; ++q) off += A[p][q] * A[p][q];
        if (!(off > 1e-34 * fro2)) break;               // off-diagonal at rounding level
        for (int p = 0; p < k; ++p)
            for (int q = p + 1; q < k; ++q) {
                if (A[p][q] == 0.0) continue;
                const double theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
                const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1));
                const double c = 1 / std::sqrt(t * t + 1), s = t * c;
                for (int r = 0; r < k; ++r) {
                    const double arp = A[r][p], arq = A[r][q];
                    A[r][p] = c * arp - s * arq; A[r][q] = s * arp + c * arq;
                }
                for (int r = 0; r < k; ++r) {
                    const double apr = A[p][r], aqr = A[q][r];
                    A[p][r] = c * apr - s * aqr; A[q][r] = s * apr + c * aqr;
                }
                for (int r = 0; r < k; ++r) {
                    const double vrp = V[r][p], vrq = V[r][q];
                    V[r][p] = c * vrp - s * vrq; V[r][q] = s * vrp + c * vrq;
                }
            }
    }
    double mx = 0.0;
    for (int i = 0; i < k; ++i) mx = std::max(mx, std::fabs(A[i][i]));
    std::vector<Vec> P(k, Vec(k, 0.0));
    for (int e = 0; e < k; ++e) {
        const double lam = A[e][e];
        if (!(std::fabs(lam) > 1e-15 * mx)) continue;
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < k; ++j) P[i][j] += V[i][e] * V[j][e] / lam;
    }
    return P;
}

// ols / wls: pinv(X'WX) (W = I for ols) depends only on the design, so it is computed once per fit
std::vector<Vec> ls_prepare(const std::vector<Vec>& X, const Vec* w) {
    const int k = static_cast<int>(X.size());
    const size_t n = X[0].size();
    std::vector<Vec> G(k, Vec(k, 0.0));
    for (int i = 0; i < k; ++i)
        for (int j = i; j < k; ++j) {
            double s = 0.0;
            for (size_t t = 0; t < n; ++t) s += (w ? X[i][t] * (*w)[t] : X[i][t]) * X[j][t];
            G[i][j] = G[j][i] = s;
        }
    return sym_pinv(G);
}

// fitted values X pinv(X'WX) X'Wy
Vec ls_apply(const std::vector<Vec>& X, const Vec& y, const Vec* w, const std::vector<Vec>& P) {
    const int k = static_cast<int>(X.size());
    const size_t n = y.size();
    Vec b(k, 0.0);
    for (int i = 0; i < k; ++i) {
        double s = 0.0;
        for (size_t t = 0; t < n; ++t) s += (w ? X[i][t] * (*w)[t] : X[i][t]) * y[t];
        b[i] = s;
    }
    Vec coef(k, 0.0);
    for (int i = 0; i < k; ++i) for (int j = 0; j < k; ++j) coef[i] += P[i][j] * b[j];
    Vec f(n, 0.0);
    for (size_t t = 0; t < n; ++t) {
        double s = 0.0;
        for (int i = 0; i < k; ++i) s += X[i][t] * coef[i];
        f[t] = s;
    }
    return f;
}

std::vector<Vec> fourier_series(int n, int sp, int K) {     // [cos(k = 1..K), sin(k = 1..K)]
    std::vector<Vec> X(2 * K, Vec(n));
    for (int k = 1; k <= K; ++k) {
        const double x = 2 * PI * k / sp;
        for (int t = 1; t <= n; ++t) {
            X[k - 1][t - 1] = std::cos(x * t);
            X[K + k - 1][t - 1] = std::sin(x * t);
        }
    }
    return X;
}

int set_fourier(int p) { return p < 10 ? 5 : (p < 70 ? 10 : 15); }

Vec fast_ols(const Vec& y) {                            // x = 0..n-1
    const long long M = static_cast<long long>(y.size());
    long long xs = 0, xx = 0;
    for (long long i = 0; i < M; ++i) { xs += i; xx += i * i; }
    const double ys = npsum(y);
    double xy = 0.0;
    for (long long i = 0; i < M; ++i) xy += static_cast<double>(i) * y[i];
    const double slope = (M * xy - static_cast<double>(xs) * ys) / static_cast<double>(M * xx - xs * xs);
    const double icpt = (ys - slope * static_cast<double>(xs)) / static_cast<double>(M);
    Vec f(y.size());
    for (long long i = 0; i < M; ++i) f[i] = slope * static_cast<double>(i) + icpt;
    return f;
}

double med_partial(std::vector<double>& v, size_t m) {  // median of v[0..m) as in mfles.cpp
    std::nth_element(v.begin(), v.begin() + m / 2, v.begin() + m);
    if (m % 2 == 1) return v[m / 2];
    const double mid = v[m / 2];
    const double left = *std::max_element(v.begin(), v.begin() + m / 2);
    return (left + mid) / 2.0;
}

Vec siegel(const Vec& y) {                              // x = 0..n-1
    const size_t n = y.size();
    if (n < 2) throw std::invalid_argument("siegel_repeated_medians requires at least 2 data points");
    Vec slopes(n);
    std::vector<double> sub(n - 1);
    for (size_t i = 0; i < n; ++i) {
        size_t k = 0;
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            const double xd = static_cast<double>(j) - static_cast<double>(i);
            sub[k++] = (xd == 0.0) ? 0.0 : (y[j] - y[i]) / xd;
        }
        slopes[i] = med_partial(sub, n - 1);
    }
    std::vector<double> sv(slopes.begin(), slopes.end());
    const double ms = med_partial(sv, n);
    std::vector<double> iv(n);
    for (size_t i = 0; i < n; ++i) iv[i] = y[i] - slopes[i] * static_cast<double>(i);
    const double mi = med_partial(iv, n);
    Vec f(n);
    for (size_t i = 0; i < n; ++i) f[i] = static_cast<double>(i) * ms + mi;
    return f;
}

std::vector<Vec> get_basis(const Vec& yin, int ncp, double decay) {   // gradient_strategy = 0; columns
    const long n = static_cast<long>(yin.size());
    if (ncp < 1) {
        Vec c(n);
        for (long i = 0; i < n; ++i) c[i] = static_cast<double>(i);
        return {c};
    }
    Vec y = yin;
    const double y0 = y[0];
    for (double& v : y) v -= y0;
    const double initial = y[0], final_point = y[n - 1];
    double s = 0.0;
    for (double v : y) s += v;
    const double mean_y = s / static_cast<double>(n);
    std::vector<Vec> C(ncp + 1, Vec(n, 0.0));
    std::vector<long> len(ncp);
    Vec last(ncp);
    Vec work = y;
    for (int idx = 0; idx < ncp; ++idx) {
        const int i = ncp - idx;
        const long sp = static_cast<long>(work.size()) / i;
        len[idx] = sp;
        last[idx] = work[sp - 1];
        work = Vec(work.begin() + sp, work.end());
    }
    long ls = 0;
    for (int i = 0; i < ncp; ++i) {
        ls += len[i];
        const double mp = last[i];
        for (long j = 0; j < ls; ++j) C[i][j] = (ls > 1) ? initial + (mp - initial) * j / (ls - 1) : initial;
        double end;
        if (std::isnan(decay)) end = final_point;
        else if (decay == -1.0) {
            double dd = mp * mp;
            if (mean_y != 0.0) dd /= (mean_y * mean_y);
            if (dd > 0.99) dd = 0.99;
            if (dd < 0.001) dd = 0.001;
            end = mp - ((mp - final_point) * (1.0 - dd));
        } else end = mp - ((mp - final_point) * (1.0 - decay));
        const long rn = n - ls + 1;
        for (long j = 1; j < rn; ++j) C[i][ls + j - 1] = (rn > 1) ? mp + (end - mp) * j / (rn - 1) : mp;
    }
    std::fill(C[ncp].begin(), C[ncp].end(), 1.0);
    return C;
}

// coreforecast: exponentially_weighted_mean / rolling_mean (min_samples = window)
Vec ses_ensemble(const Vec& y, bool smooth, int order) {
    const size_t n = y.size();
    Vec res(n, 0.0);
    if (smooth) {
        const int na = 19;                              // np.arange(0.05, 1.0, 0.05)
        for (int a = 0; a < na; ++a) {
            const double alpha = 0.05 + a * 0.05;
            double prev = y[0];
            res[0] += prev;
            for (size_t i = 1; i < n; ++i) { prev = alpha * y[i] + (1 - alpha) * prev; res[i] += prev; }
        }
        for (double& v : res) v /= na;
    } else {
        const size_t w = static_cast<size_t>(order) + 1;
        double acc = 0.0;
        for (size_t i = 0; i < std::min(w, n); ++i) { acc += y[i]; res[i] = (i + 1 < w) ? NAN : acc / w; }
        for (size_t i = w; i < n; ++i) { acc += y[i] - y[i - w]; res[i] = acc / w; }
        for (size_t i = 0; i < std::min(w, n); ++i) res[i] = y[i];
    }
    return res;
}

double calc_cov(const Vec& y, bool mult) {
    if (mult) { const double s = npstd(y); return std::sqrt(std::exp(std::log(10.0) * (s * s) - 1)); }
    double r = npstd(y);
    const double m = npmean(y);
    if (m != 0) r /= m;
    return r;
}

double calc_rsq(const Vec& y, const Vec& f) {
    const double m = npmean(y);
    Vec a(y.size()), b(y.size());
    for (size_t i = 0; i < y.size(); ++i) { a[i] = (y[i] - f[i]) * (y[i] - f[i]); b[i] = (y[i] - m) * (y[i] - m); }
    return 1 - (npsum(a) / npsum(b));
}

struct Config {
    std::vector<int> sp;                                // empty = None
    bool seasonality_weights = false, smoother = false;
    int ma = 0;                                         // 0 = None
    double cov_threshold = 0.7;
    int max_rounds = 50;
};

struct Mfles {
    int robust = -1;                                    // -1 = None (kept across fits of the same object)
    bool has_penalty = false;
    double penalty = 0.0;
    // per fit
    bool trend_penalty = true, has_const = false;
    double mean = 0, std = 0;
    Vec trend, seasonality;
    bool has_seasonality = false;

    void fit(Vec y, const Config& c) {
        const int n = static_cast<int>(y.size());
        const double cov_threshold = (c.cov_threshold == -1) ? 10000 : c.cov_threshold;
        bool mult = !c.sp.empty();
        double ymin = INFINITY;
        for (double v : y) ymin = std::min(ymin, v);
        if (ymin <= 0) mult = false;
        if (mult) { has_const = true; for (double& v : y) v = std::log(v); }
        else {
            has_const = false;
            std = npstd(y); mean = npmean(y);
            for (double& v : y) v = v - mean;
            if (std > 0) for (double& v : y) v = v / std;
        }
        bool constant = true;
        { const double m = npmean(y); for (double v : y) if (v != m) { constant = false; break; } }
        if (n < 4 || constant) {
            trend = {y[n - 1], y[n - 1]};
            seasonality.assign(n, 0.0); has_seasonality = true;
            trend_penalty = false;
            mean = y[n - 1]; std = 0;
            return;
        }
        const int ncp = static_cast<int>(0.25 * n);
        trend_penalty = true;
        Vec fitted(n, npmedian(y));
        trend = {fitted[n - 1], fitted[n - 1]};
        bool have_mse = false;
        double mse = 0;
        int equal = 0, season_pos = 0;
        std::vector<std::vector<Vec>> fs;
        std::vector<std::vector<Vec>> pinvs;
        std::vector<Vec> cw;
        has_seasonality = !c.sp.empty();
        if (has_seasonality) {
            int mx = 0;
            for (int p : c.sp) mx = std::max(mx, p);
            seasonality.assign(mx, 0.0);
            for (int p : c.sp) fs.push_back(fourier_series(n, p, set_fourier(p)));
            if (c.seasonality_weights)
                for (int p : c.sp) { Vec w(n); for (int t = 0; t < n; ++t) w[t] = 1 + t / p; cw.push_back(w); }
            for (size_t k = 0; k < c.sp.size(); ++k) pinvs.push_back(ls_prepare(fs[k], c.seasonality_weights ? &cw[k] : nullptr));
        } else seasonality.clear();
        Vec resids(n), tmp(n);
        auto plus = [&](const Vec& a) { for (int t = 0; t < n; ++t) tmp[t] = fitted[t] + a[t]; return calc_mse(y, tmp); };
        for (int i = 0; i < c.max_rounds; ++i) {
            for (int t = 0; t < n; ++t) resids[t] = y[t] - fitted[t];
            const double cur = calc_mse(y, fitted);
            if (!have_mse) { mse = cur; have_mse = true; }
            else {
                if (mse <= cur) { if (equal == 6) break; ++equal; }
                else mse = cur;
            }
            if (has_seasonality) {
                const int k = season_pos;
                season_pos = (season_pos + 1) % static_cast<int>(c.sp.size());
                Vec seas = ls_apply(fs[k], resids, c.seasonality_weights ? &cw[k] : nullptr, pinvs[k]);
                for (double& v : seas) v *= 0.9;
                const double cm = plus(seas);
                if (mse > cm) {
                    mse = cm;
                    for (int t = 0; t < n; ++t) fitted[t] += seas[t];
                    for (int t = 0; t < n; ++t) resids[t] = y[t] - fitted[t];
                    const int p = c.sp[k];
                    const int L = std::min(p, n);       // seas[-p:] is the whole array when n < p; np.resize cycles it
                    for (size_t j = 0; j < seasonality.size(); ++j) seasonality[j] += seas[n - L + (j % L)];
                }
            }
            if (i % 2) {
                Vec tren;
                if (robust == 1) tren = siegel(resids);
                else if (i == 1) tren = fast_ols(resids);
                else {
                    const int cps = std::min(ncp, static_cast<int>(0.1 * n));
                    const std::vector<Vec> lbf = get_basis(resids, cps, -1.0);
                    const Vec coef = sk_lasso(lbf, resids, 1.0);
                    tren.assign(n, 0.0);
                    for (int t = 0; t < n; ++t) {
                        double s = 0.0;
                        for (size_t j = 0; j < lbf.size(); ++j) s += lbf[j][t] * coef[j];
                        tren[t] = s * 0.9;
                    }
                }
                const double cm = plus(tren);
                if (mse > cm) {
                    mse = cm;
                    for (int t = 0; t < n; ++t) fitted[t] += tren[t];
                    trend[0] += tren[n - 2]; trend[1] += tren[n - 1];
                    if (i == 1) { penalty = calc_rsq(resids, tren); has_penalty = true; }
                }
            } else if (i > 4) {
                const int order = c.ma ? c.ma : 1;      // itertools.cycle([ma]) / cycle([1])
                Vec tren = ses_ensemble(resids, c.smoother, order);
                const double cm = plus(tren);
                if (mse > cm + 0.0001 * mse) {
                    mse = cm;
                    for (int t = 0; t < n; ++t) fitted[t] += tren[t];
                    trend[0] += tren[n - 1]; trend[1] += tren[n - 1];
                }
            }
            if (i == 0 && robust == -1) robust = (calc_cov(resids, mult) > cov_threshold) ? 1 : 0;
        }
    }

    Vec predict(int h) const {
        const double last = trend[1];
        double slope = last - trend[0];
        if (trend_penalty && has_penalty) slope = slope * std::max(0.0, penalty);
        Vec p(h);
        for (int j = 0; j < h; ++j) p[j] = slope * (j + 1) + last;
        if (has_seasonality && !seasonality.empty())
            for (int j = 0; j < h; ++j) p[j] = p[j] + seasonality[j % seasonality.size()];
        if (has_const) for (double& v : p) v = std::exp(v);
        else for (double& v : p) v = mean + (v * std);
        return p;
    }
};

std::vector<Config> default_configs(const std::vector<int>& sp) {
    std::vector<Config> out;
    if (!sp.empty()) {
        const int mn = *std::min_element(sp.begin(), sp.end());
        const int mas[3] = {mn, static_cast<int>(mn / 2.0), 0};
        for (bool sw : {true, false})
            for (bool sm : {true, false})
                for (int ma : mas)
                    for (int s = 0; s < 2; ++s) {
                        if (sm && ma != 0) continue;            // smoother with an MA order
                        if (sw && s == 0) continue;             // seasonality weights without a season
                        Config c;
                        c.seasonality_weights = sw; c.smoother = sm; c.ma = ma;
                        if (s == 1) c.sp = sp;
                        out.push_back(c);
                    }
    } else {
        for (bool sm : {true, false})
            for (double ct : {0.5, -1.0})
                for (int mr : {5, 20}) {
                    Config c;
                    c.smoother = sm; c.cov_threshold = ct; c.max_rounds = mr;
                    out.push_back(c);
                }
    }
    return out;
}

}  // namespace

Vec sk_lasso(const std::vector<Vec>& X, const Vec& y, double alpha_in, double tol, int max_iter) {
    const int nf = static_cast<int>(X.size()), ns = static_cast<int>(y.size());
    const double alpha = alpha_in * ns, beta = 0.0;     // l1_reg = alpha * l1_ratio * n_samples
    Vec w(nf, 0.0), norm2(nf, 0.0), R = y, XtA(nf, 0.0);
    for (int j = 0; j < nf; ++j) { double s = 0.0; for (int i = 0; i < ns; ++i) s += X[j][i] * X[j][i]; norm2[j] = s; }
    auto dot = [ns](const Vec& a, const Vec& b) {      // four accumulators, like a BLAS ddot kernel
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        int i = 0;
        for (; i + 4 <= ns; i += 4) { s0 += a[i] * b[i]; s1 += a[i + 1] * b[i + 1]; s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3]; }
        for (; i < ns; ++i) s0 += a[i] * b[i];
        return (s0 + s1) + (s2 + s3);
    };
    auto gap_enet = [&](double& dual_norm) {
        const double R2 = dot(R, R), Ry = dot(R, y);
        for (int j = 0; j < nf; ++j) XtA[j] = dot(X[j], R);
        dual_norm = std::fabs(XtA[0]);
        for (int j = 1; j < nf; ++j) dual_norm = std::max(dual_norm, std::fabs(XtA[j]));
        double l1 = 0.0;
        for (double v : w) l1 += std::fabs(v);
        const double primal = 0.5 * (R2 + beta * 0.0) + alpha * l1;
        const double scale = (dual_norm > alpha) ? alpha / dual_norm : 1.0;
        const double dual = -0.5 * (scale * scale) * (R2 + beta * 0.0) + scale * Ry;
        return primal - dual;
    };
    const double d_w_tol = tol;                         // set before tol is scaled
    tol *= dot(y, y);
    double dual_norm;
    double gap = gap_enet(dual_norm);
    if (gap <= tol) return w;
    std::vector<int> active;
    std::vector<char> excluded(nf, 0);
    auto screen = [&](bool first) {
        active.clear();
        for (int j = 0; j < nf; ++j) {
            if (first && norm2[j] == 0) { w[j] = 0; excluded[j] = 1; continue; }
            if (!first && excluded[j]) continue;
            const double xt = XtA[j] / std::max(alpha, dual_norm);
            const double dj = (1 - std::fabs(xt)) / std::sqrt(norm2[j] + beta);
            if (dj <= std::sqrt(2 * gap) / alpha) { active.push_back(j); excluded[j] = 0; }
            else {
                if (w[j] != 0) { for (int i = 0; i < ns; ++i) R[i] += w[j] * X[j][i]; w[j] = 0; }
                excluded[j] = 1;
            }
        }
    };
    screen(true);
    for (int it = 0; it < max_iter; ++it) {
        double w_max = 0.0, d_w_max = 0.0;
        for (int j : active) {
            if (norm2[j] == 0.0) continue;
            const double wj = w[j];
            const double t = dot(X[j], R) + wj * norm2[j];
            const double sgn = (t == 0) ? 0.0 : (t > 0 ? 1.0 : -1.0);
            w[j] = sgn * std::max(std::fabs(t) - alpha, 0.0) / (norm2[j] + beta);
            if (w[j] != wj) { const double a = wj - w[j]; for (int i = 0; i < ns; ++i) R[i] += a * X[j][i]; }
            d_w_max = std::max(d_w_max, std::fabs(w[j] - wj));
            w_max = std::max(w_max, std::fabs(w[j]));
        }
        if (w_max == 0.0 || d_w_max / w_max <= d_w_tol || it == max_iter - 1) {
            gap = gap_enet(dual_norm);
            if (gap <= tol) break;
            screen(false);
        }
    }
    return w;
}

Vec mfles_forecast(const Vec& y, const Vec& periods, int h) {
    const int n = static_cast<int>(y.size());
    std::set<int> ps;
    for (double p : periods) ps.insert(static_cast<int>(std::nearbyint(p)));
    std::vector<int> sp;
    for (int p : ps) if (p >= 2 && 2 * p <= n) sp.push_back(p);
    const std::vector<Config> configs = default_configs(sp);
    // optimize(): rolling origin, test_size = step = h, n_windows = 2, smape
    const int test = h, step = h;
    int n_steps = 2;
    const int max_steps = static_cast<int>(std::floor(static_cast<double>(n - test - 4) / step)) + 1;   // Python //
    Config best = configs[0];
    if (max_steps >= 1) {
        if (max_steps < n_steps) n_steps = max_steps;
        Vec metrics;
        for (const Config& c : configs) {
            Mfles m;
            Vec mets;
            for (int split = 0; split < n_steps; ++split) {
                const int ntr = n - (split * step + test);
                const Vec tr(y.begin(), y.begin() + ntr), te(y.begin() + ntr, y.begin() + ntr + test);
                m.fit(tr, c);
                mets.push_back(calc_smape(te, m.predict(test)));
            }
            metrics.push_back(npmean(mets));
        }
        size_t bi = 0;                                  // np.argmin: first NaN if any, else first minimum
        bool nan_found = false;
        for (size_t i = 0; i < metrics.size(); ++i) if (std::isnan(metrics[i])) { bi = i; nan_found = true; break; }
        if (!nan_found) for (size_t i = 1; i < metrics.size(); ++i) if (metrics[i] < metrics[bi]) bi = i;
        best = configs[bi];
    }
    Mfles final_model;
    final_model.fit(y, best);
    return final_model.predict(h);
}

}  // namespace tsf
