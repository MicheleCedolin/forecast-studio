// Port of R forecast 9.0.2's nnetar() / forecast.nnetar() (R/nnetar.R, GPL-3, Rob J Hyndman and contributors) and of
// the single-hidden-layer network of the R package nnet 7.3-21 (R/nnet.R and src/nnet.c: fpass, bpass, fminfn, fmingr,
// VR_dovm, VR_nntest; GPL-2 | GPL-3, W. N. Venables and B. D. Ripley), with R's vmmin (BFGS, src/appl/optim.c),
// ar.yw() / eureka.f (Levinson-Durbin), scale() and runif(). See nnetar.hpp.
//
//   inputs   lags 1..p and m of the scaled series ((x - mean) / sd)
//   network  k logistic hidden units, a linear output, weights by destination unit (bias first); least squares
//   fit      BFGS from runif(-0.7, 0.7) weights, at most 100 iterations, abstol 1e-4, reltol 1e-8
#include "nnetar.hpp"

#include <algorithm>
#include <stdexcept>

#include "autoarima.hpp"
#include "rstats.hpp"

namespace tsf {

namespace {

// exp() as R's compiled code computes it on Windows (Rtools' mingw-w64: x87 extended precision, rounded to double);
// see the note on log() in struct_ts.cpp.
inline double r_exp(double x) {
#if defined(_WIN32) && (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    return static_cast<double>(std::exp(static_cast<long double>(x)));
#else
    return std::exp(x);
#endif
}

// The network of nnet.c for ninputs -> nhidden -> 1 (linout = TRUE, no skip layer, no decay). Unit 0 is the bias.
struct Net {
    int ninputs = 0, nunits = 0, first_hidden = 0, first_output = 0, nweights = 0;
    std::vector<int> nconn, conn;
    Vec outputs, error_sums, errors, slopes;
    const double* train_in = nullptr;                   // ntrain x ninputs, column-major
    const double* train_out = nullptr;
    int ntrain = 0;
    double total_error = 0;

    Net(int nin, int nhid) : ninputs(nin) {
        nunits = 1 + nin + nhid + 1;
        first_hidden = 1 + nin;
        first_output = 1 + nin + nhid;
        nconn.assign(nunits + 1, 0);
        for (int j = first_hidden; j < nunits; ++j) {   // norm.net(): bias, then every unit of the layer below
            nconn[j] = static_cast<int>(conn.size());
            conn.push_back(0);
            if (j < first_output) for (int i = 1; i <= nin; ++i) conn.push_back(i);
            else for (int i = first_hidden; i < first_output; ++i) conn.push_back(i);
        }
        nconn[nunits] = static_cast<int>(conn.size());
        for (int j = 0; j < first_hidden; ++j) nconn[j] = 0;
        nweights = static_cast<int>(conn.size());
        outputs.assign(nunits, 0.0);
        outputs[0] = 1.0;
        error_sums.assign(nunits, 0.0);
        errors.assign(nunits, 0.0);
        slopes.assign(nweights, 0.0);
    }

    static double sigmoid(double sum) {
        if (sum < -15.0) return 0.0;
        if (sum > 15.0) return 1.0;
        return 1.0 / (1.0 + r_exp(-sum));
    }

    // fpass(input, goal, wx = 1, nr): input[i * nr] is input i. With has_goal the squared error is accumulated.
    void fpass(const double* wts, const double* input, int nr, bool has_goal, double goal) {
        for (int i = 0; i < ninputs; ++i) outputs[i + 1] = input[static_cast<size_t>(i) * nr];
        for (int j = first_hidden; j < nunits; ++j) {
            double sum = 0.0;
            for (int i = nconn[j]; i < nconn[j + 1]; ++i) sum += outputs[conn[i]] * wts[i];
            if (j < first_output) sum = sigmoid(sum);
            outputs[j] = sum;
        }
        if (has_goal) {
            const double dif = outputs[first_output] - goal;
            total_error += 1.0 * (dif * dif);
        }
    }

    void bpass(const double* wts, double goal) {
        error_sums[first_output] = 2 * (outputs[first_output] - goal);
        for (int i = first_hidden; i < first_output; ++i) error_sums[i] = 0.0;
        for (int j = nunits - 1; j >= first_hidden; --j) {
            errors[j] = error_sums[j];
            if (j < first_output) errors[j] *= outputs[j] * (1.0 - outputs[j]);
            for (int i = nconn[j]; i < nconn[j + 1]; ++i) {
                const int cix = conn[i];
                error_sums[cix] += errors[j] * wts[i];
                slopes[i] += 1.0 * errors[j] * outputs[cix];
            }
        }
    }

    double fminfn(const Vec& p) {
        total_error = 0.0;
        for (int i = 0; i < ntrain; ++i) fpass(p.data(), train_in + i, ntrain, true, train_out[i]);
        double sum1 = 0.0;
        for (int i = 0; i < nweights; ++i) sum1 += 0.0 * p[i] * p[i];      // decay = 0
        return total_error + sum1;
    }

    void fmingr(const Vec& p, Vec& df) {
        for (int j = 0; j < nweights; ++j) slopes[j] = 2 * 0.0 * p[j];
        total_error = 0.0;
        for (int i = 0; i < ntrain; ++i) {
            fpass(p.data(), train_in + i, ntrain, true, train_out[i]);
            bpass(p.data(), train_out[i]);
        }
        for (int j = 0; j < nweights; ++j) df[j] = slopes[j];
    }

    double predict(const Vec& wts, const double* row) {                     // VR_nntest for one row
        fpass(wts.data(), row, 1, false, 0.0);
        return outputs[first_output];
    }
};

// vmmin() of R's src/appl/optim.c with every parameter free, as nnet's VR_dovm calls it.
void vmmin(Net& net, Vec& b, double& Fmin, int maxit, double abstol, double reltol) {
    const double stepredn = 0.2, acctol = 0.0001, reltest = 10.0;
    const int n = static_cast<int>(b.size());
    Vec g(n), t(n), X(n), c(n);
    std::vector<Vec> B(n);
    for (int i = 0; i < n; ++i) B[i].assign(i + 1, 0.0);
    double f = net.fminfn(b);
    if (!std::isfinite(f)) throw std::runtime_error("initial value in 'vmmin' is not finite");
    Fmin = f;
    int gradcount = 1;
    net.fmingr(b, g);
    int iter = 1, ilast = gradcount, count = 0;
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
                    f = net.fminfn(b);
                    accpoint = std::isfinite(f) && (f <= Fmin + gradproj * steplength * acctol);
                    if (!accpoint) steplength *= stepredn;
                }
            } while (!(count == n || accpoint));
            const bool enough = (f > abstol) && std::fabs(f - Fmin) > reltol * (std::fabs(Fmin) + reltol);
            if (!enough) { count = n; Fmin = f; }
            if (count < n) {
                Fmin = f;
                net.fmingr(b, g);
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
}

// length(ar(x)$ar): ar.yw.default(x, aic = TRUE) with order.max = min(n - 1, floor(10 log10(n))), the
// autocovariances of acf() and the Levinson-Durbin recursion of eureka.f. Throws for a zero-variance series.
int ar_order(const Vec& x0) {
    const int n = static_cast<int>(x0.size());
    const int order_max = std::min(n - 1, static_cast<int>(std::floor(10 * std::log10(static_cast<double>(n)))));
    if (order_max < 1) throw std::runtime_error("'order.max' must be >= 1");
    Vec x = x0;
    for (int pass = 0; pass < 2; ++pass) {              // ar() removes colMeans(x), acf(demean = TRUE) once more
        const double xm = rs::col_mean(x.data(), x.size());
        for (double& v : x) v -= xm;
    }
    Vec r(order_max + 1);
    for (int lag = 0; lag <= order_max; ++lag) {
        double sum = 0.0;
        for (int i = 0; i < n - lag; ++i) sum += x[i + lag] * x[i];
        r[lag] = sum / n;
    }
    if (r[0] == 0) throw std::runtime_error("zero-variance series");
    const int lr = order_max;
    Vec a(lr + 1), var(lr + 1), fprev(lr + 1), fcur(lr + 1);            // 1-based; f(l, .) row by row
    double v = r[0], d = r[1], q;
    a[1] = 1.0;
    fprev[1] = r[1] / v;
    q = fprev[1] * r[1];
    var[1] = (1 - fprev[1] * fprev[1]) * r[0];
    for (int l = 2; l <= lr; ++l) {
        a[l] = -d / v;
        if (l > 2) {
            const int l1 = (l - 2) / 2, l2 = l1 + 1;
            for (int j = 2; j <= l2; ++j) {
                const double hold = a[j];
                const int k = l - j + 1;
                a[j] = a[j] + a[l] * a[k];
                a[k] = a[k] + a[l] * hold;
            }
            if (2 * l1 != l - 2) a[l2 + 1] = a[l2 + 1] * (1.0 + a[l]);
        }
        v = v + a[l] * d;
        fcur[l] = (r[l] - q) / v;
        for (int j = 1; j <= l - 1; ++j) fcur[j] = fprev[j] + fcur[l] * a[l - j + 1];
        var[l] = var[l - 1] * (1 - fcur[l] * fcur[l]);
        if (l == lr) break;
        d = 0.0;
        q = 0.0;
        for (int i = 1; i <= l; ++i) {
            const int k = l - i + 2;
            d = d + a[i] * r[k - 1];
            q = q + fcur[i] * r[k - 1];
        }
        fprev = fcur;
    }
    int best = 0;                                       // xaic = n log(var.pred) + 2 k + 2; the first minimum
    double best_aic = n * std::log(r[0]) + 2.0;
    for (int k = 1; k <= lr; ++k) {
        const double aic = n * std::log(var[k]) + 2.0 * k + 2.0;
        if (aic < best_aic) { best_aic = aic; best = k; }
    }
    return best;
}

struct Design {                                         // what nnetar() hands to nnet()
    Vec xx;                                             // the scaled series
    double center = 0, scale = 1;
    std::vector<int> lags;
    int size = 0, ntrain = 0;
    Vec X, target;                                      // ntrain x nlag column-major
};

Design make_design(const Vec& x, int m) {
    const int n = static_cast<int>(x.size());
    Design D;
    D.center = rs::col_mean(x.data(), x.size());        // scale(): colMeans, then sqrt(sum(v^2) / (n - 1))
    D.xx.resize(n);
    for (int i = 0; i < n; ++i) D.xx[i] = x[i] - D.center;
    long double ss = 0;
    for (double v : D.xx) ss += v * v;
    D.scale = std::sqrt(static_cast<double>(ss) / std::max(1, n - 1));
    for (double& v : D.xx) v = v / D.scale;
    int p;
    bool seasonal_lag = false;
    if (m == 1) {
        p = std::max(ar_order(D.xx), 1);
        if (p >= n) p = n - 1;
    } else {
        Vec sa = D.xx;
        if (n > 2 * m) {                                // seasadj(mstl(xx)): trend + remainder = xx - seasonal
            const StlOut f = stl(D.xx, m, 11);
            for (int i = 0; i < n; ++i) {
                const double deseas = D.xx[i] - f.seasonal[i];
                sa[i] = f.trend[i] + (deseas - f.trend[i]);
            }
        }
        p = std::max(ar_order(sa), 1);
        if (p >= n) p = n - 1;
        seasonal_lag = n >= m + 2;                      // P = 1
    }
    for (int i = 1; i <= p; ++i) D.lags.push_back(i);
    if (seasonal_lag && m > p) D.lags.push_back(m);
    const int maxlag = D.lags.back(), nlag = static_cast<int>(D.lags.size());
    D.size = static_cast<int>(std::nearbyint((nlag + 1) / 2.0));        // round(): half to even
    D.ntrain = n - maxlag;
    if (D.ntrain < 1) throw std::runtime_error("No data to fit (possibly due to NA or NaN)");
    D.X.resize(static_cast<size_t>(D.ntrain) * nlag);
    for (int c = 0; c < nlag; ++c)
        for (int i = 0; i < D.ntrain; ++i) D.X[static_cast<size_t>(c) * D.ntrain + i] = D.xx[maxlag - D.lags[c] + i];
    D.target.assign(D.xx.begin() + maxlag, D.xx.end());
    return D;
}

// avnnet(): `repeats` networks, each from runif(nwts, -0.7, 0.7) of the running random stream
std::vector<Vec> fit_nets(const Design& D, int repeats, uint32_t seed, Net& net, Vec* values = nullptr) {
    if (net.nweights > 1000) throw std::runtime_error("too many weights for nnet (more than 1000)");
    net.train_in = D.X.data();
    net.train_out = D.target.data();
    net.ntrain = D.ntrain;
    rs::RRng rng(seed);
    std::vector<Vec> all;
    all.reserve(repeats);
    for (int rep = 0; rep < repeats; ++rep) {
        Vec w(net.nweights);
        for (double& v : w) v = -0.7 + (0.7 - -0.7) * rng.unif_rand();
        double fmin = 0;
        vmmin(net, w, fmin, 100, 1.0e-4, 1.0e-8);
        if (values) values->push_back(fmin);
        all.push_back(std::move(w));
    }
    return all;
}

}  // namespace

Vec nnetar_forecast(const Vec& y, int h, int m, int repeats, uint32_t seed) {
    const int n = static_cast<int>(y.size());
    if (n < 3) throw std::runtime_error("Not enough data to fit a model");
    for (double v : y) if (!std::isfinite(v)) throw std::runtime_error("nnetar: missing values are not supported");
    if (is_constant(y)) return Vec(h, y[n - 1]);
    if (repeats < 1) throw std::invalid_argument("repeats must be at least 1");
    m = std::max(m, 1);
    const Design D = make_design(y, m);
    const int nlag = static_cast<int>(D.lags.size()), maxlag = D.lags.back();
    Net net(nlag, D.size);
    const std::vector<Vec> wts = fit_nets(D, repeats, seed, net);
    Vec flag(maxlag);                                   // rev(tail(xx, maxlag)): flag[k - 1] is the value k steps back
    for (int k = 0; k < maxlag; ++k) flag[k] = D.xx[n - 1 - k];
    Vec fc(h), row(nlag), preds(repeats);
    for (int i = 0; i < h; ++i) {
        for (int c = 0; c < nlag; ++c) row[c] = flag[D.lags[c] - 1];
        for (int r = 0; r < repeats; ++r) preds[r] = net.predict(wts[r], row.data());
        fc[i] = rs::mean(preds.data(), preds.size());
        for (int k = maxlag - 1; k >= 1; --k) flag[k] = flag[k - 1];
        flag[0] = fc[i];
    }
    for (double& v : fc) v = v * D.scale + D.center;
    return fc;
}

}  // namespace tsf
