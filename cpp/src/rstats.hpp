// Small pieces of R reproduced exactly, header-only (namespace tsf::rs):
//  - R's default random number generator: Mersenne-Twister with set.seed()'s scrambling (src/main/RNG.c), unif_rand()
//    and sample()'s "Rejection" index (R_unif_index, R >= 3.6), so sample(1:k, 1) = 1 + unif_index(k);
//  - mean() (long double, two passes), colMeans() (one pass), sd(), optimize() (Brent_fmin), BoxCox(), InvBoxCox()
//    and forecast::BoxCox.lambda(method = "guerrero").
#pragma once

#include <cmath>
#include <cstdint>

#include "util.hpp"

namespace tsf {
namespace rs {

// R sets the x87 FPU to 64-bit (extended) precision at start-up; a host process such as Python on Windows leaves it at
// 53 bits, which makes every long double sum (mean, colMeans, var, rowMeans, ...) round like a double. This guard
// switches the calling thread's x87 precision to 64 bits and restores it on exit. SSE double arithmetic is unaffected.
class X87Extended {
public:
    X87Extended() {
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
        __asm__ volatile("fnstcw %0" : "=m"(old_));
        unsigned short cw = static_cast<unsigned short>(old_ | 0x0300);   // precision control = 11 (64-bit)
        __asm__ volatile("fldcw %0" : : "m"(cw));
#endif
    }
    ~X87Extended() {
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
        __asm__ volatile("fldcw %0" : : "m"(old_));
#endif
    }
    X87Extended(const X87Extended&) = delete;
    X87Extended& operator=(const X87Extended&) = delete;

private:
    unsigned short old_ = 0;
};

class RRng {
public:
    explicit RRng(uint32_t seed) { set_seed(seed); }

    void set_seed(uint32_t seed) {                     // RNG_Init(MERSENNE_TWISTER, seed)
        for (int j = 0; j < 50; ++j) seed = 69069u * seed + 1u;
        uint32_t dummy0 = 0;
        for (int j = 0; j < 625; ++j) {
            seed = 69069u * seed + 1u;
            if (j == 0) dummy0 = seed; else mt_[j - 1] = seed;
        }
        (void)dummy0;
        mti_ = N;                                       // FixupSeeds(initial = 1): dummy[0] = 624
    }

    double unif_rand() {                                 // fixup(MT_genrand())
        const double x = genrand();
        constexpr double i2_32m1 = 2.328306437080797e-10;
        if (x <= 0.0) return 0.5 * i2_32m1;
        if (1.0 - x <= 0.0) return 1.0 - 0.5 * i2_32m1;
        return x;
    }

    // R_unif_index(dn), sample.kind = "Rejection": an integer in [0, dn)
    double unif_index(double dn) {
        if (dn <= 0) return 0.0;
        const int bits = static_cast<int>(std::ceil(std::log2(dn)));
        double dv;
        do { dv = rbits(bits); } while (dn <= dv);
        return dv;
    }

    // sample(1:k, 1)
    int sample1(int k) { return 1 + static_cast<int>(unif_index(static_cast<double>(k))); }

private:
    static constexpr int N = 624, M = 397;
    uint32_t mt_[N] = {};
    int mti_ = N + 1;

    double rbits(int bits) {
        int64_t v = 0;
        for (int n = 0; n <= bits; n += 16) {
            const int v1 = static_cast<int>(std::floor(unif_rand() * 65536));
            v = 65536 * v + v1;
        }
        const int64_t one64 = 1;
        return static_cast<double>(v & ((one64 << bits) - 1));
    }

    double genrand() {
        static const uint32_t mag01[2] = {0x0u, 0x9908b0dfu};
        constexpr uint32_t UPPER = 0x80000000u, LOWER = 0x7fffffffu;
        uint32_t y;
        if (mti_ >= N) {
            int kk;
            for (kk = 0; kk < N - M; ++kk) {
                y = (mt_[kk] & UPPER) | (mt_[kk + 1] & LOWER);
                mt_[kk] = mt_[kk + M] ^ (y >> 1) ^ mag01[y & 0x1u];
            }
            for (; kk < N - 1; ++kk) {
                y = (mt_[kk] & UPPER) | (mt_[kk + 1] & LOWER);
                mt_[kk] = mt_[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1u];
            }
            y = (mt_[N - 1] & UPPER) | (mt_[0] & LOWER);
            mt_[N - 1] = mt_[M - 1] ^ (y >> 1) ^ mag01[y & 0x1u];
            mti_ = 0;
        }
        y = mt_[mti_++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= (y >> 18);
        return static_cast<double>(y) * 2.3283064365386963e-10;
    }
};

inline double mean(const double* v, size_t n) {          // mean(): long double, second pass correction
    long double s = 0;
    for (size_t i = 0; i < n; ++i) s += v[i];
    s /= static_cast<long double>(n);
    if (std::isfinite(static_cast<double>(s))) {
        long double t = 0;
        for (size_t i = 0; i < n; ++i) t += (v[i] - s);
        s += t / static_cast<long double>(n);
    }
    return static_cast<double>(s);
}

inline double col_mean(const double* v, size_t n) {      // colMeans(): one long double pass
    long double s = 0;
    for (size_t i = 0; i < n; ++i) s += v[i];
    return static_cast<double>(s / static_cast<long double>(n));
}

inline double sd(const double* v, size_t n) {            // sd(): cov.c, mean stored as double, sums in long double
    if (n < 2) return std::nan("");
    const long double m = mean(v, n);
    long double s = 0;
    for (size_t i = 0; i < n; ++i) { const long double d = v[i] - m; s += d * d; }
    return std::sqrt(static_cast<double>(s / static_cast<long double>(n - 1)));
}

// R's x ^ y (arithmetic.c R_POW / R_pow): x * x for y == 2, 1 for x == 1 or y == 0, otherwise pow().
inline double r_pow(double x, double y) {
    if (y == 2.0) return x * x;
    if (x == 1.0 || y == 0.0) return 1.0;
    return std::pow(x, y);
}

inline double box_cox(double x, double lam) {            // forecast::BoxCox
    if (lam < 0 && x < 0) return std::nan("");
    if (lam == 0) return std::log(x);
    const double s = (x > 0) ? 1.0 : (x < 0 ? -1.0 : 0.0);
    return (s * r_pow(std::fabs(x), lam) - 1) / lam;
}

inline double inv_box_cox(double x, double lam) {        // forecast::InvBoxCox (biasadj = FALSE)
    if (lam < 0 && x > -1 / lam) return std::nan("");
    if (lam == 0) return std::exp(x);
    const double xx = x * lam + 1;
    const double s = (xx > 0) ? 1.0 : (xx < 0 ? -1.0 : 0.0);
    return s * r_pow(std::fabs(xx), 1 / lam);
}

// optimize(f, c(ax, bx)): Brent_fmin (src/library/stats/src/optimize.c), tol = .Machine$double.eps^0.25
template <class F>
double brent_fmin(double ax, double bx, F f, double tol = 1.220703125e-04) {
    const double c = (3. - std::sqrt(5.)) * .5;
    const double eps = std::sqrt(2.220446049250313e-16);
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

// forecast::BoxCox.lambda(x, "guerrero", lower, upper) for a ts of frequency freq
inline double box_cox_lambda(const Vec& x, double lower, double upper, double freq) {
    for (double v : x) if (v <= 0) { lower = std::max(lower, 0.0); break; }
    const int n = static_cast<int>(x.size());
    if (n <= 2 * freq) return 1.0;
    const int period = static_cast<int>(std::nearbyint(std::max(2.0, freq)));
    const int nyr = n / period, nobst = nyr * period;
    Vec mu(nyr), sdv(nyr), rat(nyr);
    for (int j = 0; j < nyr; ++j) {
        const double* col = x.data() + (n - nobst) + j * period;
        mu[j] = col_mean(col, period);
        sdv[j] = sd(col, period);
    }
    auto cv = [&](double lam) {
        for (int j = 0; j < nyr; ++j) rat[j] = sdv[j] / r_pow(mu[j], 1 - lam);
        return sd(rat.data(), rat.size()) / mean(rat.data(), rat.size());
    };
    return brent_fmin(lower, upper, cv);
}

}  // namespace rs
}  // namespace tsf
