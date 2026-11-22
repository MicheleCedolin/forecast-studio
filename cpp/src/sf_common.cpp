#include "sf_common.hpp"

namespace tsf {

double np_sum(const double* a, size_t n) {
    if (n < 8) {
        double res = -0.0;
        for (size_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return np_sum(a, n2) + np_sum(a + n2, n - n2);
}

Vec sm_seasonal(const Vec& x, int period, bool mult) {
    const int n = static_cast<int>(x.size());
    Vec filt;
    if (period % 2 == 0) { filt.assign(period + 1, 1.0 / period); filt.front() = filt.back() = 0.5 / period; }
    else filt.assign(period, 1.0 / period);
    const int L = static_cast<int>(filt.size());
    const int head = (L + 1) / 2 - 1;
    Vec trend(n, NAN);
    for (int j = 0; j + L <= n; ++j) {
        double s = 0.0;
        for (int k = 0; k < L; ++k) s += x[j + k] * filt[L - 1 - k];
        trend[j + head] = s;
    }
    Vec avg(period);
    for (int i = 0; i < period; ++i) {
        Vec v;
        for (int t = i; t < n; t += period)
            if (!std::isnan(trend[t])) v.push_back(mult ? x[t] / trend[t] : x[t] - trend[t]);
        avg[i] = np_mean(v.data(), v.size());
    }
    const double m = np_mean(avg.data(), avg.size());
    for (double& v : avg) v = mult ? v / m : v - m;
    Vec seas(n);
    for (int t = 0; t < n; ++t) seas[t] = avg[t % period];
    return seas;
}

}  // namespace tsf
