// Error measures, summary and WRMSSE (see metrics.hpp). The plain C interface is at the end of the file.
#include "metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

namespace tsf {
namespace metrics {

namespace {

const double NaN = std::numeric_limits<double>::quiet_NaN();
constexpr double MSMAPE_EPSILON = 0.1;

inline bool nan(double v) { return v != v; }
inline double sq(double v) { return v * v; }

// np.mean of a contiguous run.
inline double mean(const std::vector<double>& x) { return pairwise_sum(x.data(), x.size()) / static_cast<double>(x.size()); }

// np.nanmean of a run: NaN counts as 0 in the sum and is left out of the count; an all-NaN run gives NaN.
// `x` is overwritten.
double nanmean(std::vector<double>& x) {
    size_t cnt = 0;
    for (double& v : x) {
        if (nan(v)) v = 0.0;
        else ++cnt;
    }
    return pairwise_sum(x.data(), x.size()) / static_cast<double>(cnt);
}

// np.median of the values in `x` (no NaN inside): the middle one, or the mean of the two middle ones.
double median(std::vector<double>& x) {
    const size_t n = x.size();
    if (n == 0) return NaN;
    std::sort(x.begin(), x.end());
    return n % 2 ? x[n / 2] : (x[n / 2 - 1] + x[n / 2]) / 2.0;
}

// metrics._mase_scale on a training part without NaN.
double mase_scale(const std::vector<double>& tr, size_t lag, std::vector<double>& tmp) {
    if (tr.size() <= lag) return NaN;
    tmp.resize(tr.size() - lag);
    for (size_t i = lag; i < tr.size(); ++i) tmp[i - lag] = std::fabs(tr[i] - tr[i - lag]);
    return mean(tmp);
}

enum Id { SMAPE, MSMAPE, MASE, MAE, RMSE, WAPE, NRMSE_MEAN, NRMSE_RANGE, RMSSE, MSE, MAPE, MAAPE, MEDAE, RMSLE, ME, PBIAS };

const char* const NAMES[] = {"SMAPE", "mSMAPE", "MASE", "MAE", "RMSE", "WAPE", "nRMSE_Mean", "nRMSE_Range",
                             "RMSSE", "MSE",    "MAPE", "MAAPE", "MedAE", "RMSLE", "ME",       "PBias"};
constexpr int N_NAMES = sizeof NAMES / sizeof NAMES[0];

int find_metric(const std::string& name) {
    for (int i = 0; i < N_NAMES; ++i)
        if (name == NAMES[i]) return i;
    return -1;
}

// One measure for one series. f, a: the h forecasts / actuals; tr: the training part without NaN; e, e2: scratch.
double one(int id, const double* f, const double* a, size_t h, const std::vector<double>& tr, size_t lag,
           std::vector<double>& e, std::vector<double>& e2) {
    e.resize(h);
    switch (id) {
        case SMAPE:
            for (size_t j = 0; j < h; ++j) e[j] = 2.0 * std::fabs(f[j] - a[j]) / (std::fabs(f[j]) + std::fabs(a[j]));
            return nanmean(e);
        case MSMAPE:
            for (size_t j = 0; j < h; ++j) {
                const double d = std::fabs(f[j]) + std::fabs(a[j]) + MSMAPE_EPSILON;  // np.maximum keeps NaN
                e[j] = 2.0 * std::fabs(f[j] - a[j]) / (nan(d) ? d : std::max(0.5 + MSMAPE_EPSILON, d));
            }
            return nanmean(e);
        case MAE:
            for (size_t j = 0; j < h; ++j) e[j] = std::fabs(f[j] - a[j]);
            return nanmean(e);
        case MSE:
        case RMSE:
        case NRMSE_MEAN:
        case NRMSE_RANGE:
        case RMSSE: {
            for (size_t j = 0; j < h; ++j) e[j] = sq(f[j] - a[j]);
            const double m = nanmean(e);
            if (id == MSE) return m;
            if (id == RMSSE) {
                if (tr.size() < 2) return NaN;
                e2.resize(tr.size() - 1);
                for (size_t i = 1; i < tr.size(); ++i) e2[i - 1] = sq(tr[i] - tr[i - 1]);
                const double s = mean(e2);
                return (s > 0 && std::isfinite(m)) ? std::sqrt(m / s) : NaN;
            }
            const double r = std::sqrt(m);
            if (id == RMSE) return r;
            if (id == NRMSE_MEAN) {
                e2.assign(a, a + h);
                const double am = nanmean(e2);
                return am == 0 ? NaN : r / am;
            }
            double lo = NaN, hi = NaN;  // np.nanmax - np.nanmin
            for (size_t j = 0; j < h; ++j) {
                if (nan(a[j])) continue;
                if (nan(lo) || a[j] < lo) lo = a[j];
                if (nan(hi) || a[j] > hi) hi = a[j];
            }
            const double rng = hi - lo;
            return rng == 0 ? NaN : r / rng;
        }
        case WAPE:
        case PBIAS: {
            e2.resize(h);
            size_t ok = 0;
            for (size_t j = 0; j < h; ++j) {
                const bool good = !(nan(f[j]) || nan(a[j]));
                ok += good;
                e[j] = good ? (id == WAPE ? std::fabs(f[j] - a[j]) : f[j] - a[j]) : 0.0;
                e2[j] = good ? (id == WAPE ? std::fabs(a[j]) : a[j]) : 0.0;
            }
            const double num = pairwise_sum(e.data(), h), den = pairwise_sum(e2.data(), h);
            if (den == 0 || ok == 0) return NaN;
            return id == WAPE ? num / den * 100.0 : 100.0 * num / den;
        }
        case MASE: {
            e.clear();
            for (size_t j = 0; j < h; ++j)
                if (!(nan(f[j]) || nan(a[j]))) e.push_back(std::fabs(f[j] - a[j]));
            if (e.empty()) return NaN;
            const double err = mean(e);
            // R: mase <- MASE(te, f, seasonal scale); if (is.na(mase)) mase <- MASE(te, f, lag-1 scale)
            double scale = mase_scale(tr, lag, e2);
            double m = nan(scale) ? NaN : err / scale;
            if (nan(m)) {
                scale = mase_scale(tr, 1, e2);
                m = nan(scale) ? NaN : err / scale;
            }
            return std::isfinite(m) ? m : NaN;
        }
        case MAPE:
            for (size_t j = 0; j < h; ++j) e[j] = a[j] == 0 ? NaN : std::fabs(f[j] - a[j]) / std::fabs(a[j]);
            return 100.0 * nanmean(e);
        case MAAPE:
            for (size_t j = 0; j < h; ++j)
                e[j] = (a[j] == 0 && f[j] == 0) ? 0.0 : std::atan(std::fabs(f[j] - a[j]) / std::fabs(a[j]));
            return nanmean(e);
        case MEDAE:
            e.clear();
            for (size_t j = 0; j < h; ++j)
                if (!(nan(f[j]) || nan(a[j]))) e.push_back(std::fabs(f[j] - a[j]));
            return median(e);
        case RMSLE: {
            bool bad = false;
            for (size_t j = 0; j < h; ++j) {
                bad = bad || f[j] < 0 || a[j] < 0;
                e[j] = sq(std::log1p(f[j] < 0 ? NaN : f[j]) - std::log1p(a[j] < 0 ? NaN : a[j]));
            }
            const double r = std::sqrt(nanmean(e));
            return bad ? NaN : r;
        }
        case ME:
            for (size_t j = 0; j < h; ++j) e[j] = f[j] - a[j];
            return nanmean(e);
    }
    return NaN;
}

const char* const LEVEL_NAMES[WRMSSE_LEVELS] = {"Total", "State", "Store", "Category", "Department", "State x Category",
                                                "State x Department", "Store x Category", "Store x Department", "Item",
                                                "Item x State", "Item x Store"};
// columns of the hierarchy: 0 item_id, 1 dept_id, 2 cat_id, 3 store_id, 4 state_id
const std::vector<int> LEVEL_KEYS[WRMSSE_LEVELS] = {{}, {4}, {3}, {2}, {1}, {4, 2}, {4, 1}, {3, 2}, {3, 1}, {0}, {0, 4}, {0, 3}};

}  // namespace

namespace {

// numpy's DOUBLE_pairwise_sum
double pairwise(const double* a, size_t n) {
    if (n < 8) {
        double res = 0.0;
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
    return pairwise(a, n2) + pairwise(a + n2, n - n2);
}

}  // namespace

// np.add.reduce starts from +0.0, so a run of -0.0 sums to +0.0 (checked against NumPy 2.5).
double pairwise_sum(const double* a, size_t n) { return 0.0 + pairwise(a, n); }

int primary_period(const double* seasonality, int n) {
    if (!seasonality || n < 1) return 1;
    double lo = seasonality[0];
    for (int i = 1; i < n; ++i) lo = std::min(lo, seasonality[i]);
    return std::max(1, static_cast<int>(lo));
}

const std::vector<std::string>& all_metrics() {
    static const std::vector<std::string> names(NAMES, NAMES + N_NAMES);
    return names;
}

void error_table(const double* forecasts, const double* actuals, int64_t n, int64_t h, const double* train,
                 const int64_t* offsets, int lag, const std::vector<std::string>& metrics, double* out) {
    std::vector<int> ids;
    for (const auto& m : metrics) {
        const int id = find_metric(m);
        if (id < 0) throw std::invalid_argument("Unknown metric: " + m);
        ids.push_back(id);
    }
    const size_t k = ids.size(), hh = static_cast<size_t>(h);
    std::vector<double> tr, e, e2;
    for (int64_t s = 0; s < n; ++s) {
        tr.clear();
        if (train && offsets)
            for (int64_t i = offsets[s]; i < offsets[s + 1]; ++i)
                if (!nan(train[i])) tr.push_back(train[i]);
        const double* f = forecasts + s * h;
        const double* a = actuals + s * h;
        for (size_t j = 0; j < k; ++j) out[static_cast<size_t>(s) * k + j] = one(ids[j], f, a, hh, tr, static_cast<size_t>(std::max(1, lag)), e, e2);
    }
}

void series_scales(const double* train, const int64_t* offsets, int64_t n, int lag, double* out) {
    std::vector<double> tr, tmp;
    for (int64_t s = 0; s < n; ++s) {
        tr.clear();
        for (int64_t i = offsets[s]; i < offsets[s + 1]; ++i)
            if (!nan(train[i])) tr.push_back(train[i]);
        double v = mase_scale(tr, static_cast<size_t>(std::max(1, lag)), tmp);
        if (!(std::isfinite(v) && v > 0)) v = mase_scale(tr, 1, tmp);
        out[s] = (std::isfinite(v) && v > 0) ? v : NaN;
    }
}

Summary summarize(const double* values, int64_t n) {
    std::vector<double> ok;
    ok.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        if (std::isfinite(values[i])) ok.push_back(values[i]);
    Summary s;
    s.n_valid = static_cast<int64_t>(ok.size());
    s.n_excluded = n - s.n_valid;
    s.mean = ok.empty() ? NaN : mean(ok);
    s.median = median(ok);
    return s;
}

const char* wrmsse_level_name(int level) { return (level >= 0 && level < WRMSSE_LEVELS) ? LEVEL_NAMES[level] : nullptr; }

WrmsseResult wrmsse(const double* train, const int64_t* offsets, int64_t n, const double* test, const double* forecasts,
                    int64_t h, const std::vector<std::vector<std::string>>& hierarchy, const double* units) {
    if (hierarchy.size() != 5) throw std::invalid_argument("WRMSSE needs the 5 hierarchy columns");
    for (const auto& col : hierarchy)
        if (static_cast<int64_t>(col.size()) != n) throw std::invalid_argument("A hierarchy column has the wrong length");
    int64_t T = 0;
    for (int64_t i = 0; i < n; ++i) T = std::max(T, offsets[i + 1] - offsets[i]);
    if (T < WRMSSE_WINDOW) throw std::invalid_argument("WRMSSE needs at least 28 training observations");

    // wrmsse._right_align: series aligned on their last observation, np.nan_to_num applied
    const size_t TT = static_cast<size_t>(T), nn = static_cast<size_t>(n), hh = static_cast<size_t>(h);
    std::vector<double> Y(nn * TT, 0.0);
    const double big = std::numeric_limits<double>::max();
    for (size_t i = 0; i < nn; ++i) {
        const int64_t len = offsets[i + 1] - offsets[i];
        double* row = Y.data() + i * TT + (TT - static_cast<size_t>(len));
        for (int64_t t = 0; t < len; ++t) {
            const double v = train[offsets[i] + t];
            row[t] = nan(v) ? 0.0 : std::isinf(v) ? (v > 0 ? big : -big) : v;
        }
    }

    WrmsseResult res;
    std::vector<double> y_tr(TT), ts(hh), fs(hh), buf, d;
    std::vector<double> level_values;
    for (int L = 0; L < WRMSSE_LEVELS; ++L) {
        // groups in sorted key order (pandas groupby(sort=True)); '\0' joins the keys so that tuples sort like strings
        std::map<std::string, std::vector<size_t>> groups;
        for (size_t i = 0; i < nn; ++i) {
            std::string key;
            for (int c : LEVEL_KEYS[L]) {
                key += hierarchy[static_cast<size_t>(c)][i];
                key.push_back('\0');
            }
            groups[key].push_back(i);
        }
        double tot_w = 0.0, acc = 0.0;
        for (const auto& g : groups) {
            const std::vector<size_t>& idx = g.second;
            // sums over the group's rows, one row after the other (np.sum(axis=0))
            std::fill(y_tr.begin(), y_tr.end(), 0.0);
            std::fill(ts.begin(), ts.end(), 0.0);
            std::fill(fs.begin(), fs.end(), 0.0);
            for (size_t i : idx) {
                const double* yr = Y.data() + i * TT;
                for (size_t t = 0; t < TT; ++t) y_tr[t] += yr[t];
                for (size_t j = 0; j < hh; ++j) {
                    ts[j] += test[i * hh + j];
                    fs[j] += forecasts[i * hh + j];
                }
            }
            size_t first = 0;
            while (first < TT && y_tr[first] == 0) ++first;
            if (first == TT || TT - first < 2) continue;
            d.resize(TT - first - 1);
            for (size_t t = first + 1; t < TT; ++t) d[t - first - 1] = sq(y_tr[t] - y_tr[t - 1]);
            const double den = mean(d);
            buf.resize(hh);
            for (size_t j = 0; j < hh; ++j) buf[j] = sq(ts[j] - fs[j]);
            const double num = nanmean(buf);
            // the group's weight: np.sum over its (rows x 28) block
            buf.resize(idx.size() * WRMSSE_WINDOW);
            for (size_t r = 0; r < idx.size(); ++r) {
                const double* src = units ? units + idx[r] * WRMSSE_WINDOW : Y.data() + idx[r] * TT + (TT - WRMSSE_WINDOW);
                std::memcpy(buf.data() + r * WRMSSE_WINDOW, src, WRMSSE_WINDOW * sizeof(double));
            }
            const double w = pairwise_sum(buf.data(), buf.size());
            if (den > 0 && std::isfinite(num)) {
                acc += w * std::sqrt(num / den);
                tot_w += w;
            }
        }
        res.level[L] = tot_w > 0 ? acc / tot_w : NaN;
        res.n_series[L] = static_cast<int64_t>(groups.size());
        level_values.push_back(res.level[L]);
    }
    res.total = nanmean(level_values);
    return res;
}

}  // namespace metrics
}  // namespace tsf

// ================================================================================================ plain C interface
// const char* tsf_metrics_names()                      "SMAPE,mSMAPE,..." (every measure tsf_metrics_table accepts)
// int tsf_primary_period(seasonality, n)               the MASE lag
// int tsf_metrics_table(forecasts, actuals, n, h, train, offsets, lag, names, out, msg, msg_len)
//       forecasts/actuals: n x h row-major (NaN beyond a series' test length); train/offsets as in tsf_forecast_batch
//       names: comma-separated measures; out: n x (number of names), row-major. 0 = ok, -1 = unknown measure, -2 = bad arguments
// void tsf_metrics_summary(values, n, out4)            out4 = mean, median, n_valid, n_excluded (finite values only)
// int tsf_wrmsse(train, offsets, n, test, forecasts, h, hierarchy, units, level, n_series, total, msg, msg_len)
//       hierarchy: 5 * n UTF-8 strings, hierarchy[c * n + i] = column c (item, dept, cat, store, state) of series i
//       units: n x 28 weights or NULL (unit sales); level / n_series: 12 values each. 0 = ok, -1 = error, -2 = bad arguments
#if defined(_WIN32)
#define TSFM_API extern "C" __declspec(dllexport)
#else
#define TSFM_API extern "C" __attribute__((visibility("default")))
#endif

namespace {

void metrics_msg(char* msg, int32_t msg_len, const char* text) {
    if (msg && msg_len > 0) std::snprintf(msg, static_cast<size_t>(msg_len), "%s", text);
}

}  // namespace

TSFM_API const char* tsf_metrics_names() {
    static const std::string joined = [] {
        std::string s;
        for (const auto& m : tsf::metrics::all_metrics()) s += (s.empty() ? "" : ",") + m;
        return s;
    }();
    return joined.c_str();
}

TSFM_API int32_t tsf_primary_period(const double* seasonality, int32_t n) { return tsf::metrics::primary_period(seasonality, n); }

TSFM_API int32_t tsf_metrics_table(const double* forecasts, const double* actuals, int64_t n, int64_t h, const double* train,
                                   const int64_t* offsets, int32_t lag, const char* names, double* out, char* msg,
                                   int32_t msg_len) {
    metrics_msg(msg, msg_len, "");
    if (!forecasts || !actuals || !names || !out || n < 0 || h < 1) { metrics_msg(msg, msg_len, "Bad arguments"); return -2; }
    std::vector<std::string> metrics;
    for (const char* p = names;;) {
        const char* q = std::strchr(p, ',');
        metrics.emplace_back(p, q ? static_cast<size_t>(q - p) : std::strlen(p));
        if (!q) break;
        p = q + 1;
    }
    try {
        tsf::metrics::error_table(forecasts, actuals, n, h, train, offsets, lag, metrics, out);
        return 0;
    } catch (const std::exception& err) {
        metrics_msg(msg, msg_len, err.what());
        return -1;
    }
}

TSFM_API void tsf_metrics_summary(const double* values, int64_t n, double* out4) {
    if (!values || !out4 || n < 0) return;
    const tsf::metrics::Summary s = tsf::metrics::summarize(values, n);
    out4[0] = s.mean;
    out4[1] = s.median;
    out4[2] = static_cast<double>(s.n_valid);
    out4[3] = static_cast<double>(s.n_excluded);
}

TSFM_API int32_t tsf_wrmsse(const double* train, const int64_t* offsets, int64_t n, const double* test, const double* forecasts,
                            int64_t h, const char* const* hierarchy, const double* units, double* level, int64_t* n_series,
                            double* total, char* msg, int32_t msg_len) {
    metrics_msg(msg, msg_len, "");
    if (!train || !offsets || !test || !forecasts || !hierarchy || !level || !n_series || !total || n < 1 || h < 1) {
        metrics_msg(msg, msg_len, "Bad arguments");
        return -2;
    }
    try {
        std::vector<std::vector<std::string>> cols(5, std::vector<std::string>(static_cast<size_t>(n)));
        for (size_t c = 0; c < 5; ++c)
            for (size_t i = 0; i < static_cast<size_t>(n); ++i) cols[c][i] = hierarchy[c * static_cast<size_t>(n) + i];
        const tsf::metrics::WrmsseResult r = tsf::metrics::wrmsse(train, offsets, n, test, forecasts, h, cols, units);
        for (int L = 0; L < tsf::metrics::WRMSSE_LEVELS; ++L) {
            level[L] = r.level[L];
            n_series[L] = r.n_series[L];
        }
        *total = r.total;
        return 0;
    } catch (const std::exception& err) {
        metrics_msg(msg, msg_len, err.what());
        return -1;
    }
}
