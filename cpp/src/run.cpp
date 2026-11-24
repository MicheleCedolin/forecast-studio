// A complete run inside tsfcore (see run.hpp). The plain C interface is at the end of the file.
#include "run.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>

#include "metrics.hpp"

// tsfcore's forecasting entry points (capi.cpp)
extern "C" int tsf_forecast_batch(const char* method, const double* values, const int64_t* offsets, int32_t n_series, int32_t h,
                                  const int32_t* horizons, const double* periods, int32_t n_periods, int32_t n_threads,
                                  double* out, int32_t* flags, char* msg, int32_t msg_len);
extern "C" const char* tsf_version();

namespace tsf {
namespace run {

namespace {

namespace fs = std::filesystem;
using tsf::io::py_repr;

const char* const DEFAULT_METRICS[] = {"SMAPE", "mSMAPE", "MASE", "MAE", "RMSE", "WAPE", "nRMSE_Mean", "nRMSE_Range"};
const char* const INTERMITTENT[] = {"croston_classic", "croston_sba", "croston_optimized", "tsb", "adida", "imapa"};

inline bool nan(double v) { return v != v; }

#if defined(_WIN32)
const char* const PLATFORM_EOL = "\r\n";
#else
const char* const PLATFORM_EOL = "\n";
#endif

// ---------------------------------------------------------------------------------------------- missing values
// runner._apply_missing_policy; `y` has at least one value that is not missing.
void apply_missing_policy(std::vector<double>& y, const std::string& policy) {
    const size_t n = y.size();
    if (policy == "zero") {  // np.nan_to_num(y, nan=0.0): infinities become the largest finite numbers
        const double big = std::numeric_limits<double>::max();
        for (double& v : y) v = nan(v) ? 0.0 : std::isinf(v) ? (v > 0 ? big : -big) : v;
        return;
    }
    if (policy == "locf") {  // pandas ffill().bfill()
        for (size_t i = 1; i < n; ++i)
            if (nan(y[i])) y[i] = y[i - 1];
        for (size_t i = n; i-- > 1;)
            if (nan(y[i - 1])) y[i - 1] = y[i];
        return;
    }
    // "interpolate": pandas interpolate(method="linear", limit_direction="both") = np.interp over the positions
    std::vector<size_t> xp;
    for (size_t i = 0; i < n; ++i)
        if (!nan(y[i])) xp.push_back(i);
    const std::vector<double> fp = [&] {
        std::vector<double> v;
        for (size_t i : xp) v.push_back(y[i]);
        return v;
    }();
    size_t j = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!nan(y[i])) continue;
        if (i < xp.front()) { y[i] = fp.front(); continue; }
        if (i > xp.back()) { y[i] = fp.back(); continue; }
        while (xp[j + 1] < i) ++j;
        const double x = static_cast<double>(i), x0 = static_cast<double>(xp[j]), x1 = static_cast<double>(xp[j + 1]);
        const double slope = (fp[j + 1] - fp[j]) / (x1 - x0);
        double r = slope * (x - x0) + fp[j];
        if (nan(r)) {
            r = slope * (x - x1) + fp[j + 1];
            if (nan(r) && fp[j] == fp[j + 1]) r = fp[j];
        }
        y[i] = r;
    }
}

// ---------------------------------------------------------------------------------------------- CSV / JSON text
// csv.writer with QUOTE_MINIMAL
void csv_field(std::string& out, const std::string& s) {
    if (s.find_first_of(",\"\r\n") == std::string::npos) { out += s; return; }
    out.push_back('"');
    for (char c : s) {
        if (c == '"') out.push_back('"');
        out.push_back(c);
    }
    out.push_back('"');
}

void csv_number(std::string& out, double v) {
    if (!nan(v)) out += py_repr(v);  // missing = empty field
}

void json_string(std::string& out, const std::string& s) {
    out.push_back('"');
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out.push_back('\\'); out.push_back(static_cast<char>(c)); }
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); out += b; }
        else out.push_back(static_cast<char>(c));
    }
    out.push_back('"');
}

std::string jstr(const std::string& s) { std::string o; json_string(o, s); return o; }
std::string jnum(double v) { return nan(v) ? "NaN" : std::isinf(v) ? (v > 0 ? "Infinity" : "-Infinity") : py_repr(v); }
std::string jbool(bool b) { return b ? "true" : "false"; }

template <class T, class F>
std::string jlist(const std::vector<T>& v, F item) {
    std::string o = "[";
    for (size_t i = 0; i < v.size(); ++i) o += (i ? ", " : "") + item(v[i]);
    return o + "]";
}

void write_text(const fs::path& path, const std::string& text) {
    std::ofstream f(path, std::ios::binary);
    f.write(text.data(), static_cast<std::streamsize>(text.size()));
    f.flush();
    if (!f) throw RunError("Could not write " + path.u8string());
}

std::string utc_now(const char* format) {
    const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[64];
    std::strftime(buf, sizeof buf, format, &tm);
    return buf;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// ---------------------------------------------------------------------------------------------- SHA-256
struct Sha256 {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    unsigned char block[64];
    size_t fill = 0;
    uint64_t total = 0;

    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void compress(const unsigned char* p) {
        static const uint32_t K[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
            0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
            0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
            0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) | (uint32_t(p[4 * i + 2]) << 8) | uint32_t(p[4 * i + 3]);
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    void update(const unsigned char* p, size_t n) {
        total += n;
        while (n) {
            const size_t k = std::min(n, 64 - fill);
            std::memcpy(block + fill, p, k);
            fill += k; p += k; n -= k;
            if (fill == 64) { compress(block); fill = 0; }
        }
    }

    std::string hex() {
        const uint64_t bits = total * 8;
        const unsigned char one = 0x80, zero = 0;
        update(&one, 1);
        while (fill != 56) update(&zero, 1);
        unsigned char len[8];
        for (int i = 0; i < 8; ++i) len[i] = static_cast<unsigned char>(bits >> (56 - 8 * i));
        update(len, 8);
        char out[65];
        for (int i = 0; i < 8; ++i) std::snprintf(out + 8 * i, 9, "%08x", h[i]);
        return out;
    }
};

std::string plain_method(const std::string& spec) { return spec.substr(0, spec.find('?')); }

}  // namespace

// ---------------------------------------------------------------------------------------------- public
std::vector<double> seasonality_for(const tsf::io::Header& header) {
    if (!header.has_frequency) return {1.0};
    static const std::map<std::string, std::vector<double>> MAP = {
        {"4_seconds", {21600.0, 151200.0, 7889400.0}}, {"minutely", {1440.0, 10080.0, 525960.0}},
        {"10_minutes", {144.0, 1008.0, 52596.0}},      {"15_minutes", {96.0, 672.0, 35064.0}},
        {"half_hourly", {48.0, 336.0, 17532.0}},       {"hourly", {24.0, 168.0, 8766.0}},
        {"daily", {7.0}},                              {"weekly", {365.25 / 7}},
        {"monthly", {12.0}},                           {"quarterly", {4.0}},
        {"yearly", {1.0}}};
    const auto it = MAP.find(header.frequency);
    if (it == MAP.end())
        throw RunError("Invalid frequency: '" + header.frequency + "'. Valid values: 4_seconds, minutely, 10_minutes, 15_minutes, "
                       "half_hourly, hourly, daily, weekly, monthly, quarterly, yearly");
    return it->second;
}

Prepared prepare(const tsf::io::Dataset& ds, const Config& cfg, int64_t cut, int64_t min_extra) {
    const std::string& policy = cfg.missing_policy;
    if (policy != "error" && policy != "zero" && policy != "locf" && policy != "interpolate")
        throw RunError("Invalid missing-value policy: '" + policy + "' (valid: error, zero, locf, interpolate)");
    Prepared p;
    const int64_t H = cfg.horizon > 0 ? cfg.horizon : (ds.header.has_horizon ? ds.header.horizon : 0);
    if (H < 1)
        throw RunError("The forecast horizon could not be determined; the data header has no @horizon. Enter the horizon manually.");
    p.horizon = H;
    p.seasonality = cfg.seasonality.empty() ? seasonality_for(ds.header) : cfg.seasonality;

    // TSFDataset.series_names: the series_name attribute, else T1, T2, ...
    const std::vector<std::string>* name_col = nullptr;
    for (size_t a = 0; a < ds.header.attributes.size(); ++a)
        if (ds.header.attributes[a].first == "series_name") { name_col = &ds.attrs[a]; break; }
    size_t n = ds.values.size();
    if (cfg.max_series > 0) n = std::min(n, static_cast<size_t>(cfg.max_series));

    const size_t hh = static_cast<size_t>(H);
    for (size_t i = 0; i < n; ++i) {
        const std::string name = name_col ? (*name_col)[i] : "T" + std::to_string(i + 1);
        const std::vector<double>& full = ds.values[i];
        const int64_t len = static_cast<int64_t>(full.size());
        if (len < H + 2 + min_extra) {
            const std::string need = min_extra ? "horizon+2+(origins-1)*step = " + std::to_string(H + 2 + min_extra)
                                               : "horizon+2 = " + std::to_string(H + 2);
            p.excluded.push_back({static_cast<int64_t>(i), name, "series too short (length " + std::to_string(len) + " < " + need + ")"});
            continue;
        }
        const size_t used = static_cast<size_t>(len - cut);
        std::vector<double> tr(full.begin(), full.begin() + static_cast<std::ptrdiff_t>(used - hh));
        const double* te = full.data() + (used - hh);
        int64_t miss = 0;
        for (double v : tr) miss += nan(v);
        if (miss) {
            ++p.series_with_missing;
            if (policy == "error")
                throw RunError("Series '" + name + "' has " + std::to_string(miss) + " missing values in its training part. Choose a "
                               "missing-value policy: 'zero', 'locf' (carry the last value forward) or 'interpolate' (linear "
                               "interpolation). We do not change these values silently; the choice affects the results.");
            if (miss == static_cast<int64_t>(tr.size())) {
                p.excluded.push_back({static_cast<int64_t>(i), name, "training part entirely missing"});
                continue;
            }
            apply_missing_policy(tr, policy);
            p.imputed_values += miss;
        }
        for (size_t j = 0; j < hh; ++j) p.test_missing_values += nan(te[j]);
        p.train.push_back(std::move(tr));
        p.test.insert(p.test.end(), te, te + hh);
        p.names.push_back(name);
    }
    if (p.train.empty()) throw RunError("No series left to run (all were excluded).");

    const std::string method = plain_method(cfg.method);
    if (cfg.integer_conversion && std::find(std::begin(INTERMITTENT), std::end(INTERMITTENT), method) != std::end(INTERMITTENT))
        p.warnings.push_back(
            "Rounding to integers is ON (catalogue or choice): intermittent-demand methods forecast a flat demand RATE (often < 0.5), which "
            "becomes 0 when rounded; all forecasts can come out as 0. To see the rate forecast, turn rounding off (in the interface: Settings, "
            "'Round to integers' = Off; on the command line: --integer-conversion off).");
    if (!p.excluded.empty())
        p.warnings.push_back(std::to_string(p.excluded.size()) + " series excluded (details in manifest.json -> split.excluded).");
    if (p.test_missing_values)
        p.warnings.push_back("There are " + std::to_string(p.test_missing_values) +
                             " missing values in the test part; the measures exclude these pairs.");
    return p;
}

std::string table_csv(const std::string& first_header, const std::vector<std::string>& first_column,
                      const std::vector<std::string>& headers, const double* values, const std::string& eol) {
    std::string out;
    csv_field(out, first_header);
    for (const auto& hname : headers) { out.push_back(','); csv_field(out, hname); }
    out += eol;
    const size_t k = headers.size();
    for (size_t i = 0; i < first_column.size(); ++i) {
        csv_field(out, first_column[i]);
        for (size_t j = 0; j < k; ++j) { out.push_back(','); csv_number(out, values[i * k + j]); }
        out += eol;
    }
    return out;
}

std::string sha256_file(const std::string& utf8_path) {
    std::ifstream f(fs::u8path(utf8_path), std::ios::binary);
    if (!f) return "";
    Sha256 sha;
    std::vector<char> buf(1 << 20);
    while (f) {
        f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        const std::streamsize got = f.gcount();
        if (got > 0) sha.update(reinterpret_cast<const unsigned char*>(buf.data()), static_cast<size_t>(got));
    }
    return sha.hex();
}

namespace {

// ---------------------------------------------------------------------------------------------- forecasting
struct Forecasts {
    std::vector<double> values;       // training parts, concatenated
    std::vector<int64_t> offsets;
    std::vector<double> fc;           // n x H
    std::vector<int64_t> fallbacks;   // series that received the fallback forecast
    std::string message;              // first fallback reason (as the engine reports it)
    double seconds = 0.0;
};

// cpp_engine.CppEngine.run for a local method
Forecasts forecast(const Prepared& prep, const Config& cfg) {
    const size_t n = prep.train.size(), H = static_cast<size_t>(prep.horizon);
    if (n > static_cast<size_t>(std::numeric_limits<int32_t>::max()) || H > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
        throw RunError("Too many series or too long a horizon for one run.");
    Forecasts f;
    f.offsets.assign(n + 1, 0);
    for (size_t i = 0; i < n; ++i) f.offsets[i + 1] = f.offsets[i] + static_cast<int64_t>(prep.train[i].size());
    f.values.reserve(static_cast<size_t>(f.offsets[n]));
    for (const auto& tr : prep.train) f.values.insert(f.values.end(), tr.begin(), tr.end());
    f.fc.resize(n * H);
    std::vector<int32_t> flags(n, 0);
    char msg[512] = "";
    const auto t0 = std::chrono::steady_clock::now();
    const int rc = tsf_forecast_batch(cfg.method.c_str(), f.values.data(), f.offsets.data(), static_cast<int32_t>(n),
                                      static_cast<int32_t>(H), nullptr, prep.seasonality.data(),
                                      static_cast<int32_t>(prep.seasonality.size()), cfg.threads, f.fc.data(), flags.data(), msg,
                                      sizeof msg);
    f.seconds = seconds_since(t0);
    if (rc == -1)
        throw RunError("The C++ engine does not know the method '" + cfg.method + "'" +
                       (cfg.method.find('?') != std::string::npos ? " or does not take settings for it." : "."));
    if (rc != 0) throw RunError("The C++ engine rejected the job (code " + std::to_string(rc) + ").");
    if (cfg.integer_conversion)
        for (double& v : f.fc) v = std::nearbyint(v);  // round half to even, as np.round
    for (size_t i = 0; i < n; ++i)
        if (flags[i]) f.fallbacks.push_back(static_cast<int64_t>(i));
    f.message = msg;
    return f;
}

// ---------------------------------------------------------------------------------------------- prediction intervals
// Python's round(x, 6)
double round6(double x) {
    char b[64];
    std::snprintf(b, sizeof b, "%.6f", x);
    return std::strtod(b, nullptr);
}

// intervals.probs_for: the quantile probabilities of the interval bounds, ascending
std::vector<double> probs_for(const std::vector<int>& levels) {
    std::vector<double> ps;
    for (int lv : levels) {
        const double a = 1 - lv / 100.0;
        ps.push_back(round6(a / 2));
        ps.push_back(round6(1 - a / 2));
    }
    std::sort(ps.begin(), ps.end());
    ps.erase(std::unique(ps.begin(), ps.end()), ps.end());
    return ps;
}

// np.quantile(x, p) with the default (linear, type 7) method; `sorted` is not empty
double quantile7(const std::vector<double>& sorted, double p) {
    const size_t n = sorted.size();
    const double vi = static_cast<double>(n - 1) * p;
    if (vi >= static_cast<double>(n - 1)) return sorted.back();
    if (vi < 0) return sorted.front();
    const double lo = std::floor(vi), t = vi - lo;
    const double a = sorted[static_cast<size_t>(lo)], b = sorted[static_cast<size_t>(lo) + 1], diff = b - a;
    return t >= 0.5 ? b - diff * (1 - t) : a + diff * t;  // numpy's _lerp
}

// intervals.conformal_quantiles: q[p * H + h] of the scaled signed calibration errors
std::vector<double> conformal_quantiles(const std::vector<double>& cal_fc, const std::vector<double>& cal_actual,
                                        const std::vector<double>& cal_scale, const std::vector<double>& probs, size_t H) {
    const size_t n = cal_scale.size(), P = probs.size();
    std::vector<double> q(P * H, std::numeric_limits<double>::quiet_NaN()), col;
    for (size_t h = 0; h < H; ++h) {
        col.clear();
        for (size_t i = 0; i < n; ++i) {
            const double e = (cal_actual[i * H + h] - cal_fc[i * H + h]) / cal_scale[i];
            if (std::isfinite(e)) col.push_back(e);
        }
        if (col.empty()) continue;
        std::sort(col.begin(), col.end());
        for (size_t k = 0; k < P; ++k) q[k * H + h] = quantile7(col, probs[k]);
    }
    return q;
}

// intervals.interval_metrics. bounds[k] is the n x H bound of probs[k]. Appends Coverage<level>..., MSIS, WQL to `cols`.
void interval_metrics(const std::vector<double>& fc, const std::vector<double>& actual, const std::vector<double>& scale,
                      const std::vector<int>& levels, const std::vector<double>& probs,
                      const std::vector<std::vector<double>>& bounds, size_t H, std::vector<std::vector<double>>& cols) {
    const size_t n = scale.size();
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    const auto bound_of = [&](double p) -> const std::vector<double>& {
        return bounds[static_cast<size_t>(std::find(probs.begin(), probs.end(), p) - probs.begin())];
    };
    const auto no_inf = [](double v) { return std::isinf(v) ? std::numeric_limits<double>::quiet_NaN() : v; };
    for (int lv : levels) {
        const double a = 1 - lv / 100.0;
        const std::vector<double>&lo = bound_of(round6(a / 2)), &hi = bound_of(round6(1 - a / 2));
        std::vector<double> c(n);
        for (size_t i = 0; i < n; ++i) {
            int64_t inside = 0, cnt = 0;
            for (size_t h = 0; h < H; ++h) {
                const double y = actual[i * H + h], l = lo[i * H + h], u = hi[i * H + h];
                const bool ok = std::isfinite(y) && std::isfinite(l) && std::isfinite(u);
                cnt += ok;
                inside += ok && y >= l && y <= u;
            }
            c[i] = cnt > 0 ? static_cast<double>(inside) / static_cast<double>(cnt) : NaN;
        }
        cols.push_back(std::move(c));
    }
    {
        const int lv = std::find(levels.begin(), levels.end(), 95) != levels.end() ? 95 : *std::max_element(levels.begin(), levels.end());
        const double a = 1 - lv / 100.0, w = 2 / a;
        const std::vector<double>&lo = bound_of(round6(a / 2)), &hi = bound_of(round6(1 - a / 2));
        std::vector<double> c(n), row(H);
        for (size_t i = 0; i < n; ++i) {
            int64_t cnt = 0;
            for (size_t h = 0; h < H; ++h) {
                const double y = actual[i * H + h], l = lo[i * H + h], u = hi[i * H + h];
                const double score = (u - l) + w * (l - y) * (y < l ? 1.0 : 0.0) + w * (y - u) * (y > u ? 1.0 : 0.0);
                const bool ok = std::isfinite(score);
                cnt += ok;
                row[h] = ok ? score : 0.0;
            }
            const double msis = cnt > 0 ? tsf::metrics::pairwise_sum(row.data(), H) / static_cast<double>(cnt) : NaN;
            c[i] = no_inf(msis / scale[i]);
        }
        cols.push_back(std::move(c));
    }
    {
        // quantiles: the median (the point forecast) first, then the bounds in ascending order
        std::vector<double> ps = {0.5};
        std::vector<const std::vector<double>*> bs = {&fc};
        for (size_t k = 0; k < probs.size(); ++k) { ps.push_back(probs[k]); bs.push_back(&bounds[k]); }
        std::vector<double> c(n), row(H);
        for (size_t i = 0; i < n; ++i) {
            for (size_t h = 0; h < H; ++h) row[h] = std::isnan(actual[i * H + h]) ? 0.0 : std::fabs(actual[i * H + h]);
            const double denom = tsf::metrics::pairwise_sum(row.data(), H);
            double total = 0.0;
            for (size_t k = 0; k < ps.size(); ++k) {
                for (size_t h = 0; h < H; ++h) {
                    const double d = actual[i * H + h] - (*bs[k])[i * H + h];
                    row[h] = std::isfinite(d) ? std::max(ps[k] * d, (ps[k] - 1) * d) : 0.0;
                }
                total += 2 * tsf::metrics::pairwise_sum(row.data(), H) / denom;
            }
            c[i] = denom > 0 ? no_inf(total / static_cast<double>(ps.size())) : NaN;
        }
        cols.push_back(std::move(c));
    }
}

// One forecast origin: its split, forecasts and measures (n x columns, row-major).
struct Origin {
    int64_t cut = 0;
    Prepared prep;
    Forecasts f;
    std::vector<double> table;
};

void summary_rows(std::string& text, const std::string& prefix, const std::vector<std::string>& columns, const double* table,
                  size_t n, const std::string& eol) {
    const size_t k = columns.size();
    std::vector<double> col(n);
    for (size_t j = 0; j < k; ++j) {
        for (size_t i = 0; i < n; ++i) col[i] = table[i * k + j];
        const tsf::metrics::Summary s = tsf::metrics::summarize(col.data(), static_cast<int64_t>(n));
        text += prefix;
        csv_field(text, columns[j]);
        text.push_back(',');
        csv_number(text, s.mean);
        text.push_back(',');
        csv_number(text, s.median);
        text += "," + std::to_string(s.n_valid) + "," + std::to_string(s.n_excluded) + eol;
    }
}

}  // namespace

std::string run(const std::string& tsf_path, const std::string& out_dir, const Config& cfg) {
    const auto t_start = std::chrono::steady_clock::now();
    const tsf::io::Dataset ds = tsf::io::read_tsf(tsf::io::file_source(tsf_path));
    const std::string method = plain_method(cfg.method);
    const std::string eol = cfg.eol.empty() ? PLATFORM_EOL : cfg.eol;

    // ---- runner.prepare_origins
    const int64_t n_orig = cfg.origins;
    if (n_orig < 1) throw RunError("The number of forecast origins must be at least 1.");
    Prepared latest = prepare(ds, cfg);
    const int64_t step = cfg.origin_step != 0 ? cfg.origin_step : latest.horizon;
    if (step < 1) throw RunError("The step between forecast origins must be at least 1.");
    std::vector<int> levels = cfg.intervals;
    for (int lv : levels)
        if (lv < 50 || lv > 99) throw RunError("Prediction-interval levels must be whole percentages between 50 and 99.");
    std::sort(levels.begin(), levels.end());
    levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
    const int64_t extra = (n_orig - 1) * step + (levels.empty() ? 0 : latest.horizon);
    if (extra) latest = prepare(ds, cfg, 0, extra);
    const size_t H = static_cast<size_t>(latest.horizon);
    const int lag = tsf::metrics::primary_period(latest.seasonality.data(), static_cast<int>(latest.seasonality.size()));

    std::vector<std::string> metric_names = cfg.metrics;
    if (metric_names.empty()) metric_names.assign(std::begin(DEFAULT_METRICS), std::end(DEFAULT_METRICS));
    std::vector<std::string> columns = metric_names;
    for (int lv : levels) columns.push_back("Coverage" + std::to_string(lv));
    if (!levels.empty()) { columns.push_back("MSIS"); columns.push_back("WQL"); }
    const size_t k = columns.size();
    const std::vector<double> probs = probs_for(levels);

    // ---- one complete fit per origin, oldest first; the last origin is the usual split
    std::vector<Origin> origins;
    std::vector<std::vector<double>> latest_bounds;
    std::vector<std::string> notes;
    std::vector<std::pair<int64_t, std::string>> fallback_rows;  // (series index, message), all origins
    double engine_sec = 0.0;
    const auto t_engine = std::chrono::steady_clock::now();
    for (int64_t o = n_orig - 1; o >= 0; --o) {
        Origin org;
        org.cut = o * step;
        org.prep = org.cut == 0 ? latest : prepare(ds, cfg, org.cut, extra);
        org.f = forecast(org.prep, cfg);
        engine_sec += org.f.seconds;
        const size_t n = org.prep.train.size();
        std::vector<double> base(n * metric_names.size());
        try {
            tsf::metrics::error_table(org.f.fc.data(), org.prep.test.data(), static_cast<int64_t>(n), latest.horizon, org.f.values.data(),
                                      org.f.offsets.data(), lag, metric_names, base.data());
        } catch (const std::invalid_argument& err) {
            throw RunError(err.what());
        }
        std::vector<std::vector<double>> extra_cols;
        if (!levels.empty()) {
            // conformal intervals: the same model fitted one horizon earlier gives the calibration errors
            const Prepared cal = prepare(ds, cfg, org.cut + latest.horizon, extra);
            const Forecasts cf = forecast(cal, cfg);
            engine_sec += cf.seconds;
            std::vector<double> cal_scale(cal.train.size()), scale(n);
            tsf::metrics::series_scales(cf.values.data(), cf.offsets.data(), static_cast<int64_t>(cal.train.size()), lag, cal_scale.data());
            tsf::metrics::series_scales(org.f.values.data(), org.f.offsets.data(), static_cast<int64_t>(n), lag, scale.data());
            const std::vector<double> q = conformal_quantiles(cf.fc, cal.test, cal_scale, probs, H);
            std::vector<std::vector<double>> bounds(probs.size(), std::vector<double>(n * H));
            for (size_t p = 0; p < probs.size(); ++p)
                for (size_t i = 0; i < n; ++i)
                    for (size_t h = 0; h < H; ++h) bounds[p][i * H + h] = org.f.fc[i * H + h] + q[p * H + h] * scale[i];
            interval_metrics(org.f.fc, org.prep.test, scale, levels, probs, bounds, H, extra_cols);
            if (org.cut == 0) latest_bounds = std::move(bounds);
        }
        org.table.resize(n * k);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < metric_names.size(); ++j) org.table[i * k + j] = base[i * metric_names.size() + j];
            for (size_t j = 0; j < extra_cols.size(); ++j) org.table[i * k + metric_names.size() + j] = extra_cols[j][i];
        }
        const std::string message = org.f.message.empty() ? method + ": fallback" : org.f.message;
        for (int64_t i : org.f.fallbacks) fallback_rows.emplace_back(i, message);
        if (!org.f.fallbacks.empty()) {
            const std::string note = std::to_string(org.f.fallbacks.size()) + " series received a fallback (SNaive/naive) forecast "
                                     "instead of the model. First reason: " + org.f.message.substr(0, 300);
            if (std::find(notes.begin(), notes.end(), note) == notes.end()) notes.push_back(note);
        }
        origins.push_back(std::move(org));
    }
    const double engine_wall = seconds_since(t_engine);
    const Origin& last = origins.back();
    const size_t n = latest.train.size();

    // ---- measures per series: the latest origin, or the mean over the origins (pandas groupby().mean(): NaN left out,
    //      Kahan summation in the order of the rows, oldest origin first)
    std::vector<double> table = last.table;
    if (n_orig > 1) {
        std::map<std::string, size_t> row_of;
        for (size_t i = 0; i < n; ++i) row_of.emplace(latest.names[i], i);
        std::vector<double> sum(n * k, 0.0), comp(n * k, 0.0);
        std::vector<int64_t> nobs(n * k, 0);
        for (const Origin& org : origins)
            for (size_t r = 0; r < org.prep.names.size(); ++r) {
                const auto it = row_of.find(org.prep.names[r]);
                if (it == row_of.end()) continue;
                for (size_t j = 0; j < k; ++j) {
                    const double v = org.table[r * k + j];
                    if (nan(v)) continue;
                    const size_t c = it->second * k + j;
                    ++nobs[c];
                    const double y = v - comp[c], t = sum[c] + y;
                    comp[c] = t - sum[c] - y;
                    if (nan(comp[c])) comp[c] = 0.0;
                    sum[c] = t;
                }
            }
        for (size_t c = 0; c < n * k; ++c) table[c] = nobs[c] ? sum[c] / static_cast<double>(nobs[c]) : std::numeric_limits<double>::quiet_NaN();
    }

    // ---- warnings (runner.run)
    std::vector<std::string> warnings = latest.warnings;
    warnings.insert(warnings.end(), notes.begin(), notes.end());
    const double frac_fb = static_cast<double>(fallback_rows.size()) / static_cast<double>(n * static_cast<size_t>(n_orig));
    if (frac_fb > 0.5) {
        char b[160];
        std::snprintf(b, sizeof b, "%.0f%% of the series received a fallback (SNaive) forecast instead of the model; the results may not "
                                   "represent this model.", 100 * frac_fb);
        warnings.push_back(b);
    }

    // ---- run folder: <label>__cpp-<method>__<UTC time>, with -2, -3, ... if it exists
    const fs::path src = fs::u8path(tsf_path);
    const std::string label = cfg.dataset_id.empty() ? src.stem().u8string() : cfg.dataset_id;
    std::string safe = label + "__cpp-" + method;
    for (char& c : safe) {
        const unsigned char u = static_cast<unsigned char>(c);
        if (!(std::isalnum(u) || u >= 0x80 || c == '-' || c == '_')) c = '_';
    }
    const fs::path base_dir = fs::u8path(out_dir);
    std::string run_id;
    fs::path run_dir;
    for (int attempt = 1;; ++attempt) {
        run_id = safe + "__" + utc_now("%Y%m%d-%H%M%S") + (attempt > 1 ? "-" + std::to_string(attempt) : "");
        run_dir = base_dir / fs::u8path(run_id);
        std::error_code ec;
        fs::create_directories(base_dir, ec);
        if (fs::create_directory(run_dir, ec)) break;
        if (ec || attempt > 1000) throw RunError("Could not create the run folder " + run_dir.u8string());
    }

    // ---- result files (runner._write_outputs)
    std::vector<std::pair<std::string, std::string>> outputs;
    std::vector<std::string> hcols;
    for (size_t j = 1; j <= H; ++j) hcols.push_back("h" + std::to_string(j));
    write_text(run_dir / "forecasts.csv", table_csv("series_name", latest.names, hcols, last.f.fc.data(), eol));
    outputs.emplace_back("forecasts_csv", "forecasts.csv");
    write_text(run_dir / "test_actuals.csv", table_csv("series_name", latest.names, hcols, latest.test.data(), eol));
    outputs.emplace_back("test_actuals_csv", "test_actuals.csv");
    std::string text = "series_index,series_name,message" + eol;
    for (const auto& fb : fallback_rows) {
        text += std::to_string(fb.first) + ",";
        csv_field(text, fb.first < static_cast<int64_t>(n) ? latest.names[static_cast<size_t>(fb.first)] : std::string());
        text.push_back(',');
        csv_field(text, fb.second);
        text += eol;
    }
    write_text(run_dir / "fallbacks.csv", text);
    outputs.emplace_back("fallbacks_csv", "fallbacks.csv");
    text = "Metric,Mean,Median,N_valid,N_excluded" + eol;
    summary_rows(text, "", columns, table.data(), n, eol);
    write_text(run_dir / "summary.csv", text);
    outputs.emplace_back("summary_csv", "summary.csv");
    write_text(run_dir / "errors_per_series.csv", table_csv("series_name", latest.names, columns, table.data(), eol));
    outputs.emplace_back("errors_per_series_csv", "errors_per_series.csv");

    if (n_orig > 1) {  // runner._write_origin_outputs: every origin, origin 1 = oldest
        std::string fcs = "origin,cut,series_name", acs, ers = "origin,cut,series_name", sums = "origin,cut,Metric,Mean,Median,N_valid,N_excluded" + eol;
        for (const auto& hname : hcols) fcs += "," + hname;
        fcs += eol;
        acs = fcs;
        for (const auto& c : columns) { ers.push_back(','); csv_field(ers, c); }
        ers += eol;
        for (size_t o = 0; o < origins.size(); ++o) {
            const Origin& org = origins[o];
            const std::string head = std::to_string(o + 1) + "," + std::to_string(org.cut) + ",";
            for (size_t i = 0; i < org.prep.names.size(); ++i) {
                fcs += head; acs += head; ers += head;
                csv_field(fcs, org.prep.names[i]); csv_field(acs, org.prep.names[i]); csv_field(ers, org.prep.names[i]);
                for (size_t h = 0; h < H; ++h) {
                    fcs.push_back(','); csv_number(fcs, org.f.fc[i * H + h]);
                    acs.push_back(','); csv_number(acs, org.prep.test[i * H + h]);
                }
                for (size_t j = 0; j < k; ++j) { ers.push_back(','); csv_number(ers, org.table[i * k + j]); }
                fcs += eol; acs += eol; ers += eol;
            }
            summary_rows(sums, head, columns, org.table.data(), org.prep.names.size(), eol);
        }
        write_text(run_dir / "forecasts_by_origin.csv", fcs);
        outputs.emplace_back("forecasts_by_origin_csv", "forecasts_by_origin.csv");
        write_text(run_dir / "test_actuals_by_origin.csv", acs);
        outputs.emplace_back("test_actuals_by_origin_csv", "test_actuals_by_origin.csv");
        write_text(run_dir / "errors_by_origin.csv", ers);
        outputs.emplace_back("errors_by_origin_csv", "errors_by_origin.csv");
        write_text(run_dir / "summary_by_origin.csv", sums);
        outputs.emplace_back("summary_by_origin_csv", "summary_by_origin.csv");
    }
    if (!latest_bounds.empty()) {  // the bounds of the latest origin, one block of rows per quantile
        text = "series_name,quantile";
        for (const auto& hname : hcols) text += "," + hname;
        text += eol;
        for (size_t p = 0; p < probs.size(); ++p)
            for (size_t i = 0; i < n; ++i) {
                csv_field(text, latest.names[i]);
                text.push_back(',');
                csv_number(text, probs[p]);
                for (size_t h = 0; h < H; ++h) { text.push_back(','); csv_number(text, latest_bounds[p][i * H + h]); }
                text += eol;
            }
        write_text(run_dir / "intervals.csv", text);
        outputs.emplace_back("intervals_csv", "intervals.csv");
    }

    // ---- manifest (the layout of runner.run's manifest; fields the C++ run does not have are null)
    const auto num_list = [](const std::vector<double>& v) { return jlist(v, jnum); };
    const auto int_list = [](const std::vector<int>& v) { return jlist(v, [](int x) { return std::to_string(x); }); };
    std::string m = "{\n";
    m += "  \"schema\": 1,\n";
    m += "  \"run_id\": " + jstr(run_id) + ",\n";
    m += "  \"created_utc\": " + jstr(utc_now("%Y-%m-%dT%H:%M:%S+00:00")) + ",\n";
    m += "  \"config\": {\n";
    m += "    \"engine\": \"cpp\",\n    \"task\": \"local\",\n";
    m += "    \"method\": " + jstr(method) + ",\n";
    m += "    \"method_spec\": " + jstr(cfg.method) + ",\n";
    m += "    \"horizon\": " + (cfg.horizon > 0 ? std::to_string(cfg.horizon) : std::string("null")) + ",\n";
    m += "    \"lag\": null,\n";
    m += "    \"seasonality\": " + (cfg.seasonality.empty() ? std::string("null") : num_list(cfg.seasonality)) + ",\n";
    m += "    \"integer_conversion\": " + jbool(cfg.integer_conversion) + ",\n";
    m += "    \"missing_policy\": " + jstr(cfg.missing_policy) + ",\n";
    m += "    \"workers\": " + std::to_string(cfg.threads) + ",\n";
    m += "    \"max_series\": " + (cfg.max_series > 0 ? std::to_string(cfg.max_series) : std::string("null")) + ",\n";
    m += "    \"metrics\": " + jlist(metric_names, jstr) + ",\n";
    m += "    \"origins\": " + std::to_string(n_orig) + ",\n";
    m += "    \"origin_step\": " + (cfg.origin_step != 0 ? std::to_string(cfg.origin_step) : std::string("null")) + ",\n";
    m += "    \"intervals\": " + int_list(cfg.intervals) + ",\n";
    m += "    \"effective_horizon\": " + std::to_string(latest.horizon) + ",\n";
    m += "    \"effective_seasonality\": " + num_list(latest.seasonality) + ",\n";
    m += "    \"effective_integer_conversion\": " + jbool(cfg.integer_conversion) + "\n  },\n";
    m += "  \"dataset\": {\n";
    m += "    \"id\": " + jstr(label) + ",\n";
    m += "    \"source\": " + jstr(tsf_path) + ",\n";
    const std::string sha = sha256_file(tsf_path);
    m += "    \"sha256\": " + (sha.empty() ? std::string("null") : jstr(sha)) + ",\n";
    m += "    \"frequency\": " + (ds.header.has_frequency ? jstr(ds.header.frequency) : std::string("null")) + ",\n";
    m += "    \"n_series_total\": " + std::to_string(ds.values.size()) + ",\n";
    m += "    \"n_series_run\": " + std::to_string(n) + ",\n";
    m += "    \"header_horizon\": " + (ds.header.has_horizon ? std::to_string(ds.header.horizon) : std::string("null")) + "\n  },\n";
    m += "  \"split\": {\n";
    m += "    \"horizon\": " + std::to_string(latest.horizon) + ",\n";
    m += "    \"n_excluded\": " + std::to_string(latest.excluded.size()) + ",\n";
    m += "    \"excluded\": " + jlist(latest.excluded, [](const Excluded& e) {
             return "{\"index\": " + std::to_string(e.index) + ", \"name\": " + jstr(e.name) + ", \"reason\": " + jstr(e.reason) + "}";
         }) + ",\n";
    m += "    \"origins\": " + std::to_string(n_orig) + ",\n";
    m += "    \"origin_step\": " + (n_orig > 1 ? std::to_string(step) : std::string("null")) + ",\n";
    m += "    \"interval_levels\": " + int_list(levels) + ",\n";
    m += "    \"interval_method\": " + (levels.empty() ? std::string("null") : jstr("split conformal, one calibration window")) + ",\n";
    m += "    \"origin_cuts\": " + jlist(origins, [](const Origin& o) { return std::to_string(o.cut); }) + "\n  },\n";
    m += "  \"missing\": {\n";
    m += "    \"policy\": " + jstr(cfg.missing_policy) + ",\n";
    m += "    \"series_with_missing\": " + std::to_string(latest.series_with_missing) + ",\n";
    m += "    \"imputed_values\": " + std::to_string(latest.imputed_values) + ",\n";
    m += "    \"test_missing_values\": " + std::to_string(latest.test_missing_values) + "\n  },\n";
    m += "  \"engine\": {\n    \"name\": \"cpp\",\n";
    m += "    \"versions\": {\"tsfcore\": " + jstr(tsf_version()) + "},\n";
    m += "    \"elapsed_sec\": " + jnum(engine_sec) + "\n  },\n";
    m += "  \"fallbacks\": {\"count\": " + std::to_string(fallback_rows.size()) + ", \"fraction\": " + jnum(frac_fb) + ", \"na_replaced\": 0},\n";
    m += "  \"estimates\": {\"memory\": null, \"runtime\": null},\n";
    m += "  \"timing\": {\"engine_wall_sec\": " + jnum(engine_wall) + ", \"total_sec\": " + jnum(seconds_since(t_start)) + "},\n";
    m += "  \"software\": {\"tsfcore\": " + jstr(tsf_version()) + ", \"runner\": \"tsfcore (C++)\"},\n";
    m += "  \"warnings\": " + jlist(warnings, jstr) + ",\n";
    m += "  \"outputs\": {\n";
    for (size_t i = 0; i < outputs.size(); ++i)
        m += "    " + jstr(outputs[i].first) + ": " + jstr(outputs[i].second) + (i + 1 < outputs.size() ? ",\n" : "\n");
    m += "  }\n}";
    write_text(run_dir / "manifest.json", m);
    return run_dir.u8string();
}

}  // namespace run
}  // namespace tsf

// ================================================================================================ plain C interface
// int tsf_run(tsf_path, out_dir, config, run_dir, run_dir_len, msg, msg_len)
//   config: lines "key=value" (unknown keys are an error):
//     method=ets            a local method, or "name?key=value&..." (see tsf_forecast_batch)
//     dataset_id=nn5_daily  label of the run (default: the file name)
//     horizon=56            default: the header's @horizon
//     seasonality=7         comma-separated periods; default: from @frequency
//     integer_conversion=1  round the forecasts
//     missing_policy=locf   error (default) | zero | locf | interpolate
//     max_series=8          only the first N series
//     metrics=SMAPE,MASE    default: SMAPE,mSMAPE,MASE,MAE,RMSE,WAPE,nRMSE_Mean,nRMSE_Range
//     origins=3             rolling origin: number of forecast origins (default 1 = the last horizon only)
//     origin_step=7         observations between origins (default: the horizon)
//     intervals=80,95       prediction-interval levels in percent (split conformal); default: none
//     threads=4             0 = all cores
//     eol=lf                lf | crlf line ends in the CSV files (default: the platform's)
//   run_dir receives the folder that was written (UTF-8). Paths are UTF-8.
//   returns 0 = ok, -1 = the run could not be made (msg says why), -2 = bad arguments, -3 = run_dir too small
#if defined(_WIN32)
#define TSFRUN_API extern "C" __declspec(dllexport)
#else
#define TSFRUN_API extern "C" __attribute__((visibility("default")))
#endif

namespace {

void run_msg(char* msg, int32_t msg_len, const std::string& text) {
    if (msg && msg_len > 0) std::snprintf(msg, static_cast<size_t>(msg_len), "%s", text.c_str());
}

std::vector<std::string> split_list(const std::string& s) {
    std::vector<std::string> out;
    size_t b = 0;
    while (b <= s.size()) {
        const size_t e = std::min(s.find(',', b), s.size());
        if (e > b) out.push_back(s.substr(b, e - b));
        b = e + 1;
    }
    return out;
}

tsf::run::Config parse_config(const std::string& text) {
    tsf::run::Config cfg;
    size_t b = 0;
    while (b < text.size()) {
        size_t e = std::min(text.find('\n', b), text.size());
        std::string line = text.substr(b, e - b);
        b = e + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const size_t eq = line.find('=');
        if (eq == std::string::npos) throw tsf::run::RunError("Bad configuration line: " + line);
        const std::string key = line.substr(0, eq), value = line.substr(eq + 1);
        if (key == "method") cfg.method = value;
        else if (key == "dataset_id") cfg.dataset_id = value;
        else if (key == "horizon") cfg.horizon = std::atoll(value.c_str());
        else if (key == "seasonality") { for (const auto& v : split_list(value)) cfg.seasonality.push_back(std::strtod(v.c_str(), nullptr)); }
        else if (key == "integer_conversion") cfg.integer_conversion = value == "1" || value == "true";
        else if (key == "missing_policy") cfg.missing_policy = value;
        else if (key == "max_series") cfg.max_series = std::atoll(value.c_str());
        else if (key == "metrics") cfg.metrics = split_list(value);
        else if (key == "origins") cfg.origins = std::atoll(value.c_str());
        else if (key == "origin_step") cfg.origin_step = std::atoll(value.c_str());
        else if (key == "intervals") { for (const auto& v : split_list(value)) cfg.intervals.push_back(std::atoi(v.c_str())); }
        else if (key == "threads") cfg.threads = std::atoi(value.c_str());
        else if (key == "eol") cfg.eol = value == "crlf" ? "\r\n" : "\n";
        else throw tsf::run::RunError("Unknown configuration key: " + key);
    }
    return cfg;
}

}  // namespace

TSFRUN_API int32_t tsf_run(const char* tsf_path, const char* out_dir, const char* config, char* run_dir, int32_t run_dir_len,
                           char* msg, int32_t msg_len) {
    run_msg(msg, msg_len, "");
    if (!tsf_path || !out_dir || !run_dir || run_dir_len < 1) { run_msg(msg, msg_len, "Bad arguments"); return -2; }
    try {
        const std::string dir = tsf::run::run(tsf_path, out_dir, parse_config(config ? config : ""));
        if (static_cast<int64_t>(dir.size()) >= run_dir_len) { run_msg(msg, msg_len, "The run_dir buffer is too small: " + dir); return -3; }
        std::memcpy(run_dir, dir.c_str(), dir.size() + 1);
        return 0;
    } catch (const std::bad_alloc&) {
        run_msg(msg, msg_len, "Not enough memory for this run");
    } catch (const std::exception& err) {
        run_msg(msg, msg_len, err.what());
    }
    return -1;
}
