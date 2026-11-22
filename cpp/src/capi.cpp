// Plain C interface of tsfcore (stable across compilers; Python uses it through ctypes).
//
// int tsf_forecast_batch(method, values, offsets, n_series, h, horizons, periods, n_periods, n_threads, out, flags, msg, msg_len)
//   values/offsets : all series concatenated; series i is values[offsets[i] .. offsets[i+1])
//   horizons       : per-series horizon (<= h) or NULL; out is n_series x h, row-major, NaN beyond a series' horizon
//   flags          : per series, 0 = model forecast, 1 = fallback (seasonal naive, or naive) after an error or non-finite output
//   msg            : first fallback reason (truncated to msg_len - 1)
//   returns 0 on success, -1 unknown method, -2 bad arguments.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <mutex>
#include <string>
#include <thread>

#include "basic.hpp"

#if defined(_WIN32)
#define TSF_API extern "C" __declspec(dllexport)
#else
#define TSF_API extern "C" __attribute__((visibility("default")))
#endif

namespace {

using tsf::Vec;
using Fn = Vec (*)(const Vec&, int, int);

Vec m_naive(const Vec& y, int h, int) { return tsf::naive(y, h); }
Vec m_snaive(const Vec& y, int h, int m) { return tsf::snaive(y, h, m); }
Vec m_drift(const Vec& y, int h, int) { return tsf::drift(y, h); }
Vec m_ses(const Vec& y, int h, int) { return tsf::ses_forecast(y, h); }
Vec m_theta(const Vec& y, int h, int m) { return tsf::theta_forecast(y, h, m); }
Vec m_naive2(const Vec& y, int h, int m) { return tsf::naive2(y, h, m); }
Vec m_comb(const Vec& y, int h, int m) { return tsf::comb_shd(y, h, m); }
Vec m_holt(const Vec& y, int h, int) { return tsf::holt_forecast(y, h, false); }
Vec m_damped(const Vec& y, int h, int) { return tsf::holt_forecast(y, h, true); }

struct Entry { const char* name; Fn fn; };
const Entry METHODS[] = {
    {"naive", m_naive}, {"snaive", m_snaive}, {"drift", m_drift}, {"ses", m_ses}, {"theta", m_theta},
    {"naive2", m_naive2}, {"comb_shd", m_comb}, {"holt", m_holt}, {"holt_damped", m_damped},
};

Fn find_method(const char* name) {
    for (const auto& e : METHODS)
        if (std::strcmp(e.name, name) == 0) return e.fn;
    return nullptr;
}

}  // namespace

TSF_API const char* tsf_version() { return "tsfcore 0.1.0 (phase 1: naive, snaive, drift, ses, theta, naive2, holt, comb_shd)"; }

TSF_API const char* tsf_methods() { return "naive,snaive,drift,ses,theta,naive2,comb_shd,holt,holt_damped"; }

TSF_API int tsf_forecast_batch(const char* method, const double* values, const int64_t* offsets, int32_t n_series, int32_t h,
                               const int32_t* horizons, const double* periods, int32_t n_periods, int32_t n_threads,
                               double* out, int32_t* flags, char* msg, int32_t msg_len) {
    if (!method || !values || !offsets || !out || !flags || n_series < 0 || h < 1) return -2;
    const Fn fn = find_method(method);
    if (!fn) return -1;
    const int m = tsf::r_frequency(periods, n_periods);
    if (msg && msg_len > 0) msg[0] = '\0';
    std::mutex msg_mutex;
    bool msg_set = false;
    std::atomic<int32_t> next{0};

    auto work = [&]() {
        for (;;) {
            const int32_t i = next.fetch_add(1);
            if (i >= n_series) return;
            const int64_t a = offsets[i], b = offsets[i + 1];
            Vec y(values + a, values + b);
            const int hi = horizons ? std::clamp<int32_t>(horizons[i], 0, h) : h;
            double* row = out + static_cast<int64_t>(i) * h;
            std::fill(row, row + h, std::nan(""));
            flags[i] = 0;
            if (hi == 0 || y.empty()) continue;
            std::string why;
            Vec f;
            try {
                f = fn(y, hi, m);
                if (static_cast<int>(f.size()) < hi) throw std::runtime_error("the model produced an incomplete horizon");
                for (int k = 0; k < hi; ++k)
                    if (!std::isfinite(f[k])) throw std::runtime_error("the model produced a non-finite value");
            } catch (const std::exception& ex) {
                why = std::string(method) + ": " + ex.what();
            } catch (...) {
                why = std::string(method) + ": unknown error";
            }
            if (!why.empty()) {
                flags[i] = 1;
                f = tsf::snaive(y, hi, m);
                std::lock_guard<std::mutex> lk(msg_mutex);
                if (!msg_set && msg && msg_len > 0) {
                    std::strncpy(msg, why.c_str(), static_cast<size_t>(msg_len - 1));
                    msg[msg_len - 1] = '\0';
                    msg_set = true;
                }
            }
            std::copy(f.begin(), f.begin() + hi, row);
        }
    };
    int nt = n_threads > 0 ? n_threads : static_cast<int>(std::thread::hardware_concurrency());
    nt = std::max(1, std::min(nt, std::max<int32_t>(1, n_series)));
    if (nt == 1) {
        work();
    } else {
        std::vector<std::thread> pool;
        pool.reserve(nt);
        for (int t = 0; t < nt; ++t) pool.emplace_back(work);
        for (auto& th : pool) th.join();
    }
    return 0;
}

// Test hook: in-sample SSE of the Holt fit (damped != 0 for damped Holt).
TSF_API double tsf_holt_sse(const double* y, int32_t n, int32_t damped) {
    return tsf::holt_best_sse(Vec(y, y + n), damped != 0);
}
