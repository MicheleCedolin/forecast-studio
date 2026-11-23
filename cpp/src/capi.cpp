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
#include <limits>
#include <mutex>
#include <string>
#include <thread>

#include "arima.hpp"
#include "autoarima.hpp"
#include "basic.hpp"
#include "ets.hpp"
#include "intermittent.hpp"
#include "theta_sf.hpp"
#include "ces_sf.hpp"
#include "mstl_sf.hpp"
#include "mfles_sf.hpp"
#include "tbats.hpp"
#include "global.hpp"
#include "catboost_bridge.hpp"
#include "lightgbm_bridge.hpp"
#include "stlf.hpp"
#include "bagged.hpp"
#include "loess.hpp"
#include "temporal.hpp"
#include "rstats.hpp"
#include "simple_sf.hpp"

#if defined(_WIN32)
#define TSF_API extern "C" __declspec(dllexport)
#else
#define TSF_API extern "C" __attribute__((visibility("default")))
#endif

namespace {

using tsf::Vec;
using Fn = Vec (*)(const Vec&, int, int, const Vec&);   // (y, h, m = floor(max period), periods)

Vec m_naive(const Vec& y, int h, int, const Vec&) { return tsf::naive(y, h); }
Vec m_snaive(const Vec& y, int h, int m, const Vec&) { return tsf::snaive(y, h, m); }
Vec m_drift(const Vec& y, int h, int, const Vec&) { return tsf::drift(y, h); }
Vec m_ses(const Vec& y, int h, int, const Vec&) { return tsf::ses_forecast(y, h); }
Vec m_theta(const Vec& y, int h, int m, const Vec&) { return tsf::theta_forecast(y, h, m); }
Vec m_naive2(const Vec& y, int h, int m, const Vec&) { return tsf::naive2(y, h, m); }
Vec m_comb(const Vec& y, int h, int m, const Vec&) { return tsf::comb_shd(y, h, m); }
Vec m_holt(const Vec& y, int h, int, const Vec&) { return tsf::holt_forecast(y, h, false); }
Vec m_damped(const Vec& y, int h, int, const Vec&) { return tsf::holt_forecast(y, h, true); }
Vec m_ets(const Vec& y, int h, int m, const Vec&) { return tsf::ets_auto(y, m, h).forecast; }
Vec m_hw(const Vec& y, int h, int m, const Vec&) { return tsf::ets_holt_winters(y, m, h).forecast; }
Vec m_arima(const Vec& y, int h, int m, const Vec&) { return tsf::auto_arima_study(y, m, h).forecast; }
Vec m_dhr(const Vec& y, int h, int m, const Vec& per) { return tsf::dhr_arima(y, m, static_cast<int>(per.size()), 1, h).forecast; }
Vec m_crc(const Vec& y, int h, int, const Vec&) { return tsf::croston_classic(y, h); }
Vec m_crs(const Vec& y, int h, int, const Vec&) { return tsf::croston_sba(y, h); }
Vec m_cro(const Vec& y, int h, int, const Vec&) { return tsf::croston_optimized(y, h); }
Vec m_tsb(const Vec& y, int h, int, const Vec&) { return tsf::tsb(y, h); }
Vec m_adida(const Vec& y, int h, int, const Vec&) { return tsf::adida(y, h); }
Vec m_imapa(const Vec& y, int h, int, const Vec&) { return tsf::imapa(y, h); }
Vec m_dotm(const Vec& y, int h, int m, const Vec&) { return tsf::dotm_forecast(y, m, h, true); }
Vec m_ces(const Vec& y, int h, int m, const Vec&) { return tsf::ces_forecast(y, m, h); }
Vec m_atheta(const Vec& y, int h, int m, const Vec&) { return tsf::auto_theta_forecast(y, m, h, true); }
Vec m_mstl(const Vec& y, int h, int, const Vec& per) { return tsf::mstl_forecast(y, per, h); }
Vec m_mfles(const Vec& y, int h, int, const Vec& per) { return tsf::mfles_forecast(y, per, h); }
Vec m_tbats(const Vec& y, int h, int, const Vec& per) { return tsf::tbats_forecast(y, per, h); }
Vec m_holt_ets(const Vec& y, int h, int, const Vec&) { return tsf::ets_holt(y, h).forecast; }
Vec m_hist(const Vec& y, int h, int, const Vec&) { return tsf::historic_average(y, h); }
Vec m_wavg(const Vec& y, int h, int, const Vec&) { return tsf::window_average(y, h, 4); }
Vec m_swavg(const Vec& y, int h, int m, const Vec&) { return tsf::seasonal_window_average(y, h, m, 2); }
Vec m_ses_seas(const Vec& y, int h, int m, const Vec&) { return tsf::seasonal_es_optimized(y, h, m); }
Vec m_zero(const Vec&, int h, int, const Vec&) { return tsf::zero_model(h); }
Vec m_stlf_ets(const Vec& y, int h, int, const Vec& per) { return tsf::stlf_ets(y, per, h); }
Vec m_stlf_arima(const Vec& y, int h, int, const Vec& per) { return tsf::stlf_arima(y, per, h); }
Vec m_bagged_ets(const Vec& y, int h, int, const Vec& per) { return tsf::bagged_ets(y, per, h, 100, 0); }
Vec m_thief(const Vec& y, int h, int, const Vec& per) { return tsf::thief_ets(y, per, h); }
Vec m_mapa(const Vec& y, int h, int, const Vec& per) { return tsf::mapa_ets(y, per, h); }

// SCUM (Petropoulos & Svetunkov 2020): pointwise median of ets, auto_ces, arima and dotm, as the Python engine's
// forecast_scum (a failing or non-finite component is left out; fewer than two -> error). Components are the C++ ones:
// ets and arima reproduce R (the Python engine uses statsforecast's), auto_ces and dotm reproduce statsforecast.
// ARIMA uses the Python engine's arima_max_season = 200 rule (non-seasonal above it).
Vec m_scum(const Vec& y, int h, int m, const Vec&) {
    std::vector<Vec> fcs;
    std::string failed;
    auto add = [&](const char* name, auto&& fn) {
        try {
            Vec f = fn();
            bool ok = static_cast<int>(f.size()) >= h;
            for (int k = 0; ok && k < h; ++k) ok = std::isfinite(f[k]);
            if (ok) { f.resize(h); fcs.push_back(std::move(f)); return; }
            failed += std::string(failed.empty() ? "" : "; ") + name + ": non-finite";
        } catch (const std::exception& ex) {
            failed += std::string(failed.empty() ? "" : "; ") + name + ": " + ex.what();
        }
    };
    add("ets", [&] { return tsf::ets_auto(y, m, h).forecast; });
    add("auto_ces", [&] { return tsf::ces_forecast(y, m, h); });
    add("arima", [&] { return tsf::auto_arima_study(y, m <= 200 ? m : 1, h).forecast; });
    add("dotm", [&] { return tsf::dotm_forecast(y, m, h, true); });
    if (fcs.size() < 2) throw std::runtime_error("scum: fewer than two components succeeded (" + failed + ")");
    Vec out(h);
    const size_t k = fcs.size();
    Vec col(k);
    for (int j = 0; j < h; ++j) {                       // np.median: middle value, or the mean of the two middle ones
        for (size_t c = 0; c < k; ++c) col[c] = fcs[c][j];
        std::sort(col.begin(), col.end());
        out[j] = (k % 2) ? col[k / 2] : (col[k / 2 - 1] + col[k / 2]) / 2.0;
    }
    return out;
}

struct Entry { const char* name; Fn fn; };
const Entry METHODS[] = {
    {"naive", m_naive}, {"snaive", m_snaive}, {"drift", m_drift}, {"ses", m_ses}, {"theta", m_theta},
    {"naive2", m_naive2}, {"comb_shd", m_comb}, {"holt", m_holt}, {"holt_damped", m_damped}, {"ets", m_ets}, {"holt_winters", m_hw},
    {"arima", m_arima}, {"dhr_arima", m_dhr}, {"croston_classic", m_crc}, {"croston_sba", m_crs},
    {"croston_optimized", m_cro}, {"tsb", m_tsb}, {"adida", m_adida}, {"imapa", m_imapa}, {"dotm", m_dotm}, {"auto_ces", m_ces}, {"scum", m_scum}, {"auto_theta", m_atheta}, {"mstl", m_mstl}, {"auto_mfles", m_mfles}, {"tbats", m_tbats},
    {"holt_ets", m_holt_ets}, {"historic_average", m_hist}, {"window_average", m_wavg},
    {"seasonal_window_average", m_swavg}, {"seasonal_es", m_ses_seas}, {"zero_model", m_zero},
    {"stl_ets", m_stlf_ets}, {"stl_arima", m_stlf_arima}, {"bagged_ets", m_bagged_ets}, {"thief", m_thief}, {"mapa", m_mapa},
};

Fn find_method(const char* name) {
    for (const auto& e : METHODS)
        if (std::strcmp(e.name, name) == 0) return e.fn;
    return nullptr;
}

}  // namespace

TSF_API const char* tsf_version() {
    return "tsfcore 0.4.0 (naive, snaive, drift, ses, theta, naive2, holt, comb_shd, ets, holt_winters, arima, dhr_arima, croston_classic, croston_sba, croston_optimized, tsb, adida, imapa, dotm, auto_ces, scum, auto_theta, mstl, auto_mfles, tbats, holt_ets, historic_average, window_average, seasonal_window_average, seasonal_es, zero_model)";
}

TSF_API const char* tsf_methods() { return "naive,snaive,drift,ses,theta,naive2,comb_shd,holt,holt_damped,ets,holt_winters,arima,dhr_arima,"
                                         "croston_classic,croston_sba,croston_optimized,tsb,adida,imapa,dotm,auto_ces,scum,auto_theta,mstl,auto_mfles,tbats,"
                                         "holt_ets,historic_average,window_average,seasonal_window_average,seasonal_es,zero_model"; }

// Test hook for auto.arima (the study's chain). info = {p, d, q, P, D, Q, period, drift, mean, step, loglik, aicc,
// sigma2}. Returns 0, or 1 on error (message in msg).
TSF_API int tsf_auto_arima(const double* y, int32_t n, int32_t freq, int32_t h, double* fc, double* info, char* msg,
                           int32_t len) {
    tsf::rs::X87Extended fpu_guard;              // R runs with 64-bit x87 precision
    try {
        const tsf::AutoArimaResult r = tsf::auto_arima_study(Vec(y, y + n), freq, h);
        std::copy(r.forecast.begin(), r.forecast.end(), fc);
        const double v[13] = {double(r.p), double(r.d), double(r.q), double(r.P), double(r.D), double(r.Q),
                              double(r.period), r.drift ? 1.0 : 0.0, r.mean ? 1.0 : 0.0, double(r.step),
                              r.loglik, r.aicc, r.sigma2};
        std::copy(v, v + 13, info);
        if (msg && len > 0) msg[0] = '\0';
        return 0;
    } catch (const std::exception& ex) {
        if (msg && len > 0) { std::strncpy(msg, ex.what(), static_cast<size_t>(len - 1)); msg[len - 1] = '\0'; }
        return 1;
    }
}

// Test hook: Q0 of an ARMA(p, q) by method 0 (Gardner, as R) or 1 (autocovariances); out gets r*r values.
TSF_API int tsf_arima_q0(const double* phi, int32_t p, const double* theta, int32_t q, int32_t method, double* out) {
    tsf::rs::X87Extended fpu_guard;              // R runs with 64-bit x87 precision
    try {
        const Vec Q = tsf::arima_q0(Vec(phi, phi + p), Vec(theta, theta + q), method);
        std::copy(Q.begin(), Q.end(), out);
        return 0;
    } catch (...) { return 1; }
}

// Test hook for dhr_arima (K Fourier terms, one period): info as tsf_auto_arima. Returns 0, or 1 on error.
TSF_API int tsf_dhr_arima(const double* y, int32_t n, int32_t freq, int32_t K, int32_t h, double* fc, double* info,
                          char* msg, int32_t len) {
    tsf::rs::X87Extended fpu_guard;              // R runs with 64-bit x87 precision
    try {
        const tsf::AutoArimaResult r = tsf::dhr_arima(Vec(y, y + n), freq, 1, K, h);
        std::copy(r.forecast.begin(), r.forecast.end(), fc);
        const double v[13] = {double(r.p), double(r.d), double(r.q), double(r.P), double(r.D), double(r.Q),
                              double(r.period), r.drift ? 1.0 : 0.0, r.mean ? 1.0 : 0.0, double(r.step),
                              r.loglik, r.aicc, r.sigma2};
        std::copy(v, v + 13, info);
        if (msg && len > 0) msg[0] = '\0';
        return 0;
    } catch (const std::exception& ex) {
        if (msg && len > 0) { std::strncpy(msg, ex.what(), static_cast<size_t>(len - 1)); msg[len - 1] = '\0'; }
        return 1;
    }
}

// Test hook: the MSTL decomposition (xsa: n values) and the summed seasonal naive forecast (h values).
// Diagnostic hooks for bagged_ets: bld.mbb.bootstrap (out: num x n, row 0 = y; returns 0, 1 on error with msg) and
// R's RNG (out: k values of unif_rand() after set.seed(seed), then k values of sample(1:m, 1)).
TSF_API int tsf_bld_mbb(const double* y, int32_t n, int32_t freq, int32_t num, uint32_t seed, double* out, char* msg,
                        int32_t len) {
    tsf::rs::X87Extended fpu_guard;              // R runs with 64-bit x87 precision
    try {
        const std::vector<Vec> xs = tsf::bld_mbb_bootstrap(Vec(y, y + n), freq, num, seed);
        for (int32_t i = 0; i < num; ++i) std::copy(xs[i].begin(), xs[i].end(), out + static_cast<int64_t>(i) * n);
        if (msg && len > 0) msg[0] = '\0';
        return 0;
    } catch (const std::exception& ex) {
        if (msg && len > 0) { std::strncpy(msg, ex.what(), static_cast<size_t>(len - 1)); msg[len - 1] = '\0'; }
        return 1;
    }
}

// Diagnostic hook: loess(y ~ x, span, degree = 1) fitted values (R's interpolate surface, cell 0.2). Returns 0 / 1.
TSF_API int tsf_loess(const double* x, const double* y, int32_t n, double span, double* out) {
    tsf::rs::X87Extended fpu_guard;              // R runs with 64-bit x87 precision
    try {
        const Vec f = tsf::loess_fit_1d(Vec(x, x + n), Vec(y, y + n), span);
        std::copy(f.begin(), f.end(), out);
        return 0;
    } catch (...) {
        return 1;
    }
}

// Diagnostic hook: R's x ^ y as the C++ engine computes it.
TSF_API double tsf_r_pow(double x, double y) { return tsf::rs::r_pow(x, y); }

// Diagnostic hook: forecast::BoxCox.lambda(x, "guerrero", lower, upper) for frequency freq.
TSF_API double tsf_boxcox_lambda(const double* x, int32_t n, double lower, double upper, double freq) {
    tsf::rs::X87Extended fpu_guard;              // R runs with 64-bit x87 precision
    return tsf::rs::box_cox_lambda(Vec(x, x + n), lower, upper, freq);
}

TSF_API void tsf_r_rng(uint32_t seed, int32_t k, int32_t m, double* out) {
    tsf::rs::X87Extended fpu_guard;              // R runs with 64-bit x87 precision
    tsf::rs::RRng a(seed);
    for (int32_t i = 0; i < k; ++i) out[i] = a.unif_rand();
    tsf::rs::RRng b(seed);
    for (int32_t i = 0; i < k; ++i) out[k + i] = b.sample1(m);
}

// Diagnostic hook: R mstl() as stl_ets / stl_arima use it (seasonally adjusted series, seasonal forecast). Returns the
// frequency, or 0 on error.
TSF_API int tsf_stlf_parts(const double* y, int32_t n, const double* periods, int32_t n_periods, int32_t h, double* xsa,
                           double* seas_fc) {
    tsf::rs::X87Extended fpu_guard;              // R runs with 64-bit x87 precision
    try {
        const tsf::StlfParts r = tsf::r_mstl_parts(Vec(y, y + n), Vec(periods, periods + n_periods), h);
        std::copy(r.xsa.begin(), r.xsa.end(), xsa);
        std::copy(r.seas_fc.begin(), r.seas_fc.end(), seas_fc);
        return r.freq;
    } catch (...) {
        return 0;
    }
}

TSF_API int tsf_mstl_parts(const double* y, int32_t n, const double* periods, int32_t n_periods, int32_t h, double* xsa,
                           double* seas_fc) {
    tsf::rs::X87Extended fpu_guard;              // R runs with 64-bit x87 precision
    try {
        const tsf::MstlParts r = tsf::mstl_parts(Vec(y, y + n), Vec(periods, periods + n_periods), h);
        std::copy(r.xsa.begin(), r.xsa.end(), xsa);
        std::copy(r.seas_fc.begin(), r.seas_fc.end(), seas_fc);
        return 0;
    } catch (...) { return 1; }
}

// Test hook: sklearn Lasso(alpha, fit_intercept = False, tol = 1e-3, max_iter = 10000) on X (n x k, column-major).
TSF_API void tsf_lasso(const double* X, int32_t n, int32_t k, const double* y, double alpha, double* coef) {
    tsf::rs::X87Extended fpu_guard;              // R runs with 64-bit x87 precision
    std::vector<Vec> cols(k);
    for (int j = 0; j < k; ++j) cols[j] = Vec(X + static_cast<size_t>(j) * n, X + static_cast<size_t>(j + 1) * n);
    const Vec w = tsf::sk_lasso(cols, Vec(y, y + n), alpha);
    std::copy(w.begin(), w.end(), coef);
}

// Global pooled regression over all series (values/offsets as tsf_forecast_batch): out is n_series x h; aliased gets the
// 1-based lag numbers of aliased (NA) coefficients (room for lag values), n_aliased their count. Returns 0, or 1 on
// error (message in msg).
TSF_API int tsf_pooled_regression(const double* values, const int64_t* offsets, int32_t n_series, int32_t lag, int32_t h,
                                  int32_t max_obs, double* out, int32_t* aliased, int32_t* n_aliased, char* msg,
                                  int32_t len) {
    tsf::rs::X87Extended fpu_guard;              // R runs with 64-bit x87 precision
    try {
        std::vector<Vec> series(n_series);
        for (int32_t i = 0; i < n_series; ++i) series[i] = Vec(values + offsets[i], values + offsets[i + 1]);
        const tsf::PooledResult r = tsf::pooled_regression(series, lag, h, max_obs);
        for (int32_t i = 0; i < n_series; ++i) std::copy(r.forecasts[i].begin(), r.forecasts[i].end(), out + static_cast<int64_t>(i) * h);
        *n_aliased = static_cast<int32_t>(r.aliased.size());
        std::copy(r.aliased.begin(), r.aliased.end(), aliased);
        if (msg && len > 0) msg[0] = '\0';
        return 0;
    } catch (const std::exception& ex) {
        if (msg && len > 0) { std::strncpy(msg, ex.what(), static_cast<size_t>(len - 1)); msg[len - 1] = '\0'; }
        return 1;
    }
}

// Global CatBoost (training by the CatBoost command-line program, prediction by its model library). out: n_series x h.
// Returns 0, or 1 on error (message in msg).
TSF_API int tsf_catboost(const double* values, const int64_t* offsets, int32_t n_series, int32_t lag, int32_t h,
                         int32_t max_obs, const char* exe, const char* model_lib, const char* workdir, int32_t threads,
                         double* out, char* msg, int32_t len) {
    tsf::rs::X87Extended fpu_guard;              // R runs with 64-bit x87 precision
    try {
        std::vector<Vec> series(n_series);
        for (int32_t i = 0; i < n_series; ++i) series[i] = Vec(values + offsets[i], values + offsets[i + 1]);
        tsf::CatboostSetup s;
        s.exe = exe; s.model_lib = model_lib; s.workdir = workdir; s.threads = threads;
        const std::vector<Vec> f = tsf::catboost_forecast(series, lag, h, max_obs, s);
        for (int32_t i = 0; i < n_series; ++i) std::copy(f[i].begin(), f[i].end(), out + static_cast<int64_t>(i) * h);
        if (msg && len > 0) msg[0] = '\0';
        return 0;
    } catch (const std::exception& ex) {
        if (msg && len > 0) { std::strncpy(msg, ex.what(), static_cast<size_t>(len - 1)); msg[len - 1] = '\0'; }
        return 1;
    }
}

// Global LightGBM through LightGBM's C library (lib_path). params: LightGBM parameter string; rounds: iterations.
// out: n_series x h. Returns 0, or 1 on error (message in msg).
TSF_API int tsf_lightgbm(const double* values, const int64_t* offsets, int32_t n_series, int32_t lag, int32_t h,
                         int32_t max_obs, const char* lib_path, const char* params, int32_t rounds, double* out, char* msg,
                         int32_t len) {
    tsf::rs::X87Extended fpu_guard;              // R runs with 64-bit x87 precision
    try {
        std::vector<Vec> series(n_series);
        for (int32_t i = 0; i < n_series; ++i) series[i] = Vec(values + offsets[i], values + offsets[i + 1]);
        tsf::LightgbmSetup s;
        s.lib = lib_path; s.params = params; s.rounds = rounds;
        const std::vector<Vec> f = tsf::lightgbm_forecast(series, lag, h, max_obs, s);
        for (int32_t i = 0; i < n_series; ++i) std::copy(f[i].begin(), f[i].end(), out + static_cast<int64_t>(i) * h);
        if (msg && len > 0) msg[0] = '\0';
        return 0;
    } catch (const std::exception& ex) {
        if (msg && len > 0) { std::strncpy(msg, ex.what(), static_cast<size_t>(len - 1)); msg[len - 1] = '\0'; }
        return 1;
    }
}

// Global LightGBM with M5-style features (lightgbm_feat_forecast); m = seasonal period. Returns 0 / 1 (msg).
TSF_API int tsf_lightgbm_feat(const double* values, const int64_t* offsets, int32_t n_series, int32_t lag, int32_t h,
                              int32_t m, int32_t max_obs, const char* lib_path, const char* params, int32_t rounds,
                              double* out, char* msg, int32_t len) {
    tsf::rs::X87Extended fpu_guard;
    try {
        std::vector<Vec> series(n_series);
        for (int32_t i = 0; i < n_series; ++i) series[i] = Vec(values + offsets[i], values + offsets[i + 1]);
        tsf::LightgbmSetup s;
        s.lib = lib_path; s.params = params; s.rounds = rounds;
        const std::vector<Vec> f = tsf::lightgbm_feat_forecast(series, lag, h, m, max_obs, s);
        for (int32_t i = 0; i < n_series; ++i) std::copy(f[i].begin(), f[i].end(), out + static_cast<int64_t>(i) * h);
        if (msg && len > 0) msg[0] = '\0';
        return 0;
    } catch (const std::exception& ex) {
        if (msg && len > 0) { std::strncpy(msg, ex.what(), static_cast<size_t>(len - 1)); msg[len - 1] = '\0'; }
        return 1;
    }
}

// Diagnostic hook for auto.arima's differencing tests: out = {seas.heuristic, nsdiffs, ndiffs(after D), kpss stat
// of the level series}.
TSF_API void tsf_arima_tests(const double* y, int32_t n, int32_t freq, double* out) {
    tsf::rs::X87Extended fpu_guard;              // R runs with 64-bit x87 precision
    const Vec x(y, y + n);
    double s = std::numeric_limits<double>::quiet_NaN();
    try { s = tsf::seas_heuristic(x, freq); } catch (...) {}
    const int D = (n > 2 * freq && freq > 1) ? tsf::nsdiffs(x, freq) : 0;
    Vec dx = x;
    if (D > 0) { Vec t(dx.size() - freq); for (size_t i = 0; i < t.size(); ++i) t[i] = dx[i + freq] - dx[i]; dx = t; }
    out[0] = s; out[1] = D; out[2] = tsf::ndiffs(dx, 2); out[3] = tsf::kpss_stat(x);
}

// Diagnostic hook: one fixed ETS model; init gets the initial state (up to 26 values), par the smoothing parameters
// (alpha, [beta], [gamma], [phi]); returns lik (Inf on failure).
TSF_API double tsf_ets_debug(const double* y, int32_t n, int32_t m, int32_t e, int32_t t, int32_t s, int32_t d,
                             double* init, int32_t* n_init, double* par, int32_t* n_par) {
    tsf::rs::X87Extended fpu_guard;              // R runs with 64-bit x87 precision
    try {
        Vec iv, pv;
        const double lik = tsf::ets_debug(Vec(y, y + n), m, e, t, s, d != 0, iv, pv);
        *n_init = static_cast<int32_t>(std::min<size_t>(iv.size(), 26));
        std::copy(iv.begin(), iv.begin() + *n_init, init);
        *n_par = static_cast<int32_t>(pv.size());
        std::copy(pv.begin(), pv.end(), par);
        return lik;
    } catch (...) {
        *n_init = 0; *n_par = 0;
        return std::numeric_limits<double>::infinity();
    }
}

// Diagnostic hook: stats::arima with a fixed order. ord = {p,d,q,P,D,Q,period}; flags = {include_mean, drift, method
// (0 CSS, 1 CSS-ML, 2 ML)}; info = {loglik, aic, sigma2, var_ok, n_coef}; coef up to 16 values. Returns 0, or 1 on error.
TSF_API int tsf_arima_fixed(const double* y, int32_t n, const int32_t* ord, const int32_t* flags, int32_t h, double* fc,
                            double* coef, double* info, char* msg, int32_t len) {
    tsf::rs::X87Extended fpu_guard;              // R runs with 64-bit x87 precision
    try {
        tsf::ArimaSpec s;
        s.p = ord[0]; s.d = ord[1]; s.q = ord[2]; s.P = ord[3]; s.D = ord[4]; s.Q = ord[5]; s.period = ord[6];
        s.include_mean = flags[0] != 0; s.drift = flags[1] != 0;
        s.method = flags[2] == 0 ? tsf::ArimaMethod::CSS : (flags[2] == 1 ? tsf::ArimaMethod::CSS_ML : tsf::ArimaMethod::ML);
        const tsf::ArimaFit f = tsf::arima_fit(Vec(y, y + n), s);
        const Vec p = tsf::arima_forecast(f, h);
        std::copy(p.begin(), p.end(), fc);
        const size_t nc = std::min<size_t>(f.coef.size(), 16);
        std::copy(f.coef.begin(), f.coef.begin() + nc, coef);
        info[0] = f.loglik; info[1] = f.aic; info[2] = f.sigma2; info[3] = f.var_ok ? 1 : 0; info[4] = static_cast<double>(nc);
        if (msg && len > 0) msg[0] = '\0';
        return 0;
    } catch (const std::exception& ex) {
        if (msg && len > 0) { std::strncpy(msg, ex.what(), static_cast<size_t>(len - 1)); msg[len - 1] = '\0'; }
        return 1;
    }
}

// Test hook for ETS: fits one series; info = {aicc, lik, alpha, beta, gamma, phi}; model gets "ETS(x,y,z)".
// Returns 0 on success, 1 if no model could be fitted (message in model).
TSF_API int tsf_ets_fit(const double* y, int32_t n, int32_t m, int32_t h, double* fc, double* info, char* model, int32_t len) {
    tsf::rs::X87Extended fpu_guard;              // R runs with 64-bit x87 precision
    try {
        const tsf::EtsResult r = tsf::ets_auto(Vec(y, y + n), m, h);
        std::copy(r.forecast.begin(), r.forecast.end(), fc);
        const double v[6] = {r.aicc, r.lik, r.alpha, r.beta, r.gamma, r.phi};
        std::copy(v, v + 6, info);
        std::strncpy(model, r.model.c_str(), static_cast<size_t>(len - 1));
        model[len - 1] = '\0';
        return 0;
    } catch (const std::exception& ex) {
        std::strncpy(model, ex.what(), static_cast<size_t>(len - 1));
        model[len - 1] = '\0';
        return 1;
    }
}

TSF_API int tsf_forecast_batch(const char* method, const double* values, const int64_t* offsets, int32_t n_series, int32_t h,
                               const int32_t* horizons, const double* periods, int32_t n_periods, int32_t n_threads,
                               double* out, int32_t* flags, char* msg, int32_t msg_len) {
    tsf::rs::X87Extended fpu_guard;              // R runs with 64-bit x87 precision
    if (!method || !values || !offsets || !out || !flags || n_series < 0 || h < 1) return -2;
    const Fn fn = find_method(method);
    if (!fn) return -1;
    const int m = tsf::r_frequency(periods, n_periods);
    const Vec pervec = (periods && n_periods > 0) ? Vec(periods, periods + n_periods) : Vec{};
    if (msg && msg_len > 0) msg[0] = '\0';
    std::mutex msg_mutex;
    bool msg_set = false;
    std::atomic<int32_t> next{0};

    auto work = [&]() {
        tsf::rs::X87Extended fpu_guard;          // per worker thread
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
                f = fn(y, hi, m, pervec);
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
    tsf::rs::X87Extended fpu_guard;              // R runs with 64-bit x87 precision
    return tsf::holt_best_sse(Vec(y, y + n), damped != 0);
}
