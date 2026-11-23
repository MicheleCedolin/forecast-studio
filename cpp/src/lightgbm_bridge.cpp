#include "lightgbm_bridge.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "sf_common.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace tsf {

namespace {

// LightGBM c_api.h (4.x)
constexpr int kFloat32 = 0, kFloat64 = 1, kPredictNormal = 0;
using ErrFn = const char* (*)();
using DsFromMatFn = int (*)(const void*, int, int32_t, int32_t, int, const char*, const void*, void**);
using DsSetFieldFn = int (*)(void*, const char*, const void*, int, int);
using DsFreeFn = int (*)(void*);
using BCreateFn = int (*)(const void*, const char*, void**);
using BUpdateFn = int (*)(void*, int*);
using BPredictFn = int (*)(void*, const void*, int, int32_t, int32_t, int, int, int, int, const char*, int64_t*, double*);
using BFreeFn = int (*)(void*);

struct Lib {
#if defined(_WIN32)
    HMODULE h = nullptr;
    void* sym(const char* n) const { return reinterpret_cast<void*>(GetProcAddress(h, n)); }
#else
    void* h = nullptr;
    void* sym(const char* n) const { return dlsym(h, n); }
#endif
    ErrFn err = nullptr; DsFromMatFn ds_mat = nullptr; DsSetFieldFn ds_field = nullptr; DsFreeFn ds_free = nullptr;
    BCreateFn b_create = nullptr; BUpdateFn b_update = nullptr; BPredictFn b_predict = nullptr; BFreeFn b_free = nullptr;
    explicit Lib(const std::string& path) {
#if defined(_WIN32)
        h = LoadLibraryExA(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
#else
        h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
        if (!h) throw std::runtime_error("lightgbm: cannot load " + path);
        err = reinterpret_cast<ErrFn>(sym("LGBM_GetLastError"));
        ds_mat = reinterpret_cast<DsFromMatFn>(sym("LGBM_DatasetCreateFromMat"));
        ds_field = reinterpret_cast<DsSetFieldFn>(sym("LGBM_DatasetSetField"));
        ds_free = reinterpret_cast<DsFreeFn>(sym("LGBM_DatasetFree"));
        b_create = reinterpret_cast<BCreateFn>(sym("LGBM_BoosterCreate"));
        b_update = reinterpret_cast<BUpdateFn>(sym("LGBM_BoosterUpdateOneIter"));
        b_predict = reinterpret_cast<BPredictFn>(sym("LGBM_BoosterPredictForMat"));
        b_free = reinterpret_cast<BFreeFn>(sym("LGBM_BoosterFree"));
        if (!err || !ds_mat || !ds_field || !ds_free || !b_create || !b_update || !b_predict || !b_free)
            throw std::runtime_error("lightgbm: the library lacks the C API");
    }
    ~Lib() {
#if defined(_WIN32)
        if (h) FreeLibrary(h);
#else
        if (h) dlclose(h);
#endif
    }
    void check(int rc) const { if (rc != 0) throw std::runtime_error(std::string("lightgbm: ") + err()); }
};

// Frees the handles on every exit path.
struct Handles {
    const Lib& lib;
    void* ds = nullptr;
    void* booster = nullptr;
    explicit Handles(const Lib& l) : lib(l) {}
    ~Handles() { if (booster) lib.b_free(booster); if (ds) lib.ds_free(ds); }
};

// py_global.normalise
void normalise(const std::vector<Vec>& series, int lag, int max_obs, std::vector<Vec>& norm, Vec& means) {
    for (size_t i = 0; i < series.size(); ++i) {
        Vec x = series[i];
        if (max_obs > 0 && static_cast<int>(x.size()) > max_obs) x = Vec(x.end() - max_obs, x.end());
        if (static_cast<int>(x.size()) < lag + 1)
            throw std::runtime_error("Series " + std::to_string(i + 1) + " is too short for lag = " + std::to_string(lag) +
                                     " (length " + std::to_string(x.size()) + "; at least " + std::to_string(lag + 1) +
                                     " required).");
        double mu = np_mean(x.data(), x.size());
        if (mu == 0.0) mu = 1.0;
        for (double& v : x) v /= mu;
        norm.push_back(std::move(x));
        means.push_back(mu);
    }
}

// one feature row for the target at position t of hist (py_global.lgb_features)
void feature_row(const double* hist, int t, int lag, int m_eff, int w1, int w2, double* row) {
    int c = 0;
    for (int j = 1; j <= lag; ++j) row[c++] = hist[t - j];
    if (m_eff > 1) { row[c++] = hist[t - m_eff]; row[c++] = hist[t - 2 * m_eff]; }
    for (int w : {w1, w2}) {
        double s = 0.0;                                  // np.cumsum(win)[-1] / w
        for (int k = 0; k < w; ++k) s = s + hist[t - w + k];
        const double mean = s / w;
        double ss = 0.0;
        for (int k = 0; k < w; ++k) { const double d = hist[t - w + k] - mean; ss = ss + d * d; }
        row[c++] = mean;
        row[c++] = std::sqrt(ss / w);
    }
    if (m_eff > 1) row[c++] = static_cast<double>(t % m_eff);
}

}  // namespace

std::vector<Vec> lightgbm_feat_forecast(const std::vector<Vec>& series, int lag, int h, int m, int max_obs,
                                        const LightgbmSetup& s) {
    if (lag < 1) throw std::runtime_error("lag must be >= 1");
    if (s.rounds < 1) throw std::runtime_error("lightgbm: num_iterations must be >= 1");
    std::vector<Vec> norm;
    Vec means;
    normalise(series, lag, max_obs, norm, means);
    size_t nmin = norm[0].size();
    for (const Vec& x : norm) nmin = std::min(nmin, x.size());
    const int m_eff = (m > 1 && static_cast<int>(nmin) >= 3 * m) ? m : 1;
    const int w1 = std::max(4, m_eff), w2 = 2 * w1;
    const int h0 = std::max({lag, m_eff > 1 ? 2 * m_eff : 0, w2});
    const int nf = lag + (m_eff > 1 ? 3 : 0) + 4;
    for (size_t i = 0; i < norm.size(); ++i)
        if (static_cast<int>(norm[i].size()) <= h0)
            throw std::runtime_error("Series " + std::to_string(i + 1) + " is too short for the LightGBM features (at least " +
                                     std::to_string(h0 + 1) + " observations required).");
    size_t n_rows = 0;
    for (const Vec& x : norm) n_rows += x.size() - h0;
    if (n_rows > 2147483647u) throw std::runtime_error("lightgbm: more than 2^31 - 1 training rows; use max_obs");
    Lib lib(s.lib);
    Handles hd(lib);
    {
        Vec X(n_rows * static_cast<size_t>(nf));
        std::vector<float> y(n_rows);
        size_t r = 0;
        for (const Vec& x : norm)
            for (int t = h0; t < static_cast<int>(x.size()); ++t, ++r) {
                feature_row(x.data(), t, lag, m_eff, w1, w2, &X[r * nf]);
                y[r] = static_cast<float>(x[t]);
            }
        lib.check(lib.ds_mat(X.data(), kFloat64, static_cast<int32_t>(n_rows), nf, 1, s.params.c_str(), nullptr, &hd.ds));
        lib.check(lib.ds_field(hd.ds, "label", y.data(), static_cast<int>(n_rows), kFloat32));
    }
    lib.check(lib.b_create(hd.ds, s.params.c_str(), &hd.booster));
    for (int it = 0; it < s.rounds; ++it) {
        int finished = 0;
        lib.check(lib.b_update(hd.booster, &finished));
        if (finished) break;
    }
    const size_t ns = norm.size();
    std::vector<Vec> hist(ns), out(ns, Vec(h));
    for (size_t i = 0; i < ns; ++i) { hist[i] = norm[i]; hist[i].resize(norm[i].size() + h, 0.0); }
    Vec F(ns * static_cast<size_t>(nf)), pred(ns);
    for (int k = 0; k < h; ++k) {
        for (size_t i = 0; i < ns; ++i)
            feature_row(hist[i].data(), static_cast<int>(norm[i].size()) + k, lag, m_eff, w1, w2, &F[i * nf]);
        int64_t len = 0;
        lib.check(lib.b_predict(hd.booster, F.data(), kFloat64, static_cast<int32_t>(ns), nf, 1, kPredictNormal, 0, -1, "",
                                &len, pred.data()));
        for (size_t i = 0; i < ns; ++i) {
            hist[i][norm[i].size() + k] = pred[i];
            out[i][k] = pred[i] * means[i];
        }
    }
    return out;
}

// py_global.lightgbm_direct_forecast: the lightgbm_feat features at the origin t, one booster per step k with target
// x[t + k] (origins t = h0 .. n - 1 - k of every series); all boosters predict from the features at t = n.
std::vector<Vec> lightgbm_direct_forecast(const std::vector<Vec>& series, int lag, int h, int m, int max_obs,
                                          const LightgbmSetup& s) {
    if (lag < 1) throw std::runtime_error("lag must be >= 1");
    if (s.rounds < 1) throw std::runtime_error("lightgbm: num_iterations must be >= 1");
    std::vector<Vec> norm;
    Vec means;
    normalise(series, lag, max_obs, norm, means);
    size_t nmin = norm[0].size();
    for (const Vec& x : norm) nmin = std::min(nmin, x.size());
    const int m_eff = (m > 1 && static_cast<int>(nmin) >= 3 * m) ? m : 1;
    const int w1 = std::max(4, m_eff), w2 = 2 * w1;
    const int h0 = std::max({lag, m_eff > 1 ? 2 * m_eff : 0, w2});
    const int nf = lag + (m_eff > 1 ? 3 : 0) + 4;
    for (size_t i = 0; i < norm.size(); ++i)
        if (static_cast<int>(norm[i].size()) <= h0)
            throw std::runtime_error("Series " + std::to_string(i + 1) + " is too short for the LightGBM features (at least " +
                                     std::to_string(h0 + 1) + " observations required).");
    const size_t ns = norm.size();
    Vec F(ns * static_cast<size_t>(nf)), pred(ns);
    for (size_t i = 0; i < ns; ++i) {                    // features at the origin t = n (hist[n] itself is not read)
        Vec hh = norm[i];
        hh.push_back(0.0);
        feature_row(hh.data(), static_cast<int>(norm[i].size()), lag, m_eff, w1, w2, &F[i * nf]);
    }
    Lib lib(s.lib);
    std::vector<Vec> out(ns, Vec(h));
    for (int k = 0; k < h; ++k) {
        size_t n_rows = 0;
        for (const Vec& x : norm)
            if (static_cast<int>(x.size()) - h0 - k > 0) n_rows += x.size() - h0 - k;
        if (n_rows == 0)
            throw std::runtime_error("No training rows for forecast step " + std::to_string(k + 1) +
                                     ": the series are too short for this horizon.");
        if (n_rows > 2147483647u) throw std::runtime_error("lightgbm: more than 2^31 - 1 training rows; use max_obs");
        Handles hd(lib);
        {
            Vec X(n_rows * static_cast<size_t>(nf));
            std::vector<float> y(n_rows);
            size_t r = 0;
            for (const Vec& x : norm)
                for (int t = h0; t + k < static_cast<int>(x.size()); ++t, ++r) {
                    feature_row(x.data(), t, lag, m_eff, w1, w2, &X[r * nf]);
                    y[r] = static_cast<float>(x[t + k]);
                }
            lib.check(lib.ds_mat(X.data(), kFloat64, static_cast<int32_t>(n_rows), nf, 1, s.params.c_str(), nullptr, &hd.ds));
            lib.check(lib.ds_field(hd.ds, "label", y.data(), static_cast<int>(n_rows), kFloat32));
        }
        lib.check(lib.b_create(hd.ds, s.params.c_str(), &hd.booster));
        for (int it = 0; it < s.rounds; ++it) {
            int finished = 0;
            lib.check(lib.b_update(hd.booster, &finished));
            if (finished) break;
        }
        int64_t len = 0;
        lib.check(lib.b_predict(hd.booster, F.data(), kFloat64, static_cast<int32_t>(ns), nf, 1, kPredictNormal, 0, -1, "",
                                &len, pred.data()));
        for (size_t i = 0; i < ns; ++i) out[i][k] = pred[i] * means[i];
    }
    return out;
}

std::vector<Vec> lightgbm_forecast(const std::vector<Vec>& series, int lag, int h, int max_obs, const LightgbmSetup& s) {
    if (lag < 1) throw std::runtime_error("lag must be >= 1");
    if (s.rounds < 1) throw std::runtime_error("lightgbm: num_iterations must be >= 1");
    std::vector<Vec> norm;
    Vec means;
    size_t n_rows = 0;
    for (size_t i = 0; i < series.size(); ++i) {               // py_global.normalise
        Vec x = series[i];
        if (max_obs > 0 && static_cast<int>(x.size()) > max_obs) x = Vec(x.end() - max_obs, x.end());
        if (static_cast<int>(x.size()) < lag + 1)
            throw std::runtime_error("Series " + std::to_string(i + 1) + " is too short for lag = " + std::to_string(lag) +
                                     " (length " + std::to_string(x.size()) + "; at least " + std::to_string(lag + 1) +
                                     " required).");
        double mu = np_mean(x.data(), x.size());
        if (mu == 0.0) mu = 1.0;
        for (double& v : x) v /= mu;
        n_rows += x.size() - lag;
        norm.push_back(std::move(x));
        means.push_back(mu);
    }
    if (n_rows > 2147483647u) throw std::runtime_error("lightgbm: more than 2^31 - 1 training rows; use max_obs");
    Lib lib(s.lib);
    Handles hd(lib);
    {
        Vec X(n_rows * static_cast<size_t>(lag));               // py_global.lag_matrix: rows in rbind order
        std::vector<float> y(n_rows);
        size_t r = 0;
        for (const Vec& x : norm) {
            const size_t n = x.size();
            for (size_t i = 0; i + lag < n; ++i, ++r) {
                y[r] = static_cast<float>(x[i + lag]);
                for (int j = 1; j <= lag; ++j) X[r * lag + (j - 1)] = x[i + lag - j];
            }
        }
        lib.check(lib.ds_mat(X.data(), kFloat64, static_cast<int32_t>(n_rows), lag, 1, s.params.c_str(), nullptr, &hd.ds));
        lib.check(lib.ds_field(hd.ds, "label", y.data(), static_cast<int>(n_rows), kFloat32));
    }
    lib.check(lib.b_create(hd.ds, s.params.c_str(), &hd.booster));
    for (int it = 0; it < s.rounds; ++it) {
        int finished = 0;
        lib.check(lib.b_update(hd.booster, &finished));
        if (finished) break;
    }
    const size_t ns = norm.size();
    Vec lags(ns * static_cast<size_t>(lag)), pred(ns);
    std::vector<Vec> out(ns, Vec(h));
    for (size_t i = 0; i < ns; ++i)
        for (int j = 0; j < lag; ++j) lags[i * lag + j] = norm[i][norm[i].size() - 1 - j];
    for (int k = 0; k < h; ++k) {
        int64_t len = 0;
        lib.check(lib.b_predict(hd.booster, lags.data(), kFloat64, static_cast<int32_t>(ns), lag, 1, kPredictNormal, 0, -1, "",
                                &len, pred.data()));
        for (size_t i = 0; i < ns; ++i) {
            out[i][k] = pred[i] * means[i];
            double* L = lags.data() + i * lag;
            for (int j = lag - 1; j > 0; --j) L[j] = L[j - 1];
            L[0] = pred[i];
        }
    }
    return out;
}

}  // namespace tsf
