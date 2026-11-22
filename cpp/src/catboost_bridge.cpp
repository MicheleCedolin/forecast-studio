#include "catboost_bridge.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
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

using CreateFn = void* (*)();
using DeleteFn = void (*)(void*);
using ErrFn = const char* (*)();
using LoadFn = bool (*)(void*, const char*);
using PredFn = bool (*)(void*, size_t, const float**, size_t, double*, size_t);

struct ModelLib {
#if defined(_WIN32)
    HMODULE h = nullptr;
    void* sym(const char* n) const { return reinterpret_cast<void*>(GetProcAddress(h, n)); }
#else
    void* h = nullptr;
    void* sym(const char* n) const { return dlsym(h, n); }
#endif
    CreateFn create = nullptr; DeleteFn del = nullptr; ErrFn err = nullptr; LoadFn load = nullptr; PredFn pred = nullptr;
    explicit ModelLib(const std::string& path) {
#if defined(_WIN32)
        h = LoadLibraryA(path.c_str());
#else
        h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
        if (!h) throw std::runtime_error("catboost: cannot load the model library " + path);
        create = reinterpret_cast<CreateFn>(sym("ModelCalcerCreate"));
        del = reinterpret_cast<DeleteFn>(sym("ModelCalcerDelete"));
        err = reinterpret_cast<ErrFn>(sym("GetErrorString"));
        load = reinterpret_cast<LoadFn>(sym("LoadFullModelFromFile"));
        pred = reinterpret_cast<PredFn>(sym("CalcModelPredictionFlat"));
        if (!create || !del || !err || !load || !pred) throw std::runtime_error("catboost: the model library lacks the C API");
    }
    ~ModelLib() {
#if defined(_WIN32)
        if (h) FreeLibrary(h);
#else
        if (h) dlclose(h);
#endif
    }
};

int run_process(const std::string& cmdline) {
#if defined(_WIN32)
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    PROCESS_INFORMATION pi{};
    std::string cl = cmdline;
    if (!CreateProcessA(nullptr, cl.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi))
        throw std::runtime_error("catboost: cannot start the command-line program");
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return static_cast<int>(code);
#else
    return std::system(cmdline.c_str());
#endif
}

std::string q(const std::string& s) { return "\"" + s + "\""; }

}  // namespace

std::vector<Vec> catboost_forecast(const std::vector<Vec>& series, int lag, int h, int max_obs, const CatboostSetup& s) {
    if (lag < 1) throw std::runtime_error("lag must be >= 1");
    std::vector<Vec> norm;
    Vec means;
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
        norm.push_back(std::move(x));
        means.push_back(mu);
    }
    const std::string sep = (s.workdir.empty() || s.workdir.back() == '/' || s.workdir.back() == '\\') ? "" : "/";
    const std::string learn = s.workdir + sep + "learn.tsv", cd = s.workdir + sep + "learn.cd",
                      model = s.workdir + sep + "model.cbm";
    {
        // float32 values written with 9 significant digits round-trip exactly (what the Python Pool receives)
        FILE* f = std::fopen(learn.c_str(), "wb");
        if (!f) throw std::runtime_error("catboost: cannot write " + learn);
        for (const Vec& x : norm) {
            const int n = static_cast<int>(x.size());
            for (int i = 0; i + lag < n; ++i) {
                std::fprintf(f, "%.9g", static_cast<double>(static_cast<float>(x[i + lag])));
                for (int j = 1; j <= lag; ++j) std::fprintf(f, "\t%.9g", static_cast<double>(static_cast<float>(x[i + lag - j])));
                std::fputc('\n', f);
            }
        }
        std::fclose(f);
        std::ofstream c(cd);
        c << "0\tLabel\n";
    }
    const std::string cmd = q(s.exe) + " fit --learn-set " + q(learn) + " --column-description " + q(cd) +
                            " --loss-function RMSE --random-seed 0 --model-file " + q(model) + " --train-dir " +
                            q(s.workdir) + " --logging-level Silent --thread-count " + std::to_string(s.threads);
    if (run_process(cmd) != 0) throw std::runtime_error("catboost: training failed (catboost fit returned an error)");
    ModelLib lib(s.model_lib);
    void* m = lib.create();
    if (!lib.load(m, model.c_str())) { const std::string e = lib.err(); lib.del(m); throw std::runtime_error("catboost: " + e); }
    const size_t ns = norm.size();
    std::vector<Vec> lags(ns, Vec(lag)), out(ns, Vec(h));
    for (size_t i = 0; i < ns; ++i)
        for (int j = 0; j < lag; ++j) lags[i][j] = norm[i][norm[i].size() - 1 - j];
    std::vector<std::vector<float>> rows(ns, std::vector<float>(lag));
    std::vector<const float*> ptr(ns);
    Vec pred(ns);
    for (int k = 0; k < h; ++k) {
        for (size_t i = 0; i < ns; ++i) {
            for (int j = 0; j < lag; ++j) rows[i][j] = static_cast<float>(lags[i][j]);
            ptr[i] = rows[i].data();
        }
        if (!lib.pred(m, ns, ptr.data(), lag, pred.data(), ns)) { const std::string e = lib.err(); lib.del(m); throw std::runtime_error("catboost: " + e); }
        for (size_t i = 0; i < ns; ++i) {
            out[i][k] = pred[i] * means[i];
            for (int j = lag - 1; j > 0; --j) lags[i][j] = lags[i][j - 1];
            lags[i][0] = pred[i];
        }
    }
    lib.del(m);
    std::remove(learn.c_str());
    std::remove(cd.c_str());
    std::remove(model.c_str());
    return out;
}

}  // namespace tsf
