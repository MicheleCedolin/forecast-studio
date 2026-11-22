// Global CatBoost model driven from C++: the same data pipeline as py_global.catboost_forecast (mean-normalised lag
// rows of all series, float32), training by the official CatBoost command-line program (`catboost fit`, RMSE,
// random seed 0, otherwise CatBoost's defaults) and recursive prediction through CatBoost's model library
// (catboostmodel.dll / libcatboostmodel.so, C API CalcModelPredictionFlat).
#pragma once

#include <string>
#include <vector>

#include "util.hpp"

namespace tsf {

struct CatboostSetup {
    std::string exe;                // catboost(.exe)
    std::string model_lib;          // catboostmodel.dll / libcatboostmodel.so
    std::string workdir;            // existing writable directory for the training files
    int threads = -1;               // -1 = all cores (CatBoost's thread_count)
};

std::vector<Vec> catboost_forecast(const std::vector<Vec>& series, int lag, int h, int max_obs, const CatboostSetup& s);

}  // namespace tsf
