// A complete run inside tsfcore (C++17): read the .tsf file, split every series into training part and test horizon,
// forecast with one of tsfcore's local methods, compute the error measures and write the result files and the manifest.
// Mirrors src/tsfstudio/runner.py (prepare, rolling origins, _write_outputs, the manifest layout) and intervals.py (split
// conformal prediction intervals and their measures): the CSV files are byte for byte what the Python runner writes with
// the C++ engine (tools/validate_run.py). Global models, the WRMSSE file, the memory / time estimate and the Excel files
// stay with the Python runner for now.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "tsfio.hpp"

namespace tsf {
namespace run {

// A problem the user can act on (runner.RunError).
struct RunError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Config {
    std::string method = "ets";         // plain name, or "name?key=value&..." as tsf_forecast_batch takes it
    std::string dataset_id;             // label for the run folder and the manifest (default: file name without extension)
    int64_t horizon = 0;                // 0 = the header's @horizon
    std::vector<double> seasonality;    // empty = from the header's @frequency
    bool integer_conversion = false;    // round the forecasts to whole numbers
    std::string missing_policy = "error";  // error | zero | locf | interpolate (missing values in the training part)
    int64_t max_series = 0;             // 0 = all series, else the first N
    std::vector<std::string> metrics;   // empty = the default set (metrics.DEFAULT_METRICS)
    int64_t origins = 1;                // rolling origin: number of forecast origins (1 = the last horizon only)
    int64_t origin_step = 0;            // observations between origins (0 = the horizon: windows do not overlap)
    std::vector<int> intervals;         // prediction-interval levels in percent, e.g. {80, 95}; empty = none
    int threads = 0;                    // 0 = all CPU cores
    std::string eol;                    // line end of the CSV files; empty = the platform's (pandas does the same)
};

struct Excluded {
    int64_t index;
    std::string name, reason;
};

struct Prepared {
    std::vector<std::string> names;
    std::vector<std::vector<double>> train;
    std::vector<double> test;           // n x horizon, row-major
    int64_t horizon = 0;
    std::vector<double> seasonality;
    std::vector<Excluded> excluded;
    int64_t series_with_missing = 0, imputed_values = 0, test_missing_values = 0;
    std::vector<std::string> warnings;
};

// seasonality.seasonality_for: the seasonal periods of a Monash frequency name (no frequency -> 1).
std::vector<double> seasonality_for(const tsf::io::Header& header);

// runner.prepare
Prepared prepare(const tsf::io::Dataset& ds, const Config& cfg, int64_t cut = 0, int64_t min_extra = 0);

// The CSV text of a table with a leading text column, as pandas.DataFrame.to_csv(index=False) writes it.
std::string table_csv(const std::string& first_header, const std::vector<std::string>& first_column,
                      const std::vector<std::string>& headers, const double* values, const std::string& eol);

// SHA-256 of a file, lower-case hex (empty string if it cannot be read).
std::string sha256_file(const std::string& utf8_path);

// Runs the whole pipeline and returns the run folder (out_dir/<dataset>__cpp-<method>__<UTC time>).
std::string run(const std::string& tsf_path, const std::string& out_dir, const Config& cfg);

}  // namespace run
}  // namespace tsf
