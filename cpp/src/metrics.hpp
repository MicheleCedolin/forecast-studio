// Error measures, their summary and the M5 WRMSSE for tsfcore (C++17). Mirrors src/tsfstudio/metrics.py and wrmsse.py:
// the same definitions, the same treatment of undefined values (NaN per series, left out of the summary) and the same
// order of floating-point operations as NumPy (pairwise summation), so the two sides agree bit for bit wherever only
// + - * / sqrt are involved (tools/validate_metrics.py checks this). Nothing here depends on the rest of tsfcore.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tsf {
namespace metrics {

// numpy's pairwise summation (the sum behind np.sum / np.mean over a contiguous run of doubles).
double pairwise_sum(const double* a, size_t n);

// seasonality.primary_period: the MASE lag, max(1, trunc(min(seasonality))).
int primary_period(const double* seasonality, int n);

// The measures compute_error_table knows, in metrics.ALL_METRICS order.
const std::vector<std::string>& all_metrics();

// metrics.compute_error_table. forecasts / actuals: n x h, row-major, NaN beyond a series' own test length.
// train: all training parts concatenated, series i = train[offsets[i] .. offsets[i + 1]). lag: primary_period(...).
// out: n x metrics.size(), row-major. Throws std::invalid_argument for an unknown measure.
void error_table(const double* forecasts, const double* actuals, int64_t n, int64_t h, const double* train,
                 const int64_t* offsets, int lag, const std::vector<std::string>& metrics, double* out);

// intervals.series_scales: per series the in-sample MAE of the seasonal naive (lag 1 when that is zero or undefined),
// as MASE uses it; NaN when there is no positive scale. out: n values.
void series_scales(const double* train, const int64_t* offsets, int64_t n, int lag, double* out);

// metrics.summarize for one measure: mean and median over the finite values.
struct Summary {
    double mean, median;
    int64_t n_valid, n_excluded;
};
Summary summarize(const double* values, int64_t n);

// wrmsse.wrmsse. hierarchy: 5 columns (item_id, dept_id, cat_id, store_id, state_id), each with n entries.
// units: n x 28 weights of the last 28 training days (unit sales x price), or nullptr for unit sales.
constexpr int WRMSSE_LEVELS = 12;
constexpr int WRMSSE_WINDOW = 28;
struct WrmsseResult {
    double total;
    double level[WRMSSE_LEVELS];
    int64_t n_series[WRMSSE_LEVELS];
};
const char* wrmsse_level_name(int level);
WrmsseResult wrmsse(const double* train, const int64_t* offsets, int64_t n, const double* test, const double* forecasts,
                    int64_t h, const std::vector<std::vector<std::string>>& hierarchy, const double* units);

}  // namespace metrics
}  // namespace tsf
