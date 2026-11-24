// Monash .tsf reader, writer and dataset summary for tsfcore (C++17). Mirrors src/tsfstudio/tsf/reader.py, writer.py and
// catalog.dataset_stats so that the C++ and Python sides agree value for value (tools/validate_tsfio.py checks this on the
// whole data package). Nothing here depends on the rest of tsfcore, so the file also builds on its own (and for WebAssembly).
//
// Text handling follows the Python reader: bytes are read as cp1252 and, if a byte is undefined there, the file is read
// again as UTF-8 (with an optional BOM). Strings are held as UTF-8.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tsf {
namespace io {

// Invalid .tsf content (the message names the line).
struct FormatError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Header {
    bool has_relation = false, has_frequency = false, has_horizon = false;
    std::string relation, frequency;
    int64_t horizon = 0;
    int missing = -1, equal_length = -1;                          // -1 = not declared, else 0 / 1
    std::vector<std::pair<std::string, std::string>> attributes;  // (name, type: numeric | string | date)
    std::vector<std::string> comments;                            // '#' lines before @data
};

struct Dataset {
    Header header;
    std::vector<std::vector<std::string>> attrs;  // attrs[a][s]: attribute a of series s, in the writer's text form
    std::vector<std::vector<double>> values;      // one vector per series, missing = NaN

    size_t n_series() const { return values.size(); }
    bool has_missing() const;
    bool is_equal_length() const;
};

// catalog.dataset_stats
struct Stats {
    int64_t n_series = 0, n_values = 0, n_missing = 0, n_zero = 0, len_min = 0, len_max = 0;
    double len_median = 0.0, missing_share = 0.0, zero_share = 0.0;
    bool equal_length = false;
};

// Fills `buf` with up to `cap` bytes and returns how many were written (0 = end of input).
using ByteSource = std::function<size_t(char*, size_t)>;
// Returns a fresh source positioned at the start (the reader may need a second pass when it falls back to UTF-8).
using SourceFactory = std::function<ByteSource()>;

SourceFactory file_source(const std::string& utf8_path);
SourceFactory buffer_source(const char* data, size_t len);

// max_series < 0 reads everything; otherwise only the first max_series series (preview / time estimate).
Dataset read_tsf(const SourceFactory& src, int64_t max_series = -1);
Header read_tsf_header(const SourceFactory& src);

// writer.write_dataset: ASCII, '\n' line ends, missing = '?'. @missing / @equallength are written from the data.
void write_tsf(const std::string& utf8_path, const Dataset& ds);
std::string format_tsf(const Dataset& ds);  // the same text in memory (for the browser build)

Stats dataset_stats(const Dataset& ds);

// Python's repr(float): shortest digits that round-trip, in Python's layout ("0.1", "1e-05", "1.5e+16", "nan", "inf").
std::string py_repr(double v);

std::string header_json(const Header& h);
std::string stats_json(const Dataset& ds);

}  // namespace io
}  // namespace tsf
