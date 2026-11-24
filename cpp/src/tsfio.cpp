// Monash .tsf reader, writer and dataset summary (see tsfio.hpp). The plain C interface is at the end of the file.
#include "tsfio.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace tsf {
namespace io {

namespace {

const double NaN = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------------------------------------- text helpers
struct NeedUtf8 {};  // a byte that cp1252 does not define: read the file again as UTF-8 (reader.read_tsf does the same)

// cp1252 0x80..0x9F -> Unicode (0 = undefined); 0xA0..0xFF map to the same code point.
const uint16_t CP1252_HI[32] = {0x20AC, 0,      0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021, 0x02C6, 0x2030, 0x0160,
                                0x2039, 0x0152, 0,      0x017D, 0,      0,      0x2018, 0x2019, 0x201C, 0x201D, 0x2022,
                                0x2013, 0x2014, 0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0,      0x017E, 0x0178};

void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

bool has_high_byte(const std::string& s) {
    for (unsigned char c : s)
        if (c >= 0x80) return true;
    return false;
}

void cp1252_to_utf8(std::string& line) {
    if (!has_high_byte(line)) return;
    std::string out;
    out.reserve(line.size() + 16);
    for (unsigned char c : line) {
        if (c < 0x80) {
            out.push_back(static_cast<char>(c));
        } else if (c < 0xA0) {
            const uint16_t cp = CP1252_HI[c - 0x80];
            if (cp == 0) throw NeedUtf8{};
            append_utf8(out, cp);
        } else {
            append_utf8(out, c);
        }
    }
    line.swap(out);
}

// Whitespace as Python's str.strip()/split() see it, for the characters that can occur here.
inline bool is_ws(unsigned char c) { return c == ' ' || (c >= 9 && c <= 13) || (c >= 0x1C && c <= 0x1F); }

// [b, e) without leading/trailing whitespace (ASCII, NBSP and NEL, the latter two in their UTF-8 form).
void strip(const char*& b, const char*& e) {
    for (;;) {
        if (b < e && is_ws(static_cast<unsigned char>(*b))) {
            ++b;
        } else if (e - b >= 2 && static_cast<unsigned char>(b[0]) == 0xC2 &&
                   (static_cast<unsigned char>(b[1]) == 0xA0 || static_cast<unsigned char>(b[1]) == 0x85)) {
            b += 2;
        } else {
            break;
        }
    }
    for (;;) {
        if (b < e && is_ws(static_cast<unsigned char>(e[-1]))) {
            --e;
        } else if (e - b >= 2 && static_cast<unsigned char>(e[-2]) == 0xC2 &&
                   (static_cast<unsigned char>(e[-1]) == 0xA0 || static_cast<unsigned char>(e[-1]) == 0x85)) {
            e -= 2;
        } else {
            break;
        }
    }
}

std::string lower(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return s;
}

std::string at_line(int64_t lineno) { return "Line " + std::to_string(lineno) + ": "; }

bool strtobool(const std::string& value, bool& out) {
    const std::string v = lower(value);
    if (v == "y" || v == "yes" || v == "t" || v == "true" || v == "on" || v == "1") { out = true; return true; }
    if (v == "n" || v == "no" || v == "f" || v == "false" || v == "off" || v == "0") { out = false; return true; }
    return false;
}

// int(text) for the plain forms: optional whitespace and sign, decimal digits. `norm` gets str(int(text)).
bool parse_int_text(const char* b, const char* e, std::string& norm) {
    strip(b, e);
    bool neg = false;
    if (b < e && (*b == '+' || *b == '-')) neg = (*b++ == '-');
    if (b == e) return false;
    for (const char* p = b; p < e; ++p)
        if (*p < '0' || *p > '9') return false;
    while (e - b > 1 && *b == '0') ++b;
    norm.assign(b, e);
    if (neg && norm != "0") norm.insert(norm.begin(), '-');
    return true;
}

bool read_digits(const char*& p, const char* e, int min_n, int max_n, int& out) {
    int n = 0, v = 0;
    while (p < e && n < max_n && *p >= '0' && *p <= '9') { v = v * 10 + (*p++ - '0'); ++n; }
    out = v;
    return n >= min_n;
}

// datetime.strptime(text, "%Y-%m-%d %H-%M-%S"); `norm` gets the value in the same format, zero padded.
bool parse_date_text(const char* b, const char* e, std::string& norm) {
    int Y, M, D, h, m, s;
    const char* p = b;
    if (!read_digits(p, e, 4, 4, Y) || p == e || *p++ != '-') return false;
    if (!read_digits(p, e, 1, 2, M) || p == e || *p++ != '-') return false;
    if (!read_digits(p, e, 1, 2, D)) return false;
    if (p == e || !is_ws(static_cast<unsigned char>(*p))) return false;
    while (p < e && is_ws(static_cast<unsigned char>(*p))) ++p;
    if (!read_digits(p, e, 1, 2, h) || p == e || *p++ != '-') return false;
    if (!read_digits(p, e, 1, 2, m) || p == e || *p++ != '-') return false;
    if (!read_digits(p, e, 1, 2, s) || p != e) return false;
    static const int DAYS[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    const bool leap = (Y % 4 == 0 && Y % 100 != 0) || Y % 400 == 0;
    if (Y < 1 || M < 1 || M > 12 || D < 1 || D > DAYS[M - 1] + (M == 2 && leap ? 1 : 0)) return false;
    if (h > 23 || m > 59 || s > 59) return false;
    char buf[32];
    std::snprintf(buf, sizeof buf, "%04d-%02d-%02d %02d-%02d-%02d", Y, M, D, h, m, s);
    norm = buf;
    return true;
}

// One series value outside the fast path: float(text) for surrounding whitespace, a sign, nan / inf, and '?' (missing).
bool parse_value_slow(const char* b, const char* e, double& out) {
    strip(b, e);
    if (b == e) return false;
    bool neg = false;
    const char* p = b;
    if (*p == '+' || *p == '-') neg = (*p++ == '-');
    if (p == e) return false;
    const std::string rest = lower(std::string(p, e));
    if (rest == "?" || rest == "nan") { out = NaN; return true; }
    if (rest == "inf" || rest == "infinity") { out = neg ? -HUGE_VAL : HUGE_VAL; return true; }
    if (!((*p >= '0' && *p <= '9') || *p == '.')) return false;  // no second sign, no nan(...) payloads
    double v = 0.0;
    const auto r = std::from_chars(p, e, v);
    if (r.ptr != e) return false;
    if (r.ec == std::errc::result_out_of_range) v = std::strtod(std::string(p, e).c_str(), nullptr);  // overflow -> inf, underflow -> 0
    else if (r.ec != std::errc()) return false;
    out = neg ? -v : v;
    return true;
}

// ---------------------------------------------------------------------------------------------- line reader
// Universal newlines ("\n", "\r\n", "\r"), any line length.
class LineReader {
  public:
    explicit LineReader(ByteSource src) : src_(std::move(src)), buf_(1 << 20) {}

    bool next(std::string& line) {
        line.clear();
        bool got = false;
        for (;;) {
            if (pos_ == len_) {
                if (eof_) return got;
                len_ = src_(buf_.data(), buf_.size());
                pos_ = 0;
                if (len_ == 0) { eof_ = true; return got; }
            }
            if (skip_lf_) {
                skip_lf_ = false;
                if (buf_[pos_] == '\n') { ++pos_; continue; }
            }
            size_t i = pos_;
            while (i < len_ && buf_[i] != '\n' && buf_[i] != '\r') ++i;
            line.append(buf_.data() + pos_, i - pos_);
            got = true;
            if (i < len_) {
                skip_lf_ = (buf_[i] == '\r');
                pos_ = i + 1;
                return true;
            }
            pos_ = len_;
        }
    }

  private:
    ByteSource src_;
    std::vector<char> buf_;
    size_t pos_ = 0, len_ = 0;
    bool eof_ = false, skip_lf_ = false;
};

// reader._parse_meta_line
void parse_meta_line(const char* b, const char* e, Header& h, int64_t lineno) {
    std::vector<std::pair<const char*, const char*>> parts;
    for (const char* p = b; p < e;) {
        while (p < e && is_ws(static_cast<unsigned char>(*p))) ++p;
        const char* q = p;
        while (q < e && !is_ws(static_cast<unsigned char>(*q))) ++q;
        if (q > p) parts.emplace_back(p, q);
        p = q;
    }
    const std::string key = lower(std::string(parts[0].first, parts[0].second));
    const std::string text(b, e);
    if (key == "@attribute") {
        if (parts.size() != 3) throw FormatError(at_line(lineno) + "@attribute must have the form 'name type': " + text);
        const std::string type(parts[2].first, parts[2].second);
        if (type != "numeric" && type != "string" && type != "date")
            throw FormatError(at_line(lineno) + "invalid attribute type '" + type + "' (numeric|string|date)");
        h.attributes.emplace_back(std::string(parts[1].first, parts[1].second), type);
        return;
    }
    if (parts.size() < 2) throw FormatError(at_line(lineno) + "header value is missing: " + text);
    const char* vb = parts[1].first;
    const char* ve = e;
    strip(vb, ve);
    const std::string value(vb, ve);
    if (key == "@relation") {
        h.relation = value;
        h.has_relation = true;
    } else if (key == "@frequency") {
        h.frequency = value;
        h.has_frequency = true;
    } else if (key == "@horizon") {
        std::string norm;
        if (!parse_int_text(vb, ve, norm) || norm.size() > 18)
            throw FormatError(at_line(lineno) + "@horizon must be an integer: '" + value + "'");
        h.horizon = std::strtoll(norm.c_str(), nullptr, 10);
        h.has_horizon = true;
    } else if (key == "@missing" || key == "@equallength") {
        bool flag = false;
        if (!strtobool(value, flag)) throw FormatError(at_line(lineno) + "expected a boolean value but found '" + value + "'");
        (key == "@missing" ? h.missing : h.equal_length) = flag ? 1 : 0;
    }
    // unknown @tags are ignored (forward compatibility)
}

Dataset read_impl(const SourceFactory& factory, int64_t max_series, bool header_only, bool utf8) {
    Dataset ds;
    Header& h = ds.header;
    LineReader reader(factory());
    std::string line;
    std::vector<const char*> colons;
    std::vector<double> scratch;
    bool in_data = false, first = true;
    size_t n_attr = 0;
    int64_t lineno = 0;
    while (reader.next(line)) {
        ++lineno;
        if (first) {
            first = false;
            if (utf8 && line.size() >= 3 && std::memcmp(line.data(), "\xEF\xBB\xBF", 3) == 0) line.erase(0, 3);
        }
        if (!utf8) cp1252_to_utf8(line);
        const char* b = line.data();
        const char* e = b + line.size();
        strip(b, e);
        if (b == e) continue;
        if (!in_data) {
            if (*b == '@') {
                if (lower(std::string(b, std::min(e, b + 5))) == "@data") {
                    if (header_only) return ds;
                    if (h.attributes.empty())
                        throw FormatError("There is no attribute section (@attribute); it must come before the data");
                    in_data = true;
                    n_attr = h.attributes.size();
                    ds.attrs.assign(n_attr, {});
                } else {
                    parse_meta_line(b, e, h, lineno);
                }
            } else if (*b == '#') {
                h.comments.emplace_back(b, e);
            } else {
                throw FormatError(at_line(lineno) + "a data line was found before the @data tag");
            }
            continue;
        }
        if (*b == '#') continue;
        if (max_series >= 0 && static_cast<int64_t>(ds.values.size()) >= max_series) break;

        colons.clear();
        for (const char* p = b; p < e; ++p)
            if (*p == ':') colons.push_back(p);
        if (colons.size() != n_attr)
            throw FormatError(at_line(lineno) + "expected " + std::to_string(n_attr) + " attributes + values, found " +
                              std::to_string(colons.size() + 1) + " fields separated by ':'");

        // values: "v1,v2,..." with '?' for a missing value
        scratch.clear();
        bool any_value = false;
        for (const char* p = colons.empty() ? b : colons.back() + 1;;) {
            const char* q = p;
            while (q < e && *q != ',') ++q;
            double v;
            if (q - p == 1 && *p == '?') {
                v = NaN;
            } else {
                const bool plain = p < q && ((*p >= '0' && *p <= '9') || *p == '-' || *p == '.');
                const auto r = plain ? std::from_chars(p, q, v) : std::from_chars_result{p, std::errc::invalid_argument};
                if ((r.ec != std::errc() || r.ptr != q) && !parse_value_slow(p, q, v))
                    throw FormatError(at_line(lineno) + "non-numeric series value ('" + std::string(p, q) + "')");
            }
            any_value = any_value || v == v;
            scratch.push_back(v);
            if (q == e) break;
            p = q + 1;
        }
        if (!any_value) throw FormatError(at_line(lineno) + "all values in the series are missing");

        // attributes (converted first into a row so that a bad line leaves the columns untouched)
        const char* fb = b;
        std::vector<std::string> row(n_attr);
        for (size_t i = 0; i < n_attr; ++i) {
            const char* fe = colons[i];
            const std::string& type = h.attributes[i].second;
            bool ok = true;
            if (type == "numeric") ok = parse_int_text(fb, fe, row[i]);
            else if (type == "date") ok = parse_date_text(fb, fe, row[i]);
            else row[i].assign(fb, fe);
            if (!ok)
                throw FormatError(at_line(lineno) + "invalid value '" + std::string(fb, fe) + "' for the attribute '" +
                                  h.attributes[i].first + "' (" + type + ")");
            fb = fe + 1;
        }
        for (size_t i = 0; i < n_attr; ++i) ds.attrs[i].push_back(std::move(row[i]));
        ds.values.emplace_back(scratch.begin(), scratch.end());
    }
    if (!in_data) throw FormatError("The @data tag was not found");
    if (ds.values.empty()) throw FormatError("The file contains no series");
    return ds;
}

Dataset read_any(const SourceFactory& src, int64_t max_series, bool header_only) {
    try {
        return read_impl(src, max_series, header_only, false);
    } catch (const NeedUtf8&) {
        return read_impl(src, max_series, header_only, true);
    }
}

// ---------------------------------------------------------------------------------------------- writer
using Sink = std::function<void(const char*, size_t)>;

void put_text(const Sink& sink, const std::string& s) {
    if (has_high_byte(s)) throw FormatError("The .tsf writer is ASCII only; a non-ASCII character was found in: " + s);
    sink(s.data(), s.size());
}

bool valid_name(const std::string& s) {
    if (s.empty() || !((s[0] >= 'A' && s[0] <= 'Z') || (s[0] >= 'a' && s[0] <= 'z') || s[0] == '_')) return false;
    for (char c : s)
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) return false;
    return true;
}

void emit(const Dataset& ds, const Sink& sink) {
    const Header& h = ds.header;
    for (const auto& a : h.attributes) {
        if (!valid_name(a.first))
            throw FormatError("An attribute name may only contain letters/digits/underscore: '" + a.first + "'");
        if (a.second != "numeric" && a.second != "string" && a.second != "date")
            throw FormatError("Invalid attribute type: '" + a.second + "'");
    }
    const std::string relation = (h.has_relation && !h.relation.empty()) ? h.relation : "dataset";
    if (relation.find(' ') != std::string::npos) throw FormatError("The @relation value cannot contain spaces");
    for (const auto& c : h.comments) put_text(sink, (!c.empty() && c[0] == '#' ? c : "# " + c) + "\n");
    put_text(sink, "@relation " + relation + "\n");
    for (const auto& a : h.attributes) put_text(sink, "@attribute " + a.first + " " + a.second + "\n");
    if (h.has_frequency) put_text(sink, "@frequency " + h.frequency + "\n");
    if (h.has_horizon) put_text(sink, "@horizon " + std::to_string(h.horizon) + "\n");
    put_text(sink, std::string("@missing ") + (ds.has_missing() ? "true" : "false") + "\n");
    put_text(sink, std::string("@equallength ") + (ds.is_equal_length() ? "true" : "false") + "\n");
    put_text(sink, "@data\n");

    std::string out;
    char num[32];
    for (size_t s = 0; s < ds.values.size(); ++s) {
        out.clear();
        for (size_t a = 0; a < ds.attrs.size(); ++a) {
            const std::string& v = ds.attrs[a][s];
            if (v.find_first_of(":\n\r") != std::string::npos)
                throw FormatError("An attribute value cannot contain ':' or a line break: '" + v + "'");
            if (has_high_byte(v)) throw FormatError("The .tsf writer is ASCII only; a non-ASCII character was found in: " + v);
            out += v;
            out.push_back(':');
        }
        const std::vector<double>& y = ds.values[s];
        for (size_t i = 0; i < y.size(); ++i) {
            const double v = y[i];
            if (i) out.push_back(',');
            if (v != v) {
                out.push_back('?');
            } else if (std::isinf(v)) {
                throw FormatError("A series value is infinite and cannot be written");
            } else if (v == std::trunc(v) && std::fabs(v) < 1e15) {
                const auto r = std::to_chars(num, num + sizeof num, static_cast<long long>(v));
                out.append(num, r.ptr);
            } else {
                out += py_repr(v);
            }
        }
        out.push_back('\n');
        sink(out.data(), out.size());
    }
}

struct FileCloser {
    void operator()(std::FILE* f) const { if (f) std::fclose(f); }
};
using FilePtr = std::unique_ptr<std::FILE, FileCloser>;

FilePtr open_file(const std::string& utf8_path, bool write) {
#if defined(_WIN32)
    const int n = MultiByteToWideChar(CP_UTF8, 0, utf8_path.c_str(), -1, nullptr, 0);
    std::wstring w(n > 0 ? static_cast<size_t>(n) : 1, L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, utf8_path.c_str(), -1, &w[0], n);
    return FilePtr(_wfopen(w.c_str(), write ? L"wb" : L"rb"));
#else
    return FilePtr(std::fopen(utf8_path.c_str(), write ? "wb" : "rb"));
#endif
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

}  // namespace

// ---------------------------------------------------------------------------------------------- public
bool Dataset::has_missing() const {
    for (const auto& y : values)
        for (double v : y)
            if (v != v) return true;
    return false;
}

bool Dataset::is_equal_length() const {
    if (values.empty()) return false;
    for (const auto& y : values)
        if (y.size() != values[0].size()) return false;
    return true;
}

SourceFactory file_source(const std::string& utf8_path) {
    return [utf8_path]() -> ByteSource {
        std::shared_ptr<std::FILE> f(open_file(utf8_path, false).release(), FileCloser());
        if (!f) throw std::runtime_error("Cannot open the file: " + utf8_path);
        return [f](char* buf, size_t cap) { return std::fread(buf, 1, cap, f.get()); };
    };
}

SourceFactory buffer_source(const char* data, size_t len) {
    return [data, len]() -> ByteSource {
        auto pos = std::make_shared<size_t>(0);
        return [data, len, pos](char* buf, size_t cap) {
            const size_t n = std::min(cap, len - *pos);
            std::memcpy(buf, data + *pos, n);
            *pos += n;
            return n;
        };
    };
}

Dataset read_tsf(const SourceFactory& src, int64_t max_series) { return read_any(src, max_series, false); }

Header read_tsf_header(const SourceFactory& src) {
    const Dataset ds = read_any(src, -1, true);  // stops at @data; a file without @data is an error there
    if (ds.header.attributes.empty())
        throw FormatError("There is no attribute section (@attribute); it must come before the data");
    return ds.header;
}

void write_tsf(const std::string& utf8_path, const Dataset& ds) {
    FilePtr f = open_file(utf8_path, true);
    if (!f) throw std::runtime_error("Cannot write the file: " + utf8_path);
    std::vector<char> buf(1 << 20);
    std::setvbuf(f.get(), buf.data(), _IOFBF, buf.size());
    bool failed = false;
    emit(ds, [&](const char* p, size_t n) { failed = failed || std::fwrite(p, 1, n, f.get()) != n; });
    failed = std::fflush(f.get()) != 0 || failed;
    f.reset();  // close before the buffer goes away
    if (failed) throw std::runtime_error("Writing failed (disk full?): " + utf8_path);
}

std::string format_tsf(const Dataset& ds) {
    std::string out;
    emit(ds, [&](const char* p, size_t n) { out.append(p, n); });
    return out;
}

Stats dataset_stats(const Dataset& ds) {
    Stats st;
    st.n_series = static_cast<int64_t>(ds.values.size());
    std::vector<int64_t> len;
    len.reserve(ds.values.size());
    for (const auto& y : ds.values) {
        len.push_back(static_cast<int64_t>(y.size()));
        st.n_values += static_cast<int64_t>(y.size());
        for (double v : y) {
            if (v != v) ++st.n_missing;
            else if (v == 0.0) ++st.n_zero;
        }
    }
    if (len.empty()) return st;
    std::sort(len.begin(), len.end());
    st.len_min = len.front();
    st.len_max = len.back();
    const size_t n = len.size();
    st.len_median = n % 2 ? static_cast<double>(len[n / 2]) : (static_cast<double>(len[n / 2 - 1]) + static_cast<double>(len[n / 2])) / 2.0;
    st.equal_length = st.len_min == st.len_max;
    st.missing_share = static_cast<double>(st.n_missing) / static_cast<double>(st.n_values);
    st.zero_share = static_cast<double>(st.n_zero) / static_cast<double>(std::max<int64_t>(1, st.n_values - st.n_missing));
    return st;
}

std::string py_repr(double v) {
    if (v != v) return "nan";
    if (std::isinf(v)) return v < 0 ? "-inf" : "inf";
    char buf[48];
    const auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);  // shortest round-trip digits
    const char* p = buf;
    std::string out;
    if (*p == '-') { out.push_back('-'); ++p; }
    std::string digits;
    for (; p < r.ptr && *p != 'e'; ++p)
        if (*p != '.') digits.push_back(*p);
    const int x = std::atoi(std::string(p + 1, static_cast<const char*>(r.ptr)).c_str());  // decimal exponent of the first digit
    const int nd = static_cast<int>(digits.size());
    if (x < -4 || x >= 16) {
        out.push_back(digits[0]);
        if (nd > 1) { out.push_back('.'); out.append(digits, 1, std::string::npos); }
        char e[8];
        std::snprintf(e, sizeof e, "e%c%02d", x < 0 ? '-' : '+', std::abs(x));
        out += e;
    } else if (x >= 0) {
        if (nd <= x + 1) {
            out += digits;
            out.append(static_cast<size_t>(x + 1 - nd), '0');
            out += ".0";
        } else {
            out.append(digits, 0, static_cast<size_t>(x + 1));
            out.push_back('.');
            out.append(digits, static_cast<size_t>(x + 1), std::string::npos);
        }
    } else {
        out += "0.";
        out.append(static_cast<size_t>(-x - 1), '0');
        out += digits;
    }
    return out;
}

std::string header_json(const Header& h) {
    std::string o = "{\"relation\":";
    if (h.has_relation) json_string(o, h.relation); else o += "null";
    o += ",\"attributes\":[";
    for (size_t i = 0; i < h.attributes.size(); ++i) {
        o += i ? ",[" : "[";
        json_string(o, h.attributes[i].first);
        o.push_back(',');
        json_string(o, h.attributes[i].second);
        o.push_back(']');
    }
    o += "],\"frequency\":";
    if (h.has_frequency) json_string(o, h.frequency); else o += "null";
    o += ",\"horizon\":" + (h.has_horizon ? std::to_string(h.horizon) : std::string("null"));
    o += std::string(",\"missing\":") + (h.missing < 0 ? "null" : h.missing ? "true" : "false");
    o += std::string(",\"equal_length\":") + (h.equal_length < 0 ? "null" : h.equal_length ? "true" : "false");
    o += ",\"comments\":[";
    for (size_t i = 0; i < h.comments.size(); ++i) {
        if (i) o.push_back(',');
        json_string(o, h.comments[i]);
    }
    o += "]}";
    return o;
}

std::string stats_json(const Dataset& ds) {
    const Header& h = ds.header;
    const Stats st = dataset_stats(ds);
    std::string o = "{\"relation\":";
    if (h.has_relation) json_string(o, h.relation); else o += "null";
    o += ",\"frequency\":";
    if (h.has_frequency) json_string(o, h.frequency); else o += "null";
    o += ",\"horizon\":" + (h.has_horizon ? std::to_string(h.horizon) : std::string("null"));
    o += ",\"n_series\":" + std::to_string(st.n_series);
    o += ",\"len_min\":" + std::to_string(st.len_min);
    o += ",\"len_median\":" + py_repr(st.len_median);
    o += ",\"len_max\":" + std::to_string(st.len_max);
    o += std::string(",\"equal_length\":") + (st.equal_length ? "true" : "false");
    o += ",\"missing_share\":" + py_repr(st.missing_share);
    o += ",\"zero_share\":" + py_repr(st.zero_share);
    o += ",\"attributes\":[";
    for (size_t i = 0; i < h.attributes.size(); ++i) {
        if (i) o.push_back(',');
        json_string(o, h.attributes[i].first);
    }
    o += "]}";
    return o;
}

}  // namespace io
}  // namespace tsf

// ================================================================================================ plain C interface
// void* tsf_io_open(path, max_series, msg, msg_len)          handle or NULL (msg says why); max_series < 0 = all
// void* tsf_io_open_buffer(data, len, max_series, msg, len)  the same from memory (browser build)
// void  tsf_io_close(handle)
// tsf_io_n_series / tsf_io_n_attributes(handle)
// const double* tsf_io_series(handle, i, &len)               values of series i (missing = NaN), valid until close
// const char*   tsf_io_attr(handle, i, a)                    attribute a of series i as UTF-8 text
// const char*   tsf_io_header_json / tsf_io_stats_json       UTF-8 JSON, valid until close
// int tsf_io_write(handle, path, msg, msg_len)               0 = written
// int tsf_io_read_header(path, out, out_len, msg, msg_len)   header JSON only (fast on large files); 0 = ok,
//                                                            -1 = error, -3 = out too small
// Paths are UTF-8.
#if defined(_WIN32)
#define TSFIO_API extern "C" __declspec(dllexport)
#else
#define TSFIO_API extern "C" __attribute__((visibility("default")))
#endif

namespace {

struct IoHandle {
    tsf::io::Dataset ds;
    std::string header, stats;
};

void set_msg(char* msg, int32_t msg_len, const char* text) {
    if (!msg || msg_len < 1) return;
    std::snprintf(msg, static_cast<size_t>(msg_len), "%s", text);
}

void* open_from(const tsf::io::SourceFactory& src, int64_t max_series, char* msg, int32_t msg_len) {
    set_msg(msg, msg_len, "");
    try {
        std::unique_ptr<IoHandle> h(new IoHandle);
        h->ds = tsf::io::read_tsf(src, max_series);
        return h.release();
    } catch (const std::bad_alloc&) {
        set_msg(msg, msg_len, "Not enough memory to load the file");
    } catch (const std::exception& err) {
        set_msg(msg, msg_len, err.what());
    }
    return nullptr;
}

}  // namespace

TSFIO_API void* tsf_io_open(const char* path, int64_t max_series, char* msg, int32_t msg_len) {
    if (!path) { set_msg(msg, msg_len, "No path given"); return nullptr; }
    return open_from(tsf::io::file_source(path), max_series, msg, msg_len);
}

TSFIO_API void* tsf_io_open_buffer(const char* data, int64_t len, int64_t max_series, char* msg, int32_t msg_len) {
    if (!data || len < 0) { set_msg(msg, msg_len, "No data given"); return nullptr; }
    return open_from(tsf::io::buffer_source(data, static_cast<size_t>(len)), max_series, msg, msg_len);
}

TSFIO_API void tsf_io_close(void* handle) { delete static_cast<IoHandle*>(handle); }

TSFIO_API int64_t tsf_io_n_series(void* handle) {
    return handle ? static_cast<int64_t>(static_cast<IoHandle*>(handle)->ds.values.size()) : 0;
}

TSFIO_API int32_t tsf_io_n_attributes(void* handle) {
    return handle ? static_cast<int32_t>(static_cast<IoHandle*>(handle)->ds.attrs.size()) : 0;
}

TSFIO_API const double* tsf_io_series(void* handle, int64_t i, int64_t* len) {
    if (len) *len = 0;
    if (!handle) return nullptr;
    const auto& values = static_cast<IoHandle*>(handle)->ds.values;
    if (i < 0 || i >= static_cast<int64_t>(values.size())) return nullptr;
    if (len) *len = static_cast<int64_t>(values[static_cast<size_t>(i)].size());
    return values[static_cast<size_t>(i)].data();
}

TSFIO_API const char* tsf_io_attr(void* handle, int64_t i, int32_t a) {
    if (!handle) return nullptr;
    const auto& attrs = static_cast<IoHandle*>(handle)->ds.attrs;
    if (a < 0 || a >= static_cast<int32_t>(attrs.size())) return nullptr;
    const auto& col = attrs[static_cast<size_t>(a)];
    if (i < 0 || i >= static_cast<int64_t>(col.size())) return nullptr;
    return col[static_cast<size_t>(i)].c_str();
}

TSFIO_API const char* tsf_io_header_json(void* handle) {
    if (!handle) return nullptr;
    IoHandle* h = static_cast<IoHandle*>(handle);
    if (h->header.empty()) h->header = tsf::io::header_json(h->ds.header);
    return h->header.c_str();
}

TSFIO_API const char* tsf_io_stats_json(void* handle) {
    if (!handle) return nullptr;
    IoHandle* h = static_cast<IoHandle*>(handle);
    if (h->stats.empty()) h->stats = tsf::io::stats_json(h->ds);
    return h->stats.c_str();
}

TSFIO_API int32_t tsf_io_write(void* handle, const char* path, char* msg, int32_t msg_len) {
    set_msg(msg, msg_len, "");
    if (!handle || !path) { set_msg(msg, msg_len, "No dataset or path given"); return -2; }
    try {
        tsf::io::write_tsf(path, static_cast<IoHandle*>(handle)->ds);
        return 0;
    } catch (const std::exception& err) {
        set_msg(msg, msg_len, err.what());
        return -1;
    }
}

TSFIO_API int32_t tsf_io_read_header(const char* path, char* out, int32_t out_len, char* msg, int32_t msg_len) {
    set_msg(msg, msg_len, "");
    if (!path || !out || out_len < 1) { set_msg(msg, msg_len, "No path or output buffer given"); return -2; }
    try {
        const std::string json = tsf::io::header_json(tsf::io::read_tsf_header(tsf::io::file_source(path)));
        if (static_cast<int64_t>(json.size()) >= out_len) { set_msg(msg, msg_len, "The output buffer is too small"); return -3; }
        std::memcpy(out, json.c_str(), json.size() + 1);
        return 0;
    } catch (const std::exception& err) {
        set_msg(msg, msg_len, err.what());
        return -1;
    }
}
