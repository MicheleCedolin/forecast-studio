"""Checks the C++ .tsf reader / writer / summary (cpp/src/tsfio.cpp) against the Python ones on real files.

Usage:
    python tools/validate_tsfio.py --lib <tsfio.dll or tsfcore.dll> [--data-dir data/tsf] [--files a.tsf b.tsf]
                                   [--max-mb 300] [--no-write]

For every .tsf file (smallest first) it compares, C++ against Python:
  * the header (relation, attributes, frequency, horizon, flags, comments),
  * every attribute of every series (text form) and every series value, bit for bit (NaN = NaN),
  * the dataset summary (catalog.dataset_stats),
  * a preview read of the first 8 series and the header-only read,
  * the file written back by each side (SHA-256 of the two outputs must be equal).
Exit code 0 only if every file passes.

The library can be the standalone test build
    g++ -std=c++17 -O3 -shared -static -o tsfio.dll cpp/src/tsfio.cpp
or tsfcore once tsfio.cpp is part of it.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsfstudio.catalog import dataset_stats  # noqa: E402
from tsfstudio.tsf.reader import TSF_DATE_FORMAT, read_tsf, read_tsf_header  # noqa: E402
from tsfstudio.tsf.writer import write_dataset  # noqa: E402

MSG_LEN = 512


def load_lib(path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(Path(path).resolve()))
    c = ctypes
    lib.tsf_io_open.restype = c.c_void_p
    lib.tsf_io_open.argtypes = [c.c_char_p, c.c_int64, c.c_char_p, c.c_int32]
    lib.tsf_io_close.restype = None
    lib.tsf_io_close.argtypes = [c.c_void_p]
    lib.tsf_io_n_series.restype = c.c_int64
    lib.tsf_io_n_series.argtypes = [c.c_void_p]
    lib.tsf_io_n_attributes.restype = c.c_int32
    lib.tsf_io_n_attributes.argtypes = [c.c_void_p]
    lib.tsf_io_series.restype = c.POINTER(c.c_double)
    lib.tsf_io_series.argtypes = [c.c_void_p, c.c_int64, c.POINTER(c.c_int64)]
    lib.tsf_io_attr.restype = c.c_char_p
    lib.tsf_io_attr.argtypes = [c.c_void_p, c.c_int64, c.c_int32]
    lib.tsf_io_header_json.restype = c.c_char_p
    lib.tsf_io_header_json.argtypes = [c.c_void_p]
    lib.tsf_io_stats_json.restype = c.c_char_p
    lib.tsf_io_stats_json.argtypes = [c.c_void_p]
    lib.tsf_io_write.restype = c.c_int32
    lib.tsf_io_write.argtypes = [c.c_void_p, c.c_char_p, c.c_char_p, c.c_int32]
    lib.tsf_io_read_header.restype = c.c_int32
    lib.tsf_io_read_header.argtypes = [c.c_char_p, c.c_char_p, c.c_int32, c.c_char_p, c.c_int32]
    return lib


def cpp_open(lib, path: Path, max_series: int = -1):
    msg = ctypes.create_string_buffer(MSG_LEN)
    h = lib.tsf_io_open(str(path).encode("utf-8"), max_series, msg, MSG_LEN)
    if not h:
        raise RuntimeError(msg.value.decode("utf-8", "replace"))
    return h


def cpp_series(lib, h, i: int) -> np.ndarray:
    n = ctypes.c_int64(0)
    p = lib.tsf_io_series(h, i, ctypes.byref(n))
    return np.ctypeslib.as_array(p, shape=(n.value,)) if n.value else np.empty(0)


def header_dict(h) -> dict:
    return {"relation": h.relation, "attributes": [list(a) for a in h.attributes], "frequency": h.frequency,
            "horizon": h.horizon, "missing": h.missing, "equal_length": h.equal_length, "comments": list(h.comments)}


def attr_text(value, typ: str) -> str:
    if typ == "numeric":
        return str(int(value))
    if typ == "date":
        return value.strftime(TSF_DATE_FORMAT)
    return str(value)


def same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    return bool(((a.view(np.uint64) == b.view(np.uint64)) | (np.isnan(a) & np.isnan(b))).all())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def compare(lib, h, ds, fails: list[str], label: str) -> None:
    hj = json.loads(lib.tsf_io_header_json(h).decode("utf-8"))
    if hj != header_dict(ds.header):
        fails.append(f"{label}: header differs")
    n = lib.tsf_io_n_series(h)
    if n != ds.n_series:
        fails.append(f"{label}: {n} series in C++, {ds.n_series} in Python")
        return
    bad_values = sum(1 for i in range(n) if not same_bits(cpp_series(lib, h, i), ds.values[i]))
    if bad_values:
        fails.append(f"{label}: {bad_values} series with different values")
    bad_attrs = 0
    for a, (name, typ) in enumerate(ds.header.attributes):
        col = ds.attrs[name].tolist()
        for i in range(n):
            if lib.tsf_io_attr(h, i, a).decode("utf-8") != attr_text(col[i], typ):
                bad_attrs += 1
    if bad_attrs:
        fails.append(f"{label}: {bad_attrs} attribute values differ")


def check_file(lib, path: Path, write: bool, tmp: Path) -> tuple[list[str], dict]:
    fails: list[str] = []
    t = {}
    t0 = time.perf_counter()
    h = cpp_open(lib, path)
    t["cpp_read"] = time.perf_counter() - t0
    try:
        t0 = time.perf_counter()
        ds = read_tsf(path)
        t["py_read"] = time.perf_counter() - t0
        compare(lib, h, ds, fails, "full read")

        sj = json.loads(lib.tsf_io_stats_json(h).decode("utf-8"))
        if sj != dataset_stats(ds):
            fails.append("summary differs")

        buf = ctypes.create_string_buffer(1 << 20)
        msg = ctypes.create_string_buffer(MSG_LEN)
        rc = lib.tsf_io_read_header(str(path).encode("utf-8"), buf, len(buf), msg, MSG_LEN)
        if rc != 0 or json.loads(buf.value.decode("utf-8")) != header_dict(read_tsf_header(path)):
            fails.append(f"header-only read differs (rc={rc}, {msg.value.decode('utf-8', 'replace')})")

        h8 = cpp_open(lib, path, 8)
        try:
            compare(lib, h8, read_tsf(path, max_series=8), fails, "first 8 series")
        finally:
            lib.tsf_io_close(h8)

        if write:
            p_py, p_cpp = tmp / "py.tsf", tmp / "cpp.tsf"
            py_err = cpp_err = None
            try:
                write_dataset(p_py, ds)
            except Exception as exc:  # e.g. non-ASCII comments: the writer is ASCII only
                py_err = f"{type(exc).__name__}: {exc}"
            t0 = time.perf_counter()
            if lib.tsf_io_write(h, str(p_cpp).encode("utf-8"), msg, MSG_LEN) != 0:
                cpp_err = msg.value.decode("utf-8", "replace")
            t["cpp_write"] = time.perf_counter() - t0
            if (py_err is None) != (cpp_err is None):
                fails.append(f"writer: Python {'failed: ' + py_err if py_err else 'wrote'}, C++ {'failed: ' + cpp_err if cpp_err else 'wrote'}")
            elif py_err is None and sha256(p_py) != sha256(p_cpp):
                fails.append("written files differ")
            t["write"] = "both refuse" if py_err else "identical"
            for p in (p_py, p_cpp):
                p.unlink(missing_ok=True)
    finally:
        lib.tsf_io_close(h)
    return fails, t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--data-dir", default="data/tsf")
    ap.add_argument("--files", nargs="*", help="file names inside --data-dir (default: all .tsf files)")
    ap.add_argument("--max-mb", type=float, default=None, help="skip files larger than this")
    ap.add_argument("--no-write", action="store_true", help="skip the write-back comparison")
    a = ap.parse_args()

    lib = load_lib(a.lib)
    data_dir = Path(a.data_dir)
    files = [data_dir / f for f in a.files] if a.files else sorted(data_dir.glob("*.tsf"))
    files.sort(key=lambda p: p.stat().st_size)
    n_fail = n_skip = 0
    with tempfile.TemporaryDirectory() as td:
        for p in files:
            mb = p.stat().st_size / 1e6
            if a.max_mb is not None and mb > a.max_mb:
                print(f"SKIP  {p.name:55s} {mb:8.1f} MB")
                n_skip += 1
                continue
            try:
                fails, t = check_file(lib, p, not a.no_write, Path(td))
            except Exception as exc:
                fails, t = [f"{type(exc).__name__}: {exc}"], {}
            n_fail += bool(fails)
            times = f"read C++ {t.get('cpp_read', float('nan')):6.2f}s / Python {t.get('py_read', float('nan')):6.2f}s"
            print(f"{'FAIL' if fails else 'PASS'}  {p.name:55s} {mb:8.1f} MB  {times}  write: {t.get('write', '-')}", flush=True)
            for f in fails:
                print(f"      {f}")
    print(f"\n{len(files) - n_skip - n_fail} passed, {n_fail} failed, {n_skip} skipped")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
