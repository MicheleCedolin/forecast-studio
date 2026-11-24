"""Checks the C++ tsfeatures set (cpp/src/tsfeat.cpp) against the Python module it was ported from
(tsfstudio.tsfeat, through features.tsfeatures_table).

Usage:
    python tools/validate_tsfeat.py [--lib path/to/tsfcore.dll] [--node <node executable>] [--data-dir data/tsf]
                                    [--datasets nn5_weekly ...] [--max-series N]

For every dataset the script computes the table three ways:
  * Python: features.tsfeatures_table(series, tsfeat.study_periods(frequency), start timestamps);
  * native: tsf_tsfeat_new / _add / _columns / _table of the library (--lib, else the one the application loads);
  * with --node, the browser page: web/features.js (TsfFeatures.tsfeatures) on the WebAssembly build under Node.
and compares the columns kept, which series have no features, and every value. The two sides solve their least-squares
fits with different factorisations and sum in different orders, so values agree to rounding, not bit for bit; the
tolerance is 1e-6 relative (to max(1, |value|)). Two kinds of value are left out and counted: ARCH.LM of a series whose
regression fits exactly (see arch_is_noise), which is rounding noise on every side, and the position of a largest shift
that is reached at several places (see shift_ties). Exit code 0 if every dataset passes.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NODE_SCRIPT = r"""
const fs = require("fs");
const path = require("path");
const web = process.argv[2];
const createTsfCore = require(path.join(web, "tsfcore.js"));
const TF = require(path.join(web, "features.js"));
(async () => {
  const core = await createTsfCore();
  const maxSeries = Number(process.argv[4]);
  const out = {};
  for (const file of process.argv.slice(5)) {
    const bytes = fs.readFileSync(file);
    const p = core._malloc(bytes.length), msg = core._malloc(512);
    core.HEAPU8.set(bytes, p);
    const h = core._tsf_io_open_buffer(p, BigInt(bytes.length), -1n, msg, 512);
    core._free(p);
    const header = JSON.parse(core.UTF8ToString(core._tsf_io_header_json(h)));
    const t0 = Date.now();
    const f = TF.tsfeatures(core, h, header, null, maxSeries || undefined);
    const ms = Date.now() - t0;
    core._tsf_io_close(h);
    out[path.basename(file)] = { columns: f.columns, table: Array.from(f.table, (v) => (Number.isFinite(v) ? v : null)), ms };
  }
  fs.writeFileSync(process.argv[3], JSON.stringify(out));
})();
"""


def native_table(lib, series, periods, startx):
    dp = ctypes.POINTER(ctypes.c_double)
    lib.tsf_tsfeat_new.restype = ctypes.c_void_p
    lib.tsf_tsfeat_new.argtypes = [dp, ctypes.c_int32]
    lib.tsf_tsfeat_add.restype = ctypes.c_int32
    lib.tsf_tsfeat_add.argtypes = [ctypes.c_void_p, dp, ctypes.c_int64, ctypes.c_int32]
    lib.tsf_tsfeat_columns.restype = ctypes.c_int32
    lib.tsf_tsfeat_columns.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32]
    lib.tsf_tsfeat_table.restype = ctypes.c_int32
    lib.tsf_tsfeat_table.argtypes = [ctypes.c_void_p, dp]
    lib.tsf_tsfeat_free.restype = None
    lib.tsf_tsfeat_free.argtypes = [ctypes.c_void_p]
    per = np.asarray(periods, dtype=np.float64)
    h = lib.tsf_tsfeat_new(per.ctypes.data_as(dp), per.size)
    try:
        for y, sx in zip(series, startx):
            yc = np.ascontiguousarray(y, dtype=np.float64)
            lib.tsf_tsfeat_add(h, yc.ctypes.data_as(dp), yc.size, int(sx))
        buf = ctypes.create_string_buffer(8192)
        k = lib.tsf_tsfeat_columns(h, buf, 8192)
        cols = buf.value.decode().split("\n") if k > 0 else []
        out = np.full((len(series), max(k, 1)), np.nan)
        lib.tsf_tsfeat_table(h, out.ctypes.data_as(dp))
        return cols, out[:, :max(k, 0)]
    finally:
        lib.tsf_tsfeat_free(h)


def arch_is_noise(y) -> bool:
    """True when ARCH.LM of this series is rounding noise: the regression of the squared series on its 12 lags fits
    exactly (an intermittent series whose last values are all equal), so R^2 = MSS / (MSS + RSS) is 0 / 0 up to
    rounding and no two implementations agree on it."""
    from tsfstudio import tsfeat
    x = np.asarray(y, float)
    x = x[np.isfinite(x)]
    if x.size <= 13:
        return False
    z = tsfeat.scalets(x)
    x2 = (z - z.mean()) ** 2
    n, lags = x2.size, 12
    yy = x2[lags:]
    X = np.column_stack([np.ones(n - lags)] + [x2[lags - j: n - j] for j in range(1, lags + 1)])
    res = yy - X @ np.linalg.lstsq(X, yy, rcond=None)[0]
    fitted = yy - res
    total = float(np.sum((fitted - fitted.mean()) ** 2) + np.sum(res ** 2))
    return total <= 1e-20 * max(1.0, float(np.sum(yy ** 2)))


def shift_ties(cols, tab_a, tab_b, tol):
    """Cells of the time_*_shift columns where the two sides name different positions for the same largest shift: the
    shift is reached at several places (an intermittent series repeats itself) and rounding decides which comes first."""
    a, b = np.array(tab_a, float), np.array(tab_b, float)
    ties = np.zeros(a.shape, dtype=bool)
    if a.shape != b.shape:
        return ties
    for kind in ("kl", "level", "var"):
        if f"time_{kind}_shift" in cols and f"max_{kind}_shift" in cols:
            jt, jm = cols.index(f"time_{kind}_shift"), cols.index(f"max_{kind}_shift")
            with np.errstate(invalid="ignore"):
                same_max = np.abs(a[:, jm] - b[:, jm]) <= tol * np.maximum(1.0, np.abs(a[:, jm]))
                ties[:, jt] = same_max & (a[:, jt] != b[:, jt]) & ~np.isnan(a[:, jt]) & ~np.isnan(b[:, jt])
    return ties


def compare(name, cols_a, tab_a, cols_b, tab_b, noise=None):
    """Problems and the largest relative difference between table a (reference) and b; noise: cells left out."""
    problems = []
    if cols_a != cols_b:
        return [f"{name}: columns differ: only in Python {sorted(set(cols_a) - set(cols_b))}, only in {name} "
                f"{sorted(set(cols_b) - set(cols_a))}"], float("inf"), None
    a, b = np.array(tab_a, float), np.array(tab_b, float)
    a[~np.isfinite(a)] = np.nan
    b[~np.isfinite(b)] = np.nan
    if a.shape != b.shape:
        return [f"{name}: table shapes differ {a.shape} vs {b.shape}"], float("inf"), None
    nan_diff = np.isnan(a) != np.isnan(b)
    if nan_diff.any():
        i, j = np.argwhere(nan_diff)[0]
        problems.append(f"{name}: {int(nan_diff.sum())} cells are missing on one side only (first: series {i}, {cols_a[j]})")
    ok = ~np.isnan(a) & ~np.isnan(b)
    if noise is not None:
        ok &= ~noise
    d = np.zeros_like(a)
    d[ok] = np.abs(a[ok] - b[ok]) / np.maximum(1.0, np.abs(a[ok]))
    worst = float(d.max()) if d.size else 0.0
    where = None
    if d.size and worst > 0:
        i, j = np.unravel_index(int(d.argmax()), d.shape)
        where = f"{cols_a[j]} of series {i}: {a[i, j]!r} vs {b[i, j]!r}"
    return problems, worst, where


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib")
    ap.add_argument("--node")
    ap.add_argument("--data-dir", default="data/tsf")
    ap.add_argument("--datasets", nargs="*", default=["nn5_weekly", "nn5_daily", "m1_yearly", "m1_quarterly", "m1_monthly",
                                                      "tourism_quarterly", "hospital", "car_parts", "covid_deaths"])
    ap.add_argument("--max-series", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()
    if a.lib:
        os.environ["TSFSTUDIO_CPP_LIB"] = str(Path(a.lib).resolve())
    from tsfstudio import catalog as C
    from tsfstudio import features as F
    from tsfstudio import tsfeat
    from tsfstudio.engines import cpp_engine

    lib = cpp_engine.load_lib()
    if lib is None or not hasattr(lib, "tsf_tsfeat_new"):
        print("The native library has no tsf_tsfeat_new (build cpp/ and pass --lib).")
        return 1
    cat = C.load_catalog()
    data_dir = Path(a.data_dir)
    entries = [cat.get(d) for d in a.datasets]
    web = {}
    if a.node:
        files = [str((data_dir / e.file).resolve()) for e in entries]
        with tempfile.TemporaryDirectory() as td:
            script, out = Path(td) / "run.cjs", Path(td) / "out.json"
            script.write_text(NODE_SCRIPT, encoding="utf-8")
            r = subprocess.run([a.node, str(script), str(ROOT / "web"), str(out), str(a.max_series), *files],
                               capture_output=True, text=True)
            if r.returncode != 0 or not out.is_file():
                print("Node failed:\n" + r.stderr[-2000:])
                return 1
            web = json.loads(out.read_text(encoding="utf-8"))
    n_fail, worst_all = 0, 0.0
    for e in entries:
        ds = C.load_entry(e, data_dir)
        series = list(ds.values)
        starts = list(ds.attrs["start_timestamp"]) if "start_timestamp" in ds.attrs.columns else [None] * len(series)
        if a.max_series:
            series, starts = series[:a.max_series], starts[:a.max_series]
        periods = tsfeat.study_periods(ds.frequency)
        t0 = time.perf_counter()
        ref = F.tsfeatures_table(series, periods, starts)
        t_py = time.perf_counter() - t0
        cols = list(ref.columns)
        startx = [tsfeat.study_startx(s, periods) for s in starts]
        t0 = time.perf_counter()
        ncols, ntab = native_table(lib, series, periods, startx)
        t_cpp = time.perf_counter() - t0
        noise = np.zeros(ref.shape, dtype=bool)
        if "ARCH.LM" in cols:
            noise[:, cols.index("ARCH.LM")] = [arch_is_noise(y) for y in series]
        ties = shift_ties(cols, ref.to_numpy(float), ntab, a.tol) if cols == ncols else np.zeros(ref.shape, dtype=bool)
        problems, worst, where = compare("C++", cols, ref.to_numpy(float), ncols, ntab, noise | ties)
        line = f"{e.id:20s} {len(series):5d} series {len(cols):3d} features  C++ {worst:.1e} (Python {t_py:.1f} s, C++ {t_cpp:.1f} s)"
        if noise.any():
            line += f"  [{int(noise.sum())} ARCH.LM values are rounding noise, not compared]"
        if a.node:
            w = web[e.file]
            wt = np.array([np.nan if v is None else v for v in w["table"]], float).reshape(len(series), -1) if w["columns"] else np.zeros((len(series), 0))
            wcols = [c[4:] if c == "tsf_mean" else c for c in w["columns"]]
            ties2 = shift_ties(cols, ref.to_numpy(float), wt, a.tol) if cols == wcols else np.zeros(ref.shape, dtype=bool)
            ties |= ties2
            p2, worst2, where2 = compare("page", cols, ref.to_numpy(float), wcols, wt, noise | ties2)
            problems += p2
            line += f"  page {worst2:.1e} ({w['ms'] / 1000:.1f} s)"
            if worst2 > worst:
                worst, where = worst2, where2
        if ties.any():
            line += f"  [{int(ties.sum())} shift positions are ties, not compared]"
        if worst > a.tol:
            problems.append(f"values differ by {worst:.2e}: {where}")
        worst_all = max(worst_all, worst if np.isfinite(worst) else 0.0)
        n_fail += bool(problems)
        print(("FAIL  " if problems else "PASS  ") + line, flush=True)
        for p in problems:
            print("      " + p)
    print(f"\n{'ALL PASS' if not n_fail else str(n_fail) + ' dataset(s) failed'}; largest difference {worst_all:.2e}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
