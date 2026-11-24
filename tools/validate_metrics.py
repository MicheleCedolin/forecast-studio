"""Checks the C++ error measures, summary and WRMSSE (cpp/src/metrics.cpp) against the Python ones.

Usage:
    python tools/validate_metrics.py --lib <dll with metrics.cpp> [--data-dir data/tsf] [--datasets nn5_daily ...]
                                     [--random 400] [--no-m5]

Cases:
  * random tables (several horizons, ragged training parts, NaN padding, zeros, negative values, constant series),
  * real datasets from the catalogue with a seasonal-naive forecast on the last `horizon` points,
  * metrics.summarize on every column,
  * WRMSSE on a small random hierarchy (unit and custom weights) and, unless --no-m5, on the M5 item x store file.
Every value is compared bit for bit (NaN = NaN; the one exception is the sign of a zero median). For each measure the script prints how many values are identical and
the largest relative difference of the others. Exit code 0 only if everything is identical.

Test build of the library:
    g++ -std=c++17 -O3 -shared -static -o tsfmetrics.dll cpp/src/metrics.cpp
"""
from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsfstudio import catalog as C  # noqa: E402
from tsfstudio import wrmsse as W  # noqa: E402
from tsfstudio.metrics import ALL_METRICS, compute_error_table, summarize  # noqa: E402
from tsfstudio.seasonality import primary_period, seasonality_for  # noqa: E402

c = ctypes
PD = c.POINTER(c.c_double)
PI64 = c.POINTER(c.c_int64)


def load_lib(path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(Path(path).resolve()))
    lib.tsf_metrics_names.restype = c.c_char_p
    lib.tsf_primary_period.restype = c.c_int32
    lib.tsf_primary_period.argtypes = [PD, c.c_int32]
    lib.tsf_metrics_table.restype = c.c_int32
    lib.tsf_metrics_table.argtypes = [PD, PD, c.c_int64, c.c_int64, PD, PI64, c.c_int32, c.c_char_p, PD, c.c_char_p, c.c_int32]
    lib.tsf_metrics_summary.restype = None
    lib.tsf_metrics_summary.argtypes = [PD, c.c_int64, PD]
    lib.tsf_wrmsse.restype = c.c_int32
    lib.tsf_wrmsse.argtypes = [PD, PI64, c.c_int64, PD, PD, c.c_int64, c.POINTER(c.c_char_p), PD, PD, PI64, PD, c.c_char_p, c.c_int32]
    return lib


def dptr(a: np.ndarray):
    return a.ctypes.data_as(PD)


def flat_train(train):
    off = np.zeros(len(train) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(t) for t in train])
    values = np.ascontiguousarray(np.concatenate(train) if len(train) else np.empty(0), dtype=np.float64)
    if values.size == 0:
        values = np.zeros(1)
    return values, off


def cpp_table(lib, f, a, train, lag, metrics) -> np.ndarray:
    f = np.ascontiguousarray(f, dtype=np.float64)
    a = np.ascontiguousarray(a, dtype=np.float64)
    values, off = flat_train(train)
    out = np.empty((f.shape[0], len(metrics)))
    msg = c.create_string_buffer(256)
    rc = lib.tsf_metrics_table(dptr(f), dptr(a), f.shape[0], f.shape[1], dptr(values), off.ctypes.data_as(PI64), lag,
                               ",".join(metrics).encode(), dptr(out), msg, 256)
    if rc != 0:
        raise RuntimeError(msg.value.decode())
    return out


def same_bits(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    return (x.view(np.uint64) == y.view(np.uint64)) | (np.isnan(x) & np.isnan(y))


class Tally:
    def __init__(self):
        self.n: dict[str, int] = {}
        self.bad: dict[str, int] = {}
        self.worst: dict[str, float] = {}

    def add(self, name: str, py, cpp, zero_sign: bool = True) -> None:
        py = np.atleast_1d(np.asarray(py, dtype=np.float64))
        cpp = np.atleast_1d(np.asarray(cpp, dtype=np.float64))
        ok = same_bits(py, cpp)
        if not zero_sign:
            ok |= (py == 0) & (cpp == 0)
        self.n[name] = self.n.get(name, 0) + ok.size
        self.bad[name] = self.bad.get(name, 0) + int((~ok).sum())
        if (~ok).any():
            with np.errstate(all="ignore"):
                rel = np.abs(py[~ok] - cpp[~ok]) / np.maximum(np.abs(py[~ok]), 1e-300)
            rel = np.where(np.isnan(rel), np.inf, rel)          # NaN on one side only, or inf vs finite
            self.worst[name] = max(self.worst.get(name, 0.0), float(rel.max()))

    def report(self) -> int:
        total_bad = 0
        for name in self.n:
            bad = self.bad[name]
            total_bad += bad
            note = "" if not bad else f"   largest relative difference {self.worst[name]:.3g}"
            print(f"  {'PASS' if not bad else 'DIFF'}  {name:22s} {self.n[name] - bad:>9d} / {self.n[name]:<9d} identical{note}")
        return total_bad


def check_table(lib, tally: Tally, f, a, train, seasonality) -> None:
    metrics = list(ALL_METRICS)
    py = compute_error_table(f, a, train, seasonality, metrics=metrics)
    lag = primary_period(seasonality)
    s = np.ascontiguousarray(np.atleast_1d(np.asarray(seasonality, dtype=np.float64)))
    if lib.tsf_primary_period(dptr(s), s.size) != lag:
        raise AssertionError(f"primary_period differs for {seasonality}")
    cpp = cpp_table(lib, f, a, train, lag, metrics)
    out4 = np.empty(4)
    for j, m in enumerate(metrics):
        tally.add(m, py[m].to_numpy(), cpp[:, j])
        col = np.ascontiguousarray(cpp[:, j])
        lib.tsf_metrics_summary(dptr(col), col.size, dptr(out4))
        row = summarize(pd.DataFrame({m: col})).iloc[0]
        tally.add("summary: mean", row["Mean"], out4[0])
        # A column that holds both 0.0 and -0.0 has no defined order between them, so the median may be either zero:
        # NumPy's partition and std::sort pick differently. Only here the sign of a zero is not compared.
        tally.add("summary: median", row["Median"], out4[1], zero_sign=False)
        tally.add("summary: counts", [row["N_valid"], row["N_excluded"]], out4[2:])


def random_case(rng: np.random.Generator):
    n = int(rng.integers(1, 40))
    h = int(rng.choice([1, 2, 5, 7, 8, 9, 12, 28, 56, 127, 128, 129, 200, 400]))
    kind = rng.choice(["positive", "counts", "mixed", "tiny"])
    def draw(shape):
        if kind == "positive":
            return rng.lognormal(3, 1.5, shape)
        if kind == "counts":
            return rng.poisson(0.6, shape).astype(float)
        if kind == "mixed":
            return rng.normal(0, 50, shape)
        return rng.normal(0, 1e-9, shape)
    a, f = draw((n, h)), draw((n, h))
    train = []
    for i in range(n):
        t = draw(int(rng.choice([0, 1, 2, 3, 10, 60, 129, 500, 3000])))
        if t.size and rng.random() < 0.2:
            t[rng.random(t.size) < 0.1] = np.nan
        if t.size and rng.random() < 0.1:
            t[:] = t[0]                                         # constant training part: zero scale
        train.append(t)
        r = rng.random()
        if r < 0.15:
            k = int(rng.integers(0, h + 1))                     # shorter test part: NaN padding on the right
            a[i, k:] = np.nan
            f[i, k:] = np.nan
        elif r < 0.25:
            a[i] = 0.0
        elif r < 0.30:
            f[i] = a[i]
        elif r < 0.35:
            f[i, rng.random(h) < 0.3] = np.nan
        elif r < 0.40:
            a[i] = a[i, 0]                                      # constant actuals: zero range
    seasonality = [(7.0,), (365.25 / 7,), (12.0,), (1.0,), (24.0, 168.0, 8766.0), 4.0][int(rng.integers(0, 6))]
    return f, a, train, seasonality


def real_case(ds_id: str, data_dir: Path):
    entry = C.load_catalog().get(ds_id)
    ds = C.load_entry(entry, data_dir)
    h = int(entry.horizon_override or ds.horizon or 12)
    seasonality = seasonality_for(ds.frequency)
    lag = primary_period(seasonality)
    train, rows_a, rows_f = [], [], []
    for y in ds.values:
        if len(y) <= h:
            continue
        tr, te = y[:-h], y[-h:]
        base = tr[-lag:] if len(tr) >= lag else tr[-1:]
        train.append(tr)
        rows_a.append(te)
        rows_f.append(np.resize(base, h))                        # seasonal naive
    return np.array(rows_f), np.array(rows_a), train, seasonality


def cpp_wrmsse(lib, train, test, fc, attrs: pd.DataFrame, units):
    values, off = flat_train(train)
    test = np.ascontiguousarray(test, dtype=np.float64)
    fc = np.ascontiguousarray(fc, dtype=np.float64)
    n = len(train)
    strings = [str(v).encode("utf-8") for col in W.HIERARCHY for v in attrs[col].tolist()]
    arr = (c.c_char_p * len(strings))(*strings)
    level, groups, total = np.empty(12), np.empty(12, dtype=np.int64), c.c_double(0)
    msg = c.create_string_buffer(256)
    u = None if units is None else dptr(np.ascontiguousarray(units, dtype=np.float64))
    rc = lib.tsf_wrmsse(dptr(values), off.ctypes.data_as(PI64), n, dptr(test), dptr(fc), test.shape[1], arr, u,
                        dptr(level), groups.ctypes.data_as(PI64), c.byref(total), msg, 256)
    if rc != 0:
        raise RuntimeError(msg.value.decode())
    return total.value, level, groups


def check_wrmsse(lib, tally: Tally, label: str, train, test, fc, attrs) -> None:
    py = W.wrmsse(train, test, fc, attrs)
    total, level, groups = cpp_wrmsse(lib, train, test, fc, attrs, None)
    tally.add(f"WRMSSE {label}: total", py["WRMSSE"], total)
    tally.add(f"WRMSSE {label}: levels", [r["WRMSSE"] for r in py["levels"]], level)
    tally.add(f"WRMSSE {label}: groups", [r["n_series"] for r in py["levels"]], groups)


def random_hierarchy(rng: np.random.Generator):
    rows = []
    for s in range(3):
        for st in range(int(rng.integers(1, 4))):
            for cat in ("FOODS", "HOBBIES"):
                for dept in (1, 2):
                    for item in range(int(rng.integers(1, 5))):
                        rows.append({"item_id": f"{cat}_{dept}_{item:03d}", "dept_id": f"{cat}_{dept}", "cat_id": cat,
                                     "store_id": f"S{s}_{st}", "state_id": f"S{s}"})
    attrs = pd.DataFrame(rows)
    n, h = len(attrs), 28
    train = []
    for _ in range(n):
        t = rng.poisson(1.2, int(rng.integers(40, 400))).astype(float)
        t[: int(rng.integers(0, 30))] = 0.0                      # leading zeros: the scale starts at the first sale
        train.append(t)
    return train, rng.poisson(1.2, (n, h)).astype(float), rng.gamma(2.0, 0.6, (n, h)), attrs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--data-dir", default="data/tsf")
    ap.add_argument("--datasets", nargs="*", default=["nn5_daily", "nn5_weekly", "m3_monthly", "m4_hourly", "tourism_quarterly",
                                                      "car_parts", "hospital", "covid_deaths", "m1_yearly", "traffic_weekly",
                                                      "kdd_cup", "fred_md"])
    ap.add_argument("--random", type=int, default=400, help="number of random tables")
    ap.add_argument("--no-m5", action="store_true", help="skip WRMSSE on the M5 item x store file")
    a = ap.parse_args()

    lib = load_lib(a.lib)
    names = lib.tsf_metrics_names().decode().split(",")
    if names != list(ALL_METRICS):
        print(f"FAIL  the measure lists differ:\n  C++   : {names}\n  Python: {list(ALL_METRICS)}")
        return 1
    total_bad = 0

    rng = np.random.default_rng(20261123)
    tally = Tally()
    for _ in range(a.random):
        check_table(lib, tally, *random_case(rng))
    print(f"Random tables ({a.random}):")
    total_bad += tally.report()

    data_dir = Path(a.data_dir)
    tally = Tally()
    done = []
    for ds_id in a.datasets:
        try:
            if C.availability(C.load_catalog().get(ds_id), data_dir) != "ok":
                continue
            check_table(lib, tally, *real_case(ds_id, data_dir))
            done.append(ds_id)
        except Exception as exc:
            print(f"  ERROR {ds_id}: {type(exc).__name__}: {exc}")
            total_bad += 1
    print(f"\nReal datasets, seasonal-naive forecast ({', '.join(done)}):")
    total_bad += tally.report()

    tally = Tally()
    for _ in range(20):
        train, test, fc, attrs = random_hierarchy(rng)
        check_wrmsse(lib, tally, "random", train, test, fc, attrs)
        # custom weights: Python multiplies the unit sales by a price matrix, C++ receives the product
        P = rng.gamma(2.0, 2.0, (len(train), 28))
        Y = W._right_align(train)
        units = Y[:, Y.shape[1] - 28:] * P
        orig = W._right_align
        py = None
        try:
            W._right_align = lambda t, _Y=Y: _Y                 # same Y; weights replaced below through `units`
            py = _wrmsse_with_units(train, test, fc, attrs, units)
        finally:
            W._right_align = orig
        total, level, _ = cpp_wrmsse(lib, train, test, fc, attrs, units)
        tally.add("WRMSSE weights: total", py["WRMSSE"], total)
        tally.add("WRMSSE weights: levels", [r["WRMSSE"] for r in py["levels"]], level)
    if not a.no_m5:
        entry = C.load_catalog().get("m5_item_store")
        if C.availability(entry, data_dir) == "ok":
            ds = C.load_entry(entry, data_dir)
            h = 28
            train = [y[:-h] for y in ds.values]
            test = np.array([y[-h:] for y in ds.values])
            fc = np.array([np.resize(t[-7:], h) for t in train])
            check_wrmsse(lib, tally, "M5", train, test, fc, ds.attrs)
    print("\nWRMSSE:")
    total_bad += tally.report()

    print(f"\n{'ALL IDENTICAL' if not total_bad else str(total_bad) + ' value(s) differ'}")
    return 1 if total_bad else 0


def _wrmsse_with_units(train, test, fc, attrs, units) -> dict:
    """wrmsse.wrmsse with a given weight matrix (the function body, with `units` supplied instead of unit sales)."""
    Y = W._right_align(train)
    a = attrs[list(W.HIERARCHY)].astype(str).reset_index(drop=True)
    per_level = []
    for name, keys in W.LEVELS:
        groups = a.groupby(list(keys), sort=True).indices if keys else {"all": np.arange(len(a))}
        tot_w = acc = 0.0
        for _, idx in groups.items():
            y_tr = Y[idx].sum(axis=0)
            nz = np.flatnonzero(y_tr != 0)
            if nz.size == 0:
                continue
            y_tr = y_tr[nz[0]:]
            if y_tr.size < 2:
                continue
            den = np.mean(np.diff(y_tr) ** 2)
            num = np.nanmean((test[idx].sum(axis=0) - fc[idx].sum(axis=0)) ** 2)
            w = float(units[idx].sum())
            if den > 0 and np.isfinite(num):
                acc += w * np.sqrt(num / den)
                tot_w += w
        per_level.append({"level": name, "n_series": len(groups), "WRMSSE": acc / tot_w if tot_w > 0 else np.nan})
    return {"WRMSSE": float(np.nanmean([r["WRMSSE"] for r in per_level])), "levels": per_level}


if __name__ == "__main__":
    sys.exit(main())
