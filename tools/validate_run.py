"""Checks a complete C++ run (cpp/src/run.cpp, tsf_run) against the Python runner with the C++ engine.

Usage:
    python tools/validate_run.py --lib <tsfcore.dll with run.cpp> [--data-dir data/tsf] [--quick]

For every case both sides read the same .tsf file, split it, forecast with the same tsfcore method, compute the measures
and write their run folder. The script then compares
  * forecasts.csv, test_actuals.csv, errors_per_series.csv, summary.csv and fallbacks.csv byte for byte,
  * the manifest fields that describe the data and the split (horizon, seasonality, SHA-256, series counts, excluded
    series, missing-value counts, fallback count) and the warnings,
and, for cases that must be refused, that both sides refuse with the same message.
Exit code 0 only if every case passes.

Both sides load the library given with --lib (the Python engine through TSFSTUDIO_CPP_LIB).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import tempfile
from pathlib import Path

FILES = ("forecasts.csv", "test_actuals.csv", "errors_per_series.csv", "summary.csv", "fallbacks.csv")


def cpp_run(lib, path: Path, out_dir: Path, config: dict) -> Path:
    text = "\n".join(f"{k}={v}" for k, v in config.items())
    run_dir = ctypes.create_string_buffer(4096)
    msg = ctypes.create_string_buffer(2048)
    rc = lib.tsf_run(str(path).encode("utf-8"), str(out_dir).encode("utf-8"), text.encode("utf-8"), run_dir, 4096, msg, 2048)
    if rc != 0:
        raise RuntimeError(msg.value.decode("utf-8", "replace"))
    return Path(run_dir.value.decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--data-dir", default="data/tsf")
    ap.add_argument("--quick", action="store_true", help="only the first 60 series of each catalogue dataset")
    a = ap.parse_args()

    lib_path = str(Path(a.lib).resolve())
    os.environ["TSFSTUDIO_CPP_LIB"] = lib_path                      # before the engine is imported
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import numpy as np
    from tsfstudio import catalog as C
    from tsfstudio.engines.cpp_engine import CppEngine, method_with_params
    from tsfstudio.metrics import ALL_METRICS, DEFAULT_METRICS
    from tsfstudio.runner import RunConfig, run
    from tsfstudio.tsf.reader import read_tsf

    lib = ctypes.CDLL(lib_path)
    lib.tsf_run.restype = ctypes.c_int32
    lib.tsf_run.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32,
                            ctypes.c_char_p, ctypes.c_int32]
    data_dir = Path(a.data_dir)
    cat = C.load_catalog()
    n_fail = 0

    def one(label: str, path: Path, entry, method: str, params: dict | None = None, metrics=DEFAULT_METRICS,
            missing_policy: str = "error", max_series: int | None = None, integer: bool | None = None,
            horizon: int | None = None, expect_error: bool = False, origins: int = 1, origin_step: int | None = None,
            intervals: tuple = ()) -> None:
        nonlocal n_fail
        problems: list[str] = []
        with tempfile.TemporaryDirectory() as td:
            py_dir, cpp_dir = Path(td) / "py", Path(td) / "cpp"
            cfg = RunConfig(engine="cpp", task="local", method=method, params=params or {}, metrics=tuple(metrics),
                            missing_policy=missing_policy, max_series=max_series, integer_conversion=integer, horizon=horizon,
                            output_dir=py_dir, write_xlsx=False, force=True, origins=origins, origin_step=origin_step,
                            intervals=tuple(intervals))
            ds = read_tsf(path)
            eff_h = horizon or (entry.effective_horizon(ds.header) if entry else None)
            eff_int = integer if integer is not None else bool(entry.integer_conversion if entry else False)
            config = {"method": method_with_params(method, params), "dataset_id": entry.id if entry else path.stem,
                      "integer_conversion": int(eff_int), "missing_policy": missing_policy, "metrics": ",".join(metrics)}
            if eff_h:
                config["horizon"] = int(eff_h)
            if max_series:
                config["max_series"] = max_series
            if origins != 1:
                config["origins"] = origins
            if origin_step:
                config["origin_step"] = origin_step
            if intervals:
                config["intervals"] = ",".join(str(v) for v in intervals)
            py_err = cpp_err = None
            res = cdir = None
            try:
                res = run(ds, cfg, entry=entry, engine=CppEngine())
            except Exception as exc:
                py_err = str(exc)
            try:
                cdir = cpp_run(lib, path, cpp_dir, config)
            except Exception as exc:
                cpp_err = str(exc)
            if py_err or cpp_err:
                if not expect_error:
                    problems.append(f"Python: {py_err or 'ran'} | C++: {cpp_err or 'ran'}")
                elif py_err != cpp_err:
                    problems.append(f"messages differ\n        Python: {py_err}\n        C++   : {cpp_err}")
            elif expect_error:
                problems.append("both ran, but an error was expected")
            else:
                py_files = sorted(f.name for f in res.run_dir.glob("*.csv") if f.name != "wrmsse.csv")
                cpp_files = sorted(f.name for f in cdir.glob("*.csv"))
                if py_files != cpp_files or not set(FILES) <= set(py_files):
                    problems.append(f"different files: Python {py_files}, C++ {cpp_files}")
                for name in sorted(set(py_files) & set(cpp_files)):
                    if (res.run_dir / name).read_bytes() != (cdir / name).read_bytes():
                        problems.append(f"{name} differs")
                mp = res.manifest
                mc = json.loads((cdir / "manifest.json").read_text(encoding="utf-8"))
                for sect, key in (("config", "effective_horizon"), ("config", "effective_seasonality"),
                                  ("config", "effective_integer_conversion"), ("dataset", "id"), ("dataset", "sha256"),
                                  ("dataset", "frequency"), ("dataset", "n_series_total"), ("dataset", "n_series_run"),
                                  ("dataset", "header_horizon"), ("split", "horizon"), ("split", "n_excluded"),
                                  ("split", "excluded"), ("split", "origins"), ("split", "origin_step"),
                                  ("split", "interval_levels"), ("split", "interval_method"), ("split", "origin_cuts"),
                                  ("missing", "policy"), ("missing", "series_with_missing"),
                                  ("missing", "imputed_values"), ("missing", "test_missing_values"), ("fallbacks", "count"),
                                  ("fallbacks", "fraction")):
                    if mp[sect][key] != mc[sect][key]:
                        problems.append(f"manifest {sect}.{key}: Python {mp[sect][key]!r}, C++ {mc[sect][key]!r}")
                # the Python runner may add warnings of its memory / time estimate; the others must be the same
                missing_w = [w for w in mc["warnings"] if w not in mp["warnings"]]
                extra_w = [w for w in mp["warnings"] if w not in mc["warnings"]]
                if missing_w:
                    problems.append(f"C++ warnings not in the Python run: {missing_w}")
                if extra_w:
                    info.append(f"{label}: Python-only warning(s): {[w[:70] for w in extra_w]}")
                rows = len(res.names)
                detail = (f"{rows} series, {int(mc['fallbacks']['count'])} fallbacks, {mc['split']['n_excluded']} excluded, "
                          f"{len(py_files)} files")
        n_fail += bool(problems)
        tail = "both refuse, same message" if expect_error and not problems else (detail if not problems else "")
        print(f"{'FAIL' if problems else 'PASS'}  {label:58s} {tail}", flush=True)
        for p in problems:
            print(f"      {p}")

    info: list[str] = []
    top = 60 if a.quick else None
    cases = [("nn5_daily", "ets", None, DEFAULT_METRICS, top), ("nn5_daily", "tbats", None, DEFAULT_METRICS, 12),
             ("nn5_weekly", "theta", None, ALL_METRICS, top), ("nn5_weekly", "arima", None, DEFAULT_METRICS, 20),
             ("m3_monthly", "ets", None, DEFAULT_METRICS, 200 if not a.quick else 60),
             ("m1_yearly", "comb_shd", None, ALL_METRICS, top), ("tourism_quarterly", "naive2", None, DEFAULT_METRICS, top),
             ("m4_hourly", "snaive", None, ALL_METRICS, top), ("hospital", "ses", None, ALL_METRICS, top),
             ("car_parts", "croston_classic", None, ALL_METRICS, top),
             ("car_parts", "tsb", {"alpha_d": 0.2, "alpha_p": 0.3}, DEFAULT_METRICS, 300 if not a.quick else 60),
             ("covid_deaths", "drift", None, DEFAULT_METRICS, top), ("fred_md", "holt_winters", None, DEFAULT_METRICS, top),
             ("traffic_weekly", "auto_theta", None, DEFAULT_METRICS, 100 if not a.quick else 30),
             ("kdd_cup", "naive", None, ALL_METRICS, top), ("m5_item_state", "adida", None, ALL_METRICS, 500 if not a.quick else 60)]
    for ds_id, method, params, metrics, max_series in cases:
        entry = cat.get(ds_id)
        if C.availability(entry, data_dir) != "ok":
            print(f"SKIP  {ds_id}: file not in {data_dir}")
            continue
        label = f"{ds_id} / {method}" + (f" {params}" if params else "") + (f" (first {max_series})" if max_series else "")
        one(label, data_dir / entry.file, entry, method, params, metrics, max_series=max_series)

    # rolling origins and prediction intervals
    extra = [("nn5_daily", "ets", dict(origins=3), 40), ("nn5_daily", "theta", dict(intervals=(80, 95)), 60),
             ("nn5_weekly", "ets", dict(origins=3, origin_step=4, intervals=(90,)), 60),
             ("m3_monthly", "theta", dict(origins=2, intervals=(50, 80, 95, 99)), 150),
             ("m1_yearly", "naive2", dict(origins=2, origin_step=1), None),
             ("tourism_quarterly", "ets", dict(intervals=(80,)), 80),
             ("car_parts", "croston_sba", dict(origins=2, origin_step=3, intervals=(80, 95)), 200),
             ("hospital", "ses", dict(origins=4, origin_step=6, intervals=(95,)), 120),
             ("m4_hourly", "snaive", dict(origins=2, intervals=(80, 95)), 40)]
    for ds_id, method, kw, max_series in extra:
        entry = cat.get(ds_id)
        if C.availability(entry, data_dir) != "ok":
            continue
        if a.quick and max_series:
            max_series = min(max_series, 40)
        one(f"{ds_id} / {method} {kw}" + (f" (first {max_series})" if max_series else ""), data_dir / entry.file, entry, method,
            metrics=ALL_METRICS, max_series=max_series, **kw)

    # a file with missing values, short series, an all-missing training part, names that need quoting, no @horizon
    with tempfile.TemporaryDirectory() as td:
        rng = np.random.default_rng(5)
        p = Path(td) / "gaps.tsf"
        lines = ["# synthetic", "@relation gaps", "@attribute series_name string", "@attribute code numeric", "@frequency monthly",
                 "@missing true", "@equallength false", "@data"]
        def fmt(y):
            return ",".join("?" if v != v else repr(float(v)) for v in y)
        for i in range(40):
            y = rng.gamma(2.0, 30.0, int(rng.integers(30, 90)))
            y[rng.random(y.size) < 0.15] = np.nan
            if i % 7 == 0:
                y[:3] = np.nan                                       # leading gap
            if i % 9 == 0:
                y[-14:-12] = np.nan                                  # gap that ends the training part
            if i % 11 == 0:
                y[-5] = np.nan                                       # missing value in the test part
            y[10] = 12.5
            lines.append(f'name, with "quotes" {i}:{i}:{fmt(y)}')
        lines.append("short:100:1,2,3,4,5")
        lines.append("all_missing_train:101:" + ",".join(["?"] * 20 + ["1"] * 12))
        lines.append("constant:102:" + ",".join(["7"] * 50))
        p.write_text("\n".join(lines) + "\n", encoding="ascii")
        for policy in ("zero", "locf", "interpolate"):
            one(f"synthetic gaps / ets, policy {policy}", p, None, "ets", metrics=ALL_METRICS, missing_policy=policy, horizon=12)
        one("synthetic gaps / theta, rounded, policy locf", p, None, "theta", missing_policy="locf", horizon=12, integer=True)
        one("synthetic gaps / croston_sba, rounded (warning)", p, None, "croston_sba", missing_policy="zero", horizon=6, integer=True)
        one("synthetic gaps / ets, 3 origins + intervals, policy interpolate", p, None, "ets", metrics=ALL_METRICS,
            missing_policy="interpolate", horizon=6, origins=3, origin_step=2, intervals=(80, 95))
        one("synthetic gaps / naive, 2 origins, policy locf", p, None, "naive", missing_policy="locf", horizon=12, origins=2)
        one("synthetic gaps / interval level out of range", p, None, "naive", missing_policy="locf", horizon=6, intervals=(30,),
            expect_error=True)
        one("synthetic gaps / policy error", p, None, "ets", horizon=12, expect_error=True)
        one("synthetic gaps / no horizon", p, None, "ets", missing_policy="locf", expect_error=True)
        one("synthetic gaps / horizon too long for every series", p, None, "naive", missing_policy="locf", horizon=500,
            expect_error=True)

    for line in info:
        print("note:", line)
    print(f"\n{'ALL CASES PASSED' if not n_fail else str(n_fail) + ' case(s) failed'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
