"""Command-line helpers: `python -m tsfstudio <command>`.

  data-check   compares the presence and byte sizes of the bundled TSF files with the catalogue
  index        writes a summary of the existing files (number of series, length, share of missing/zero) to _index.json
  hash         produces SHA256SUMS.txt (data fingerprint; used in experiment manifests)
  template     writes the empty Excel template
  convert      converts Excel -> .tsf and prints the validation report
  engines      availability of the registered engines (R, baseline, ...)
  estimate     memory/time estimate for a run (without running)
  run          runs a model on a dataset: forecasts + error measures + Excel outputs + manifest
  doctor       environment check: Python, R, packages, memory, data; a quick live test of every method of every engine
  golden       compares an engine's forecasts with your own R outputs (NN5): forecast difference + effect on SMAPE/MASE
  rank         ranks models from several runs; Friedman, Nemenyi CD and bootstrap ranking stability
  app          starts the local web interface (opens in the browser; no extra installation needed)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from . import catalog as C


def _cmd_data_check(a) -> int:
    cat = C.load_catalog()
    rep = C.check_bundle(cat, a.data_dir)
    total = len(cat.entries)
    print(f"Data folder: {C._dir(a.data_dir).resolve()}")
    print(f"OK: {len(rep['ok'])}/{total} | missing: {len(rep['missing'])} | size mismatch: {len(rep['size_mismatch'])}")
    for k, label in (("missing", "MISSING"), ("size_mismatch", "SIZE MISMATCH")):
        for i in rep[k]:
            e = cat.get(i)
            print(f"  {label:14s} {e.id:30s} {e.file}  (expected {e.bytes:,} bytes)")
    if cat.not_bundled:
        print("Datasets mentioned in your code but not in the package:", ", ".join(m["id"] for m in cat.not_bundled))
    return 0 if (not a.strict or (not rep["missing"] and not rep["size_mismatch"])) else 1


def _cmd_index(a) -> int:
    def prog(i, k, n):
        print(f"[{k}/{n}] {i}", flush=True)
    idx = C.build_index(a.data_dir, a.ids or None, prog)
    print(f"{len(idx)} datasets summarised -> {C._dir(a.data_dir) / C.INDEX_FILENAME}")
    return 0


def _cmd_hash(a) -> int:
    d = C._dir(a.data_dir)
    lines = []
    for e in C.load_catalog().entries:
        p = d / e.file
        if not p.is_file():
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {e.file}")
        print(lines[-1], flush=True)
    (d / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def _cmd_template(a) -> int:
    from .excel import build_template
    print(build_template(a.out))
    return 0


def _cmd_convert(a) -> int:
    from .excel import excel_to_tsf
    ds, rep = excel_to_tsf(a.xlsx, a.out, fill_gaps=a.fill_gaps, lang=a.lang)
    if rep.text():
        print(rep.text())
    if ds is None:
        return 2
    print(f"Done: {ds.n_series} series, length {int(ds.lengths.min())}–{int(ds.lengths.max())}, frequency {ds.frequency}, horizon {ds.horizon} -> {a.out}")
    return 0


# ---------------------------------------------------------------------------------------------
# Run commands
# ---------------------------------------------------------------------------------------------
def _parse_params(items: list[str] | None) -> dict:
    out: dict = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--param must have the form 'key=value': {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except ValueError:
            out[k.strip()] = v
    return out


def _load_for_run(a):
    from .runner import RunConfig
    from .tsf.reader import read_tsf

    entry = None
    if a.dataset:
        entry = C.load_catalog().get(a.dataset)
        if C.availability(entry, a.data_dir) != "ok":
            raise SystemExit(f"The data file of '{a.dataset}' was not found or does not match: run `data-check` first.")
        ds = C.load_entry(entry, a.data_dir, max_series=a.max_series)
    else:
        ds = read_tsf(a.file, max_series=a.max_series)
    cfg = RunConfig(engine=a.engine, task=a.task, method=a.method, horizon=a.horizon, lag=a.lag,
                    missing_policy=a.missing_policy, params=_parse_params(a.param), seed=a.seed, workers=a.workers,
                    max_series=a.max_series, output_dir=a.out, force=a.force,
                    integer_conversion={"auto": None, "on": True, "off": False}[a.integer_conversion])
    return ds, cfg, entry


def _print_preflight(rep, n_series: int) -> None:
    print(f"Number of series: {n_series}")
    if rep.memory:
        print("Memory :", rep.memory.message)
    if rep.runtime:
        print("Time   :", rep.runtime.message)
    for w in rep.warnings:
        print("WARNING:", w)


def _cmd_estimate(a) -> int:
    from .engines.registry import get_engine
    from .estimate import preflight
    from .runner import build_job, prepare

    ds, cfg, entry = _load_for_run(a)
    prep = prepare(ds, cfg, entry)
    job = build_job(prep, cfg, entry)
    eng = get_engine(cfg.engine)
    rep = preflight(job, engine=eng, probe=a.probe)
    _print_preflight(rep, len(prep.train))
    return 3 if rep.blocked else 0


def _cmd_run(a) -> int:
    from .engines.registry import get_engine
    from .estimate import preflight
    from .runner import RunError, build_job, prepare, run

    ds, cfg, entry = _load_for_run(a)
    prep = prepare(ds, cfg, entry)
    job = build_job(prep, cfg, entry)
    rep = preflight(job, engine=get_engine(cfg.engine))
    _print_preflight(rep, len(prep.train))
    if rep.blocked and not cfg.force:
        print("Run not started (memory). See the details above; you can force it with --force (not recommended).")
        return 3
    if rep.long_running and not a.yes:
        if sys.stdin.isatty():
            if input("This may take long. Continue? [y/N] ").strip().lower() not in ("y", "yes"):
                print("Cancelled.")
                return 1
        else:
            print("Long run: no interactive confirmation. Pass --yes to continue (or limit the number of series with --max-series).")
            return 1

    def prog(stage, done, total):
        print(f"[{stage}] {done}/{total}", flush=True)

    try:
        res = run(ds, cfg, entry=entry, dataset_id=a.dataset, engine=get_engine(cfg.engine), progress=prog)
    except RunError as exc:
        print("ERROR:", exc)
        return 2
    print(f"\nDone -> {res.run_dir}")
    print(res.summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    for w in res.manifest["warnings"]:
        print("WARNING:", w)
    return 0


def _cmd_engines(a) -> int:
    from .engines.registry import engine_status
    for name, (ok, msg) in engine_status().items():
        print(f"{name:16s} {'AVAILABLE' if ok else 'MISSING':15s} {msg}")
    return 0


def _doctor_engine(eng, train, versions_out: dict, train_intermittent=None) -> int:
    from .engines import EngineJob
    from .engines.py_sf import INTERMITTENT

    bad = 0
    ok, msg = eng.available()
    print(f"\nEngine '{eng.name}': {'ok' if ok else 'MISSING'} — {msg}")
    if not ok:
        return 0                      # a missing engine is not critical (without R the Python engine is enough)
    import numpy as np
    for task, methods in (("local", eng.methods()["local"]), ("global", eng.methods()["global"])):
        for m in methods:
            try:
                tr = train_intermittent if (train_intermittent is not None and m in INTERMITTENT) else train
                r = eng.run(EngineJob(task=task, method=m, train=tr, horizon=6, seasonality=(12.0,),
                                      lag=12 if task == "global" else None))
                fin = bool(np.isfinite(r.forecasts).all())
                note = f"fallback={len(r.fallbacks)}" + ("" if fin else " | NON-FINITE VALUE")
                if r.fallbacks:
                    note += f" ({r.fallbacks[0]['message'][:200]})"
                print(f"  {task}/{m:18s} ok     {r.elapsed_sec:6.2f} s   {note}")
                for n_ in [x for x in r.notes if "allback" not in x][:3]:
                    print(f"      i {n_[:200]}")
                bad += 0 if (fin and not r.fallbacks) else 1
                versions_out[eng.name] = r.versions
            except Exception as exc:  # noqa: BLE001
                first = str(exc).strip().splitlines()[0]
                if m == "catboost":
                    print(f"  {task}/{m:18s} WARNING: {first}")
                else:
                    print(f"  {task}/{m:18s} ERROR: {first}")
                    bad += 1
    return bad


def _cmd_doctor(a) -> int:
    """Environment check. Exit code 0: no critical problem. The Python engine is always tried; the R engine too if it exists."""
    import platform

    import numpy as np

    from .engines import PythonEngine, REngine
    from .estimate import available_memory_bytes

    print(f"Python {platform.python_version()} | numpy {np.__version__} | {platform.platform()}")
    print(f"CPU cores: {os.cpu_count()} | OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}")
    mem = available_memory_bytes()
    print("Available memory:", f"{mem / 1024 ** 3:.1f} GB" if mem else "could not be read")
    rep = C.check_bundle(C.load_catalog(), a.data_dir)
    print(f"Data ({C._dir(a.data_dir).resolve()}): {len(rep['ok'])} ok, {len(rep['missing'])} missing, {len(rep['size_mismatch'])} mismatched")
    rng = np.random.default_rng(1)
    t = np.arange(120)
    train = [50 + 5 * np.sin(2 * np.pi * t / 12) + 0.05 * t + rng.normal(0, 1, 120) + 10 * i for i in range(3)]
    versions: dict = {}
    sparse = [np.where(rng.random(120) < 0.3, rng.poisson(4, 120) + 1.0, 0.0) for _ in range(3)]   # for the intermittent models
    bad = _doctor_engine(PythonEngine(), train, versions, sparse)
    bad += _doctor_engine(REngine(), train, versions)
    if versions:
        print("\nVersions:", json.dumps(versions, ensure_ascii=False))
    print("RESULT:", "no critical problem" if bad == 0 else f"{bad} problem(s) (see above)")
    return 0 if bad == 0 else 1


def _cmd_golden(a) -> int:
    import pandas as pd

    from .engines.registry import get_engine
    from .golden import available_cases, compare, golden_root, load_case

    eng = get_engine(a.engine)
    ok, msg = eng.available()
    if not ok:
        print(f"Engine '{a.engine}' is not available: {msg}")
        return 2
    cases = a.case or available_cases()
    if not cases:
        print(f"Golden dataset not found: {golden_root()}")
        return 2
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    frames = []
    for c in cases:
        print(f"\n== {c} | engine: {a.engine} | {msg}", flush=True)
        df = compare(eng, load_case(c), a.methods, a.workers, progress=lambda m: print(f"   {m} ...", flush=True), limit=a.limit)
        df.insert(0, "case", c)
        print(df.drop(columns=["case", "fb_reason", "notes"], errors="ignore").to_string(index=False, float_format=lambda x: f"{x:.3g}"))
        for _, r in df.iterrows():
            if r.get("fallbacks", 0) and r["fallbacks"] > 0:
                print(f"   !! {r['method']}: {int(r['fallbacks'])}/{int(r['n_series'])} series fell back to the (SNaive) forecast -> "
                      f"the measures in this row measure the fallback, NOT the method. First reason: {r['fb_reason']}")
            note = r.get("notes", "")
            if isinstance(note, str) and note:
                print(f"   i  {r['method']}: {note}")
        frames.append(df)
    if a.csv:
        pd.concat(frames).to_csv(a.csv, index=False)
        print("Written:", a.csv)
    print("\nNote: if median_rel is ~1e-5 or smaller the algorithm matches R; n_gt_1e-3 is usually the series where the optimiser "
          "stops at a bound. dSMAPE_%/dMASE_% is the effect of the difference on the accuracy measure.")
    return 0


def _cmd_app(a) -> int:
    from .app.server import serve
    from .app.service import AppService
    svc = AppService(data_dir=a.data_dir, runs_dir=a.runs_dir, uploads_dir=a.uploads_dir)
    return serve(svc, port=a.port, open_browser=not a.no_browser)


def _cmd_rank(a) -> int:
    """Ranking and ranking reliability from the per-series errors of several runs."""
    import numpy as np
    import pandas as pd

    from . import ranking as RK

    runs = {}
    for it in a.run:
        name, sep, path = it.partition("=")
        if not sep:
            path, name = it, Path(it).name
        runs[name] = path
    if len(runs) < 2:
        print("At least two --run entries are needed.")
        return 2
    e = RK.errors_from_runs(runs, a.metric)
    if a.higher_is_better:
        e = -e
    mr = RK.mean_ranks(e)
    print(f"Metric: {a.metric} | series: {len(e)} | models: {e.shape[1]}")
    print("\nMean rank (1 = best):")
    print(mr.to_string(float_format=lambda x: f"{x:.3f}"))
    if e.shape[1] >= 3:
        fr = RK.friedman(e)
        cd = RK.nemenyi_cd(fr.n_models, fr.n_blocks, a.alpha)
        print(f"\nFriedman chi2 = {fr.statistic:.4f}, p = {fr.p_value:.3g} (complete blocks {fr.n_blocks}, dropped {fr.n_dropped})")
        print(f"Nemenyi critical difference (alpha = {a.alpha}): {cd:.3f}")
    st = RK.rank_stability(e, n_boot=a.boot, top_k=a.top_k, seed=a.seed)
    tab = pd.DataFrame({"P(first)": st.p_first, f"P(top {st.k})": st.p_top_k}).join(st.mean_rank_ci).sort_values("p50")
    print(f"\nBootstrap over series ({st.n_boot} resamples):")
    print(tab.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"Kendall tau vs full ranking: median {float(np.median(st.kendall_tau)):.3f}, "
          f"5th percentile {float(np.percentile(st.kendall_tau, 5)):.3f}")
    if a.csv:
        tab.join(mr.rename("mean_rank")).to_csv(a.csv)
        print("Written:", a.csv)
    return 0


def _safe_streams() -> None:
    """On Windows, when output is redirected to a pipe (| Tee-Object) or a file, Python uses the default encoding (cp1252) and
    CRASHES with UnicodeEncodeError on characters such as ı, ş, ğ, İ. To avoid a crash, characters that cannot be written become '?'."""
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001
            pass


def main(argv: list[str] | None = None) -> int:
    _safe_streams()
    p = argparse.ArgumentParser(prog="tsfstudio", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, helptext in (("data-check", _cmd_data_check, "package status"), ("index", _cmd_index, "data summary"), ("hash", _cmd_hash, "SHA-256")):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--data-dir", default=None, help="default: $TSFSTUDIO_DATA_DIR or data/tsf")
        sp.set_defaults(fn=fn)
        if name == "data-check":
            sp.add_argument("--strict", action="store_true", help="exit with 1 if a file is missing or mismatched")
        if name == "index":
            sp.add_argument("--ids", nargs="*", help="only these datasets")
    sp = sub.add_parser("template", help="write an empty Excel template")
    sp.add_argument("out", type=Path)
    sp.set_defaults(fn=_cmd_template)
    sp = sub.add_parser("convert", help="Excel -> TSF")
    sp.add_argument("xlsx", type=Path)
    sp.add_argument("out", type=Path)
    sp.add_argument("--fill-gaps", action="store_true", help="fill missing dates with missing values")
    sp.add_argument("--lang", choices=("tr", "en"), default="en", help="language of the validation messages")
    sp.set_defaults(fn=_cmd_convert)
    sp = sub.add_parser("golden", help="compare an engine with your own R outputs (NN5)")
    sp.add_argument("--engine", default="python")
    sp.add_argument("--case", action="append", help="golden dataset (default: all); e.g. nn5_weekly")
    sp.add_argument("--methods", nargs="*", help="only these methods")
    sp.add_argument("--workers", type=int, default=1)
    sp.add_argument("--limit", type=int, default=None, help="only the first N series (quick trial / time measurement)")
    sp.add_argument("--csv", default=None, help="save the result table as CSV")
    sp.set_defaults(fn=_cmd_golden)
    sp = sub.add_parser("app", help="start the web interface")
    sp.add_argument("--port", type=int, default=8765, help="starting port (the next ones are tried if it is busy)")
    sp.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    sp.add_argument("--data-dir", default=None, help="default: $TSFSTUDIO_DATA_DIR or data/tsf")
    sp.add_argument("--runs-dir", default=None, help="default: $TSFSTUDIO_RUNS_DIR or ./runs")
    sp.add_argument("--uploads-dir", default=None, help="default: $TSFSTUDIO_UPLOADS_DIR or ./uploads")
    sp.set_defaults(fn=_cmd_app)
    sp = sub.add_parser("rank", help="rank models from several runs + ranking reliability (Friedman, Nemenyi, bootstrap)")
    sp.add_argument("--run", action="append", required=True, help="name=run_dir (repeat for each model); run_dir holds metrics.xlsx")
    sp.add_argument("--metric", default="MASE")
    sp.add_argument("--higher-is-better", action="store_true", help="for metrics where larger is better")
    sp.add_argument("--alpha", type=float, default=0.05)
    sp.add_argument("--boot", type=int, default=1000)
    sp.add_argument("--top-k", type=int, default=3)
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--csv", default=None)
    sp.set_defaults(fn=_cmd_rank)
    sp = sub.add_parser("engines", help="engine status")
    sp.set_defaults(fn=_cmd_engines)
    sp = sub.add_parser("doctor", help="environment check + quick live R test")
    sp.add_argument("--data-dir", default=None)
    sp.set_defaults(fn=_cmd_doctor)
    for name, fn, helptext in (("estimate", _cmd_estimate, "memory/time estimate"), ("run", _cmd_run, "run a model")):
        sp = sub.add_parser(name, help=helptext)
        g = sp.add_mutually_exclusive_group(required=True)
        g.add_argument("--dataset", help="catalogue id (e.g. nn5_weekly)")
        g.add_argument("--file", type=Path, help="direct path to a .tsf file")
        sp.add_argument("--data-dir", default=None)
        sp.add_argument("--engine", default="python", help="python | r | python-baseline")
        sp.add_argument("--task", choices=("local", "global"), default="local")
        sp.add_argument("--method", required=True)
        sp.add_argument("--horizon", type=int, default=None)
        sp.add_argument("--lag", type=int, default=None)
        sp.add_argument("--max-series", type=int, default=None, help="only the first N series (quick trial)")
        sp.add_argument("--missing-policy", choices=("error", "zero", "locf", "interpolate"), default="error")
        sp.add_argument("--param", action="append", help="model parameter key=value (JSON); repeatable")
        sp.add_argument("--workers", type=int, default=1)
        sp.add_argument("--integer-conversion", choices=("auto", "on", "off"), default="auto",
                        help="round forecasts to integers: auto = the value in the catalogue (ON for count data such as car_parts and M5)")
        sp.add_argument("--seed", type=int, default=1)
        sp.add_argument("--out", default=None, help="output folder (default: $TSFSTUDIO_RUNS_DIR or ./runs)")
        sp.add_argument("--force", action="store_true", help="override the memory block (not recommended)")
        sp.set_defaults(fn=fn)
        if name == "estimate":
            sp.add_argument("--probe", action="store_true", help="measure the time by making a small trial run")
        else:
            sp.add_argument("--yes", action="store_true", help="skip the long-run confirmation")
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
