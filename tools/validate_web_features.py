"""Checks the browser page's series characteristics and PCA (web/features.js on the WebAssembly build of tsfcore)
against the desktop application's (tsfstudio.features).

Usage:
    python tools/validate_web_features.py --node <node executable> [--data-dir data/tsf] [--datasets nn5_weekly ...]

Needs web/tsfcore.js and web/tsfcore.wasm (cpp/build_wasm.ps1) and the native library for the Python side. For every
dataset the script runs web/features.js under Node and compares
  * the basic features and catch22, value by value (largest relative difference),
  * the demand classes,
  * the PCA: explained variance, the features used and left out, loadings and scores.
Exit code 0 if every difference is below 1e-6 (the two sides use different summation orders and math libraries, so the
values agree to rounding, not bit for bit).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfstudio import catalog as C  # noqa: E402
from tsfstudio import features as F  # noqa: E402
from tsfstudio.seasonality import primary_period, seasonality_for  # noqa: E402

NODE_SCRIPT = r"""
const fs = require("fs");
const path = require("path");
const web = process.argv[2];
const createTsfCore = require(path.join(web, "tsfcore.js"));
const TF = require(path.join(web, "features.js"));
(async () => {
  const core = await createTsfCore();
  const out = {};
  for (const file of process.argv.slice(4)) {
    const bytes = fs.readFileSync(file);
    const p = core._malloc(bytes.length), msg = core._malloc(512);
    core.HEAPU8.set(bytes, p);
    const h = core._tsf_io_open_buffer(p, BigInt(bytes.length), -1n, msg, 512);
    core._free(p);
    const header = JSON.parse(core.UTF8ToString(core._tsf_io_header_json(h)));
    const t0 = Date.now();
    const f = TF.compute(core, h, header);
    const ms = Date.now() - t0;
    let pca = null, err = null;
    try { pca = TF.pca(f.table, f.names.length, f.columns); } catch (e) { err = String(e.message); }
    core._tsf_io_close(h);
    const num = (a) => Array.from(a, (v) => (Number.isFinite(v) ? v : null));
    out[path.basename(file)] = { names: f.names, columns: f.columns, table: num(f.table), classes: f.classes, ms, err,
      pca: pca && { k: pca.k, scores: num(pca.scores), explained: pca.explained, loadings: num(pca.loadings), features: pca.features, dropped: pca.dropped } };
  }
  fs.writeFileSync(process.argv[3], JSON.stringify(out));
})();
"""


def rel(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or (np.isnan(a) != np.isnan(b)).any():
        return float("inf")
    ok = ~np.isnan(a)
    if not ok.any():
        return 0.0
    return float(np.max(np.abs(a[ok] - b[ok]) / np.maximum(1.0, np.abs(a[ok]))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--node", required=True)
    ap.add_argument("--data-dir", default="data/tsf")
    ap.add_argument("--datasets", nargs="*", default=["nn5_weekly", "nn5_daily", "m1_yearly", "m1_quarterly", "m1_monthly",
                                                      "tourism_quarterly", "hospital", "car_parts", "covid_deaths"])
    a = ap.parse_args()
    cat = C.load_catalog()
    data_dir = Path(a.data_dir)
    entries = [cat.get(d) for d in a.datasets]
    files = [str((data_dir / e.file).resolve()) for e in entries]
    with tempfile.TemporaryDirectory() as td:
        script, out = Path(td) / "run.cjs", Path(td) / "out.json"
        script.write_text(NODE_SCRIPT, encoding="utf-8")
        r = subprocess.run([a.node, str(script), str(ROOT / "web"), str(out), *files], capture_output=True, text=True)
        if r.returncode != 0 or not out.is_file():
            print("Node failed:\n" + r.stderr[-2000:])
            return 1
        web = json.loads(out.read_text(encoding="utf-8"))
    worst = 0.0
    n_fail = 0
    for e in entries:
        w = web[e.file]
        ds = C.load_entry(e, data_dir)
        m = primary_period(seasonality_for(ds.frequency))
        df = F.compute(ds.values, m=m, sets=("basic", "catch22"), names=list(ds.series_names))
        problems = []
        if w["names"] != [str(n) for n in df.index]:
            problems.append("series names differ")
        cols = [c for c in df.columns if c != "demand_class"]
        if w["columns"] != cols:
            problems.append(f"columns differ: {w['columns'][:12]} vs {cols[:12]}")
        tw = np.array([np.nan if v is None else v for v in w["table"]], float).reshape(len(w["names"]), -1)
        tp = df[cols].to_numpy(float)
        tp[~np.isfinite(tp)] = np.nan
        d_feat = rel(tw, tp)
        if w["classes"] != [None if pd.isna(v) else v for v in df["demand_class"]]:
            problems.append("demand classes differ")
        d_pca = 0.0
        try:
            res = F.pca(df)
        except ValueError as exc:
            res = None
            if not w["err"]:
                problems.append(f"Python refuses the PCA ({exc}), the page does not")
        if res is not None:
            if w["pca"] is None:
                problems.append(f"the page refuses the PCA: {w['err']}")
            else:
                pw = w["pca"]
                k = pw["k"]
                if pw["features"] != res["features"] or pw["dropped"] != res["dropped"]:
                    problems.append("PCA features used / left out differ")
                else:
                    sc = np.array(pw["scores"], float).reshape(-1, k)
                    ld = np.array(pw["loadings"], float).reshape(-1, k)
                    scale = max(1.0, float(np.abs(res["scores"]).max()))
                    d_pca = max(rel(np.array(pw["explained"]), res["explained"]), rel(ld, res["loadings"]),
                                float(np.max(np.abs(sc - res["scores"])) / scale))
        d = max(d_feat, d_pca)
        worst = max(worst, d)
        if d > 1e-6:
            problems.append(f"values differ: features {d_feat:.2e}, PCA {d_pca:.2e}")
        n_fail += bool(problems)
        print(f"{'FAIL' if problems else 'PASS'}  {e.id:20s} {len(w['names']):5d} series  features {d_feat:.1e}  PCA {d_pca:.1e}  "
              f"({w['ms']} ms in WebAssembly)")
        for p in problems:
            print("      " + p)
    print(f"\n{'ALL PASS' if not n_fail else str(n_fail) + ' dataset(s) failed'}; largest difference {worst:.2e}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
