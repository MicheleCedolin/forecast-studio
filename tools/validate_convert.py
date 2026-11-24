"""Checks the browser page's table converter (web/convert.js) on round trips.

Usage:
    python tools/validate_convert.py --node <node executable> [--data-dir web/data] [--max-series 40]

Every .tsf file of the data directory is read with the Python reader and written out as the kinds of table a user
would have:
  * long CSV (series, date, value), comma separated, ISO dates;
  * long CSV, semicolon separated, dates as dd.mm.yyyy and decimal commas, rows shuffled;
  * wide CSV (one row per date, one column per series), tab separated;
  * wide Excel workbook with real date cells;
  * long Excel workbook with real date cells;
  * columns only (no dates), where the series have equal lengths.
web/convert.js turns each back into a .tsf under Node, the Python reader reads the result, and the script compares
the frequency, the series count, each start (as a period for monthly, quarterly and yearly data) and every value.
Leading and trailing missing values are dropped by the converter, so the reference is trimmed the same way.
Exit code 0 if everything agrees.
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

NODE_SCRIPT = r"""
const fs = require("fs");
const path = require("path");
const C = require(path.join(process.argv[2], "convert.js"));
(async () => {
  const jobs = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
  const out = [];
  for (const j of jobs) {
    try {
      const b = fs.readFileSync(j.src);
      const r = await C.convert(path.basename(j.src), b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength), j.options || {});
      fs.writeFileSync(j.dst, r.text);
      out.push({ ok: true, report: r.report });
    } catch (e) {
      out.push({ ok: false, error: String(e && e.message || e) });
    }
  }
  fs.writeFileSync(process.argv[4], JSON.stringify(out));
})();
"""

CALENDAR = {"monthly": "M", "quarterly": "Q", "yearly": "Y"}


def fmt(v: float, comma: bool = False) -> str:
    if np.isnan(v):
        return ""
    s = repr(float(v))
    s = s[:-2] if s.endswith(".0") else s
    return s.replace(".", ",") if comma else s


def trimmed(values: np.ndarray, start: pd.Timestamp, stamps: pd.DatetimeIndex):
    ok = np.flatnonzero(~np.isnan(values))
    if not ok.size:
        return None
    return values[ok[0]: ok[-1] + 1], stamps[ok[0]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--node", required=True)
    ap.add_argument("--data-dir", default="web/data")
    ap.add_argument("--max-series", type=int, default=40)
    a = ap.parse_args()
    from openpyxl import Workbook

    from tsfstudio.seasonality import timestamps
    from tsfstudio.tsf import read_tsf

    files = sorted(Path(a.data_dir).glob("*.tsf"))
    rng = np.random.default_rng(1)
    n_fail = 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        jobs, refs = [], []
        for f in files:
            ds = read_tsf(f, max_series=a.max_series)
            freq = ds.frequency
            names = ds.series_names
            starts = [pd.Timestamp(s) for s in ds.attrs["start_timestamp"]]
            stamps = [timestamps(s, len(v), freq) for s, v in zip(starts, ds.values)]
            ref = {}
            for n, v, s, st in zip(names, ds.values, starts, stamps):
                t = trimmed(np.asarray(v, float), s, st)
                if t is not None:
                    ref[n] = t
            intraday = freq not in ("daily", "weekly", "monthly", "quarterly", "yearly")
            iso = (lambda t: t.strftime("%Y-%m-%d %H:%M:%S")) if intraday else (lambda t: t.strftime("%Y-%m-%d"))
            dmy = (lambda t: t.strftime("%d.%m.%Y %H:%M:%S")) if intraday else (lambda t: t.strftime("%d.%m.%Y"))
            stem = f.stem

            def job(kind, path, options=None, ref_=ref):
                jobs.append({"src": str(path), "dst": str(td / f"{stem}__{kind}.tsf"), "options": options or {}})
                refs.append((stem, kind, freq, ref_))

            rows = [(n, t, x) for n, v, st in zip(names, ds.values, stamps) for t, x in zip(st, v)]
            p = td / f"{stem}__long.csv"
            p.write_text("series,date,value\n" + "".join(f"{n},{iso(t)},{fmt(x)}\n" for n, t, x in rows), encoding="utf-8")
            job("long_csv", p)

            order = rng.permutation(len(rows))
            p = td / f"{stem}__long_eu.csv"
            p.write_text("Tarih;Ürün;Satış\r\n" + "".join(f"{dmy(rows[i][1])};{rows[i][0]};{fmt(rows[i][2], True)}\r\n" for i in order),
                         encoding="utf-8-sig")
            job("long_eu_csv", p)

            wide = pd.DataFrame({n: pd.Series(np.asarray(v, float), index=st) for n, v, st in zip(names, ds.values, stamps)}).sort_index()
            # a wide table holds one row per date: series that share a period but not the day cannot sit in one
            same_grid = len(wide) == len(pd.DatetimeIndex(wide.index).to_period(CALENDAR[freq]).unique()) if freq in CALENDAR \
                else bool(((wide.index[1:] - wide.index[:-1]) % (stamps[0][1] - stamps[0][0]) == pd.Timedelta(0)).all())
            if same_grid:
                p = td / f"{stem}__wide.tsv"
                with p.open("w", encoding="utf-8", newline="") as fh:
                    fh.write("date\t" + "\t".join(wide.columns) + "\n")
                    for t, r in zip(wide.index, wide.to_numpy()):
                        fh.write(iso(t) + "\t" + "\t".join(fmt(x) for x in r) + "\n")
                job("wide_tsv", p)

                wb = Workbook()
                ws = wb.active
                ws.title = "Sales"
                ws.append(["Date", *wide.columns])
                for t, r in zip(wide.index, wide.to_numpy()):
                    ws.append([t.to_pydatetime(), *[None if np.isnan(x) else float(x) for x in r]])
                p = td / f"{stem}__wide.xlsx"
                wb.save(p)
                job("wide_xlsx", p)

            wb = Workbook()
            ws = wb.active
            ws.append(["item_id", "timestamp", "target"])
            for n, t, x in rows[:60000]:
                ws.append([n, t.to_pydatetime(), None if np.isnan(x) else float(x)])
            kept = {}
            for n, t, x in rows[:60000]:
                kept.setdefault(n, []).append((t, x))
            ref_x = {}
            for n, tv in kept.items():
                t = trimmed(np.array([x for _, x in tv], float), tv[0][0], pd.DatetimeIndex([t for t, _ in tv]))
                if t is not None:
                    ref_x[n] = t
            p = td / f"{stem}__long.xlsx"
            wb.save(p)
            job("long_xlsx", p, ref_=ref_x)

            if ds.is_equal_length() and not ds.has_missing():
                p = td / f"{stem}__columns.csv"
                mat = np.column_stack([np.asarray(v, float) for v in ds.values])
                p.write_text(",".join(names) + "\n" + "".join(",".join(fmt(x) for x in r) + "\n" for r in mat), encoding="utf-8")
                job("columns_csv", p, {"frequency": freq}, {n: (np.asarray(v, float), None) for n, v in zip(names, ds.values)})

        script, jf, of = td / "run.cjs", td / "jobs.json", td / "out.json"
        script.write_text(NODE_SCRIPT, encoding="utf-8")
        jf.write_text(json.dumps(jobs), encoding="utf-8")
        r = subprocess.run([a.node, str(script), str(ROOT / "web"), str(jf), str(of)], capture_output=True, text=True)
        if r.returncode != 0 or not of.is_file():
            print("Node failed:\n" + r.stderr[-3000:])
            return 1
        results = json.loads(of.read_text(encoding="utf-8"))

        for j, res, (stem, kind, freq, ref) in zip(jobs, results, refs):
            problems = []
            if not res["ok"]:
                problems.append("converter: " + res["error"])
            else:
                got = read_tsf(j["dst"])
                rep = res["report"]
                if got.frequency != freq:
                    problems.append(f"frequency {got.frequency} instead of {freq}")
                gnames = got.series_names
                if sorted(gnames) != sorted(ref):
                    problems.append(f"series differ: {len(gnames)} read, {len(ref)} expected "
                                    f"(first difference: {sorted(set(gnames) ^ set(ref))[:3]})")
                else:
                    gstart = dict(zip(gnames, [pd.Timestamp(s) for s in got.attrs["start_timestamp"]]))
                    for n, v in zip(gnames, got.values):
                        rv, rs = ref[n]
                        if len(v) != len(rv) or not np.array_equal(np.asarray(v, float), rv, equal_nan=True):
                            problems.append(f"values of {n} differ (length {len(v)} vs {len(rv)})")
                            break
                        if rs is None:
                            continue
                        s = gstart[n]
                        same = (s.to_period(CALENDAR[freq]) == rs.to_period(CALENDAR[freq])) if freq in CALENDAR \
                            else (s == (rs.normalize() if freq in ("daily", "weekly") else rs))
                        if not same:
                            problems.append(f"start of {n}: {s} instead of {rs}")
                            break
                note = f"{rep['layout']:7s} {rep['nSeries']:4d} series, {rep['frequency']} ({rep['frequencySource']})" if res["ok"] else ""
            n_fail += bool(problems)
            print(("FAIL  " if problems else "PASS  ") + f"{stem:28s} {kind:12s} " + (note if res["ok"] else ""), flush=True)
            for p in problems:
                print("      " + p)
    print(f"\n{'ALL PASS' if not n_fail else str(n_fail) + ' file(s) failed'} ({len(jobs)} files)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
