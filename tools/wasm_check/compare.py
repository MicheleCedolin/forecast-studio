"""Compares out_wasm/<job> with out_native/<job>: forecasts.csv (rel. diff |a-b|/max(|b|,1e-12), b = native) and fallbacks.csv."""
import csv, json, math, os, statistics, sys
W = "wasm_work/wasm_check/"
jobs = json.load(open(W + "jobs.json"))
def load(side, name):
    d = os.path.join(W, side, name)
    r = json.load(open(d + "/result.json")) if os.path.exists(d + "/result.json") else None
    fc, fb = {}, {}
    if r and r["rc"] == 0:
        with open(d + "/forecasts.csv") as f:
            for row in csv.reader(f):
                if row[0] == "series_name": continue
                fc[row[0]] = [float(x) if x else math.nan for x in row[1:]]
        with open(d + "/fallbacks.csv") as f:
            for row in list(csv.reader(f))[1:]: fb[row[1]] = row[2]
    return r, fc, fb
rows = []
for j in jobs:
    rn, fn, bn = load("out_native", j["name"]); rw, fw, bw = load("out_wasm", j["name"])
    model, ds = j["name"].split("__", 1)
    if not rw or not rn or rn["rc"] or rw["rc"]:
        rows.append((model, ds, "run failed: native rc=%s wasm rc=%s %s" % (rn and rn["rc"], rw and rw["rc"], (rw or {}).get("message", "missing")))); continue
    per, worst = [], (0, None)
    nanmis = 0
    for s, b in fn.items():
        a = fw.get(s)
        if a is None: nanmis += 1; continue
        m = 0.0
        for x, y in zip(a, b):
            if math.isnan(x) != math.isnan(y): m = math.inf; continue
            if math.isnan(x): continue
            m = max(m, abs(x - y) / max(abs(y), 1e-12))
        per.append(m)
        if m > worst[0]: worst = (m, s)
    only_n = sorted(set(bn) - set(bw)); only_w = sorted(set(bw) - set(bn))
    both = len(set(bn) & set(bw))
    diffmsg = [s for s in set(bn) & set(bw) if bn[s] != bw[s]]
    rows.append(dict(model=model, ds=ds, n=len(per), max=max(per), med=statistics.median(per), gt=sum(x > 1e-6 for x in per),
                     exact=sum(x == 0 for x in per), fb_both=both, only_n=only_n, only_w=only_w, worst=worst[1], missing=nanmis,
                     msgdiff=len(diffmsg), tn=rn["seconds"], tw=rw["seconds"]))
json.dump(rows, open(W + "compare.json", "w"), indent=1, default=str)
print("| model | dataset | n | max rel diff | median rel diff | n > 1e-6 | n identical | fallbacks (both) | fallback one side only | native s | wasm s |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    if isinstance(r, tuple): print("| %s | %s | %s |" % r); continue
    one = ("native only: " + ",".join(r["only_n"]) if r["only_n"] else "") + (" wasm only: " + ",".join(r["only_w"]) if r["only_w"] else "")
    print(f"| {r['model']} | {r['ds']} | {r['n']} | {r['max']:.2e} | {r['med']:.2e} | {r['gt']} | {r['exact']} | {r['fb_both']} | {one or '-'} | {r['tn']:.1f} | {r['tw']:.1f} |"
          + (f" missing in wasm: {r['missing']}" if r["missing"] else "") + (f" fallback msg differs: {r['msgdiff']}" if r["msgdiff"] else ""))
