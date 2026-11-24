"""Writes web/samples/nn5_weekly.tsf, the sample data set of the web edition: the 111 NN5 Weekly series (training part and
the 8 held-out weeks) from the validation fixtures in tests/golden/nn5_weekly, so the page's results on the sample can be
checked against the study's R forecasts stored in the same folder.

    python tools/web/make_sample.py
"""
from __future__ import annotations

from pathlib import Path

root = Path(__file__).resolve().parents[2]
g = root / "tests" / "golden" / "nn5_weekly"
rows = lambda name: [[v for v in line.strip().split(",") if v] for line in (g / name).read_text().splitlines() if line.strip()]
train, test = rows("train.csv"), rows("actuals.csv")
assert len(train) == len(test) == 111
lines = ["# NN5 Weekly: 111 series of weekly cash withdrawals at UK cash machines (NN5 forecasting competition),",
         "# as distributed in the Monash Time Series Forecasting Archive. The last 8 values are the test part.",
         "@relation nn5_weekly", "@attribute series_name string", "@frequency weekly", "@horizon 8", "@missing false",
         "@equallength true", "@data"]
lines += [f"T{i + 1}:{','.join(tr + te)}" for i, (tr, te) in enumerate(zip(train, test))]
dest = root / "web" / "samples" / "nn5_weekly.tsf"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print(dest, dest.stat().st_size, "bytes")
