"""Synthetic intermittent-demand series for tests/golden/r_intermittent (200 series, seed 0).

    python tools/r_reference/make_intermittent_series.py train.csv
    Rscript tools/r_reference/cran_reference.R train.csv 1 12 out fc_croston,ti_croston,ti_sba,ti_tsb
"""
import sys

import numpy as np

rng = np.random.default_rng(0)
rows = []
for _ in range(200):
    n = int(rng.integers(40, 201))
    p = rng.uniform(0.1, 0.6)
    lam = rng.uniform(1, 10)
    while True:
        y = (rng.random(n) < p) * (1 + rng.poisson(lam, n)).astype(float)
        if (y > 0).sum() >= 3:
            break
    rows.append(y)
out = np.full((len(rows), max(map(len, rows))), np.nan)
for i, y in enumerate(rows):
    out[i, :len(y)] = y
np.savetxt(sys.argv[1], out, delimiter=",", fmt="%.17g")
