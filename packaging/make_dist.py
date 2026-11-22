"""Finish the PyInstaller output into the release folder and zip it.

    python packaging/make_dist.py --dist dist/ForecastStudio --version v0.1.1 --out dist

Adds README.txt (from packaging/README_dist.txt), examples/nn5_weekly.tsf (built from tests/golden/nn5_weekly) and an empty
data/tsf folder, removes folders left by test runs (runs, uploads, cache), then writes ForecastStudio-<version>-win64.zip
(top folder ForecastStudio/) and prints its size and SHA-256. Used by .github/workflows/windows-package.yml and by hand.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", required=True, help="PyInstaller output folder (contains ForecastStudio.exe)")
    ap.add_argument("--version", required=True, help="e.g. v0.1.1 or 0.1.1")
    ap.add_argument("--out", required=True, help="folder for the zip")
    a = ap.parse_args()
    dist = Path(a.dist)
    if not (dist / "ForecastStudio.exe").is_file():
        print(f"ForecastStudio.exe not found in {dist}", file=sys.stderr)
        return 2
    version = a.version.lstrip("vV")

    for leftover in ("runs", "uploads", "cache"):
        shutil.rmtree(dist / leftover, ignore_errors=True)
    (dist / "data" / "tsf").mkdir(parents=True, exist_ok=True)
    (dist / "README.txt").write_text((ROOT / "packaging" / "README_dist.txt").read_text(encoding="utf-8").format(version=version),
                                     encoding="utf-8")

    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    from tsfstudio.tsf.writer import write_tsf

    g = ROOT / "tests" / "golden" / "nn5_weekly"
    full = np.hstack([np.loadtxt(g / "train.csv", delimiter=","), np.loadtxt(g / "actuals.csv", delimiter=",")])
    (dist / "examples").mkdir(exist_ok=True)
    write_tsf(dist / "examples" / "nn5_weekly.tsf", relation="nn5_weekly", attributes=[("series_name", "string")],
              attr_rows=[(f"T{i + 1}",) for i in range(len(full))], values=list(full), frequency="weekly", horizon=8,
              missing=False, equal_length=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    zpath = out / f"ForecastStudio-{version}-win64.zip"
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(dist.rglob("*")):
            arc = Path("ForecastStudio") / f.relative_to(dist)
            if f.is_dir():
                if not any(f.iterdir()):
                    z.writestr(arc.as_posix() + "/", "")
            else:
                z.write(f, arc.as_posix())
    h = hashlib.sha256()
    with open(zpath, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    size = zpath.stat().st_size
    (out / f"{zpath.name}.sha256").write_text(f"{h.hexdigest().upper()}  {zpath.name}\n", encoding="ascii")
    print(f"{zpath.name}: {size:,} bytes ({size / 2 ** 20:.1f} MB)")
    print(f"SHA-256: {h.hexdigest().upper()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
