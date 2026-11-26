"""Builds the archive to upload to a cluster for the engine speed study (docs/HPC_SPEED.md).

    python tools/make_hpc_package.py                       # dist/forecast-studio-hpc-speed-<date>.tar.gz
    python tools/make_hpc_package.py --no-data             # code only (copy the .tsf files yourself)

Contents: the tracked files of the folders the cluster needs (source, C++ core, tools, tests, HPC scripts, documents),
taken from the working copy; the speed-study files even when they are not committed yet; the .tsf files of the study's
plan from data/tsf; a VERSION file with the commit the working copy is based on. Shell and job scripts are stored with
LF line endings and as executable.
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEEP = ("src/", "cpp/", "tools/", "tests/", "scripts/hpc/", "docs/HPC", "docs/METRICS.md", "pyproject.toml", "LICENSE",
        "NOTICE", "README.md", ".gitattributes")
EXTRA = ["tools/engine_speed_study.py", "tools/make_hpc_package.py", "scripts/hpc/run_speed.sh", "scripts/hpc/speed.sge",
         "scripts/hpc/speed.sbatch", "scripts/hpc/setup_r.sh", "scripts/hpc/preflight.sh", "docs/HPC_SPEED.md"]
LF = (".sh", ".sge", ".sbatch", ".py", ".R", ".yaml", ".yml", ".txt", ".md", ".toml", ".cpp", ".hpp", ".c", ".h")
EXEC = (".sh", ".sge", ".sbatch")
TOP = "forecast-studio"


def git() -> str:
    exe = shutil.which("git")
    if exe:
        return exe
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "GitHubDesktop"
    found = sorted(base.glob("app-*/resources/app/git/cmd/git.exe"))
    if not found:
        sys.exit("git not found (not on PATH and no GitHub Desktop)")
    return str(found[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-data", action="store_true")
    ap.add_argument("--data-only", action="store_true",
                    help="a second archive with every catalogue file of data/tsf that is NOT in the study's plan "
                         "(unpack it in the same place: tar -xzf <file> from the folder that holds forecast-studio/)")
    ap.add_argument("--out", default=str(ROOT / "dist"))
    a = ap.parse_args()
    g = git()
    tracked = subprocess.run([g, "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    commit = subprocess.run([g, "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    files = sorted({f for f in tracked if f.startswith(KEEP) or f in KEEP} | {f for f in EXTRA if (ROOT / f).is_file()})
    missing = [f for f in EXTRA if not (ROOT / f).is_file()]
    if missing:
        sys.exit(f"missing: {missing}")
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "tools"))
    data = []
    if not a.no_data:
        import engine_speed_study as ess
        from tsfstudio.catalog import load_catalog
        cat = load_catalog()
        entries = {e.id: e for e in (cat.entries.values() if isinstance(cat.entries, dict) else cat.entries)}
        for ds in ess.PLAN:
            p = ROOT / "data" / "tsf" / entries[ds].file
            if not p.is_file():
                sys.exit(f"data file not found: {p}")
            data.append(p)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.data_only:
        plan_files = {p.name for p in data}
        rest = [ROOT / "data" / "tsf" / e.file for e in entries.values() if e.file not in plan_files]
        have = [p for p in rest if p.is_file()]
        target = out / f"forecast-studio-hpc-data-{time.strftime('%Y%m%d')}.tar.gz"
        with tarfile.open(target, "w:gz", compresslevel=6) as tar:
            for p in sorted(have, key=lambda p: p.stat().st_size):
                tar.add(p, arcname=f"{TOP}/data/tsf/{p.name}")
                print(f"  {p.name}  {p.stat().st_size / 1e6:.1f} MB", flush=True)
        print(f"{target}  ({target.stat().st_size / 1e6:.1f} MB; {len(have)} data files, "
              f"{sum(p.stat().st_size for p in have) / 1e6:.0f} MB unpacked; not on this machine: "
              f"{[p.name for p in rest if not p.is_file()]})")
        return 0
    target = out / f"forecast-studio-hpc-speed-{time.strftime('%Y%m%d')}.tar.gz"
    now = time.time()

    def add(tar: tarfile.TarFile, name: str, raw: bytes, mode: int = 0o644) -> None:
        ti = tarfile.TarInfo(f"{TOP}/{name}")
        ti.size, ti.mtime, ti.mode = len(raw), now, mode
        tar.addfile(ti, io.BytesIO(raw))

    n_crlf = 0
    with tarfile.open(target, "w:gz") as tar:
        for f in files:
            raw = (ROOT / f).read_bytes()
            if f.endswith(LF) and b"\r\n" in raw:
                raw = raw.replace(b"\r\n", b"\n")
                n_crlf += 1
            add(tar, f, raw, 0o755 if f.endswith(EXEC) else 0o644)
        for p in data:
            tar.add(p, arcname=f"{TOP}/data/tsf/{p.name}")
        add(tar, "data/tsf/.gitkeep", b"")
        add(tar, "VERSION", f"commit {commit} plus the working copy of {time.strftime('%Y-%m-%d %H:%M')}\n".encode())
        add(tar, "HPC_README.txt", (ROOT / "docs" / "HPC_SPEED.md").read_bytes().replace(b"\r\n", b"\n"))
    print(f"{target}  ({target.stat().st_size / 1e6:.1f} MB; {len(files)} files, {len(data)} data files, "
          f"{n_crlf} converted to LF; based on commit {commit})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
