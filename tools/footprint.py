"""Install footprint of Forecast Studio and of the library stacks it replaces.

Prints the size of the C++ core (DLL, WebAssembly), and for the Python extras the number of installed distributions
they pull in and their size on disk (walking requirements with importlib.metadata in the running interpreter's
environment), and for R the packages and size behind forecast + jsonlite. Run with the project's virtual environment:

    python tools/footprint.py [--rscript PATH]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import tomllib
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MB = 1024 * 1024


def _name(req: str) -> str:
    return re.split(r"[ ;<>=!~\[(]", req.strip(), maxsplit=1)[0].lower().replace("_", "-")


def walk(reqs: list[str]) -> dict[str, int]:
    """Installed distributions reachable from `reqs` (markers for extras are ignored) -> bytes on disk."""
    seen: dict[str, int] = {}
    todo = [_name(r) for r in reqs]
    while todo:
        n = todo.pop()
        if n in seen:
            continue
        try:
            d = metadata.distribution(n)
        except metadata.PackageNotFoundError:
            continue
        size = 0
        for f in d.files or []:
            p = Path(d.locate_file(f))
            if p.is_file():
                size += p.stat().st_size
        seen[n] = size
        for r in d.requires or []:
            if "extra ==" not in r:
                todo.append(_name(r))
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rscript", default=os.environ.get("TSFSTUDIO_RSCRIPT", "Rscript"))
    a = ap.parse_args()

    for p in ["src/tsfstudio/native/tsfcore.dll", "web/tsfcore.wasm", "web/tsfcore.js"]:
        f = ROOT / p
        print(f"{p}: {f.stat().st_size / MB:.2f} MB" if f.exists() else f"{p}: missing")

    proj = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    core = proj.get("dependencies", [])
    extras = proj.get("optional-dependencies", {})
    stacks = {"core": core}
    for e in ("pyengine", "deep"):
        if e in extras:
            stacks[f"core+{e}"] = core + extras[e]
    if "pyengine" in extras and "deep" in extras:
        stacks["core+pyengine+deep"] = core + extras["pyengine"] + extras["deep"]
    for k, reqs in stacks.items():
        s = walk(reqs)
        print(f"Python {k}: {len(s)} installed packages, {sum(s.values()) / MB:.0f} MB")

    r = ("ip <- installed.packages(); base <- rownames(ip)[!is.na(ip[,'Priority']) & ip[,'Priority']=='base'];"
         "d <- tools::package_dependencies(c('forecast','jsonlite'), db=ip, recursive=TRUE);"
         "p <- setdiff(unique(c('forecast','jsonlite', unlist(d))), base);"
         "sz <- function(x) sum(file.info(list.files(x, recursive=TRUE, full.names=TRUE))$size);"
         "cat(length(p), 'non-base packages,', round(sum(sapply(p, function(x) sz(find.package(x))))/2^20), 'MB; ');"
         "extra <- rownames(ip)[ip[,'LibPath']==.Library & is.na(ip[,'Priority'])];"   # installed later, not by R's installer
         "cat('R itself (base install)', round((sz(R.home()) - sum(sapply(extra, function(x) sz(file.path(.Library, x)))))/2^20), 'MB\\n')")
    try:
        out = subprocess.run([a.rscript, "-e", r], capture_output=True, text=True, timeout=300)
        print("R forecast+jsonlite:", (out.stdout or out.stderr).strip())
    except (OSError, subprocess.TimeoutExpired) as e:
        print("R: not measured:", e)


if __name__ == "__main__":
    main()
