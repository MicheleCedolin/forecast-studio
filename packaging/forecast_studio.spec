# PyInstaller spec for the standalone Windows package (one folder: dist\ForecastStudio\).
# Build:  .venv-build\Scripts\pyinstaller.exe packaging\forecast_studio.spec --noconfirm
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

SRC = Path(SPECPATH).resolve().parent / "src"
sys.path.insert(0, str(SRC))                                  # so collect_submodules sees the package

datas, binaries, hiddenimports = [], [], []
# Package data (same set as pyproject package-data): catalogue, web UI, R worker.
PKG = SRC / "tsfstudio"
for pattern in ("data/*.yaml", "r/*.R", "r/lib/*.R", "app/static/*.*", "app/static/fonts/*"):
    for f in PKG.glob(pattern):
        if f.is_file():
            datas.append((str(f), str(Path("tsfstudio") / f.parent.relative_to(PKG))))
hiddenimports += collect_submodules("tsfstudio")
# Native C++ library for the "cpp" engine (built by cpp/build.ps1); optional: the package works without it.
for f in (PKG / "native").glob("tsfcore.*"):
    binaries.append((str(f), "tsfstudio/native"))
for pkg in ("statsforecast", "catboost", "numba", "llvmlite", "sklearn", "coreforecast", "utilsforecast"):
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception:
        pass

a = Analysis(
    ["launcher.py"],
    pathex=["../src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython", "jupyter", "pytest", "torch"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="ForecastStudio", console=True, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="ForecastStudio", upx=False)
