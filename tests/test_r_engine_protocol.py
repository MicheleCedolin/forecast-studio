"""R motoru iş protokolü testleri.

Gerçek R bu ortamda yoktur; bu testler `Rscript` yerine protokolü taklit eden bir betik (stub) kullanır.
Doğrulananlar: job.json/series.bin yazımı, sütun-ana tahmin okuma, yedek/NA sayaçları, ilerleme geri çağrısı,
hata ve çökme mesajları, iptal. Gerçek R işçisi `tests/test_r_engine_live.py` ile Docker içinde sınanır.
"""
import json
import os
import stat
import sys
import threading
import time

import numpy as np
import pytest

from tsfstudio.engines import EngineCancelled, EngineError, EngineJob, REngine

STUB = r'''#!{python}
import json, os, sys, time
import numpy as np
if "--version" in sys.argv:
    sys.stderr.write("R version 0.0.0 (stub)\n"); sys.exit(0)
job_dir = sys.argv[-1]
mode = os.environ.get("STUB_MODE", "ok")
job = json.load(open(os.path.join(job_dir, "job.json")))
lengths = job["lengths"]
vals = np.fromfile(os.path.join(job_dir, "series.bin"), dtype="<f8")
assert vals.size == sum(lengths), "series.bin boyutu"
if mode == "error":
    open(os.path.join(job_dir, "error.txt"), "w").write("Bilinmeyen yontem: foo"); sys.exit(2)
if mode == "crash":
    sys.stdout.write("Killed\n"); sys.exit(137)
if mode == "slow":
    time.sleep(30)
ends = np.cumsum(lengths); starts = ends - np.array(lengths)
H = job["horizon"]; n = len(lengths)
F = np.empty((n, H))
for i in range(n):
    F[i, :] = vals[ends[i] - 1]
if job["integer_conversion"]:
    F = np.round(F)
json.dump({{"stage": "forecasting", "done": n, "total": n}}, open(os.path.join(job_dir, "progress.json"), "w"))
F.flatten(order="F").astype("<f8").tofile(os.path.join(job_dir, "forecasts.bin"))
json.dump({{"n_series": n, "horizon": H, "elapsed_sec": 1.5, "na_replaced": 2, "n_fallbacks": 1,
          "fallbacks": [{{"index": 1, "message": "ets: stub hatasi"}}],
          "versions": {{"R": "R stub", "forecast": "0.0", "jsonlite": "0.0", "catboost": None}}}},
          open(os.path.join(job_dir, "summary.json"), "w"))
'''


@pytest.fixture
def engine(tmp_path):
    worker = tmp_path / "worker.R"
    worker.write_text("# stub")
    if os.name == "nt":
        # Windows ignores shebangs: run the Python stub through a .cmd launcher (like the real Rscript.exe,
        # it is a parent process that starts the actual worker as a child, which also exercises tree-kill on cancel).
        script = tmp_path / "Rscript_stub.py"
        script.write_text(STUB.format(python=sys.executable), encoding="utf-8")
        stub = tmp_path / "Rscript.cmd"
        py = sys.executable.replace("%", "%%")
        stub.write_text(f'@echo off\r\n"{py}" "%~dp0Rscript_stub.py" %*\r\nexit /b %ERRORLEVEL%\r\n', encoding="ascii")
    else:
        stub = tmp_path / "Rscript"
        stub.write_text(STUB.format(python=sys.executable))
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return REngine(rscript=str(stub), worker=worker, workdir=str(tmp_path))


def job(**kw):
    train = [np.array([1.0, 2.0, 3.5]), np.array([10.0, 20.0, 30.2, 40.4])]
    base = dict(task="local", method="ets", train=train, horizon=3, seasonality=(12.0,))
    base.update(kw)
    return EngineJob(**base)


def test_available_reports_version(engine):
    ok, msg = engine.available()
    assert ok and "stub" in msg


def test_unavailable_when_rscript_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("TSFSTUDIO_RSCRIPT", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    ok, msg = REngine(rscript=None, worker=tmp_path / "w.R").available()
    assert not ok and "Rscript not found" in msg


def test_roundtrip_column_major_and_counters(engine):
    seen = []
    res = engine.run(job(), progress=lambda s, d, t: seen.append((s, d, t)))
    np.testing.assert_allclose(res.forecasts, [[3.5] * 3, [40.4] * 3])      # sütun-ana düzen doğru çözülüyor
    assert res.forecasts.shape == (2, 3)
    assert res.fallbacks == [{"index": 1, "message": "ets: stub hatasi"}]
    assert res.na_replaced == 2 and res.elapsed_sec == 1.5
    assert res.versions["R"] == "R stub" and any("fallback" in n for n in res.notes)
    assert seen and seen[-1][0] == "finished"


def test_job_spec_is_written_as_expected(engine, tmp_path):
    engine.keep_job_dir = True
    engine.run(job(integer_conversion=True, horizons=[3, 2], params={"fourier_k": 2}, seed=7, workers=3))
    d = next(p for p in tmp_path.iterdir() if p.name.startswith("tsfjob_"))
    spec = json.loads((d / "job.json").read_text())
    assert spec["lengths"] == [3, 4] and spec["horizons"] == [3, 2] and spec["seed"] == 7 and spec["workers"] == 3
    assert spec["integer_conversion"] is True and spec["params"] == {"fourier_k": 2} and spec["seasonality"] == [12.0]
    raw = np.fromfile(d / "series.bin", dtype="<f8")
    np.testing.assert_allclose(raw, [1, 2, 3.5, 10, 20, 30.2, 40.4])


def test_job_dir_is_cleaned_up(engine, tmp_path):
    engine.run(job())
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("tsfjob_")]


def test_worker_error_message_is_surfaced(engine, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "error")
    with pytest.raises(EngineError, match="Bilinmeyen yontem: foo"):
        engine.run(job())


def test_crash_without_error_file_shows_log_tail(engine, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "crash")
    with pytest.raises(EngineError, match=r"(?s)code 137.*Killed"):
        engine.run(job())


def test_cancel_terminates_worker(engine, monkeypatch, tmp_path):
    monkeypatch.setenv("STUB_MODE", "slow")
    ev = threading.Event()
    threading.Timer(1.0, ev.set).start()
    t0 = time.time()
    with pytest.raises(EngineCancelled):
        engine.run(job(), cancel=ev)
    assert time.time() - t0 < 15
    # A surviving (grand)child would still hold worker.log open and block the cleanup on Windows.
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("tsfjob_")]


def test_unsupported_method_and_bad_jobs(engine):
    with pytest.raises(EngineError, match="does not support"):
        engine.run(job(method="naive"))
    with pytest.raises(EngineError, match="lag"):
        engine.run(job(task="global", method="catboost"))
    with pytest.raises(EngineError, match="horizons"):
        engine.run(job(horizons=[1]))


def test_r_sources_are_packaged():
    from tsfstudio.engines.r_engine import default_worker_path
    w = default_worker_path()
    assert w.is_file() and (w.parent / "lib" / "local_models.R").is_file() and (w.parent / "lib" / "global_models.R").is_file()
