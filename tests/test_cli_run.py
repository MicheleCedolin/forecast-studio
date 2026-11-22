"""CLI `run` / `estimate` / `engines` / `doctor` (R yokken) — NN5 haftalık altın verisinden üretilen bir TSF ile."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsfstudio.cli import main
from tsfstudio.tsf.writer import write_tsf

G = Path(__file__).parent / "golden" / "nn5_weekly"


@pytest.fixture(scope="module")
def nn5_tsf(tmp_path_factory):
    train = np.loadtxt(G / "train.csv", delimiter=",")
    act = np.loadtxt(G / "actuals.csv", delimiter=",")
    full = np.hstack([train, act])
    p = tmp_path_factory.mktemp("nn5") / "nn5_weekly_test.tsf"
    write_tsf(p, relation="nn5_weekly", attributes=[("series_name", "string")],
              attr_rows=[(f"T{i + 1}",) for i in range(len(full))], values=list(full),
              frequency="weekly", horizon=8, missing=False, equal_length=True)
    return p


def test_engines_command(capsys):
    assert main(["engines"]) == 0
    out = capsys.readouterr().out
    assert "python-baseline" in out and "AVAILABLE" in out


def test_run_baseline_on_nn5_matches_golden_metrics(nn5_tsf, tmp_path, capsys):
    rc = main(["run", "--file", str(nn5_tsf), "--engine", "python-baseline", "--method", "snaive",
               "--out", str(tmp_path / "out")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Done ->" in out and "SMAPE" in out
    run_dir = next((tmp_path / "out").iterdir())
    m = pd.read_excel(run_dir / "metrics.xlsx", sheet_name="Summary").set_index("Metric")
    # SNaive (periyot 52) ile NN5 haftalık 111 seri: yalnızca bütünlük/ballpark kontrolü ve eksiksiz seri sayısı
    assert m.loc["SMAPE", "N_valid"] == 111 and 0 < m.loc["SMAPE", "Mean"] < 1
    assert (run_dir / "manifest.json").is_file()


def test_estimate_command_prints_prior(nn5_tsf, capsys):
    assert main(["estimate", "--file", str(nn5_tsf), "--engine", "python-baseline", "--method", "snaive"]) == 0
    out = capsys.readouterr().out
    assert "Number of series: 111" in out and "Time" in out


def test_long_run_requires_confirmation_when_not_interactive(nn5_tsf, tmp_path, capsys, monkeypatch):
    # TBATS tahmini 111 seri x 105 uzunluk için ~12 dk (<15 dk) olduğundan uzun sayılmaz; daha çok seri sahtelenemez,
    # bu yüzden eşiği düşürerek mantığı sınarız.
    import tsfstudio.estimate as E
    monkeypatch.setattr(E.PreflightReport, "long_running", property(lambda self: True))
    rc = main(["run", "--file", str(nn5_tsf), "--engine", "python-baseline", "--method", "snaive", "--out", str(tmp_path / "o")])
    assert rc == 1 and "--yes" in capsys.readouterr().out
    rc = main(["run", "--file", str(nn5_tsf), "--engine", "python-baseline", "--method", "snaive", "--out", str(tmp_path / "o"), "--yes"])
    assert rc == 0


def test_doctor_without_r_still_passes_with_python_engine(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("TSFSTUDIO_RSCRIPT", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert main(["doctor", "--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Engine 'python': ok" in out and "Engine 'r': MISSING" in out and "local/ses" in out


def test_golden_command_runs_python_engine(capsys, tmp_path):
    csv = tmp_path / "g.csv"
    assert main(["golden", "--engine", "python", "--case", "nn5_weekly", "--methods", "pooled_regression", "ses",
                 "--csv", str(csv)]) == 0
    out = capsys.readouterr().out
    assert "pooled_regression" in out and "median_rel" in out and csv.is_file()


def test_doctor_uses_intermittent_series_for_intermittent_methods_and_prints_reasons(monkeypatch, capsys):
    import numpy as np

    from tsfstudio import cli
    from tsfstudio.engines import EngineResult
    from tsfstudio.engines.base import Engine

    seen = {}

    class Fake(Engine):
        name = "fake"
        LOCAL_METHODS = ("tsb", "ses")

        def available(self):
            return True, "sahte"

        def run(self, job, progress=None, cancel=None):
            seen[job.method] = float(np.mean([np.mean(s == 0) for s in job.train]))
            fb = [{"index": 0, "message": "ses: ValueError: kaboom"}] if job.method == "ses" else []
            return EngineResult(np.ones((len(job.train), job.horizon)), 0.0, fallbacks=fb, notes=["2 series: seasonality dropped"])

    train = [np.arange(1.0, 100.0)]
    sparse = [np.zeros(100)]
    bad = cli._doctor_engine(Fake(), train, {}, sparse)
    out = capsys.readouterr().out
    assert seen["tsb"] == 1.0 and seen["ses"] == 0.0           # aralıklı yöntem aralıklı seriyle denendi
    assert bad == 1 and "kaboom" in out and "seasonality dropped" in out


def test_integer_conversion_flag_overrides_catalog(nn5_tsf, tmp_path):
    import json
    for flag, expected in (("on", True), ("off", False)):
        out = tmp_path / flag
        assert main(["run", "--file", str(nn5_tsf), "--engine", "python-baseline", "--method", "snaive",
                     "--integer-conversion", flag, "--out", str(out)]) == 0
        man = json.loads((next(out.iterdir()) / "manifest.json").read_text(encoding="utf-8"))
        assert man["config"]["effective_integer_conversion"] is expected
