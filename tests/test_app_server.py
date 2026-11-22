"""Local interface server: security (token, Host, path), end-to-end run, cancel, upload, download."""
from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.parse

import numpy as np
import pytest

from tsfstudio.app.server import make_server
from tsfstudio.app.service import AppService
from tsfstudio.tsf.writer import write_tsf

TOKEN = "test-token"


def _make_tsf(path, n=30, length=80, horizon=6, seed=1, name="demo"):
    rng = np.random.default_rng(seed)
    vals = [60 + np.cumsum(rng.normal(size=length)) + 8 * np.sin(np.arange(length) * 2 * np.pi / 12) for _ in range(n)]
    write_tsf(path, relation=name, attributes=[("series_name", "string")], attr_rows=[(f"S{i + 1}",) for i in range(n)],
              values=vals, frequency="monthly", horizon=horizon, missing=False, equal_length=True)


@pytest.fixture()
def app(tmp_path):
    data, runs, ups = tmp_path / "data", tmp_path / "runs", tmp_path / "uploads"
    data.mkdir(); ups.mkdir()
    _make_tsf(ups / "demo.tsf")
    svc = AppService(data_dir=data, runs_dir=runs, uploads_dir=ups)
    srv, _ = make_server(svc, 0, token=TOKEN)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()

    class C:
        service = svc

        @staticmethod
        def raw(method, path, body=None, headers=None, host=None):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
            h = dict(headers or {})
            if host:
                h["Host"] = host
            conn.request(method, path, body=body, headers=h)
            r = conn.getresponse()
            data = r.read()
            conn.close()
            return r.status, dict(r.getheaders()), data

        @classmethod
        def call(cls, path, body=None, token=TOKEN):
            hdr = {"X-TSF-Token": token} if token else {}
            if body is not None:
                hdr["Content-Type"] = "application/json"
                body = json.dumps(body).encode()
            st, _, data = cls.raw("POST" if body is not None else "GET", "/api/" + path, body, hdr)
            return st, json.loads(data or b"null")

    yield C
    srv.shutdown()
    srv.server_close()


def _wait(C, job_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st, j = C.call("jobs/" + job_id)
        assert st == 200
        if j["status"] != "running":
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish in time")


CFG = {"dataset": "upload:demo", "engine": "python", "method": "ses", "metrics": ["SMAPE", "MASE", "MAE"]}


# ------------------------------------------------------------------------------------------------ güvenlik
def test_index_embeds_token_and_security_headers(app):
    st, h, body = app.raw("GET", "/")
    assert st == 200 and TOKEN.encode() in body and b"{{TOKEN}}" not in body
    low = {k.lower(): v for k, v in h.items()}
    assert low["x-content-type-options"] == "nosniff" and low["cache-control"] == "no-store"


def test_static_files_served_and_traversal_blocked(app):
    for p in ("/static/app.js", "/static/app.css", "/static/fonts/LMRoman-Regular.woff"):
        st, h, body = app.raw("GET", p)
        assert st == 200 and len(body) > 1000, p
    for p in ("/static/../server.py", "/static/%2e%2e/server.py", "/static/..%2fserver.py", "/static/fonts/../../service.py"):
        st, _, body = app.raw("GET", p)
        assert st == 404 and b"def make_server" not in body, p


def test_api_requires_token(app):
    assert app.call("system", token=None)[0] == 403
    assert app.call("system", token="yanlis")[0] == 403
    assert app.call("system")[0] == 200
    st, _, _ = app.raw("GET", "/api/system?t=" + TOKEN)
    assert st == 200                                   # ?t= is also accepted for download links


def test_host_header_allow_list(app):
    st, _, _ = app.raw("GET", "/", host="evil.example.com")
    assert st == 403
    st, _, _ = app.raw("GET", "/api/system", headers={"X-TSF-Token": TOKEN}, host="evil.example.com:8765")
    assert st == 403
    st, _, _ = app.raw("GET", "/api/system", headers={"X-TSF-Token": TOKEN}, host="localhost:1234")
    assert st == 200


def test_client_cannot_name_server_paths(app):
    for bad in ("../../etc/passwd", "/etc/passwd", "C:\\Windows\\win.ini", "upload:../../secret", "upload:..%2f..%2fsecret"):
        st, j = app.call("dataset/" + urllib.parse.quote(bad, safe=""))
        assert st in (400, 404) and "error" in j, bad
        st, j = app.call("estimate", {"config": {**CFG, "dataset": bad}})
        assert st in (400, 404), bad
    for bad in ("../x", "..%2f..%2fx", "nope"):
        st, _ = app.call("runs/" + bad)
        assert st in (400, 404), bad


# ------------------------------------------------------------------------------------------------ veri ve model
def test_datasets_models_and_detail(app):
    st, d = app.call("datasets")
    ids = {i["id"]: i for i in d["items"]}
    assert st == 200 and ids["upload:demo"]["status"] == "user"
    assert sum(1 for i in d["items"] if i["kind"] == "catalog") == 43          # 41 Monash + 2 M5
    st, det = app.call("dataset/upload%3Ademo")
    assert st == 200 and det["frequency"] == "monthly" and det["horizon"] == 6 and det["stats"]["n_series"] == 30
    assert det["preview"]["profile"]["counts"]["smooth"] == 30
    st, m = app.call("models")
    assert st == 200 and {x["id"] for x in m["models"]} >= {"naive", "ses", "theta", "croston_sba", "pooled_regression"}
    assert m["metric_defaults"] and m["ref_table"]["syntetos2005cat"]["verified"] is True


def test_estimate_and_param_validation(app):
    st, e = app.call("estimate", {"config": CFG})
    assert st == 200 and e["ok"] and e["n_series_run"] == 30 and e["horizon"] == 6 and e["runtime"]
    st, e = app.call("estimate", {"config": {**CFG, "max_series": 5}})
    assert e["n_series_run"] == 5
    assert app.call("estimate", {"config": {**CFG, "params": {"yok_boyle": 1}}})[0] == 400
    assert app.call("estimate", {"config": {**CFG, "method": "yok"}})[0] == 400
    assert app.call("estimate", {"config": {**CFG, "horizon": 0}})[0] == 400
    assert app.call("estimate", {"config": {**CFG, "metrics": ["SAHTE"]}})[0] == 400
    assert app.call("estimate", {"config": {**CFG, "engine": "r", "method": "croston_sba"}})[0] == 400   # R'da yok


def test_params_are_written_explicitly_and_nested():
    from tsfstudio.app import models_info as MI
    out = AppService._params(MI.MODELS["catboost"], "python", {"catboost.depth": 4, "max_obs": 50})
    assert out["catboost"]["depth"] == 4 and out["max_obs"] == 50
    with pytest.raises(Exception):
        AppService._params(MI.MODELS["catboost"], "python", {"catboost.depth": 999})


# ------------------------------------------------------------------------------------------------ koşu
def test_full_run_results_and_downloads(app):
    st, j = app.call("run", {"config": {**CFG, "max_series": 12}})
    assert st == 200 and j["status"] in ("running", "done")
    j = _wait(app, j["id"])
    assert j["status"] == "done", j
    rid = j["run_id"]
    st, d = app.call("runs/" + rid)
    assert st == 200 and d["n_series"] == 12 and d["horizon"] == 6
    summ = {r["Metric"]: r for r in d["summary"]}
    assert set(summ) == {"SMAPE", "MASE", "MAE"} and summ["MASE"]["N_valid"] == 12
    for k in ("zero_forecast_share", "zero_actual_share", "bias_pct", "mean_forecast", "mean_actual"):
        assert k in d["diagnostics"]
    st, names = app.call(f"runs/{rid}/names")
    assert len(names["names"]) == 12
    st, e = app.call(f"runs/{rid}/errors?sort=MASE&dir=desc&limit=5")
    vals = [r["MASE"] for r in e["rows"]]
    assert e["total"] == 12 and vals == sorted(vals, reverse=True) and len(vals) == 5
    st, e = app.call(f"runs/{rid}/errors?q=S1&limit=50")
    assert all("S1" in r["series_name"] for r in e["rows"]) and e["total"] >= 1
    st, h = app.call(f"runs/{rid}/hist?metric=MASE")
    assert sum(h["counts"]) + h.get("clipped", 0) == 12
    st, s = app.call(f"runs/{rid}/series?name=S1")
    assert len(s["forecast"]) == 6 == len(s["test"]) == s["horizon"] and set(s["errors"]) == {"SMAPE", "MASE", "MAE"}
    assert app.call(f"runs/{rid}/series?name=YOK")[0] in (400, 404)
    # files: only the outputs listed in the manifest
    for key in ("summary_csv", "errors_per_series_csv", "manifest"):
        st, h, body = app.raw("GET", f"/api/runs/{rid}/file/{key}?t={TOKEN}")
        assert st == 200 and body and "attachment" in {k.lower(): v for k, v in h.items()}["content-disposition"], key
    for key in ("..%2f..%2fmanifest.json", "yok", "../manifest"):
        st, _, _ = app.raw("GET", f"/api/runs/{rid}/file/{key}?t={TOKEN}")
        assert st in (400, 404), key
    st, runs = app.call("runs")
    assert any(r["run_id"] == rid and r["summary"]["MASE"]["mean"] is not None for r in runs["runs"])


def test_single_concurrent_job_and_cancel(app, monkeypatch):
    from tsfstudio.engines import py_engine
    real = py_engine.forecast_one

    def slow(*a, **k):
        time.sleep(0.03)
        return real(*a, **k)

    monkeypatch.setattr(py_engine, "forecast_one", slow)
    _make_tsf(app.service.uploads_dir / "buyuk.tsf", n=400, name="buyuk")
    st, j = app.call("run", {"config": {**CFG, "dataset": "upload:buyuk"}})
    assert st == 200
    st, j2 = app.call("run", {"config": CFG})
    assert st == 409 and "in progress" in j2["error"]
    time.sleep(0.5)
    st, v = app.call(f"jobs/{j['id']}/cancel", {})
    assert st == 200 and v["cancelling"] is True
    v = _wait(app, j["id"])
    assert v["status"] == "cancelled"
    assert 0 < v["done"] < 400
    st, runs = app.call("runs")
    assert not any(r["dataset"] == "upload:buyuk" for r in runs["runs"])           # a half-finished run does not enter the history


def test_long_run_needs_confirmation(app, monkeypatch):
    from tsfstudio.estimate import PreflightReport

    monkeypatch.setattr(PreflightReport, "long_running", property(lambda self: True))
    st, j = app.call("run", {"config": CFG, "confirm_long": False})
    j = _wait(app, j["id"])
    assert j["status"] == "error" and "Long run" in j["error"]
    st, j = app.call("run", {"config": {**CFG, "max_series": 4}, "confirm_long": True})
    assert _wait(app, j["id"])["status"] == "done"


def test_diagnostics_function():
    f = np.array([[0, 0, 0], [0, 2, 0], [np.nan, 0, 0.0]])
    a = np.array([[1, 0, 0], [0, 3, 1], [5, 0, 0.0]])
    d = AppService.diagnostics(f, a)
    assert d["zero_forecast_share"] == pytest.approx(7 / 8)          # 8 valid points, 7 of them zero forecasts
    assert d["series_all_zero_forecast_share"] == pytest.approx(2 / 3)
    assert d["bias_pct"] == pytest.approx((2 - (1 + 3 + 1)) / 5 * 100)


def test_integer_rounding_diagnostic_flags_zero_forecasts(app):
    """With intermittent demand rounding pulls forecasts to 0; the diagnostics must show this (the car_parts lesson)."""
    rng = np.random.default_rng(5)
    vals = [(rng.random(90) < 0.3).astype(float) for _ in range(20)]
    write_tsf(app.service.uploads_dir / "seyrek.tsf", relation="seyrek", attributes=[("series_name", "string")],
              attr_rows=[(f"Z{i}",) for i in range(20)], values=vals, frequency="monthly", horizon=6, missing=False, equal_length=True)
    out = {}
    for ic in ("on", "off"):
        cfg = {"dataset": "upload:seyrek", "engine": "python", "method": "ses", "integer_conversion": ic}
        j = _wait(app, app.call("run", {"config": cfg})[1]["id"])
        assert j["status"] == "done"
        out[ic] = app.call("runs/" + j["run_id"])[1]["diagnostics"]
    assert out["on"]["series_all_zero_forecast_share"] > out["off"]["series_all_zero_forecast_share"]
    assert out["on"]["mean_forecast"] < out["off"]["mean_forecast"]
    assert out["on"]["zero_forecast_share"] > 0.5


# ------------------------------------------------------------------------------------------------ yükleme
def test_upload_tsf_xlsx_and_rejections(app, tmp_path):
    body = (app.service.uploads_dir / "demo.tsf").read_bytes()
    st, _, raw = app.raw("POST", "/api/upload?name=yeni%20veri.tsf", body, {"X-TSF-Token": TOKEN})
    j = json.loads(raw)
    assert st == 200 and j["ok"] and j["id"].startswith("upload:") and (app.service.uploads_dir / f"{j['id'][7:]}.tsf").is_file()
    st, _, raw = app.raw("POST", "/api/upload?name=demo.tsf", body, {"X-TSF-Token": TOKEN})           # same name: not overwritten
    assert json.loads(raw)["id"] == "upload:demo-2" and (app.service.uploads_dir / "demo.tsf").read_bytes() == body
    st, _, raw = app.raw("POST", "/api/upload?name=../../kotu.tsf", body, {"X-TSF-Token": TOKEN})   # name traversal
    j = json.loads(raw)
    assert j["ok"] and ".." not in j["id"] and not (tmp_path / "kotu.tsf").exists()
    st, _, raw = app.raw("POST", "/api/upload?name=a.exe", b"MZ", {"X-TSF-Token": TOKEN})
    assert json.loads(raw)["ok"] is False
    st, _, raw = app.raw("POST", "/api/upload?name=bos.tsf", b"@relation x\n@data\n", {"X-TSF-Token": TOKEN})
    assert json.loads(raw)["ok"] is False                                                           # @frequency yok
    st, _, tpl = app.raw("GET", f"/api/template.xlsx?t={TOKEN}")
    assert st == 200 and tpl[:2] == b"PK"
    st, _, raw = app.raw("POST", "/api/upload?name=sablon.xlsx", tpl, {"X-TSF-Token": TOKEN})
    j = json.loads(raw)
    assert j["ok"] and j["id"] == "upload:sablon"
    est = app.call("estimate", {"config": {**CFG, "dataset": j["id"]}})[1]
    assert est["ok"]
    st, _, _ = app.raw("POST", "/api/upload?name=x.tsf", body, {})                                     # without token
    assert st == 403


def test_shutdown_requires_token(app):
    assert app.call("shutdown", {}, token="yanlis")[0] == 403
