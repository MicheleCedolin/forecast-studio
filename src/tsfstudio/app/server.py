"""Local HTTP server (standard library only). Binds to 127.0.0.1; never exposed to the network.

Protections (even a local app can be probed by another web page open in the browser):
  * The Host header may only be 127.0.0.1/localhost (against DNS rebinding).
  * Every /api request needs the random token generated at start-up (X-TSF-Token or ?t=).
  * No file path is accepted from the client (see service.py).
"""
from __future__ import annotations

import json
import mimetypes
import os
import secrets
import shutil
import sys
import tempfile
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from .service import MAX_UPLOAD, AppError, AppService, clean

STATIC = Path(__file__).resolve().parent / "static"
mimetypes.add_type("font/woff", ".woff")
mimetypes.add_type("text/javascript", ".js")
ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]")


class Handler(BaseHTTPRequestHandler):
    server_version = "ForecastStudio"
    service: AppService
    token: str

    # --------------------------------------------------------------------------- yardımcılar
    def log_message(self, fmt, *args):  # noqa: D401 - sessiz
        if os.environ.get("TSFSTUDIO_APP_LOG"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(clean(obj), ensure_ascii=False, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

    def _err(self, code: int, msg: str, detail: str | None = None) -> None:
        self._json({"error": msg, "detail": detail}, code)

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").strip().lower()
        name = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
        return name in ALLOWED_HOSTS

    def _auth_ok(self, qs: dict) -> bool:
        tok = self.headers.get("X-TSF-Token") or (qs.get("t") or [""])[0]
        return secrets.compare_digest(tok, self.token)

    def _body_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 2_000_000:
            raise AppError("Request body too large.", 413)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw.decode("utf-8") or "{}")
        except ValueError:
            raise AppError("Invalid JSON.") from None

    # --------------------------------------------------------------------------- yönlendirme
    def do_GET(self):  # noqa: N802
        self._dispatch("GET")

    def do_POST(self):  # noqa: N802
        self._dispatch("POST")

    def _discard_body(self, limit: int = 1 << 20) -> None:
        """Read and drop an unread request body before an early error reply. On Windows, closing a socket that still has
        unread data sends a reset and the client can lose the response (seen as WinError 10053 in the tests)."""
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if 0 < n <= limit:
            try:
                self.rfile.read(n)
            except OSError:
                pass
        elif n > limit:
            self.close_connection = True

    def _dispatch(self, method: str) -> None:
        try:
            if not self._host_ok():
                self._discard_body()
                return self._err(403, "Invalid Host header.")
            u = urlparse(self.path)
            path, qs = u.path, parse_qs(u.query)
            if method == "GET" and (path == "/" or path == "/index.html"):
                return self._index()
            if method == "GET" and path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if not path.startswith("/api/"):
                self._discard_body()
                return self._err(404, "Not found.")
            if not self._auth_ok(qs):
                self._discard_body()
                return self._err(403, "Invalid session token. Reload the page.")
            parts = [unquote(p) for p in path[len("/api/"):].split("/")]
            self._api(method, parts, qs)
        except AppError as exc:
            self._err(exc.status, str(exc))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):   # client went away (WinError 10053)
            pass
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            try:
                self.service.runs_dir.mkdir(parents=True, exist_ok=True)
                with open(self.service.runs_dir / "_app_errors.log", "a", encoding="utf-8") as fh:
                    fh.write(f"\n=== {self.command} {self.path}\n{tb}")
            except OSError:
                pass
            self._err(500, f"Unexpected error: {type(exc).__name__}: {exc}", tb.splitlines()[-1])

    def _index(self) -> None:
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        from .. import __version__
        html = html.replace("{{TOKEN}}", self.token).replace("{{VERSION}}", __version__)
        self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")

    def _static(self, rel: str) -> None:
        p = (STATIC / rel).resolve()
        if STATIC.resolve() not in p.parents or not p.is_file():
            return self._err(404, "Not found.")
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/json", "image/svg+xml"):
            ctype += "; charset=utf-8"
        self._send(200, p.read_bytes(), ctype)

    def _api(self, method: str, parts: list[str], qs: dict) -> None:
        s = self.service
        q = lambda k, d=None: (qs.get(k) or [d])[0]  # noqa: E731
        head = parts[0]
        if method == "GET":
            if parts == ["system"]:
                return self._json(s.system(refresh=q("refresh") == "1"))
            if parts == ["models"]:
                return self._json(s.models())
            if parts == ["datasets"]:
                return self._json(s.datasets())
            if head == "dataset" and len(parts) == 2:
                return self._json(s.dataset_detail(parts[1]))
            if parts == ["runs"]:
                return self._json(s.runs())
            if head == "runs" and len(parts) >= 2:
                rid = parts[1]
                if len(parts) == 2:
                    return self._json(s.run_detail(rid))
                sub = parts[2]
                if sub == "errors":
                    return self._json(s.run_errors(rid, q("sort"), q("dir") == "desc", q("q", ""), int(q("offset", 0)), int(q("limit", 25))))
                if sub == "hist":
                    return self._json(s.run_hist(rid, q("metric", ""), int(q("bins", 28))))
                if sub == "series":
                    return self._json(s.run_series(rid, q("name", "")))
                if sub == "names":
                    return self._json({"names": s.run_series_names(rid)})
                if sub == "file" and len(parts) == 4:
                    p, dl = s.run_file(rid, parts[3])
                    ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
                    if p.suffix == ".xlsx":
                        ctype = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    return self._send(200, p.read_bytes(), ctype, {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(dl)}"})
            if head == "jobs" and len(parts) == 2:
                return self._json(s.job(parts[1]))
            if parts == ["features"]:
                return self._json(s.features(self._feat_query(qs)))
            if parts == ["features", "file"]:
                body, name = s.features_file(self._feat_query(qs))
                return self._send(200, body, "text/csv; charset=utf-8",
                                  {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"})
            if parts == ["template.xlsx"]:
                return self._send(200, s.template_xlsx(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                  {"Content-Disposition": "attachment; filename*=UTF-8''forecast_studio_template.xlsx"})
        else:
            if parts == ["estimate"]:
                body = self._body_json()
                return self._json(s.estimate(body.get("config") or {}, probe=bool(body.get("probe"))))
            if parts == ["run"]:
                body = self._body_json()
                return self._json(s.start_run(body.get("config") or {}, confirm_long=bool(body.get("confirm_long"))))
            if parts == ["features"]:
                return self._json(s.start_features(self._body_json()))
            if head == "jobs" and len(parts) == 3 and parts[2] == "cancel":
                return self._json(s.cancel_job(parts[1]))
            if parts == ["upload"]:
                return self._upload(qs)
            if parts == ["shutdown"]:
                self._json({"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
        raise AppError("Not found.", 404)

    @staticmethod
    def _feat_query(qs: dict) -> dict:
        g = lambda k: (qs.get(k) or [""])[0]  # noqa: E731
        return {"dataset": g("dataset"), "sets": g("sets"), "study_fft": g("study") == "1"}

    def _upload(self, qs: dict) -> None:
        name = (qs.get("name") or [""])[0]
        n = int(self.headers.get("Content-Length") or 0)
        if not name:
            raise AppError("File name is missing.")
        if n <= 0:
            raise AppError("The file is empty.")
        if n > MAX_UPLOAD:
            raise AppError(f"File too large (at most {MAX_UPLOAD // 1_000_000} MB).", 413)
        fd, tmp = tempfile.mkstemp(suffix=Path(name).suffix.lower(), prefix="tsfs_up_")
        tmp_path = Path(tmp)
        try:
            with os.fdopen(fd, "wb") as fh:
                left = n
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    fh.write(chunk)
                    left -= len(chunk)
            if left > 0:
                raise AppError("Upload was interrupted.")
            res = self.service.upload(name, tmp_path, fill_gaps=(qs.get("fill_gaps") or ["0"])[0] == "1")
            self._json(res)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    shutil.rmtree(tmp_path, ignore_errors=True)


def make_server(service: AppService, port: int = 8765, token: str | None = None) -> tuple[ThreadingHTTPServer, str]:
    token = token or secrets.token_urlsafe(18)
    handler = type("BoundHandler", (Handler,), {"service": service, "token": token})
    ThreadingHTTPServer.daemon_threads = True
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    return srv, token


def serve(service: AppService, port: int = 8765, open_browser: bool = True) -> int:
    srv = None
    for p in range(port, port + 11):
        try:
            srv, token = make_server(service, p)
            port = p
            break
        except OSError:
            continue
    if srv is None:
        print(f"ERROR: ports {port}-{port + 10} are all in use. Try another port with `--port`.")
        return 2
    url = f"http://127.0.0.1:{port}/"
    print(f"Forecast Studio interface: {url}")
    print(f"  data: {service.data_dir.resolve()}\n  runs: {service.runs_dir.resolve()}\n  uploads: {service.uploads_dir.resolve()}")
    print("To stop, press Ctrl+C in this window or use the 'Quit application' button in the interface.")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    print("Stopped.")
    return 0
