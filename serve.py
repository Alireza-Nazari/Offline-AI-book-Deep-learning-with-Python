#!/usr/bin/env python3
"""Tiny local server for the companion (no internet, no AI, no dependencies).

Browsers block fetch()/XHR on file:// URLs, which PDF.js and the JSON data need, so the app is served
from http://127.0.0.1:<port>/ instead. It also stores your highlights and notes:
GET/PUT /api/annotations  <->  data/user/annotations.json   (highlights & notes)
GET/PUT /api/study        <->  data/user/study.json         (review deck & progress)
(loopback only, same-origin only).

Usage:  python3 serve.py [port] [--no-browser] [--user-dir DIR] [--annotations PATH]      (default port 8765)
"""
import http.server
import json
import os
import socketserver
import sys
import tempfile
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
ARGS = sys.argv[1:]
PORT = next((int(a) for a in ARGS if a.isdigit()), 8765)
USER_DIR = os.path.join(ROOT, "data", "user")
if "--user-dir" in ARGS:
    USER_DIR = os.path.abspath(ARGS[ARGS.index("--user-dir") + 1])
USER_FILES = {"/api/annotations": os.path.join(USER_DIR, "annotations.json"),   # highlights & notes
              "/api/study": os.path.join(USER_DIR, "study.json")}               # review deck & progress
if "--annotations" in ARGS:
    USER_FILES["/api/annotations"] = os.path.abspath(ARGS[ARGS.index("--annotations") + 1])
MAX_BODY = 20 * 1024 * 1024


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".js": "application/javascript",
                      ".mjs": "application/javascript", ".json": "application/json", ".pdf": "application/pdf",
                      ".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf"}

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):  # quiet
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _same_origin(self):
        # Refuse writes from other web pages: only this app's own origin may save.
        allowed = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}
        if self.headers.get("Host") not in allowed:
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin in {f"http://{h}" for h in allowed}

    def do_GET(self):
        target = USER_FILES.get(self.path.split("?")[0])
        if target:
            try:
                with open(target, encoding="utf-8") as f:
                    return self._json(200, json.load(f))
            except FileNotFoundError:
                return self._json(200, {"version": 1, "items": []} if target.endswith("annotations.json") else {"version": 1})
            except (OSError, ValueError) as e:
                return self._json(500, {"error": str(e)})
        return super().do_GET()

    def do_PUT(self):
        target = USER_FILES.get(self.path.split("?")[0])
        if not target:
            return self._json(404, {"error": "not found"})
        if not self._same_origin() or "application/json" not in (self.headers.get("Content-Type") or ""):
            return self._json(403, {"error": "forbidden"})
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BODY:
            return self._json(413, {"error": "bad size"})
        try:
            data = json.loads(self.rfile.read(n))
            if not isinstance(data, dict) or (target == USER_FILES["/api/annotations"] and not isinstance(data.get("items"), list)):
                raise ValueError("expected a JSON object (annotations need {items: [...]})")
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        os.makedirs(os.path.dirname(target), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(target), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, target)  # atomic: a crash never leaves a half-written file
        return self._json(200, {"ok": True})

    do_POST = do_PUT


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


if __name__ == "__main__":
    with Server(("127.0.0.1", PORT), Handler) as httpd:
        url = f"http://127.0.0.1:{PORT}/"
        print(f"Book companion: {url}   (Ctrl+C to stop)  ·  your notes & progress are saved in {USER_DIR}")
        if "--no-browser" not in ARGS:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
