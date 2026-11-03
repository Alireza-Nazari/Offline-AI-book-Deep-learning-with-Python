#!/usr/bin/env python3
"""Tiny local static server for the companion (no internet, no AI, no dependencies).

Browsers block fetch()/XHR on file:// URLs, which PDF.js and the JSON data need, so the app is served
from http://127.0.0.1:<port>/ instead. Usage:  python3 serve.py [port]   (default 8765)
"""
import http.server
import os
import socketserver
import sys
import webbrowser

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
ROOT = os.path.dirname(os.path.abspath(__file__))


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


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


if __name__ == "__main__":
    with Server(("127.0.0.1", PORT), Handler) as httpd:
        url = f"http://127.0.0.1:{PORT}/"
        print(f"Book companion: {url}   (Ctrl+C to stop)")
        if "--no-browser" not in sys.argv:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
