"""Capture docs screenshots for the critic loop (AI/docs_audit/LOOP.md).

Usage: uv run --with playwright python scripts/capture_docs_pages.py OUT_DIR PATH [PATH ...]
PATH is a site-relative URL such as "" (home), "learn/", "reference/".
Serves ./site statically; run `zensical build` first.
"""

import http.server
import socketserver
import sys
import threading
from functools import partial
from pathlib import Path

from playwright.sync_api import sync_playwright

out = Path(sys.argv[1])
paths = sys.argv[2:]
handler = partial(http.server.SimpleHTTPRequestHandler, directory="site")
handler.log_message = lambda *a, **k: None
srv = socketserver.TCPServer(("127.0.0.1", 0), handler)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()

shots = [
    ("desktop-light", 1440, 900, "light"),
    ("desktop-dark", 1440, 900, "dark"),
    ("mobile", 390, 844, "light"),
]
with sync_playwright() as p:
    browser = p.chromium.launch()
    for path in paths:
        name = path.strip("/").replace("/", "_") or "home"
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        for label, w, h, scheme in shots:
            ctx = browser.new_context(
                viewport={"width": w, "height": h}, color_scheme=scheme
            )
            page = ctx.new_page()
            page.goto(f"http://127.0.0.1:{port}/{path}", wait_until="networkidle")
            page.screenshot(path=d / f"{label}-fold.png")
            page.screenshot(path=d / f"{label}-full.png", full_page=True)
            ctx.close()
    browser.close()
