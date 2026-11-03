#!/usr/bin/env python3
"""Stage 6 - visualization verification.

Visualizations are *declared* inside each page package (widget + params, Mermaid source, Chart.js config
or inline SVG) by the generation stage, and rendered by assets/js/widgets/widgets.js at runtime. This
script renders every declared visualization and every KaTeX formula in a headless Chromium (Playwright)
against the local server, and records any that throw, so that broken diagrams are caught before delivery.

Usage: python scripts/generate_visualizations.py [--pages 61,62] [--port 8766]
Writes reports/render_report.json. Requires `playwright` (already installed in the project venv).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from common import PAGES, REPORTS, ROOT, dump_json, load_json, now_iso

HARNESS = """
<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="/assets/math/katex.min.css"><link rel="stylesheet" href="/assets/css/app.css"></head>
<body><div id="out"></div>
<script src="/assets/math/katex.min.js"></script><script src="/assets/math/auto-render.min.js"></script>
<script src="/assets/vendor/marked.min.js"></script><script src="/assets/vendor/purify.min.js"></script>
<script src="/assets/vendor/chart.umd.js"></script><script src="/assets/vendor/d3.min.js"></script>
<script src="/assets/vendor/mermaid.min.js"></script><script src="/assets/js/widgets/widgets.js"></script>
<script>
window.__results = [];
async function runPage(pkg) {
  const out = document.getElementById('out');
  for (const v of pkg.visualizations || []) {
    const box = document.createElement('div'); box.className = 'viz widget'; out.append(box);
    const rec = {page: pkg.pdf_page, id: v.id, type: v.type, widget: v.widget || null, ok: true, error: null};
    try {
      if (v.type === 'widget') {
        if (!window.Widgets[v.widget]) throw new Error('unknown widget ' + v.widget);
        window.Widgets[v.widget](box, v.params || {}, {});
        if (!box.querySelector('svg, table, canvas, .grid, .info')) throw new Error('widget rendered nothing');
        // exercise controls once
        for (const inp of box.querySelectorAll('input[type=range]')) { inp.value = inp.min; inp.dispatchEvent(new Event('input')); }
        for (const b of [...box.querySelectorAll('button')].slice(0, 3)) b.click();
      } else if (v.type === 'mermaid') {
        mermaid.initialize({startOnLoad: false, securityLevel: 'strict'});
        const {svg} = await mermaid.render('m' + Math.random().toString(36).slice(2), v.mermaid);
        if (!svg.includes('<svg')) throw new Error('no svg');
      } else if (v.type === 'chart') {
        const c = document.createElement('canvas'); box.append(c);
        const cfg = JSON.parse(JSON.stringify(v.chart)); cfg.options = Object.assign({}, cfg.options, {animation: false, responsive: false});
        const ch = new Chart(c, cfg); ch.destroy();
      } else if (v.type === 'svg') {
        box.innerHTML = DOMPurify.sanitize(v.svg, {USE_PROFILES: {svg: true}});
        if (!box.querySelector('svg')) throw new Error('svg did not parse');
      }
    } catch (e) { rec.ok = false; rec.error = String(e && e.message || e).slice(0, 300); }
    window.__results.push(rec);
  }
  for (const m of pkg.math || []) {
    const rec = {page: pkg.pdf_page, id: m.id, type: 'katex', ok: true, error: null};
    try { katex.renderToString(m.latex, {displayMode: true, throwOnError: true}); } catch (e) { rec.ok = false; rec.error = String(e.message).slice(0, 300); }
    window.__results.push(rec);
  }
}
</script></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright

    harness = ROOT / "work" / "render_harness.html"
    harness.write_text(HARNESS, encoding="utf-8")
    files = sorted(PAGES.glob("page_*.json"))
    if args.pages:
        want = {int(x) for x in args.pages.split(",")}
        files = [f for f in files if int(f.stem.split("_")[1]) in want]
    srv = subprocess.Popen([sys.executable, str(ROOT / "serve.py"), str(args.port), "--no-browser"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    results = []
    try:
        with sync_playwright() as pw:
            try:
                b = pw.chromium.launch(channel="chrome")       # installed Google Chrome
            except Exception:
                b = pw.chromium.launch()
            pg = b.new_page()
            errors = []
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(f"http://127.0.0.1:{args.port}/work/render_harness.html")
            for i, f in enumerate(files):
                pkg = json.load(open(f, encoding="utf-8"))
                if not pkg.get("visualizations") and not pkg.get("math"):
                    continue
                pg.evaluate("pkg => runPage(pkg)", pkg)
                if i % 40 == 39:
                    results.extend(pg.evaluate("window.__results.splice(0)"))
                    pg.evaluate("document.getElementById('out').innerHTML = ''")
            results.extend(pg.evaluate("window.__results.splice(0)"))
            b.close()
    finally:
        srv.terminate()
    failures = [r for r in results if not r["ok"]]
    rep = {"generated_at": now_iso(), "pages_checked": len(files), "items_rendered": len(results), "failures": failures,
           "by_type": {t: sum(1 for r in results if r["type"] == t) for t in {r["type"] for r in results}},
           "uncaught_page_errors": errors[:20]}
    dump_json(REPORTS / "render_report.json", rep, indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k != "failures"}, indent=1))
    for f in failures:
        print("FAIL", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
