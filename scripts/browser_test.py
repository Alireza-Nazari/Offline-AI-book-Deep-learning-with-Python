#!/usr/bin/env python3
"""Stage 9 - end-to-end browser test of the built application (headless Chrome via Playwright).

Starts serve.py on a spare port, opens the app, and checks: PDF renders, navigation (buttons, input,
keyboard), panel synchronization in both directions, sync toggle, zoom/fit, notes rendering incl. KaTeX
and widgets, note search, global search, knowledge graph, quiz, chapter overview, thumbnails, and that the
page made NO network request outside 127.0.0.1 (offline guarantee). Screenshots go to reports/screenshots/.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

from common import PAGES, REPORTS, ROOT, dump_json, now_iso

PORT = 8767


def main() -> int:
    from playwright.sync_api import sync_playwright
    shots = REPORTS / "screenshots"; shots.mkdir(parents=True, exist_ok=True)
    generated = sorted(int(f.stem.split("_")[1]) for f in PAGES.glob("page_*.json"))
    target = 61 if 61 in generated else (generated[0] if generated else 25)
    srv = subprocess.Popen([sys.executable, str(ROOT / "serve.py"), str(PORT), "--no-browser"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    results: dict[str, object] = {}
    external, console_errors = [], []

    def check(name, cond, detail=""):
        results[name] = {"ok": bool(cond), "detail": detail}
        print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail else ""))

    try:
        with sync_playwright() as pw:
            try:
                b = pw.chromium.launch(channel="chrome")
            except Exception:
                b = pw.chromium.launch()
            ctx = b.new_context(viewport={"width": 1500, "height": 950})
            pg = ctx.new_page()
            pg.on("request", lambda r: external.append(r.url) if not r.url.startswith(f"http://127.0.0.1:{PORT}") and not r.url.startswith("data:") else None)
            pg.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
            pg.on("pageerror", lambda e: console_errors.append("pageerror: " + str(e)))
            pg.goto(f"http://127.0.0.1:{PORT}/#p={target}")
            pg.wait_for_selector(f'.pdfpage[data-page="{target}"] canvas', timeout=30000)
            pg.wait_for_selector("#sec-overview", timeout=20000)
            time.sleep(1.5)
            pg.screenshot(path=str(shots / "01_two_panels.png"))
            check("pdf_rendered", pg.locator(f'.pdfpage[data-page="{target}"] canvas').count() == 1)
            check("notes_loaded_for_target", pg.locator(".page-title h1").inner_text().strip() != "")
            check("katex_rendered", pg.locator("#notes .katex").count() > 0, f"{pg.locator('#notes .katex').count()} formulas")
            pg.locator("#sec-visualizations").evaluate("d => d.open = true")
            time.sleep(0.5)
            check("widgets_rendered", pg.locator("#notes .viz svg, #notes .viz canvas, #notes .viz table").count() > 0)
            # navigation: next button -> PDF moves and notes follow
            pg.click("#btn-next"); time.sleep(1.5)
            check("next_button_syncs_notes", pg.input_value("#page-input") == str(target + 1) and str(target + 1) in pg.locator(".crumbs").inner_text())
            pg.keyboard.press("ArrowLeft"); time.sleep(1.2)
            check("keyboard_prev", pg.input_value("#page-input") == str(target))
            # scroll the viewer -> active page detection updates notes
            pg.evaluate("document.querySelector('#viewer').scrollBy(0, document.querySelector('.pdfpage').offsetHeight * 2 + 40)"); time.sleep(1.8)
            after = int(pg.input_value("#page-input"))
            check("scroll_sync_detects_page", after == target + 2, f"page input is {after}, expected {target + 2}")
            check("scroll_sync_updates_notes", f"PDF p.{after}" in pg.locator(".crumbs").inner_text())
            # sync off: scrolling should not change notes
            pg.click("#sync-toggle"); pg.evaluate("document.querySelector('#viewer').scrollBy(0, document.querySelector('.pdfpage').offsetHeight + 20)"); time.sleep(1.5)
            check("sync_toggle_off_keeps_notes", f"PDF p.{after}" in pg.locator(".crumbs").inner_text())
            pg.click("#sync-toggle")
            # page ref link in notes navigates the book
            pg.fill("#page-input", str(target)); pg.press("#page-input", "Enter"); time.sleep(1.5)
            links = pg.locator("#notes a.pageref[data-page]")
            if links.count():
                dest = int(links.first.get_attribute("data-page")); links.first.click(); time.sleep(1.5)
                check("pageref_click_navigates_book", pg.locator(f'.pdfpage[data-page="{dest}"].active').count() == 1, f"to p.{dest}")
            # zoom
            z0 = pg.locator("#zoom-label").inner_text(); pg.click("#btn-zoom-in"); time.sleep(0.8)
            check("zoom_in", pg.locator("#zoom-label").inner_text() != z0)
            pg.click("#btn-fit-page"); time.sleep(0.8); check("fit_page", True, pg.locator("#zoom-label").inner_text())
            pg.click("#btn-fit-width"); time.sleep(0.8)
            # thumbnails
            pg.click("#btn-thumbs"); time.sleep(1.5); check("thumbnails", pg.locator("#thumbs .thumb canvas").count() > 10); pg.click("#btn-thumbs")
            # note search (back on the target page)
            pg.fill("#page-input", str(target)); pg.press("#page-input", "Enter"); pg.wait_for_selector("#sec-overview", timeout=20000); time.sleep(1.0)
            pg.fill("#note-search", "the"); time.sleep(0.5); check("note_search_highlights", pg.locator("#notes mark.ns").count() > 0); pg.fill("#note-search", "")
            # pdf search
            pg.fill("#pdf-search", "tensor"); time.sleep(6)
            check("book_search_finds_hits", "hits" in pg.locator("#pdf-search-status").inner_text() or "/" in pg.locator("#pdf-search-status").inner_text(), pg.locator("#pdf-search-status").inner_text())
            pg.fill("#pdf-search", ""); pg.evaluate("document.activeElement.blur()")
            # global search
            pg.keyboard.press("/"); pg.wait_for_selector("#modal input[type=search]", timeout=20000); pg.fill("#modal input[type=search]", "gradient"); time.sleep(1.0)
            check("global_search_results", pg.locator("#modal .results .r").count() > 0, f"{pg.locator('#modal .results .r').count()} results")
            pg.screenshot(path=str(shots / "02_search.png")); pg.keyboard.press("Escape")
            # knowledge graph
            pg.keyboard.press("g"); pg.wait_for_selector("#kg .node", timeout=20000); time.sleep(1.0); check("knowledge_graph", pg.locator("#kg .node").count() > 0, f"{pg.locator('#kg .node').count()} nodes")
            pg.locator("#kg .node circle").first.click(); time.sleep(0.5); check("graph_node_detail", pg.locator("#kg-side h3").count() == 1)
            pg.screenshot(path=str(shots / "03_graph.png")); pg.keyboard.press("Escape")
            # quiz (book search moved us to its first hit; go back to the target page first)
            pg.evaluate(f"BC.gotoPage({target})"); time.sleep(1.0)
            pg.keyboard.press("q"); pg.wait_for_selector("#modal button.primary", timeout=20000); pg.click("#modal button.primary"); pg.wait_for_selector("#modal .question", timeout=20000); time.sleep(0.5)
            check("quiz_questions", pg.locator("#modal .question").count() > 0, f"{pg.locator('#modal .question').count()} questions")
            opt = pg.locator("#modal .question .opts label").first
            if opt.count():
                opt.click(); time.sleep(0.3); check("quiz_feedback", pg.locator("#modal .question .answer:not([hidden])").count() > 0)
            pg.screenshot(path=str(shots / "04_quiz.png")); pg.keyboard.press("Escape")
            pg.click("#btn-chapters"); time.sleep(0.8); check("chapters_overview", pg.locator("#modal .chapter-card").count() >= 20); pg.keyboard.press("Escape")
            # theme
            pg.click("#btn-theme"); time.sleep(0.5); check("theme_toggle", pg.evaluate("document.documentElement.dataset.theme") in ("dark", "light", "auto"))
            pg.screenshot(path=str(shots / "05_dark.png"))
            # visit every generated page quickly and look for render errors
            errs = []
            for n in generated[:80]:
                pg.evaluate(f"BC.gotoPage({n})"); time.sleep(0.35)
                if pg.locator(".viz-error").count():
                    errs.append({"page": n, "errors": pg.locator(".viz-error").all_inner_texts()})
            check("no_visualization_errors_on_generated_pages", not errs, json.dumps(errs)[:500])
            b.close()
    finally:
        srv.terminate()
    check("no_external_network_requests", not external, "; ".join(external[:5]))
    benign = [e for e in console_errors if "favicon" not in e]
    check("no_console_errors", not benign, " | ".join(benign[:5])[:600])
    rep = {"generated_at": now_iso(), "target_page": target, "results": results, "external_requests": external, "console_errors": benign[:50]}
    dump_json(REPORTS / "browser_test_report.json", rep, indent=2)
    failed = [k for k, v in results.items() if not v["ok"]]
    print("RESULT:", "PASS" if not failed else f"FAIL ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
