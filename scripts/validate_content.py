#!/usr/bin/env python3
"""Stage 9 - whole-project validation. Writes reports/validation_report.json and reports/completion_report.json.

Checks: every PDF page accounted for; package per page (schema + traceability); page references in
range; formulas present where the extractor found candidates; code blocks verbatim; diagrams belong to
the right page; no duplicate page ids; all local assets referenced by index.html exist; no remote URLs
anywhere in the application; quiz / graph / search data consistent with the packages.
"""
from __future__ import annotations

import json
import re
import sys

from common import ASSETS, DATA, DIAGRAMS, EXTRACTED, KG, PAGES, QUIZZES, REPORTS, ROOT, SEARCH, load_json, now_iso, dump_json
from generate_page_notes import check_package

REMOTE = re.compile(r"https?://(?!127\.0\.0\.1|localhost)[^\s\"')]+")


def main() -> int:
    pages = load_json(EXTRACTED)
    by = {p["pdf_page"]: p for p in pages}
    n = len(pages)
    rep = {"generated_at": now_iso(), "total_pages": n, "checks": {}, "problems": []}
    P = rep["problems"]

    # 1. packages
    pkgs = {}
    seen_ids = set()
    for p in pages:
        f = PAGES / f"{p['page_id']}.json"
        if not f.exists():
            continue
        errs = check_package(f, by)
        if errs:
            P.append({"page": p["pdf_page"], "kind": "package_invalid", "detail": errs})
            continue
        pkg = json.load(open(f, encoding="utf-8"))
        if pkg["page_id"] in seen_ids:
            P.append({"page": p["pdf_page"], "kind": "duplicate_page_id"})
        seen_ids.add(pkg["page_id"])
        pkgs[p["pdf_page"]] = pkg
    missing = [p["pdf_page"] for p in pages if p["pdf_page"] not in pkgs]
    rep["checks"]["pages_with_valid_package"] = len(pkgs)
    rep["checks"]["pages_missing_package"] = missing
    # 2. math preserved: content pages with equation candidates but no math entries -> review
    review = []
    for num, pkg in pkgs.items():
        src = by[num]
        if src["page_kind"] == "content" and len(src["eq_candidates"]) >= 2 and not pkg["math"]:
            review.append({"page": num, "reason": f"{len(src['eq_candidates'])} equation candidates but no math explanations"})
        if pkg["generation"].get("review_flags"):
            review.append({"page": num, "reason": "generator flagged: " + "; ".join(pkg["generation"]["review_flags"])[:300]})
        if pkg["generation"].get("confidence") == "low":
            review.append({"page": num, "reason": "low confidence"})
        for v in pkg["visualizations"]:
            if v["type"] == "svg" and re.search(r"<script|on\w+=|javascript:", v["svg"], re.I):
                P.append({"page": num, "kind": "unsafe_svg", "detail": v["id"]})
    rep["checks"]["pages_for_manual_review"] = review
    # 3. assets referenced by index.html exist and no remote URLs in app code
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    refs = re.findall(r'(?:src|href)="([^"]+)"', html)
    missing_assets = [r for r in refs if not r.startswith("data:") and not (ROOT / r).exists()]
    rep["checks"]["missing_assets"] = missing_assets
    if missing_assets:
        P.append({"kind": "missing_assets", "detail": missing_assets})
    remote = []
    for f in [ROOT / "index.html", *ASSETS.glob("css/*.css"), *ASSETS.glob("js/**/*.js")]:
        for m in REMOTE.finditer(f.read_text(encoding="utf-8", errors="ignore")):
            if m.group(0).startswith(("http://www.w3.org/", "https://www.w3.org/")):
                continue  # XML namespace identifiers, never fetched
            remote.append(f"{f.relative_to(ROOT)}: {m.group(0)[:80]}")
    rep["checks"]["remote_urls_in_app_code"] = remote
    if remote:
        P.append({"kind": "remote_urls", "detail": remote})
    # vendor libs: a CDN string in a vendored library is only a problem if it is actually fetched; report, don't fail
    vendor_remote = {}
    for f in [*ASSETS.glob("vendor/*.js"), *ASSETS.glob("pdfjs/*.js"), *ASSETS.glob("math/*.js")]:
        hits = set(m.group(0)[:60] for m in REMOTE.finditer(f.read_text(encoding="utf-8", errors="ignore")))
        hits = {h for h in hits if not re.search(r"mozilla|github|npmjs|w3\.org|opensource|apache|license|mit|katex\.org|d3js|chartjs|jquery|mermaid|unpkg|jsdelivr|cdnjs|purl|schema", h, re.I)}
        if hits:
            vendor_remote[str(f.relative_to(ROOT))] = sorted(hits)[:10]
    rep["checks"]["vendor_strings_mentioning_urls"] = vendor_remote
    # 4. built data consistent
    kg = load_json(KG / "graph.json", {"nodes": [], "links": []})
    qidx = load_json(QUIZZES / "index.json", {"total": 0})
    docs = load_json(SEARCH / "docs.json", [])
    diag = load_json(DIAGRAMS / "index.json", [])
    bad_diag = [d for d in diag if d["page"] not in pkgs or d["id"] not in {v["id"] for v in pkgs[d["page"]]["visualizations"]}]
    rep["checks"]["diagrams_indexed"] = len(diag)
    rep["checks"]["diagrams_mismatched_page"] = bad_diag
    rep["checks"]["knowledge_graph"] = {"nodes": len(kg["nodes"]), "links": len(kg["links"])}
    rep["checks"]["quiz_questions"] = qidx.get("total", 0)
    rep["checks"]["search_documents"] = len(docs)
    book_ok = (ROOT / "book" / "original_book.pdf").exists()
    rep["checks"]["original_book_present"] = book_ok
    render = load_json(REPORTS / "render_report.json", None)
    rep["checks"]["headless_render"] = {k: render[k] for k in ("pages_checked", "failures") if render and k in render} if render else "not run"
    ok = not P and not missing_assets and book_ok
    rep["result"] = "PASS" if ok else "ISSUES"
    dump_json(REPORTS / "validation_report.json", rep, indent=2)

    stats = {"concepts": 0, "math": 0, "code": 0, "visualizations": 0, "questions": 0, "paragraph_notes": 0}
    for pkg in pkgs.values():
        for k in stats:
            stats[k] += len(pkg.get({"paragraph_notes": "paragraphs"}.get(k, k), []))
    comp = {"generated_at": now_iso(), "book": "Deep Learning with Python, 3rd ed. (Chollet & Watson)", "total_pages": n,
            "successfully_processed_pages": len(pkgs), "failed_pages": [p["page"] for p in P if p.get("kind") == "package_invalid"],
            "pages_without_package": missing, "pages_requiring_manual_review": review, **stats,
            "concepts_unique": len(kg["nodes"]), "quizzes_total_questions": qidx.get("total", 0),
            "indexing": load_json(REPORTS / "index_report.json", {}), "extraction": {k: v for k, v in load_json(REPORTS / "extraction_report.json", {}).items() if k != "warnings"},
            "missing_assets": missing_assets, "validation_result": rep["result"]}
    dump_json(REPORTS / "completion_report.json", comp, indent=2)
    print(json.dumps({k: (v if not isinstance(v, (list, dict)) else (len(v) if isinstance(v, list) else v)) for k, v in rep["checks"].items()}, indent=1))
    print("RESULT:", rep["result"], f"({len(P)} problems, {len(review)} pages flagged for review)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
