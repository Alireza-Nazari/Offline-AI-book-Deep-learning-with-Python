#!/usr/bin/env python3
"""Stage 5 - page-by-page educational content generation (preparation phase only).

Sub-commands
  context  [--pages 30,44 | --range 25-80]   write work/context/page_NNN.md generation bundles
  pending  [--range a-b]                     list pages that still lack a valid data/pages/page_NNN.json
  check    <file.json ...>                   validate package files against the schema + traceability rules
  status                                     progress summary (written to reports/generation_report.json)
  api      [--range a-b] [--workers n]       OPTIONAL unattended generation through the Anthropic API
                                             (needs ANTHROPIC_API_KEY; the default workflow in this project
                                             is Claude Code subagents reading the context bundles instead)

The bundle given to the generator contains: the page's extracted elements (with ids), listings, figures
with their labels, equation candidates, the previous/next page text, the chapter outline, the top retrieved
chunks from elsewhere in the book (hybrid BM25 + dense), and the path of the rendered page image, which
the generator should look at to reconstruct formulas, tables and diagrams accurately.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from common import (CHUNKS, CONTEXT, EXTRACTED, INDEXES, PAGE_IMAGES, PAGES, RELATED, REPORTS, ROOT, STRUCTURE,
                    dump_json, load_json, now_iso, page_id, tokenize, validate_package)

GUIDE = ROOT / "scripts" / "GENERATION_GUIDE.md"


def parse_pages(args, n_pages: int) -> list[int]:
    if args.pages:
        return [int(x) for x in args.pages.split(",")]
    if args.range:
        a, b = args.range.split("-")
        return list(range(int(a), int(b) + 1))
    return list(range(1, n_pages + 1))


class Retriever:
    def __init__(self):
        self.chunks = load_json(CHUNKS)
        with open(INDEXES / "keyword_index" / "bm25.pkl", "rb") as f:
            d = pickle.load(f)
        self.bm25 = d["bm25"]
        self.vecs = np.load(INDEXES / "vector_index" / "chunks.npy")
        self._model = None

    def embed(self, text: str) -> np.ndarray:
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir=str(INDEXES / "vector_index" / "model_cache"))
        v = np.asarray(list(self._model.embed([text]))[0], dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    def search(self, query: str, k: int = 8, exclude_pages: set[int] = frozenset(), chapter: str | None = None) -> list[dict]:
        q = tokenize(query)[:500]
        bm = self.bm25.get_scores(q) if q else np.zeros(len(self.chunks))
        sem = self.vecs @ self.embed(query[:2000])
        order_bm = np.argsort(-bm); order_sem = np.argsort(-sem)
        rank_bm = {int(i): r for r, i in enumerate(order_bm[:200])}
        rank_sem = {int(i): r for r, i in enumerate(order_sem[:200])}
        cand = set(rank_bm) | set(rank_sem)
        scored = []
        for i in cand:
            c = self.chunks[i]
            if c["pdf_page"] in exclude_pages:
                continue
            if chapter and c["chapter"] != chapter:
                continue
            s = 1 / (60 + rank_bm.get(i, 999)) + 1 / (60 + rank_sem.get(i, 999))
            scored.append((s, i))
        scored.sort(reverse=True)
        out, seen_pages = [], {}
        for s, i in scored:
            c = self.chunks[i]
            if seen_pages.get(c["pdf_page"], 0) >= 2:       # at most two chunks per page for diversity
                continue
            seen_pages[c["pdf_page"]] = seen_pages.get(c["pdf_page"], 0) + 1
            out.append({**c, "score": round(s, 5)})
            if len(out) >= k:
                break
        return out


def chapter_outline(structure: dict, chapter_title: str) -> str:
    ch = next((c for c in structure["chapters"] if c["title"] == chapter_title), None)
    if not ch:
        return ""
    lines = [f"Chapter: {ch['title']} (PDF pages {ch['start']}-{ch['end']})"]
    for s in ch["sections"]:
        lines.append(f"  - {s['title']} (p.{s['start']})")
        for ss in s["subsections"]:
            lines.append(f"      - {ss['title']} (p.{ss['start']})")
    return "\n".join(lines)


def page_text_brief(p: dict, limit: int = 2500) -> str:
    t = p["body_text"]
    return t if len(t) <= limit else t[:limit] + " …"


def build_context(p: dict, pages: list[dict], structure: dict, related: dict, retr: Retriever) -> str:
    n = p["pdf_page"]
    prev_p = pages[n - 2] if n >= 2 else None
    next_p = pages[n] if n < len(pages) else None
    out = []
    out.append(f"# Generation bundle for {p['page_id']}  (PDF page {n}, printed page {p['printed_page'] or '—'})\n")
    out.append(f"Page kind: {p['page_kind']}\nChapter: {p['chapter']}\nSection: {p['section']}\nSubsection: {p['subsection']}")
    out.append(f"Rendered page image (LOOK AT IT for formulas, figures, tables, layout): {PAGE_IMAGES / (p['page_id'] + '.png')}")
    out.append(f"Output file to write: {PAGES / (p['page_id'] + '.json')}\n")
    out.append("## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)\n")
    for e in p["elements"]:
        if e["kind"] in ("running_head", "page_number"):
            continue
        txt = e["text"]
        if e["kind"] == "code":
            out.append(f"[{e['id']}] ({e['kind']})\n```\n{txt}\n```")
        else:
            out.append(f"[{e['id']}] ({e['kind']}) {txt}")
    if p["code_callouts"]:
        out.append("\n## Code annotations printed next to listings (callouts, in order)\n- " + "\n- ".join(p["code_callouts"]))
    if p["listings"]:
        out.append("\n## Listings\n" + "\n".join(f"- Listing {L['number']}: {L['title']} (code elements {', '.join(L['code_ids']) or 'continues on the next page'})" for L in p["listings"]))
    if p["figures"]:
        out.append("\n## Figures\n" + "\n".join(f"- Figure {f['number']}: {f['caption']}  | labels inside the figure: {', '.join(f['labels']) or '(none - image)'}" for f in p["figures"]))
    if p["tables"]:
        out.append("\n## Table extraction (raw, may be noisy - verify against the page image)\n" + json.dumps(p["tables"], ensure_ascii=False)[:3000])
    if p["eq_candidates"]:
        out.append("\n## Equation candidates (formulas written inline in the text)\n" + "\n".join(f"- [{q['element']}] {q['text']}" for q in p["eq_candidates"]))
    if p["warnings"]:
        out.append("\n## Extraction warnings\n- " + "\n- ".join(p["warnings"]))
    out.append("\n## Chapter outline\n" + chapter_outline(structure, p["chapter"]))
    if prev_p:
        out.append(f"\n## Previous page (PDF {n - 1}) text, for continuity\n" + page_text_brief(prev_p, 1800))
    if next_p:
        out.append(f"\n## Next page (PDF {n + 1}) text, for continuity\n" + page_text_brief(next_p, 1800))
    # Retrieval: related chunks from elsewhere in the book
    query = (p["section"] + " " + p["subsection"] + "\n" + p["body_text"][:3000]).strip()
    hits = retr.search(query, k=8, exclude_pages={n - 1, n, n + 1}) if query else []
    rel = related.get(str(n), [])[:8]
    out.append("\n## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)\n"
               + "\n".join(f"- PDF p.{r['page']} — {r['chapter']} › {r['section']}" for r in rel))
    out.append("\n## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)\n")
    for h in hits:
        out.append(f"### PDF p.{h['pdf_page']} (printed {h['printed_page']}) — {h['chapter']} › {h['section']} › {h['heading']}\n{h['text'][:1200]}\n")
    return "\n".join(out)


def cmd_context(args):
    pages = load_json(EXTRACTED)
    structure = load_json(STRUCTURE)
    related = load_json(RELATED, {})
    retr = Retriever()
    CONTEXT.mkdir(parents=True, exist_ok=True)
    todo = parse_pages(args, len(pages))
    for n in todo:
        out = CONTEXT / f"{page_id(n)}.md"
        if out.exists() and not args.force:
            continue
        out.write_text(build_context(pages[n - 1], pages, structure, related, retr), encoding="utf-8")
    print(f"context bundles written to {CONTEXT} ({len(list(CONTEXT.glob('*.md')))} total)")


# ------------------------------------------------------------------------------------------------
def check_package(path, pages_by_num: dict[int, dict] | None = None) -> list[str]:
    errs = []
    try:
        obj = json.loads(open(path, encoding="utf-8").read())
    except Exception as ex:
        return [f"invalid JSON: {ex}"]
    try:
        pkg = validate_package(obj)
    except Exception as ex:
        return [f"schema: {str(ex)[:800]}"]
    m = re.match(r"page_(\d{3})\.json$", os.path.basename(path))
    if not m or int(m.group(1)) != pkg.pdf_page or pkg.page_id != page_id(pkg.pdf_page):
        errs.append("page_id / pdf_page / filename mismatch")
    if pages_by_num:
        src = pages_by_num.get(pkg.pdf_page)
        if src:
            if pkg.printed_page != src["printed_page"]:
                errs.append(f"printed_page should be {src['printed_page']!r}")
            ids = {e["id"] for e in src["elements"]}
            texts = {e["id"]: e["text"] for e in src["elements"]}
            for par in pkg.paragraphs:
                if par.ref not in ids:
                    errs.append(f"paragraph ref {par.ref} is not an element of this page")
                else:
                    probe = re.sub(r"[^a-z0-9]", "", par.excerpt.lower())[:18]
                    if probe and probe not in re.sub(r"[^a-z0-9]", "", texts[par.ref].lower()):
                        errs.append(f"paragraph {par.ref}: excerpt does not match the element text")
            n_text = sum(1 for e in src["elements"] if e["kind"] in ("body", "bullet", "sidebar", "quote") and len(e["text"].split()) > 25)
            if src["page_kind"] == "content" and pkg.generation.status == "complete" and n_text and len(pkg.paragraphs) < min(n_text, 2):
                errs.append(f"only {len(pkg.paragraphs)} paragraph notes for {n_text} substantial paragraphs")
            code_src = "".join(e["text"] for e in src["elements"] if e["kind"] == "code")
            for c in pkg.code:
                probe = re.sub(r"\s+", "", c.original_code)[:40]
                if probe and probe not in re.sub(r"\s+", "", code_src) and probe not in re.sub(r"\s+", "", src["body_text"]):
                    errs.append(f"code {c.id}: original_code does not match the page's extracted code")
            if src["page_kind"] == "content" and pkg.generation.status == "complete":
                n_code = sum(1 for e in src["elements"] if e["kind"] == "code" and len(e["text"]) > 30)
                if n_code and not pkg.code:
                    errs.append("page has code elements but no code explanations")
                if len(pkg.questions) < 4:
                    errs.append("fewer than 4 review questions")
                if len(pkg.takeaways) < 2:
                    errs.append("fewer than 2 takeaways")
                if len(pkg.summary.split()) < 120:
                    errs.append("summary shorter than 120 words")
    for q in pkg.questions:
        if q.type == "mcq" and (len(q.options) < 3 or q.answer not in q.options):
            errs.append(f"mcq {q.id}: answer must be one of the options")
        if any(not 1 <= r <= 650 for r in q.page_refs):
            errs.append(f"question {q.id}: page_refs out of range")
    WIDGETS = {"gradient_descent", "gradient_descent_1d", "nn_architecture", "backprop_graph", "activation_functions", "loss_functions",
               "softmax_temperature", "convolution", "pooling", "tensor_shapes", "attention", "embedding_space", "rnn_unroll",
               "training_curves", "overfit_underfit", "kfold", "data_split", "matrix_product", "broadcasting", "derivative_tangent",
               "vector_geometry", "confusion_matrix", "residual_block", "transformer_block", "dropout", "learning_rate_schedule",
               "token_sequence", "noise_schedule", "image_tensor"}
    for v in pkg.visualizations:
        if v.type == "widget" and v.widget not in WIDGETS:
            errs.append(f"visualization {v.id}: unknown widget {v.widget!r}")
    for link in (pkg.connections.earlier + pkg.connections.later + pkg.connections.prerequisites
                 + pkg.connections.related_math + pkg.connections.related_code + pkg.connections.related_figures):
        if not 1 <= link.page <= 650:
            errs.append(f"connection page {link.page} out of range")
    for v in pkg.visualizations:
        if v.type == "widget" and not v.widget:
            errs.append(f"visualization {v.id}: widget name missing")
        if v.type == "mermaid" and not v.mermaid:
            errs.append(f"visualization {v.id}: mermaid source missing")
        if v.type == "chart" and not v.chart:
            errs.append(f"visualization {v.id}: chart spec missing")
        if v.type == "svg" and not v.svg:
            errs.append(f"visualization {v.id}: svg missing")
    return errs


def cmd_check(args):
    pages = load_json(EXTRACTED)
    by = {p["pdf_page"]: p for p in pages}
    bad = 0
    for f in args.files:
        errs = check_package(f, by)
        if errs:
            bad += 1
            print(f"FAIL {f}\n  - " + "\n  - ".join(errs))
        else:
            print(f"ok   {f}")
    sys.exit(1 if bad else 0)


def cmd_pending(args):
    pages = load_json(EXTRACTED)
    by = {p["pdf_page"]: p for p in pages}
    todo = parse_pages(args, len(pages))
    pending = []
    for n in todo:
        f = PAGES / f"{page_id(n)}.json"
        if not f.exists() or check_package(f, by):
            pending.append(n)
    print(" ".join(map(str, pending)))
    print(f"{len(pending)} pending of {len(todo)}", file=sys.stderr)


def cmd_status(args):
    pages = load_json(EXTRACTED)
    by = {p["pdf_page"]: p for p in pages}
    ok, failed, missing, review = [], {}, [], []
    stats = {"concepts": 0, "math": 0, "code": 0, "visualizations": 0, "questions": 0, "paragraph_notes": 0}
    for p in pages:
        f = PAGES / f"{p['page_id']}.json"
        if not f.exists():
            missing.append(p["pdf_page"]); continue
        errs = check_package(f, by)
        if errs:
            failed[p["pdf_page"]] = errs; continue
        ok.append(p["pdf_page"])
        pkg = json.load(open(f, encoding="utf-8"))
        if pkg["generation"].get("review_flags") or pkg["generation"].get("confidence") == "low":
            review.append(p["pdf_page"])
        stats["concepts"] += len(pkg["concepts"]); stats["math"] += len(pkg["math"]); stats["code"] += len(pkg["code"])
        stats["visualizations"] += len(pkg["visualizations"]); stats["questions"] += len(pkg["questions"])
        stats["paragraph_notes"] += len(pkg["paragraphs"])
    rep = {"generated_at": now_iso(), "total_pages": len(pages), "generated_valid": len(ok), "failed_validation": failed,
           "missing": missing, "pages_flagged_for_review": review, **stats}
    dump_json(REPORTS / "generation_report.json", rep, indent=2)
    print(json.dumps({k: (v if not isinstance(v, (list, dict)) else len(v)) for k, v in rep.items()}, indent=1))


# ------------------------------------------------------------------------------------------------
def cmd_api(args):
    """Optional: unattended generation through the Anthropic API."""
    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic")
    import base64
    client = anthropic.Anthropic()
    model = args.model
    pages = load_json(EXTRACTED)
    by = {p["pdf_page"]: p for p in pages}
    guide = GUIDE.read_text(encoding="utf-8")
    todo = [n for n in parse_pages(args, len(pages)) if not (PAGES / f"{page_id(n)}.json").exists()
            or check_package(PAGES / f"{page_id(n)}.json", by)]

    def one(n: int):
        ctx = (CONTEXT / f"{page_id(n)}.md").read_text(encoding="utf-8")
        img = base64.b64encode((PAGE_IMAGES / f"{page_id(n)}.png").read_bytes()).decode()
        msg = client.messages.create(
            model=model, max_tokens=16000,
            system=guide,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}},
                {"type": "text", "text": ctx + f"\n\nWrite the complete JSON package for {page_id(n)} now. "
                                          f"Use model name '{model}'. Output only JSON."}]}])
        text = "".join(b.text for b in msg.content if b.type == "text")
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        obj = json.loads(text)
        out = PAGES / f"{page_id(n)}.json"
        dump_json(out, obj)
        errs = check_package(out, by)
        return n, errs

    PAGES.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as ex:
        for fut in as_completed([ex.submit(one, n) for n in todo]):
            try:
                n, errs = fut.result()
                print(f"page {n}: {'ok' if not errs else errs}")
            except Exception as e:
                print(f"error: {e}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("context", "pending", "status", "api"):
        s = sub.add_parser(name)
        s.add_argument("--pages"); s.add_argument("--range"); s.add_argument("--force", action="store_true")
        if name == "api":
            s.add_argument("--workers", type=int, default=4)
            s.add_argument("--model", default="claude-opus-5-5")
    s = sub.add_parser("check"); s.add_argument("files", nargs="+")
    args = ap.parse_args()
    {"context": cmd_context, "pending": cmd_pending, "check": cmd_check, "status": cmd_status, "api": cmd_api}[args.cmd](args)


if __name__ == "__main__":
    main()
