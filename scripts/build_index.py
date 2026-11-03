#!/usr/bin/env python3
"""Stage 4 - hierarchical, hybrid knowledge index.

Level 1  structure      book > chapter > section > subsection > page      (work/structure.json, metadata.db)
Level 2  semantic chunks element-aware chunks that never split a code listing, bullet list, sidebar
                        or paragraph; long paragraphs are split at sentence boundaries with a small
                        overlap                                             (work/chunks.json, metadata.db)
Level 3  relationships   per-page top-k related pages (hybrid score)        (work/related.json)
Level 4  retrieval       BM25 (rank-bm25, pickled) + dense vectors (fastembed BAAI/bge-small-en-v1.5,
                        numpy matrix - FAISS is unnecessary for ~4k vectors) + metadata filters,
                        fused with reciprocal-rank fusion                   (indexes/)

The dense model is used only here, during preparation. The runtime application uses a prebuilt
MiniSearch (BM25-style) index written by build_application.py and needs no model.
"""
from __future__ import annotations

import pickle
import re
import sqlite3
import sys

import numpy as np
from rank_bm25 import BM25Okapi

from common import CHUNKS, EXTRACTED, INDEXES, RELATED, REPORTS, STRUCTURE, dump_json, load_json, now_iso, tokenize

CHUNK_TARGET = 220     # words
CHUNK_MAX = 380
OVERLAP_SENT = 1       # sentences of overlap when a long paragraph is split

TEXT_KINDS = {"body", "bullet", "quote", "sidebar", "footnote", "table_text", "other"}
HEAD_KINDS = {"heading_section", "heading_subsection", "heading_subsubsection", "chapter_title", "chapter_covers"}


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z`(\"“])", text) if s.strip()]


def build_chunks(pages: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for p in pages:
        if p["page_kind"] == "blank":
            continue
        ctx = {"pdf_page": p["pdf_page"], "printed_page": p["printed_page"], "chapter": p["chapter"],
               "chapter_number": p["chapter_number"], "section": p["section"], "subsection": p["subsection"]}
        heading = ""
        buf: list[dict] = []
        buf_words = 0

        def flush(kind="text"):
            nonlocal buf, buf_words
            if not buf:
                return
            text = "\n\n".join(e["text"] for e in buf)
            chunks.append({"chunk_id": f"c{p['pdf_page']:03d}_{len([c for c in chunks if c['pdf_page'] == p['pdf_page']]) + 1:02d}",
                           **ctx, "heading": heading, "kind": kind, "element_ids": [e["id"] for e in buf],
                           "text": text, "words": len(text.split())})
            buf, buf_words = [], 0

        for e in p["elements"]:
            k = e["kind"]
            if k in ("running_head", "page_number", "chapter_number", "figure_label"):
                continue
            if k in HEAD_KINDS:
                flush()
                heading = e["text"]
                continue
            if k == "code":
                flush()
                # attach the listing caption that precedes it, if any
                cap = next((L for L in p["listings"] if e["id"] in L["code_ids"]), None)
                title = f"Listing {cap['number']} {cap['title']}\n" if cap else ""
                chunks.append({"chunk_id": f"c{p['pdf_page']:03d}_{len([c for c in chunks if c['pdf_page'] == p['pdf_page']]) + 1:02d}",
                               **ctx, "heading": heading, "kind": "code", "element_ids": [e["id"]],
                               "text": title + e["text"], "words": len(e["text"].split())})
                continue
            if k in ("listing_caption", "figure_caption", "table_caption"):
                buf.append(e); buf_words += len(e["text"].split())
                continue
            if k not in TEXT_KINDS:
                continue
            w = len(e["text"].split())
            if w > CHUNK_MAX:                         # very long paragraph: split by sentences with overlap
                flush()
                sents = sentences(e["text"])
                cur: list[str] = []
                for s in sents:
                    cur.append(s)
                    if len(" ".join(cur).split()) >= CHUNK_TARGET:
                        buf = [{"id": e["id"], "text": " ".join(cur)}]; flush()
                        cur = cur[-OVERLAP_SENT:]
                if cur and len(" ".join(cur).split()) > 25:
                    buf = [{"id": e["id"], "text": " ".join(cur)}]; flush()
                continue
            if buf_words + w > CHUNK_MAX or (buf_words >= CHUNK_TARGET and k != "bullet"):
                flush()
            buf.append(e); buf_words += w
        flush()
    return chunks


def embed(texts: list[str]) -> np.ndarray:
    from fastembed import TextEmbedding
    model = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir=str(INDEXES / "vector_index" / "model_cache"))
    vecs = np.asarray(list(model.embed(texts, batch_size=64)), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return vecs


def write_sqlite(structure: dict, pages: list[dict], chunks: list[dict]) -> None:
    db = INDEXES / "metadata.db"
    if db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    con.executescript("""
    CREATE TABLE chapters(number INTEGER, title TEXT, start_page INTEGER, end_page INTEGER);
    CREATE TABLE sections(chapter_title TEXT, title TEXT, start_page INTEGER, level INTEGER);
    CREATE TABLE pages(pdf_page INTEGER PRIMARY KEY, printed_page TEXT, page_kind TEXT, chapter TEXT,
                       chapter_number INTEGER, section TEXT, subsection TEXT, char_count INTEGER,
                       n_listings INTEGER, n_figures INTEGER, n_eq_candidates INTEGER);
    CREATE TABLE elements(id TEXT PRIMARY KEY, pdf_page INTEGER, kind TEXT, text TEXT, x0 REAL, y0 REAL, x1 REAL, y1 REAL);
    CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY, pdf_page INTEGER, chapter TEXT, section TEXT, subsection TEXT,
                        heading TEXT, kind TEXT, element_ids TEXT, text TEXT, words INTEGER);
    CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, text);
    CREATE TABLE listings(number TEXT, title TEXT, pdf_page INTEGER, caption_id TEXT, code_ids TEXT);
    CREATE TABLE figures(number TEXT, caption TEXT, pdf_page INTEGER, caption_id TEXT, labels TEXT);
    """)
    for ch in structure["chapters"]:
        con.execute("INSERT INTO chapters VALUES (?,?,?,?)", (ch["number"], ch["title"], ch["start"], ch["end"]))
        for s in ch["sections"]:
            con.execute("INSERT INTO sections VALUES (?,?,?,?)", (ch["title"], s["title"], s["start"], 2))
            for ss in s["subsections"]:
                con.execute("INSERT INTO sections VALUES (?,?,?,?)", (ch["title"], ss["title"], ss["start"], ss["level"]))
    for p in pages:
        con.execute("INSERT INTO pages VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (p["pdf_page"], p["printed_page"], p["page_kind"], p["chapter"], p["chapter_number"], p["section"],
                     p["subsection"], p["char_count"], len(p["listings"]), len(p["figures"]), len(p["eq_candidates"])))
        con.executemany("INSERT INTO elements VALUES (?,?,?,?,?,?,?,?)",
                        [(e["id"], p["pdf_page"], e["kind"], e["text"], *e["bbox"]) for e in p["elements"]])
        con.executemany("INSERT INTO listings VALUES (?,?,?,?,?)",
                        [(L["number"], L["title"], p["pdf_page"], L["caption_id"], ",".join(L["code_ids"])) for L in p["listings"]])
        con.executemany("INSERT INTO figures VALUES (?,?,?,?,?)",
                        [(f["number"], f["caption"], p["pdf_page"], f["caption_id"], " | ".join(f["labels"])) for f in p["figures"]])
    con.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [(c["chunk_id"], c["pdf_page"], c["chapter"], c["section"], c["subsection"], c["heading"], c["kind"],
                      ",".join(c["element_ids"]), c["text"], c["words"]) for c in chunks])
    con.executemany("INSERT INTO chunks_fts VALUES (?,?)", [(c["chunk_id"], c["text"]) for c in chunks])
    con.commit(); con.close()


def main() -> None:
    pages = load_json(EXTRACTED)
    structure = load_json(STRUCTURE)
    chunks = build_chunks(pages)
    dump_json(CHUNKS, chunks)
    print(f"{len(chunks)} chunks", file=sys.stderr)

    (INDEXES / "keyword_index").mkdir(parents=True, exist_ok=True)
    (INDEXES / "vector_index").mkdir(parents=True, exist_ok=True)
    tok = [tokenize(c["heading"] + " " + c["text"]) for c in chunks]
    bm25 = BM25Okapi(tok)
    with open(INDEXES / "keyword_index" / "bm25.pkl", "wb") as f:
        pickle.dump({"ids": [c["chunk_id"] for c in chunks], "bm25": bm25, "tokens": tok}, f)

    print("embedding chunks (local ONNX model, preparation phase only)…", file=sys.stderr)
    vecs = embed([f"{c['chapter']} / {c['heading']}\n{c['text']}" for c in chunks])
    np.save(INDEXES / "vector_index" / "chunks.npy", vecs)
    dump_json(INDEXES / "vector_index" / "chunk_ids.json", [c["chunk_id"] for c in chunks], indent=None)
    write_sqlite(structure, pages, chunks)

    # Level 3: related pages. Page vector = mean of its chunk vectors; score pages by cosine + BM25 of page text.
    by_page: dict[int, list[int]] = {}
    for i, c in enumerate(chunks):
        by_page.setdefault(c["pdf_page"], []).append(i)
    page_nums = sorted(by_page)
    pv = np.stack([vecs[by_page[p]].mean(axis=0) for p in page_nums])
    pv /= np.linalg.norm(pv, axis=1, keepdims=True) + 1e-9
    sims = pv @ pv.T
    related = {}
    for a, pa in enumerate(page_nums):
        q = tokenize(" ".join(chunks[i]["text"] for i in by_page[pa]))[:400]
        bm = bm25.get_scores(q) if q else np.zeros(len(chunks))
        bm_page = {}
        for i, c in enumerate(chunks):
            bm_page[c["pdf_page"]] = max(bm_page.get(c["pdf_page"], 0.0), float(bm[i]))
        order_sem = np.argsort(-sims[a])
        rank_sem = {page_nums[j]: r for r, j in enumerate(order_sem)}
        order_bm = sorted(bm_page, key=lambda p: -bm_page[p])
        rank_bm = {p: r for r, p in enumerate(order_bm)}
        scored = []
        for pb in page_nums:
            if abs(pb - pa) <= 1:
                continue
            rrf = 1 / (60 + rank_sem.get(pb, 999)) + 1 / (60 + rank_bm.get(pb, 999))
            scored.append((rrf, pb))
        scored.sort(reverse=True)
        related[str(pa)] = [{"page": pb, "score": round(s, 5), "chapter": structure["page_map"][str(pb)]["chapter"],
                             "section": structure["page_map"][str(pb)]["section"]} for s, pb in scored[:12]]
    dump_json(RELATED, related)

    report = {"generated_at": now_iso(), "chunks": len(chunks),
              "chunk_kinds": {k: sum(1 for c in chunks if c["kind"] == k) for k in {c["kind"] for c in chunks}},
              "avg_words": round(sum(c["words"] for c in chunks) / len(chunks), 1),
              "max_words": max(c["words"] for c in chunks),
              "pages_with_chunks": len(by_page), "embedding_model": "BAAI/bge-small-en-v1.5 (384-d, preparation only)",
              "vector_shape": list(vecs.shape), "bm25_vocab": len({t for ts in tok for t in ts}),
              "related_pages_per_page": 12}
    dump_json(REPORTS / "index_report.json", report, indent=2)
    print(report)


if __name__ == "__main__":
    main()
