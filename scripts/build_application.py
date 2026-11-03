#!/usr/bin/env python3
"""Stage 7/8 - integrate everything into the static application.

Writes (all consumed by assets/js/app.js at runtime, no server logic needed):
  data/book_metadata.json          chapters, page map, which pages have packages
  data/search/docs.json            documents for the MiniSearch index (book chunks + notes)
  data/concepts/concepts.json      concept registry aggregated across pages
  data/knowledge_graph/graph.json  nodes (concepts) + links (related concepts)
  data/quizzes/index.json, data/quizzes/chapter_NN.json
  data/chapters/index.json, data/chapters/chapter_NN.json
  data/diagrams/index.json         every visualization with its page (for validation)
Re-run after any page package changes; it is idempotent and fast.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

from common import (CHAPTERS, CHUNKS, CONCEPTS, DATA, DIAGRAMS, EXTRACTED, KG, PAGES, QUIZZES, REPORTS, SEARCH, STRUCTURE,
                    dump_json, load_json, now_iso, slugify)

STRIP_MD = re.compile(r"[*_`#>\[\]()$\\]")


def snippet(text: str, n: int = 180) -> str:
    t = STRIP_MD.sub("", re.sub(r"\s+", " ", text or "")).strip()
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "…"


def main() -> None:
    structure = load_json(STRUCTURE)
    pages = load_json(EXTRACTED)
    chunks = load_json(CHUNKS)
    packages: dict[int, dict] = {}
    for f in sorted(PAGES.glob("page_*.json")):
        try:
            pkg = json.load(open(f, encoding="utf-8"))
            packages[pkg["pdf_page"]] = pkg
        except Exception as ex:  # noqa
            print("skipping", f, ex)

    # ---- metadata ---------------------------------------------------------------------------
    chapters = []
    for ch in structure["chapters"]:
        chapters.append({"title": ch["title"], "number": ch["number"], "start": ch["start"], "end": ch["end"],
                         "sections": [{"title": s["title"], "start": s["start"]} for s in ch["sections"]]})
    page_map = {int(k): v for k, v in structure["page_map"].items()}
    for n, v in page_map.items():
        v["has_notes"] = n in packages
        v["status"] = packages[n]["generation"]["status"] if n in packages else None
    meta = {"book_id": structure["book_id"], "title": structure["title"], "authors": structure["authors"],
            "page_count": structure["page_count"], "chapters": chapters, "page_map": page_map,
            "built_at": now_iso(), "pages_with_notes": len(packages)}
    dump_json(DATA / "book_metadata.json", meta, indent=None)

    # ---- search documents -------------------------------------------------------------------
    docs = []
    for c in chunks:
        docs.append({"id": c["chunk_id"], "kind": "book", "title": c["heading"] or c["section"] or c["chapter"],
                     "text": c["text"], "page": c["pdf_page"], "chapter": c["chapter"], "section": c["section"],
                     "snippet": snippet(c["text"]), "anchor": None})
    for ch in structure["chapters"]:
        docs.append({"id": f"ch{ch['start']}", "kind": "chapter", "title": ch["title"], "text": " ".join(s["title"] for s in ch["sections"]),
                     "page": ch["start"], "chapter": ch["title"], "section": "", "snippet": f"Chapter starts at PDF p.{ch['start']}", "anchor": None})
    for n, pkg in packages.items():
        pm = page_map[n]
        base = {"page": n, "chapter": pm["chapter"], "section": pm["section"]}
        o = pkg["overview"]
        docs.append({"id": f"pg{n}", "kind": "page", "title": o["title"], "text": f"{o['main_topic']} {o['central_idea']} {pkg['summary']} " + " ".join(pkg.get("takeaways", [])),
                     "snippet": snippet(pkg["summary"]), "anchor": "sec-summary", **base})
        for c in pkg.get("concepts", []):
            docs.append({"id": f"co{n}_{c['id']}", "kind": "concept", "title": c["name"], "text": f"{c['definition']} {c['intuition']} {c['technical']} {c['example']}",
                         "snippet": snippet(c["definition"]), "anchor": f"concept-{c['id']}", **base})
        for m in pkg.get("math", []):
            docs.append({"id": f"ma{n}_{m['id']}", "kind": "math", "title": m["title"], "text": f"{m['source_text']} {m['latex']} {m['purpose']} {m['interpretation']} " + " ".join(v["symbol"] + " " + v["meaning"] for v in m["variables"]),
                         "snippet": snippet(m["source_text"] + " — " + m["purpose"]), "anchor": f"math-{m['id']}", **base})
        for c in pkg.get("code", []):
            docs.append({"id": f"cd{n}_{c['id']}", "kind": "code", "title": (f"Listing {c['listing']} " if c.get("listing") else "") + c["title"], "text": f"{c['original_code']} {c['what_it_does']} {c['algorithm']}",
                         "snippet": snippet(c["what_it_does"]), "anchor": f"code-{c['id']}", **base})
        for q in pkg.get("questions", []):
            docs.append({"id": f"q{n}_{q['id']}", "kind": "question", "title": snippet(q["question"], 90), "text": f"{q['question']} {q['answer']}",
                         "snippet": snippet(q["answer"]), "anchor": "sec-questions", **base})
    SEARCH.mkdir(parents=True, exist_ok=True)
    dump_json(SEARCH / "docs.json", docs, indent=None)

    # ---- concepts + knowledge graph ---------------------------------------------------------
    registry: dict[str, dict] = {}
    links: dict[tuple[str, str], int] = defaultdict(int)
    for n in sorted(packages):
        pkg = packages[n]
        for c in pkg.get("concepts", []):
            cid = slugify(c["id"]) or slugify(c["name"])
            r = registry.setdefault(cid, {"id": cid, "name": c["name"], "definition": c["definition"], "first_page": n,
                                          "chapter_number": page_map[n]["chapter_number"], "chapter": page_map[n]["chapter"], "pages": [], "related": set(), "aliases": set()})
            r["pages"].append(n)
            r["aliases"].add(c["name"])
            for rel in c.get("related", []):
                rid = slugify(rel)
                if rid and rid != cid:
                    r["related"].add(rid)
                    links[tuple(sorted((cid, rid)))] += 1
    # links only between concepts that exist as nodes; dangling related ids become light nodes if referenced >= 2 times
    ref_count: dict[str, int] = defaultdict(int)
    for (a, b), w in links.items():
        ref_count[a] += w; ref_count[b] += w
    for cid, cnt in ref_count.items():
        if cid not in registry and cnt >= 2:
            registry[cid] = {"id": cid, "name": cid.replace("_", " "), "definition": "(referenced as a related concept; no dedicated entry yet)",
                             "first_page": None, "chapter_number": None, "chapter": "", "pages": [], "related": set(), "aliases": set()}
    nodes = []
    for r in registry.values():
        nodes.append({"id": r["id"], "name": r["name"], "definition": r["definition"], "first_page": r["first_page"], "chapter_number": r["chapter_number"],
                      "chapter": r["chapter"], "pages": sorted(set(r["pages"])), "aliases": sorted(r["aliases"])})
    edge_list = [{"source": a, "target": b, "weight": w} for (a, b), w in links.items() if a in registry and b in registry]
    KG.mkdir(parents=True, exist_ok=True); CONCEPTS.mkdir(parents=True, exist_ok=True)
    dump_json(KG / "graph.json", {"nodes": nodes, "links": edge_list, "built_at": now_iso()}, indent=None)
    dump_json(CONCEPTS / "concepts.json", nodes, indent=None)

    # ---- quizzes & chapters -----------------------------------------------------------------
    QUIZZES.mkdir(parents=True, exist_ok=True); CHAPTERS.mkdir(parents=True, exist_ok=True)
    qindex, cindex = [], []
    for ch in structure["chapters"]:
        num = ch["number"] if ch["number"] is not None else 0
        fname = f"chapter_{num:02d}_{slugify(ch['title'])[:30]}.json"
        qs, pages_in, concepts_in = [], [], set()
        for n in range(ch["start"], ch["end"] + 1):
            if n in packages:
                pages_in.append(n)
                for q in packages[n].get("questions", []):
                    qs.append({**q, "page": n})
                concepts_in.update(c["id"] for c in packages[n].get("concepts", []))
        dump_json(QUIZZES / fname, {"chapter": ch["title"], "questions": qs}, indent=None)
        qindex.append({"title": ch["title"], "file": fname, "count": len(qs)})
        # chapter overview: the chapter opener package summary if present, else the first page summary
        opener = next((packages[n] for n in range(ch["start"], ch["end"] + 1) if n in packages and packages[n]["overview"].get("page_kind") == "chapter_opener"), None)
        overview = (opener or (packages.get(pages_in[0]) if pages_in else None) or {}).get("summary", "")
        centry = {"title": ch["title"], "number": ch["number"], "start": ch["start"], "end": ch["end"], "pages": ch["end"] - ch["start"] + 1,
                  "generated": len(pages_in), "concepts": len(concepts_in), "questions": len(qs), "overview": overview,
                  "sections": [{"title": s["title"], "start": s["start"], "subsections": s["subsections"]} for s in ch["sections"]],
                  "page_titles": {n: packages[n]["overview"]["title"] for n in pages_in}}
        dump_json(CHAPTERS / fname, centry, indent=None)
        cindex.append({k: v for k, v in centry.items() if k not in ("sections", "page_titles")} | {"file": fname})
    dump_json(QUIZZES / "index.json", {"chapters": qindex, "total": sum(q["count"] for q in qindex)}, indent=None)
    dump_json(CHAPTERS / "index.json", {"chapters": cindex}, indent=None)

    # ---- diagrams index ---------------------------------------------------------------------
    DIAGRAMS.mkdir(parents=True, exist_ok=True)
    viz = [{"page": n, "id": v["id"], "type": v["type"], "widget": v.get("widget"), "title": v["title"]} for n, pkg in packages.items() for v in pkg.get("visualizations", [])]
    dump_json(DIAGRAMS / "index.json", viz, indent=None)

    rep = {"built_at": now_iso(), "pages_with_notes": len(packages), "search_docs": len(docs), "concept_nodes": len(nodes),
           "concept_links": len(edge_list), "quiz_questions": sum(q["count"] for q in qindex), "visualizations": len(viz),
           "visualization_types": {t: sum(1 for v in viz if v["type"] == t) for t in {v["type"] for v in viz}},
           "widgets_used": sorted({v["widget"] for v in viz if v["widget"]})}
    dump_json(REPORTS / "build_report.json", rep, indent=2)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
