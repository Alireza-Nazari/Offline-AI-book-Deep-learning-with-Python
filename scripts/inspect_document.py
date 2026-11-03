#!/usr/bin/env python3
"""Stage 1 - inspect the input directory and the book, and report what the pipeline will face.

Usage: python scripts/inspect_document.py [input_dir_or_file]
Default: the project's book/ directory. Writes reports/inspection_report.json.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf

from common import BOOK_PDF, REPORTS, dump_json, now_iso

DOC_EXT = {".pdf": "pdf", ".epub": "epub", ".docx": "docx", ".txt": "text", ".md": "text", ".ipynb": "notebook",
           ".py": "code", ".png": "image", ".jpg": "image", ".jpeg": "image", ".svg": "image"}


def inspect_pdf(path: Path) -> dict:
    doc = pymupdf.open(path)
    toc = doc.get_toc()
    fonts: Counter = Counter()
    low_text, labels, sizes = [], [], Counter()
    listings = figures = tables = words = 0
    vector_pages = 0
    for i in range(doc.page_count):
        pg = doc[i]
        t = pg.get_text()
        words += len(t.split())
        if len(t.strip()) < 50:
            low_text.append(i + 1)
        labels.append(pg.get_label())
        sizes[(round(pg.rect.width), round(pg.rect.height))] += 1
        for f in pg.get_fonts():
            fonts[f[3].split("+")[-1]] += 1
        listings += len(re.findall(r"^Listing \d+\.\d+", t, re.M))
        figures += len(re.findall(r"^Figure \d+\.\d+", t, re.M))
        tables += len(re.findall(r"^Table \d+\.\d+", t, re.M))
        if len(pg.get_drawings()) > 30:
            vector_pages += 1
    chapters = [(t, p) for lvl, t, p in toc if lvl == 1]
    return {
        "format": "pdf", "pdf_version": doc.metadata.get("format"), "encrypted": doc.is_encrypted,
        "page_count": doc.page_count, "page_sizes": {f"{w}x{h}": n for (w, h), n in sizes.items()},
        "has_text_layer": len(low_text) < doc.page_count * 0.1, "pages_with_little_text": low_text,
        "ocr_needed": False if len(low_text) < doc.page_count * 0.1 else True,
        "printed_page_labels": {"first": labels[:3], "last": labels[-3:], "front_matter_pages": sum(1 for l in labels if l and not l.isdigit())},
        "bookmarks": len(toc), "chapters": chapters, "fonts_top": fonts.most_common(12),
        "approx_words": words, "listing_captions": listings, "figure_captions": figures, "table_captions": tables,
        "embedded_images": sum(len(doc[i].get_images()) for i in range(doc.page_count)),
        "pages_with_vector_figures": vector_pages,
        "challenges": [
            "Equations are typeset inline as text (e.g. 'output = relu(dot(input, W) + b)'); some appear only inside figures.",
            "Vector figures contain text labels that interleave with body text in naive extraction.",
            "Code uses a dedicated monospace font (JetBrains Mono) which allows font-based detection.",
            "Ligatures / soft hyphens need normalisation.",
            "Tables are rare (7) and need visual verification.",
        ],
    }


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else BOOK_PDF.parent
    files = [target] if target.is_file() else sorted(p for p in target.rglob("*") if p.is_file() and not p.name.startswith("."))
    inventory = [{"path": str(p), "type": DOC_EXT.get(p.suffix.lower(), "other"), "bytes": p.stat().st_size} for p in files]
    pdfs = [p for p in files if p.suffix.lower() == ".pdf"]
    rep = {"generated_at": now_iso(), "input": str(target), "inventory": inventory,
           "by_type": dict(Counter(i["type"] for i in inventory)), "selected_pipeline": None, "book": None}
    if pdfs:
        main_pdf = max(pdfs, key=lambda p: p.stat().st_size)
        rep["selected_pipeline"] = "pdf (PyMuPDF font/position-based structured extraction, no OCR)"
        rep["book"] = {"file": str(main_pdf), **inspect_pdf(main_pdf)}
    elif any(p.suffix.lower() == ".epub" for p in files):
        rep["selected_pipeline"] = "epub (not implemented in this project: the book is a PDF)"
    dump_json(REPORTS / "inspection_report.json", rep, indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k != "inventory"}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
