#!/usr/bin/env python3
"""Stage 3 - structured extraction of every page of the book.

Output: work/extracted_pages.json  (list of page records, one per PDF page)
        work/structure.json        (book, chapters, sections, page -> location map)
        work/page_images/page_NNN.png  (rendered pages for the generation phase)
        reports/extraction_report.json

Classification is font-driven (this is a born-digital Manning PDF):
  body text        NewBaskervilleStd-Roman 10 (italic / mono spans inline)
  code listing     JetBrainsMono-Regular 8 (whole line mono)
  code callout     HumanistMann521-BoldCond 9  (annotations next to code)
  section heading  ITCFranklinGothicStd-Demi 12.5   ("1.1  Title")
  subsection       ITCFranklinGothicStd-Demi 10.5   ("2.3.6  Title")
  sub-subsection   ITCFranklinGothicStd-Demi 9.5
  listing caption  ITCFranklinGothicStd-Demi 9  starting with "Listing"
  figure caption   ITCFranklinGothicStd-Demi 8  starting with "Figure"
  figure labels    Arial* 6-9 pt (text inside vector figures)
  sidebar/note     ITCFranklinGothicStd-Book 9.5 (grey boxes)
  footnote         NewBaskervilleStd-Roman 8.5
  bullet           Wingdings2 "¡" marker
  running head     y < 40
Nothing is discarded: lines that match no rule are kept as "other" and counted in the report.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict

import pymupdf

from common import (BOOK_PDF, EXTRACTED, PAGE_IMAGES, REPORTS, STRUCTURE, WORK, clean_text, dump_json,
                    now_iso, page_id)

FRONT_MATTER_LAST = 24


def font_key(span: dict) -> tuple[str, float]:
    return span["font"].split("+")[-1], round(span["size"], 1)


def classify_line(line: dict, page_h: float) -> tuple[str, dict]:
    spans = [s for s in line["spans"] if s["text"].strip()]
    if not spans:
        return "empty", {}
    text = "".join(s["text"] for s in line["spans"])
    chars = sum(len(s["text"]) for s in spans)
    by_font: dict[tuple[str, float], int] = {}
    for sp in spans:
        by_font[font_key(sp)] = by_font.get(font_key(sp), 0) + len(sp["text"].strip())
    f0, sz0 = max(by_font.items(), key=lambda kv: kv[1])[0]
    if f0.startswith("ITCFranklinGothicStd-Dem") and sz0 == 7.6:   # small caps: lowercase set at 80% of 9.5
        sz0 = 9.5
    y0, y1 = line["bbox"][1], line["bbox"][3]
    x0 = line["bbox"][0]
    nonspace = sum(len(s["text"].replace(" ", "")) for s in spans)
    mono_chars = sum(len(s["text"].replace(" ", "")) for s in spans if "JetBrainsMono" in s["font"] or "Courier" in s["font"])
    mono_frac = mono_chars / max(nonspace, 1)
    info = {"mono_frac": round(mono_frac, 2), "font": f0, "size": sz0}

    if y1 < 40:
        return "running_head", info
    if y0 > page_h - 70 and f0 == "NewBaskervilleStd-Bold" and sz0 == 9.0 and text.strip().isdigit():
        return "page_number", info
    if "Wingdings2" in f0 or text.lstrip().startswith("¡"):
        return "bullet", info
    if f0.startswith("ITCFranklinGothicStd-Dem"):
        t = text.strip()
        if sz0 == 9.0 and t.startswith("Listing"):
            return "listing_caption", info
        if sz0 == 9.0 and t.startswith("Table"):
            return "table_caption", info
        if sz0 == 8.0 and t.startswith("Figure"):
            return "figure_caption", info
        if sz0 == 8.0 and t.startswith("Table"):
            return "table_caption", info
        if sz0 == 8.0:
            return "figure_caption_cont", info
        if sz0 >= 12.4:
            return "heading_section", info
        if sz0 == 12.0:
            return "chapter_covers", info
        if sz0 == 10.5:
            return "heading_subsection", info
        if 9.2 <= sz0 <= 10.0:
            return "heading_subsubsection", info
        if sz0 == 9.0:
            return "running_head", info
        return "heading_other", info
    if f0 == "ITCFranklinGothicStd-Boo" and sz0 in (9.8,):
        return "chapter_covers_item", info
    if f0 == "ITCFranklinGothicStd-Boo" and sz0 == 9.5:
        return "sidebar", info
    if f0 == "HumanistMann521-BoldCond":
        return "code_callout", info
    if f0.startswith("Arial"):
        return "figure_label", info
    if f0 == "FranklinGothic-Book" and sz0 < 9:
        return "table_text", info
    if f0 == "NewBaskervilleStd-Italic" and sz0 == 30.0:
        return "chapter_title", info
    if f0 == "NewBaskervilleStd-Italic" and sz0 > 100:
        return "chapter_number", info
    if mono_frac >= 0.97 and x0 >= 100:
        return "code", info
    if f0 == "NewBaskervilleStd-Roman" and sz0 == 8.5:
        return "footnote", info
    if f0 == "NewBaskervilleStd-Italic" and sz0 == 10.0 and x0 >= 106:
        return "quote", info
    if f0.startswith("NewBaskerville") and sz0 in (10.0, 9.0, 11.0):
        return "body", info
    if mono_frac >= 0.97:
        return "code", info
    if any(font_key(sp)[0].startswith("NewBaskerville") for sp in spans) or "JetBrainsMono" in f0:
        return "body", info
    return "other", info


def line_text(line: dict, keep_inline_fonts: bool = True) -> str:
    """Join spans; mark inline code spans with backticks so that the generator sees them."""
    out = []
    for s in line["spans"]:
        t = s["text"]
        if not t:
            continue
        if keep_inline_fonts and ("JetBrainsMono" in s["font"] or "Courier" in s["font"]) and t.strip():
            out.append("`" + t.strip("\t") + "`")
        else:
            out.append(t)
    s = "".join(out).replace("``", "")
    return s


def merge_lines(lines: list[str]) -> str:
    """Join wrapped lines, removing soft-hyphen breaks."""
    text = ""
    for ln in lines:
        ln = ln.rstrip()
        if text.endswith("­") or text.endswith("-") and not text.endswith(" -"):
            if text.endswith("­"):
                text = text[:-1] + ln.lstrip()
            else:
                # real hyphen at line end: keep hyphen (e.g. "self-", "high-dimensional")
                text = text + ln.lstrip()
        else:
            text = (text + " " + ln.lstrip()) if text else ln
    return clean_text(text)


def extract_page(doc: pymupdf.Document, pno: int, toc_map: dict) -> tuple[dict, list[str]]:
    page = doc[pno]
    pdf_page = pno + 1
    pid = page_id(pdf_page)
    warnings: list[str] = []
    d = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_LIGATURES | pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_PRESERVE_IMAGES)
    W, H = page.rect.width, page.rect.height

    raw_lines = []  # (y, x, kind, text, info, block_no)
    image_blocks = []
    for b in d["blocks"]:
        if b["type"] == 1:
            image_blocks.append([round(v, 1) for v in b["bbox"]])
            continue
        for ln in b["lines"]:
            kind, info = classify_line(ln, H)
            if kind == "empty":
                continue
            raw_lines.append({"y": ln["bbox"][1], "x": ln["bbox"][0], "y1": ln["bbox"][3], "x1": ln["bbox"][2],
                              "kind": kind, "text": line_text(ln, kind not in ("code", "figure_label")),
                              "raw": "".join(s["text"] for s in ln["spans"]), "info": info, "block": b["number"]})

    # Reading order: Manning pages are single column; figure labels can be anywhere. Sort by y then x,
    # but keep figure labels separate so they do not interleave body paragraphs.
    labels = [l for l in raw_lines if l["kind"] == "figure_label"]
    callouts = [l for l in raw_lines if l["kind"] == "code_callout"]
    flow = [l for l in raw_lines if l["kind"] not in ("figure_label", "code_callout")]
    flow.sort(key=lambda l: (round(l["y"] / 4), l["x"]))

    elements: list[dict] = []
    cur: dict | None = None

    def flush():
        nonlocal cur
        if cur is None:
            return
        kind = cur["kind"]
        if kind == "code":
            cur["text"] = "\n".join(clean_text(t) if False else t.rstrip() for t in cur["lines"])
            cur["text"] = cur["text"].replace(" ", " ")
        else:
            cur["text"] = merge_lines(cur["lines"])
        cur["bbox"] = [round(v, 1) for v in cur["bbox"]]
        del cur["lines"]
        if cur["text"].strip():
            elements.append(cur)
        cur = None

    para_starts = {"heading_section", "heading_subsection", "heading_subsubsection", "listing_caption",
                   "figure_caption", "table_caption", "chapter_title", "chapter_number", "chapter_covers",
                   "bullet", "running_head", "page_number", "chapter_covers_item"}
    for l in flow:
        kind = l["kind"]
        t = l["text"]
        if kind in ("figure_caption_cont",):
            kind = "figure_caption"
        if kind == "chapter_covers_item":
            kind = "bullet"
        if kind == "bullet":
            t = re.sub(r"^\s*¡\s*", "", t)
        new_para = (
            cur is None
            or kind in para_starts
            or cur["kind"] != kind
            or (kind == "body" and l["x"] - cur["x_left"] > 8 and cur["lines"])   # indented first line
            or (kind == "code" and l["y"] - cur["last_y"] > 22)                    # blank gap in code
            or (kind in ("body", "sidebar", "quote", "footnote") and l["y"] - cur["last_y"] > 20)
        )
        # bullet continuation lines are "body" lines indented at the bullet text start
        if cur is not None and cur["kind"] == "bullet" and kind == "body" and l["x"] >= cur["x_left"] + 6 \
                and l["y"] - cur["last_y"] < 20:
            new_para = False
            kind = "bullet"
        if cur is not None and cur["kind"] == "figure_caption" and kind == "figure_caption":
            new_para = False
        if cur is not None and kind == "code" and cur["kind"] in ("body", "bullet", "sidebar") \
                and l["y"] - cur["last_y"] < 16 and l["x"] < 140 and not cur["lines"][-1].rstrip().endswith(":"):
            new_para = False
            kind = cur["kind"]
            t = "`" + t.strip() + "`"
        if kind == "other" and (pdf_page <= FRONT_MATTER_LAST or toc_map[pdf_page]["chapter"] == "index"):
            kind = "body"
        if new_para:
            flush()
            cur = {"kind": kind, "lines": [], "bbox": [l["x"], l["y"], l["x1"], l["y1"]], "x_left": l["x"],
                   "last_y": l["y"], "info": l["info"]}
        cur["lines"].append(t)
        cur["bbox"] = [min(cur["bbox"][0], l["x"]), min(cur["bbox"][1], l["y"]), max(cur["bbox"][2], l["x1"]),
                       max(cur["bbox"][3], l["y1"])]
        cur["last_y"] = l["y"]
    flush()

    # Give ids and strip helper keys
    for i, e in enumerate(elements):
        e["id"] = f"p{pdf_page:03d}_e{i + 1:02d}"
        e.pop("x_left", None); e.pop("last_y", None)
        e["font"] = e.pop("info", {}).get("font", "")

    # ---- headings / captions / listings / figures ---------------------------------------------
    headings = [{"level": {"heading_section": 2, "heading_subsection": 3, "heading_subsubsection": 4,
                           "chapter_title": 1}.get(e["kind"], 5), "text": e["text"], "id": e["id"]}
                for e in elements if e["kind"].startswith("heading") or e["kind"] == "chapter_title"]

    listings = []
    for i, e in enumerate(elements):
        if e["kind"] == "listing_caption":
            m = re.match(r"Listing\s+(\d+\.\d+)\s*(.*)", e["text"])
            code_ids = []
            for f in elements[i + 1:]:
                if f["kind"] == "code":
                    code_ids.append(f["id"])
                elif f["kind"] in ("listing_caption", "heading_section", "heading_subsection", "figure_caption"):
                    break
                elif f["kind"] == "body" and code_ids:
                    break
            listings.append({"number": m.group(1) if m else None, "title": clean_text(m.group(2)) if m else e["text"],
                             "caption_id": e["id"], "code_ids": code_ids})
            if not code_ids:
                warnings.append(f"listing caption {e['id']} has no code block on this page (may continue on next page)")

    # Figure captions + vector drawing clusters + image bboxes + nearby labels
    drawings = page.get_drawings()
    draw_rects = [dr["rect"] for dr in drawings if dr["rect"].width > 2 and dr["rect"].height > 2]
    figures = []
    prev_caption_y = 0.0
    for e in elements:
        if e["kind"] == "figure_caption":
            m = re.match(r"Figure\s+(\d+\.\d+)\s*(.*)", e["text"])
            cy = e["bbox"][1]
            # figure region: drawings / images whose bottom is above the caption and within 320pt
            region = None
            for r in draw_rects + [pymupdf.Rect(*ib) for ib in image_blocks]:
                if r.y1 <= cy + 40 and r.y1 > cy - 330 and r.y0 > prev_caption_y - 5 and r.width < W - 40:
                    region = r if region is None else region | r
            labs = []
            if region is not None:
                rr = pymupdf.Rect(region.x0 - 6, region.y0 - 6, region.x1 + 6, region.y1 + 6)
                labs = [clean_text(l["raw"]) for l in labels if pymupdf.Rect(l["x"], l["y"], l["x1"], l["y1"]).intersects(rr)]
            prev_caption_y = e["bbox"][3]
            figures.append({"number": m.group(1) if m else None, "caption": clean_text(m.group(2)) if m else e["text"],
                            "caption_id": e["id"], "bbox": [round(v, 1) for v in region] if region else None,
                            "labels": labs})
    orphan_labels = [clean_text(l["raw"]) for l in labels
                     if not any(lab in f["labels"] for f in figures for lab in [clean_text(l["raw"])])]

    tables = []
    try:
        if any(e["kind"] == "table_caption" for e in elements):
            for t in page.find_tables().tables:
                rows = t.extract()
                tables.append({"bbox": [round(v, 1) for v in t.bbox],
                               "rows": [[clean_text(c or "") for c in row] for row in rows]})
    except Exception as ex:  # noqa
        warnings.append(f"table extraction failed: {ex}")

    # Equation candidates: body sentences with '=' and math-looking operators, and short mono lines with '='
    eq_candidates = []
    for e in elements:
        if e["kind"] == "body" or e["kind"] == "bullet":
            for sent in re.split(r"(?<=[.!?])\s+", e["text"]):
                if "=" in sent and re.search(r"[A-Za-z_]\w*\s*[=+*/^@]|sqrt|sum|log|exp|\\", sent) and len(sent) < 400:
                    eq_candidates.append({"element": e["id"], "text": sent.strip()})
        elif e["kind"] == "code" and "=" in e["text"] and len(e["text"]) < 160 \
                and not re.search(r"^\s*(>>>|def |import |from |class |for |if |\.\.\.)", e["text"], re.M) \
                and not any(e["id"] in L["code_ids"] for L in listings):
            eq_candidates.append({"element": e["id"], "text": e["text"]})

    body_text = "\n\n".join(e["text"] for e in elements
                            if e["kind"] in ("body", "bullet", "quote", "sidebar", "footnote", "heading_section",
                                             "heading_subsection", "heading_subsubsection", "chapter_title",
                                             "chapter_covers", "listing_caption", "figure_caption", "table_caption",
                                             "table_text", "other"))
    code_text = "\n\n".join(e["text"] for e in elements if e["kind"] == "code")
    other = [e["id"] for e in elements if e["kind"] == "other"]
    if other:
        warnings.append(f"{len(other)} unclassified text elements kept as 'other'")
    nchars = len(body_text) + len(code_text)
    if nchars < 50 and not image_blocks and len(draw_rects) < 3:
        kind = "blank"
    elif pdf_page <= FRONT_MATTER_LAST:
        kind = "front_matter"
    elif any(e["kind"] == "chapter_title" for e in elements):
        kind = "chapter_opener"
    elif toc_map[pdf_page]["chapter"].startswith("index"):
        kind = "index"
    elif nchars < 200 and (image_blocks or figures):
        kind = "figure_only"
    else:
        kind = "content"
    loc = toc_map[pdf_page]
    # Refine section from headings on the page itself (first heading wins for the page start context)
    rec = {
        "page_id": pid, "pdf_page": pdf_page, "printed_page": page.get_label() or "", "page_kind": kind,
        "width": round(W, 1), "height": round(H, 1),
        "chapter": loc["chapter"], "section": loc["section"], "subsection": loc["subsection"],
        "chapter_number": loc["chapter_number"],
        "headings_on_page": headings,
        "elements": [{k: e[k] for k in ("id", "kind", "text", "bbox", "font")} for e in elements],
        "listings": listings, "figures": figures, "tables": tables, "eq_candidates": eq_candidates,
        "images": image_blocks, "n_vector_drawings": len(draw_rects), "orphan_figure_labels": orphan_labels,
        "code_callouts": [clean_text(c["raw"]) for c in sorted(callouts, key=lambda c: (c["y"], c["x"]))],
        "body_text": body_text, "code_text": code_text,
        "char_count": nchars, "warnings": warnings,
    }
    return rec, warnings


def build_toc_map(doc: pymupdf.Document) -> tuple[dict, list[dict]]:
    toc = doc.get_toc()
    n = doc.page_count
    chapters = []
    cur_ch = {"title": "front matter", "number": None, "start": 1, "sections": []}
    for lvl, title, pg in toc:
        title = clean_text(title)
        if lvl == 1:
            cur_ch = {"title": title, "number": None, "start": pg, "sections": []}
            m = re.match(r"(\d+)\s+(.*)", title)
            if m:
                cur_ch["number"] = int(m.group(1))
            chapters.append(cur_ch)
        elif lvl == 2:
            cur_ch["sections"].append({"title": title, "start": pg, "subsections": []})
        elif lvl >= 3 and cur_ch["sections"]:
            cur_ch["sections"][-1]["subsections"].append({"title": title, "start": pg, "level": lvl})
    for i, ch in enumerate(chapters):
        ch["end"] = chapters[i + 1]["start"] - 1 if i + 1 < len(chapters) else n
    toc_map = {}
    for p in range(1, n + 1):
        ch = next((c for c in reversed(chapters) if c["start"] <= p), None)
        sec = sub = ""
        if ch:
            s = next((s for s in reversed(ch["sections"]) if s["start"] <= p), None)
            if s:
                sec = s["title"]
                ss = next((x for x in reversed(s["subsections"]) if x["start"] <= p), None)
                if ss:
                    sub = ss["title"]
        toc_map[p] = {"chapter": ch["title"] if ch else "", "chapter_number": ch["number"] if ch else None,
                      "section": sec, "subsection": sub}
    return toc_map, chapters


def main(render_images: bool = True) -> None:
    doc = pymupdf.open(BOOK_PDF)
    toc_map, chapters = build_toc_map(doc)
    pages, all_warnings = [], {}
    kinds_count = defaultdict(int)
    PAGE_IMAGES.mkdir(parents=True, exist_ok=True)
    for pno in range(doc.page_count):
        rec, warns = extract_page(doc, pno, toc_map)
        pages.append(rec)
        if warns:
            all_warnings[rec["page_id"]] = warns
        for e in rec["elements"]:
            kinds_count[e["kind"]] += 1
        if render_images:
            out = PAGE_IMAGES / f"{rec['page_id']}.png"
            if not out.exists():
                doc[pno].get_pixmap(dpi=110).save(out)
        if (pno + 1) % 50 == 0:
            print(f"  extracted {pno + 1}/{doc.page_count}", file=sys.stderr)
    dump_json(EXTRACTED, pages)
    structure = {
        "book_id": "dlwp3", "title": "Deep Learning with Python, Third Edition",
        "authors": ["François Chollet", "Matthew Watson"], "publisher": "Manning",
        "page_count": doc.page_count, "front_matter_pages": FRONT_MATTER_LAST,
        "chapters": chapters,
        "page_map": {str(p["pdf_page"]): {"printed": p["printed_page"], "chapter": p["chapter"],
                                           "chapter_number": p["chapter_number"], "section": p["section"],
                                           "subsection": p["subsection"], "kind": p["page_kind"]} for p in pages},
    }
    dump_json(STRUCTURE, structure)
    report = {
        "generated_at": now_iso(), "source": str(BOOK_PDF), "page_count": doc.page_count,
        "pages_extracted": len(pages),
        "page_kinds": dict(defaultdict(int, {k: sum(1 for p in pages if p["page_kind"] == k)
                                            for k in set(p["page_kind"] for p in pages)})),
        "element_kinds": dict(kinds_count),
        "listings": sum(len(p["listings"]) for p in pages),
        "figures": sum(len(p["figures"]) for p in pages),
        "tables": sum(len(p["tables"]) for p in pages),
        "equation_candidates": sum(len(p["eq_candidates"]) for p in pages),
        "embedded_images": sum(len(p["images"]) for p in pages),
        "pages_with_warnings": len(all_warnings), "warnings": all_warnings,
        "low_text_pages": [p["pdf_page"] for p in pages if p["char_count"] < 50],
        "page_images_rendered": len(list(PAGE_IMAGES.glob("*.png"))),
        "ocr_used": False,
        "notes": "Born-digital PDF with a complete text layer; OCR not required (checked per page: only p.4 is blank).",
    }
    dump_json(REPORTS / "extraction_report.json", report, indent=2)
    print(f"extracted {len(pages)} pages -> {EXTRACTED}")
    print({k: v for k, v in report.items() if k not in ("warnings",)})


if __name__ == "__main__":
    main(render_images="--no-images" not in sys.argv)
