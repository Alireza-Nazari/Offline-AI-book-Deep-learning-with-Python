"""Shared paths, helpers and the page-package schema for the book companion pipeline."""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parent.parent
BOOK_PDF = ROOT / "book" / "original_book.pdf"
DATA = ROOT / "data"
PAGES = DATA / "pages"
CHAPTERS = DATA / "chapters"
CONCEPTS = DATA / "concepts"
DIAGRAMS = DATA / "diagrams"
QUIZZES = DATA / "quizzes"
KG = DATA / "knowledge_graph"
SEARCH = DATA / "search"
INDEXES = ROOT / "indexes"
WORK = ROOT / "work"                  # intermediate artefacts (not needed at runtime)
PAGE_IMAGES = WORK / "page_images"
CONTEXT = WORK / "context"            # per-page generation bundles
REPORTS = ROOT / "reports"
ASSETS = ROOT / "assets"
SCRIPTS = ROOT / "scripts"

EXTRACTED = WORK / "extracted_pages.json"
STRUCTURE = WORK / "structure.json"
CHUNKS = WORK / "chunks.json"
RELATED = WORK / "related.json"

BOOK_ID = "dlwp3"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def page_id(pdf_page: int) -> str:
    return f"page_{pdf_page:03d}"


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: Any, indent: int | None = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    tmp.replace(path)


LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
             " ": " ", " ": " ", " ": " ", " ": " ", "\t": " "}


def clean_text(s: str) -> str:
    for k, v in LIGATURES.items():
        s = s.replace(k, v)
    s = s.replace("­", "")            # soft hyphen
    s = re.sub(r"[ ]{2,}", " ", s)
    return s.strip()


def slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s[:60]


STOPWORDS = set("""a an the and or of to in on for with by is are was were be been being it its this that these
those as at from into than then there their they them we our you your i he she his her not no but if so
such can could would should will may might must do does did done have has had also which who whom what
when where why how all any each more most other some only own same too very just over under again
further once here both few between through during before after above below up down out off about
let s t ll ve re d m us via per e g i.e""".split())
TOKEN_RE = re.compile(r"[a-z][a-z0-9_.]*[a-z0-9]|[a-z]")


def tokenize(text: str) -> list[str]:
    return [w for w in TOKEN_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]


# ----------------------------------------------------------------------------------------------
# Page package schema (Pydantic). Every generated data/pages/page_NNN.json must validate.
# ----------------------------------------------------------------------------------------------
Source = Literal["book", "ai_explanation", "analogy", "inference"]


class Term(BaseModel):
    term: str
    meaning: str


class ParagraphNote(BaseModel):
    ref: str = Field(description="element id from extraction, e.g. p061_e07")
    excerpt: str = Field(description="first ~12 words of the original paragraph, verbatim")
    what_it_says: str
    what_it_means: str
    why_it_matters: str
    technical_idea: Optional[str] = None
    terminology: list[Term] = []
    connections: list[str] = []


class Concept(BaseModel):
    id: str
    name: str
    definition: str
    intuition: str
    technical: str
    significance: str
    example: str
    misconception: str
    related: list[str] = []
    source: Source = "book"


class Variable(BaseModel):
    symbol: str
    meaning: str


class MathNote(BaseModel):
    id: str
    title: str
    latex: str
    source_text: str = Field(description="how the formula appears in the book text")
    variables: list[Variable]
    operations: str
    purpose: str
    interpretation: str
    intuition: str
    numerical_example: Optional[str] = None
    source: Source = "book"
    book_ref: str = ""


class LineExplanation(BaseModel):
    lines: str
    explanation: str


class Parameter(BaseModel):
    name: str
    role: str


class CodeNote(BaseModel):
    id: str
    listing: Optional[str] = None
    title: str
    original_code: str
    language: str = "python"
    what_it_does: str
    line_explanations: list[LineExplanation]
    inputs_outputs: str
    algorithm: str
    theory_link: str
    parameters: list[Parameter] = []
    pitfalls: list[str] = []
    illustrative_code: Optional[str] = Field(default=None, description="AI-written extra code, never mixed with original")


class Deeper(BaseModel):
    title: str
    kind: Literal["analogy", "step_by_step", "application", "numerical", "breakdown", "alternative"]
    body: str
    source: Source = "ai_explanation"


class Visualization(BaseModel):
    id: str
    type: Literal["widget", "mermaid", "chart", "svg"]
    title: str
    why: str = Field(description="what on this page the visual explains")
    widget: Optional[str] = None
    params: dict[str, Any] = {}
    mermaid: Optional[str] = None
    chart: Optional[dict[str, Any]] = None
    svg: Optional[str] = None

    @field_validator("params")
    @classmethod
    def _params_small(cls, v: dict[str, Any]) -> dict[str, Any]:
        if len(json.dumps(v)) > 20000:
            raise ValueError("params too large")
        return v


class Misconception(BaseModel):
    claim: str
    correction: str


class PageLink(BaseModel):
    page: int = Field(description="PDF page number")
    note: str


class Connections(BaseModel):
    earlier: list[PageLink] = []
    later: list[PageLink] = []
    prerequisites: list[PageLink] = []
    related_math: list[PageLink] = []
    related_code: list[PageLink] = []
    related_figures: list[PageLink] = []


class Question(BaseModel):
    id: str
    type: Literal["short", "conceptual", "math", "mcq", "self"]
    question: str
    options: list[str] = []
    answer: str
    explanation: str
    page_refs: list[int] = []

    @field_validator("options")
    @classmethod
    def _opts(cls, v: list[str]) -> list[str]:
        if v and len(v) < 3:
            raise ValueError("mcq needs >= 3 options")
        return v


class Overview(BaseModel):
    title: str
    chapter: str
    section: str = ""
    subsection: str = ""
    main_topic: str
    central_idea: str
    importance: str
    learning_objectives: list[str]
    prerequisites: list[str] = []
    page_kind: Literal["content", "chapter_opener", "front_matter", "index", "blank", "figure_only"] = "content"


class Generation(BaseModel):
    status: Literal["complete", "light"] = "complete"
    model: str
    generated_at: str
    confidence: Literal["high", "medium", "low"] = "high"
    review_flags: list[str] = []
    extraction_quality: str = ""


class PagePackage(BaseModel):
    page_id: str
    pdf_page: int
    printed_page: str
    book_id: str = BOOK_ID
    generation: Generation
    overview: Overview
    summary: str
    paragraphs: list[ParagraphNote] = []
    concepts: list[Concept] = []
    math: list[MathNote] = []
    code: list[CodeNote] = []
    deeper: list[Deeper] = []
    visualizations: list[Visualization] = []
    misconceptions: list[Misconception] = []
    connections: Connections = Connections()
    takeaways: list[str] = []
    questions: list[Question] = []


def validate_package(obj: dict) -> PagePackage:
    return PagePackage.model_validate(obj)
