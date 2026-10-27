"""chunking.py

Split one article (a Markdown transcription) into sections and chunks for the article index.

Responsibilities:
- Find sections from the ## to #### headings. section_id is the heading number when there is one ("3.2"), otherwise a
  slug of the heading path ("results/growth"), because unnumbered titles repeat across Methods, Results and Discussion.
  Text before the first ## heading (title, authors, cover pages) is the section "front".
- Group the paragraphs of each section into chunks of about TARGET_CHARS, never across sections or across kinds of
  block. A chunk's text is always the exact source slice source[char_start:char_end].
- Keep tables whole; a table longer than MAX_CHARS is split by rows, and each later piece carries the header as overlap.
- Carry the last sentence of the previous text chunk of the same section as overlap, stored apart from the chunk text.
- Track printed pages from the [p. N] markers and label editor-written blocks ([Figure description ...]).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

TARGET_CHARS = 1000
MAX_CHARS = 1500
OVERLAP_MAX_CHARS = 200

HEADING = re.compile(r"^(#{1,4})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
PAGE_MARKER = re.compile(r"\[p\. (\d+)\]")
NUMBERED_TITLE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")
BLANK_LINES = re.compile(r"\n[ \t]*\n")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
REFERENCE_TITLES = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+)?(references|literature cited)\b", re.IGNORECASE)
FIGURE_PREFIXES = ("[Figure description", "[Figure data")

KINDS = ("text", "table", "figure_description", "editor_note", "references")


@dataclass
class Section:
    section_id: str
    title: str
    level: int
    path: str
    order: int
    char_start: int
    char_end: int
    page_start: int | None = None
    page_end: int | None = None
    n_chunks: int = 0


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    section_id: str
    section_path: str
    seq: int
    kind: str
    char_start: int
    char_end: int
    text: str
    overlap: str = ""
    page_start: int | None = None
    page_end: int | None = None


@dataclass
class Block:
    start: int
    end: int
    kind: str


@dataclass
class ArticleChunks:
    document_id: str
    sections: list[Section] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)


def slugify(text: str, limit: int = 40) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return slug[:limit].rstrip("-") or "section"


def pages_between(markers: list[tuple[int, int]], start: int, end: int) -> tuple[int | None, int | None]:
    """(first, last) printed page of a span: the page in force at its start and the last [p. N] marker inside it."""
    before = [number for position, number in markers if position <= start]
    inside = [number for position, number in markers if start < position < end]
    first = before[-1] if before else (inside[0] if inside else None)
    return first, (inside[-1] if inside else first)


def find_sections(source: str) -> list[Section]:
    body_headings = [h for h in HEADING.finditer(source) if len(h.group(1)) >= 2]
    used: set[str] = set()

    def unique(section_id: str) -> str:
        candidate, n = section_id, 2
        while candidate in used:
            candidate, n = f"{section_id}-{n}", n + 1
        used.add(candidate)
        return candidate

    front_end = body_headings[0].start() if body_headings else len(source)
    sections = [Section(unique("front"), "Front matter", 1, "Front matter", 0, 0, front_end)]
    stack: list[str] = []
    for order, match in enumerate(body_headings, start=1):
        level, title = len(match.group(1)), match.group(2).strip()
        stack = stack[: level - 2] + [title]
        number = NUMBERED_TITLE.match(title)
        section_id = number.group(1) if number else "/".join(slugify(part) for part in stack)
        end = body_headings[order].start() if order < len(body_headings) else len(source)
        sections.append(Section(unique(section_id), title, level, " > ".join(stack), order, match.start(), end))
    return sections


def block_kind(text: str, in_references: bool) -> str:
    if text.startswith("|"):
        return "table"
    if text.startswith(FIGURE_PREFIXES):
        return "figure_description"
    if text.startswith("[") and text.endswith("]"):
        return "editor_note"
    return "references" if in_references else "text"


def section_blocks(source: str, section: Section) -> list[Block]:
    """Paragraph blocks of a section body (after its heading line), with absolute offsets; page markers alone are skipped."""
    if section.section_id == "front":
        body_start = section.char_start
    else:
        line_end = source.find("\n", section.char_start, section.char_end)
        body_start = section.char_end if line_end == -1 else line_end + 1
    body = source[body_start : section.char_end]
    spans, position = [], 0
    for separator in BLANK_LINES.finditer(body):
        spans.append((position, separator.start()))
        position = separator.end()
    spans.append((position, len(body)))

    in_references = bool(REFERENCE_TITLES.match(section.title))
    blocks = []
    for start, end in spans:
        piece = body[start:end]
        stripped = piece.strip()
        if not stripped or PAGE_MARKER.fullmatch(stripped):
            continue
        absolute = body_start + start + (len(piece) - len(piece.lstrip()))
        blocks.append(Block(absolute, absolute + len(stripped), block_kind(stripped, in_references)))
    return blocks


def split_long_block(source: str, block: Block) -> list[tuple[int, int, str]]:
    """(start, end, overlap) pieces of a block longer than MAX_CHARS: tables by rows, text by sentences."""
    text = source[block.start : block.end]
    if block.kind == "table":
        lines = text.split("\n")
        header = "\n".join(lines[:2])
        pieces: list[tuple[int, int]] = []
        piece_start, piece_len, rows = block.start, len(header) + 1, 0
        offset = block.start + len(header) + 1
        for line in lines[2:]:
            if rows and piece_len + len(line) + 1 > MAX_CHARS:
                pieces.append((piece_start, offset - 1))
                piece_start, piece_len, rows = offset, len(header) + 1, 0
            piece_len += len(line) + 1
            offset += len(line) + 1
            rows += 1
        pieces.append((piece_start, block.end))
        return [(start, end, "" if start == block.start else header) for start, end in pieces]

    cuts = [block.start + match.start() for match in SENTENCE_END.finditer(text)] + [block.end]
    pieces, start, last_cut = [], block.start, block.start
    for cut in cuts:
        if cut - start > TARGET_CHARS and last_cut > start:
            pieces.append((start, last_cut))
            start = last_cut
        last_cut = cut
    pieces.append((start, block.end))
    trimmed = []
    for start, end in pieces:
        piece = source[start:end]
        trimmed.append((start + len(piece) - len(piece.lstrip()), end - (len(piece) - len(piece.rstrip())), ""))
    return trimmed


def last_sentence(text: str) -> str:
    sentences = SENTENCE_END.split(text.strip())
    return (sentences[-1] if sentences else "")[-OVERLAP_MAX_CHARS:]


def chunk_article(document_id: str, source: str) -> ArticleChunks:
    markers = [(match.start(), int(match.group(1))) for match in PAGE_MARKER.finditer(source)]
    result = ArticleChunks(document_id)
    seq = 0
    for section in find_sections(source):
        section.page_start, section.page_end = pages_between(markers, section.char_start, section.char_end)
        groups: list[tuple[str, int, int, str]] = []  # kind, start, end, overlap
        for block in section_blocks(source, section):
            if block.end - block.start > MAX_CHARS:
                groups.extend((block.kind, start, end, overlap) for start, end, overlap in split_long_block(source, block))
                continue
            if groups:
                kind, start, end, overlap = groups[-1]
                small = end - start < TARGET_CHARS // 3
                fits = block.end - start <= TARGET_CHARS or (small and block.end - start <= MAX_CHARS)
                if kind == block.kind and kind != "table" and fits:
                    groups[-1] = (kind, start, block.end, overlap)
                    continue
            groups.append((block.kind, block.start, block.end, ""))

        previous_text = ""
        for kind, start, end, overlap in groups:
            text = source[start:end]
            if not overlap and kind in ("text", "references") and previous_text:
                overlap = last_sentence(previous_text)
            page_start, page_end = pages_between(markers, start, end)
            seq += 1
            result.chunks.append(
                Chunk(
                    chunk_id=f"{document_id}:{seq:03d}",
                    document_id=document_id,
                    section_id=section.section_id,
                    section_path=section.path,
                    seq=seq,
                    kind=kind,
                    char_start=start,
                    char_end=end,
                    text=text,
                    overlap=overlap,
                    page_start=page_start,
                    page_end=page_end,
                )
            )
            previous_text = text if kind in ("text", "references") else ""
            section.n_chunks += 1
        result.sections.append(section)
    return result


def chunk_file(document_id: str, path: Path) -> ArticleChunks:
    return chunk_article(document_id, path.read_text(encoding="utf-8"))
