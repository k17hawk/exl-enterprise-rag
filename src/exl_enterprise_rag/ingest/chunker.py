"""Heading-based chunker with merge/split heuristics.

Strategy:
  1. Split markdown into sections at each ## / ### heading.
  2. Split sections that exceed max_tokens at paragraph boundaries.
  3. Merge adjacent tiny sections that share a parent heading.
  4. Prepend the breadcrumb (heading_path) to each chunk's text.
"""

from __future__ import annotations

import re

import tiktoken

from src.exl_enterprise_rag.config.config_entity import Chunk, RawSection

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Token count using cl100k_base. Shared across the pipeline."""
    return len(_TOKENIZER.encode(text, disallowed_special=()))


# ------------------------------------------------------------------
# Step 1: split into sections by heading
# ------------------------------------------------------------------

def _match_heading(line: str) -> tuple[int, str] | None:
    """Return (level, text) if the line is an ATX heading, else None."""
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return None
    hashes = 0
    for ch in stripped:
        if ch == "#":
            hashes += 1
        else:
            break
    if hashes == 0 or hashes > 6:
        return None
    rest = stripped[hashes:]
    if rest and not rest[0].isspace():
        return None
    return hashes, rest.strip()


def split_into_sections(markdown_body: str) -> list[RawSection]:
    """Walk lines, track heading stack, emit one RawSection per heading block."""
    sections: list[RawSection] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_path: tuple[str, ...] = ()

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append(RawSection(
                heading_path=current_path,
                body=body,
            ))

    for line in markdown_body.splitlines():
        m = _match_heading(line)
        if m is None:
            current_lines.append(line)
            continue

        flush()
        current_lines = []

        level, text = m
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, text))
        current_path = tuple(t for _, t in heading_stack)

    flush()
    return sections


# ------------------------------------------------------------------
# Step 2: split and merge
# ------------------------------------------------------------------

def _split_by_sentence(text: str, max_tokens: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for s in sentences:
        s_tokens = count_tokens(s)
        if buf_tokens + s_tokens > max_tokens and buf:
            out.append(" ".join(buf))
            buf, buf_tokens = [], 0
        buf.append(s)
        buf_tokens += s_tokens
    if buf:
        out.append(" ".join(buf))
    return out


def _split_by_paragraph(body: str, max_tokens: int) -> list[str]:
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    out: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for para in paragraphs:
        p_tokens = count_tokens(para)
        if p_tokens > max_tokens:
            if buf:
                out.append("\n\n".join(buf))
                buf, buf_tokens = [], 0
            out.extend(_split_by_sentence(para, max_tokens))
            continue
        if buf_tokens + p_tokens > max_tokens and buf:
            out.append("\n\n".join(buf))
            buf, buf_tokens = [], 0
        buf.append(para)
        buf_tokens += p_tokens
    if buf:
        out.append("\n\n".join(buf))
    return out


def build_chunks(
    sections: list[RawSection],
    *,
    min_tokens: int = 100,
    max_tokens: int = 800,
) -> list[Chunk]:
    """Split oversized sections, merge tiny ones, prepend breadcrumb."""

    # Pass 1: split sections that are too big
    expanded: list[RawSection] = []
    for sec in sections:
        if count_tokens(sec.body) <= max_tokens:
            expanded.append(sec)
        else:
            for piece in _split_by_paragraph(sec.body, max_tokens):
                expanded.append(RawSection(
                    heading_path=sec.heading_path,
                    body=piece,
                ))

    # Pass 2: merge adjacent tiny sections that share a parent
    merged: list[RawSection] = []
    i = 0
    while i < len(expanded):
        sec = expanded[i]
        if count_tokens(sec.body) >= min_tokens:
            merged.append(sec)
            i += 1
            continue

        buf = [sec]
        buf_tokens = count_tokens(sec.body)
        j = i + 1
        while j < len(expanded) and buf_tokens < min_tokens:
            nxt = expanded[j]
            if nxt.heading_path[:-1] != sec.heading_path[:-1]:
                break
            buf.append(nxt)
            buf_tokens += count_tokens(nxt.body)
            j += 1

        merged.append(RawSection(
            heading_path=sec.heading_path,
            body="\n\n".join(s.body for s in buf),
        ))
        i = j

    # Pass 3: prepend breadcrumb, produce final chunks
    chunks: list[Chunk] = []
    for sec in merged:
        breadcrumb = " > ".join(sec.heading_path) if sec.heading_path else ""
        content = f"{breadcrumb}\n\n{sec.body}" if breadcrumb else sec.body
        chunks.append(Chunk(
            heading_path=breadcrumb,
            content=content,
            token_count=count_tokens(content),
        ))
    return chunks