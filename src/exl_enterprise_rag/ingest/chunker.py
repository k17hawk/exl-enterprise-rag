"""Heading-based chunker with merge/split heuristics.

Strategy:
  1. Split markdown into sections at each heading — but never inside a
     fenced code block (``` or ~~~), where '# comment' lines would
     otherwise be misread as H1 headings.
  2. Split sections that exceed max_tokens at paragraph boundaries,
     keeping fenced code blocks and tables atomic. Oversized tables are
     split by rows (header repeated); oversized code blocks are split by
     lines (fence re-opened/closed).
  3. LAST RESORT: any single unit (sentence, table row, code line) that
     alone exceeds max_tokens is hard-split on token windows via
     _force_split. This guarantees the size invariant holds for ANY
     input — mermaid diagrams, minified JSON, unclosed fences, etc.
  4. Merge adjacent tiny sections; fall back to merging backward.
  5. Root every heading path at the document title and prepend the
     breadcrumb to each chunk's text.

INVARIANT (asserted at the end of build_chunks):
    every chunk's token_count <= max_tokens + _BREADCRUMB_ALLOWANCE
A run can never silently emit an oversized chunk — it crashes at the
guilty document and names it.
"""

from __future__ import annotations

import math
import re

import tiktoken

from exl_enterprise_rag.config.config_entity import Chunk, RawSection

_TOKENIZER = tiktoken.get_encoding("cl100k_base")

_FENCE_PREFIXES = ("```", "~~~")

# Breadcrumbs are prepended AFTER size control, so allow headroom for them.
_BREADCRUMB_ALLOWANCE = 50


def count_tokens(text: str) -> int:
    """Token count using cl100k_base. Shared across the pipeline."""
    return len(_TOKENIZER.encode(text, disallowed_special=()))


# ------------------------------------------------------------------
# Step 1: split into sections by heading (fence-aware)
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


def _fence_marker(line: str) -> str | None:
    """Return the fence prefix ('```' or '~~~') if the line opens/closes a
    fenced code block, else None."""
    stripped = line.lstrip()
    for prefix in _FENCE_PREFIXES:
        if stripped.startswith(prefix):
            return prefix
    return None


def split_into_sections(markdown_body: str) -> list[RawSection]:
    """Walk lines, track heading stack, emit one RawSection per heading block.

    Lines inside fenced code blocks are never treated as headings.
    """
    sections: list[RawSection] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_path: tuple[str, ...] = ()
    in_fence = False
    fence = ""

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append(RawSection(
                heading_path=current_path,
                body=body,
            ))

    for line in markdown_body.splitlines():
        if in_fence:
            current_lines.append(line)
            if _fence_marker(line) == fence:
                in_fence = False
            continue

        marker = _fence_marker(line)
        if marker is not None:
            in_fence = True
            fence = marker
            current_lines.append(line)
            continue

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

def _force_split(text: str, max_tokens: int) -> list[str]:
    """LAST RESORT: hard token-window split.

    Used when no structural boundary (sentence, row, line) exists below
    max_tokens. Guarantees no returned piece exceeds max_tokens.
    """
    token_ids = _TOKENIZER.encode(text, disallowed_special=())
    if len(token_ids) <= max_tokens:
        return [text]
    # Balanced windows: 856 tokens at cap 800 becomes 2x428, not 800+56.
    n_pieces = math.ceil(len(token_ids) / max_tokens)
    window = math.ceil(len(token_ids) / n_pieces)
    return [
        _TOKENIZER.decode(token_ids[i:i + window])
        for i in range(0, len(token_ids), window)
    ]


def _split_by_sentence(text: str, max_tokens: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for s in sentences:
        s_tokens = count_tokens(s)
        if s_tokens > max_tokens:
            # A single "sentence" bigger than the cap: mermaid diagram,
            # minified blob, unclosed fence swallowed whole, etc. No
            # punctuation to split on — force-split by token window.
            if buf:
                out.append(" ".join(buf))
                buf, buf_tokens = [], 0
            out.extend(_force_split(s, max_tokens))
            continue
        if buf_tokens + s_tokens > max_tokens and buf:
            out.append(" ".join(buf))
            buf, buf_tokens = [], 0
        buf.append(s)
        buf_tokens += s_tokens
    if buf:
        out.append(" ".join(buf))
    return out


def _is_table(paragraph: str) -> bool:
    """True if every non-blank line of the paragraph is a Markdown table row."""
    lines = [line for line in paragraph.splitlines() if line.strip()]
    return bool(lines) and all(line.lstrip().startswith("|") for line in lines)


def _is_code_block(paragraph: str) -> bool:
    """True if the paragraph is a single fenced code block."""
    lines = paragraph.splitlines()
    if len(lines) < 2:
        return False
    opening = _fence_marker(lines[0])
    return opening is not None and _fence_marker(lines[-1]) == opening


_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def _split_table(paragraph: str, max_tokens: int) -> list[str]:
    """Split an oversized Markdown table by rows, repeating the header
    (and separator row, if present) on every piece. A single row larger
    than the cap is force-split."""
    lines = paragraph.splitlines()
    if len(lines) < 2:
        return [paragraph]

    header_lines = [lines[0]]
    body_start = 1
    if _TABLE_SEPARATOR_RE.match(lines[1]):
        header_lines.append(lines[1])
        body_start = 2

    header_block = "\n".join(header_lines)
    header_tokens = count_tokens(header_block)

    out: list[str] = []
    buf: list[str] = []
    buf_tokens = header_tokens

    def flush_buf() -> None:
        nonlocal buf, buf_tokens
        if buf:
            out.append(header_block + "\n" + "\n".join(buf))
            buf, buf_tokens = [], header_tokens

    for row in lines[body_start:]:
        row_tokens = count_tokens(row)
        if header_tokens + row_tokens > max_tokens:
            # Single monster row — no structural boundary left.
            flush_buf()
            out.extend(_force_split(row, max_tokens))
            continue
        if buf and buf_tokens + row_tokens > max_tokens:
            flush_buf()
        buf.append(row)
        buf_tokens += row_tokens
    flush_buf()
    return out or [paragraph]


def _split_code_block(paragraph: str, max_tokens: int) -> list[str]:
    """Split an oversized fenced code block by lines, re-opening and
    closing the fence on every piece. A single line larger than the cap
    is force-split."""
    lines = paragraph.splitlines()
    opening, closing = lines[0], lines[-1]
    body_lines = lines[1:-1]
    frame_tokens = count_tokens(opening) + count_tokens(closing)

    out: list[str] = []
    buf: list[str] = []
    buf_tokens = frame_tokens

    def flush_buf() -> None:
        nonlocal buf, buf_tokens
        if buf:
            out.append("\n".join([opening, *buf, closing]))
            buf, buf_tokens = [], frame_tokens

    for line in body_lines:
        line_tokens = count_tokens(line)
        if frame_tokens + line_tokens > max_tokens:
            # Single monster line (minified JS, base64 blob...).
            flush_buf()
            out.extend(_force_split(line, max_tokens))
            continue
        if buf and buf_tokens + line_tokens > max_tokens:
            flush_buf()
        buf.append(line)
        buf_tokens += line_tokens
    flush_buf()
    return out or [paragraph]


def _split_into_paragraph_blocks(body: str) -> list[str]:
    """Like body.split('\\n\\n'), but a fenced code block is one atomic
    block even when it contains blank lines."""
    blocks: list[str] = []
    buf: list[str] = []
    in_fence = False
    fence = ""

    for line in body.splitlines():
        if in_fence:
            buf.append(line)
            if _fence_marker(line) == fence:
                in_fence = False
            continue

        marker = _fence_marker(line)
        if marker is not None:
            in_fence = True
            fence = marker
            buf.append(line)
            continue

        if not line.strip():
            if buf:
                blocks.append("\n".join(buf))
                buf = []
            continue

        buf.append(line)

    if buf:
        blocks.append("\n".join(buf))
    return blocks


def _split_by_paragraph(body: str, max_tokens: int) -> list[str]:
    paragraphs = [p.strip() for p in _split_into_paragraph_blocks(body) if p.strip()]
    out: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for para in paragraphs:
        p_tokens = count_tokens(para)
        if p_tokens > max_tokens:
            if buf:
                out.append("\n\n".join(buf))
                buf, buf_tokens = [], 0
            if _is_table(para):
                out.extend(_split_table(para, max_tokens))
            elif _is_code_block(para):
                out.extend(_split_code_block(para, max_tokens))
            else:
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


def _common_prefix(paths: list[tuple[str, ...]]) -> tuple[str, ...]:
    """Longest shared prefix across a list of heading paths."""
    if not paths:
        return ()
    shortest = min(len(p) for p in paths)
    prefix: list[str] = []
    for i in range(shortest):
        if len({p[i] for p in paths}) == 1:
            prefix.append(paths[0][i])
        else:
            break
    return tuple(prefix)


def build_chunks(
    sections: list[RawSection],
    *,
    title: str | None = None,
    min_tokens: int = 100,
    max_tokens: int = 800,
) -> list[Chunk]:
    """Split oversized sections, merge tiny ones, prepend breadcrumb.

    Args:
        title: document title, prepended as the root of every section's
            heading path before splitting/merging. Callers should always
            resolve a non-None title (frontmatter -> first H1 -> path).

    Raises:
        AssertionError: if any produced chunk exceeds
            max_tokens + _BREADCRUMB_ALLOWANCE. This should be impossible
            with _force_split in place; if it fires, a new splitter hole
            has been found and must be fixed — never silence the assert.
    """

    if title:
        sections = [
            RawSection(heading_path=(title,) + sec.heading_path, body=sec.body)
            for sec in sections
        ]

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

    # Pass 2: merge adjacent tiny sections (forward, then backward fallback).
    merged: list[RawSection] = []
    i = 0
    while i < len(expanded):
        sec = expanded[i]
        sec_tokens = count_tokens(sec.body)

        if sec_tokens >= min_tokens:
            merged.append(sec)
            i += 1
            continue

        buf = [sec]
        buf_tokens = sec_tokens
        j = i + 1
        while j < len(expanded) and buf_tokens < min_tokens:
            nxt = expanded[j]
            nxt_tokens = count_tokens(nxt.body)
            if buf_tokens + nxt_tokens > max_tokens:
                break
            buf.append(nxt)
            buf_tokens += nxt_tokens
            j += 1

        if buf_tokens < min_tokens and merged:
            prev = merged[-1]
            prev_tokens = count_tokens(prev.body)
            if prev_tokens + buf_tokens <= max_tokens:
                merged[-1] = RawSection(
                    heading_path=_common_prefix(
                        [prev.heading_path] + [s.heading_path for s in buf]
                    ),
                    body=prev.body + "\n\n" + "\n\n".join(s.body for s in buf),
                )
                i = j
                continue

        merged_path = _common_prefix([s.heading_path for s in buf])
        merged.append(RawSection(
            heading_path=merged_path,
            body="\n\n".join(s.body for s in buf),
        ))
        i = j

    # Pass 2.5: EXACT size enforcement.
    # All buffers above track size as sum(count_tokens(part)), but parts
    # are joined with separators whose tokens are never counted, and
    # tokenization shifts across joined boundaries. Over a 50-row table
    # that drift reaches tens of tokens. This pass recounts every final
    # body EXACTLY and force-splits any that crept over the cap.
    sized: list[RawSection] = []
    for sec in merged:
        if count_tokens(sec.body) <= max_tokens:
            sized.append(sec)
        else:
            for piece in _force_split(sec.body, max_tokens):
                sized.append(RawSection(
                    heading_path=sec.heading_path,
                    body=piece,
                ))
    merged = sized

    # Pass 3: prepend breadcrumb, produce final chunks.
    chunks: list[Chunk] = []
    for sec in merged:
        breadcrumb = " > ".join(sec.heading_path) if sec.heading_path else ""
        content = f"{breadcrumb}\n\n{sec.body}" if breadcrumb else sec.body
        chunks.append(Chunk(
            heading_path=breadcrumb,
            content=content,
            token_count=count_tokens(content),
        ))

    # INVARIANT: no oversized chunk can leave this function silently.
    # Body is capped exactly by Pass 2.5; the per-chunk cap is that plus
    # this chunk's actual breadcrumb tokens plus join slack.
    for ch in chunks:
        bc_tokens = count_tokens(ch.heading_path) if ch.heading_path else 0
        cap = max_tokens + bc_tokens + 4
        assert ch.token_count <= cap, (
            f"chunk exceeds cap ({ch.token_count} > {cap} tokens) — "
            f"breadcrumb: {ch.heading_path[:100]!r} — "
            f"splitter hole, do not silence this assert"
        )

    return chunks