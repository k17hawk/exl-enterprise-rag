"""Title resolution and stub detection — shared by the phase-one stats
CLI and the ingestion pipeline.

Extracted from scripts/ingest.py so pipeline code never imports from a
CLI script.
"""

from __future__ import annotations

import re
from pathlib import Path


def is_stub_index(body: str) -> bool:
    """True if an _index.md is just a navigation stub (<200 chars of prose)."""
    no_links = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", body)
    prose_lines = [
        line for line in no_links.splitlines()
        if line.strip() and not line.strip().startswith(("-", "*", "{"))
    ]
    return sum(len(line) for line in prose_lines) < 200


def resolve_title(frontmatter_title: str | None, sections, path: Path) -> str:
    """Document title with fallback chain: frontmatter -> first H1 -> path.

    Guarantees a non-None title so every chunk's breadcrumb is rooted.
    For _index.md the path fallback uses the FOLDER name
    ('accounting/_index.md' -> 'Accounting').
    """
    if frontmatter_title:
        return frontmatter_title
    for sec in sections:
        if len(sec.heading_path) == 1:
            return sec.heading_path[0]
    stem = path.parent.name if path.stem == "_index" else path.stem
    return stem.replace("-", " ").replace("_", " ").title()