from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path



@dataclass(frozen=True)
class SourceFile:
    """One Markdown file, with its department already resolved."""
    path: Path                     # absolute path on disk
    relative_path: str             # e.g. 'finance/expenses.md'
    source_path: str               # e.g. 'content/handbook/finance/expenses.md'
    folder: str                    # immediate child of content/handbook/, e.g. 'finance'
    department: str                # canonical department, e.g. 'finance'


@dataclass(frozen=True)
class ParsedFile:
    """A Markdown file after frontmatter extraction."""
    frontmatter: dict
    body: str
    content_hash: str              # sha256 of the raw file bytes
    title: str | None

@dataclass
class RawSection:
    """A section captured between two headings."""
    heading_path: tuple[str, ...]    # ('Finance', 'Expenses', 'Travel')
    body: str                        # markdown text (no heading line)


@dataclass
class Chunk:
    heading_path: str                # 'Finance > Expenses > Travel'
    content: str                     # breadcrumb-prefixed body
    token_count: int
