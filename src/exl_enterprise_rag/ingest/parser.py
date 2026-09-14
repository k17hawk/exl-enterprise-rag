"""Parse Markdown files: split frontmatter from body."""

from __future__ import annotations

import hashlib
from pathlib import Path

import frontmatter

from exl_enterprise_rag.config.config_entity import ParsedFile


def parse_markdown(path: Path) -> ParsedFile:
    """Read a Markdown file, split frontmatter, hash the raw content.

    The hash is computed on raw bytes (before any parsing) so it detects
    any change to the file, including whitespace.
    """
    raw_bytes = path.read_bytes()
    content_hash = hashlib.sha256(raw_bytes).hexdigest()

    raw_text = raw_bytes.decode("utf-8")
    post = frontmatter.loads(raw_text)

    fm = dict(post.metadata)
    body = post.content

    title = fm.get("title")
    if isinstance(title, str):
        title = title.strip()
    else:
        title = None

    return ParsedFile(
        frontmatter=fm,
        body=body,
        content_hash=content_hash,
        title=title,
    )