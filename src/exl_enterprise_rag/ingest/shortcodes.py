"""Strip or resolve Hugo shortcodes while preserving content.

Rules derived from the actual corpus (see grep of {{% %}} and {{< >}}):
  - Wrappers (alert, panel, details, card): strip tags, keep inner text.
  - Semantic inline ({{< yes >}}, {{< no >}}, {{< label >}}): replace with text.
  - Includes ({{% include "path" %}}): resolve by reading the file, or mark inline.
  - Anything else: strip.
"""

from __future__ import annotations

import re
from pathlib import Path

# Matches {{% ... %}} and {{< ... >}} — non-greedy, DOTALL for multiline
_SHORTCODE_RE = re.compile(r"\{\{[%<].*?[%>]\}\}", re.DOTALL)

# Inline replacements: semantic shortcodes → text
_INLINE_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\{\{<\s*yes\s*>\}\}"), "Yes"),
    (re.compile(r"\{\{<\s*no\s*>\}\}"), "No"),
    (re.compile(r'\{\{<\s*label\s+name="([^"]+)"[^>]*>\}\}'), r"[\1]"),
]

# Wrapper tags whose inner text should be kept
_WRAPPER_TAGS = ("alert", "panel", "details", "card")


def clean_shortcodes(body: str, includes_dir: Path | None = None) -> str:
    """Remove shortcode syntax, keep inner content where it matters.

    Order matters:
      1. Resolve {{% include %}} (inline the referenced file)
      2. Replace semantic inline shortcodes ({{< yes >}} → Yes)
      3. Strip wrapper open/close tags ({{% alert %}}, {{% /alert %}})
      4. Strip any remaining shortcodes (catch-all)
    """

    # 1. Includes
    def _resolve_include(match: re.Match[str]) -> str:
        rel = match.group(1)
        if includes_dir is not None:
            candidate = includes_dir / rel
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8")
        return f"[included: {rel}]"

    body = re.sub(
        r'\{\{%\s*include\s+"([^"]+)"\s*%\}\}',
        _resolve_include,
        body,
    )

    # 2. Inline semantic replacements
    for pattern, replacement in _INLINE_REPLACEMENTS:
        body = pattern.sub(replacement, body)

    # 3. Strip wrapper tags (keep inner text)
    for tag in _WRAPPER_TAGS:
        body = re.sub(rf"\{{\{{%\s*/?{tag}\b[^%]*%\}}\}}", "", body)

    # 4. Catch-all
    body = _SHORTCODE_RE.sub("", body)

    # Collapse 3+ newlines created by removing tags
    body = re.sub(r"\n{3,}", "\n\n", body)

    return body.strip()