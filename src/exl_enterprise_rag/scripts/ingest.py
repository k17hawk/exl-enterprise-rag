"""CLI: ingest handbook Markdown into chunks (phase one — no DB writes).

Phase one purpose: prove the chunker produces good chunks before any data
touches the database. Run this, eyeball the samples, tune the thresholds.
Phase two (embed + insert) is a separate script.
"""

from __future__ import annotations

import random
import re
from collections import Counter

import typer
from rich.console import Console
from rich.table import Table

from src.exl_enterprise_rag.config.settings import get_departments, get_settings
from src.exl_enterprise_rag.ingest.chunker import (
    build_chunks,
    count_tokens,
    split_into_sections,
)
from src.exl_enterprise_rag.ingest.parser import parse_markdown
from src.exl_enterprise_rag.ingest.shortcodes import clean_shortcodes
from src.exl_enterprise_rag.ingest.walker import walk_corpus

app = typer.Typer(add_completion=False)
console = Console()


def _is_stub_index(body: str) -> bool:
    """True if an _index.md is just a navigation stub (<200 chars of prose)."""
    no_links = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", body)
    prose_lines = [
        line for line in no_links.splitlines()
        if line.strip() and not line.strip().startswith(("-", "*", "{"))
    ]
    return sum(len(line) for line in prose_lines) < 200


@app.command()
def main(
    department: str = typer.Option(
        "all", "--department", "-d",
        help="Department to ingest, or 'all' for the full allowlist.",
    ),
    sample: int = typer.Option(
        5, "--sample", "-s",
        help="How many random chunks to print for eyeballing.",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for sampling."),
    min_tokens: int = typer.Option(100, "--min-tokens"),
    max_tokens: int = typer.Option(800, "--max-tokens"),
) -> None:
    """Walk → parse → clean → chunk. Print stats. No DB writes."""
    settings = get_settings()
    departments = get_departments()

    only = None if department == "all" else {department}
    if only is not None and department not in departments.canonical():
        console.print(f"[red]Unknown department: {department}[/red]")
        console.print(f"Valid: {sorted(departments.canonical())}")
        raise typer.Exit(code=1)

    files = walk_corpus(settings, departments, only_departments=only)
    console.print(f"[bold]Found {len(files)} files[/bold] in allowlisted folders\n")

    files_walked = 0
    files_skipped_stub = 0
    files_parsed = 0
    files_errored = 0
    chunks_total = 0
    chunks_by_department: Counter[str] = Counter()
    tokens_by_chunk: list[int] = []
    all_chunks: list[tuple[str, str, str]] = []  # (dept, source_path, content)

    includes_dir = settings.handbook_root / "content" / "includes"

    for sf in files:
        files_walked += 1
        try:
            parsed = parse_markdown(sf.path)
        except Exception as e:
            files_errored += 1
            console.print(f"[red]parse error[/red] {sf.source_path}: {e}")
            continue

        body = clean_shortcodes(parsed.body, includes_dir=includes_dir)

        if sf.path.name == "_index.md" and _is_stub_index(body):
            files_skipped_stub += 1
            continue

        raw_sections = split_into_sections(body)
        chunks = build_chunks(
            raw_sections,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )

        if not chunks:
            continue

        files_parsed += 1
        for ch in chunks:
            chunks_total += 1
            chunks_by_department[sf.department] += 1
            tokens_by_chunk.append(ch.token_count)
            all_chunks.append((sf.department, sf.source_path, ch.content))

    # ---- Stats report ----
    tokens = sorted(tokens_by_chunk)
    if tokens:
        def pct(p: float) -> int:
            idx = max(0, min(len(tokens) - 1, int(len(tokens) * p)))
            return tokens[idx]
        p50, p95, tmax = pct(0.50), pct(0.95), tokens[-1]
        tiny = sum(1 for t in tokens if t < 50)
        huge = sum(1 for t in tokens if t > 1000)
    else:
        p50 = p95 = tmax = tiny = huge = 0

    table = Table(title="Phase One Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta", justify="right")
    table.add_row("Files walked", str(files_walked))
    table.add_row("Files skipped (stub _index)", str(files_skipped_stub))
    table.add_row("Files parsed", str(files_parsed))
    table.add_row("Files errored", str(files_errored))
    table.add_row("Total chunks", str(chunks_total))
    table.add_row("Tokens p50", str(p50))
    table.add_row("Tokens p95", str(p95))
    table.add_row("Tokens max", str(tmax))
    table.add_row("Chunks <50 tokens", str(tiny))
    table.add_row("Chunks >1000 tokens", str(huge))
    console.print(table)

    dept_table = Table(title="Chunks per Department")
    dept_table.add_column("Department")
    dept_table.add_column("Chunks", justify="right")
    for dept, n in chunks_by_department.most_common():
        dept_table.add_row(dept, str(n))
    console.print(dept_table)

    # ---- Random samples ----
    if sample > 0 and all_chunks:
        console.rule("[bold]Random Samples[/bold]")
        rng = random.Random(seed)
        picks = rng.sample(all_chunks, min(sample, len(all_chunks)))
        for i, (dept, src, content) in enumerate(picks, 1):
            console.rule(f"[bold yellow]Sample {i} — {dept}[/bold yellow]")
            console.print(f"[dim]{src}[/dim]")
            console.print(f"[dim]tokens: {count_tokens(content)}[/dim]\n")
            console.print(content[:2000])
            if len(content) > 2000:
                console.print("[dim]...[truncated][/dim]")
            console.print()


if __name__ == "__main__":
    app()