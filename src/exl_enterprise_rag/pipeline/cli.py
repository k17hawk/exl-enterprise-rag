"""Thin CLI for the ingestion pipeline.

All logic lives in pipeline/ingestion.py and db/repositories.py — this
file only wires components together and parses flags. Keeping the
entrypoint dumb is what lets a scheduler (cron, Airflow, Temporal) call
the same IngestionPipeline.run() without going through a shell.
"""

from __future__ import annotations

import logging
import os

import typer
from psycopg_pool import ConnectionPool
from rich.console import Console
from rich.table import Table

from src.exl_enterprise_rag.config.settings import get_departments, get_settings
from src.exl_enterprise_rag.db.repository import (
    DocumentRepository,
    EmbeddingModelRegistry,
    IngestionRunRepository,
)
from src.exl_enterprise_rag.pipeline.ingestion import (
    DocumentTransformer,
    EmbeddingService,
    IngestionPipeline,
)

app = typer.Typer(add_completion=False)
console = Console()


def build_pipeline(pool: ConnectionPool, *, batch_size: int,
                   min_tokens: int, max_tokens: int) -> IngestionPipeline:
    """Composition root: the ONLY place components are wired together."""
    settings = get_settings()
    departments = get_departments()
    registry = EmbeddingModelRegistry(pool)
    active = registry.active()
    return IngestionPipeline(
        settings=settings,
        departments=departments,
        registry=registry,
        runs=IngestionRunRepository(pool),
        documents=DocumentRepository(pool, active.table_name),
        embedder=EmbeddingService(active.model_name, active.dimensions,
                                  batch_size=batch_size),
        transformer=DocumentTransformer(
            includes_dir=settings.handbook_root / "content" / "includes",
            min_tokens=min_tokens, max_tokens=max_tokens),
    )


@app.command()
def ingest(
    department: str = typer.Option("all", "--department", "-d"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    wave_size: int = typer.Option(200, "--wave-size"),
    batch_size: int = typer.Option(64, "--batch-size"),
    min_tokens: int = typer.Option(100, "--min-tokens"),
    max_tokens: int = typer.Option(800, "--max-tokens"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("Set DATABASE_URL")

    only = None if department == "all" else {department}
    pool = ConnectionPool(dsn, min_size=1, max_size=8, open=True)
    try:
        pipeline = build_pipeline(pool, batch_size=batch_size,
                                  min_tokens=min_tokens,
                                  max_tokens=max_tokens)
        stats = pipeline.run(only_departments=only, force=force,
                             dry_run=dry_run, wave_size=wave_size)

        table = Table(title="Ingestion Run" + (" (dry run)" if dry_run else ""))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta", justify="right")
        for k, v in stats.as_dict().items():
            table.add_row(k, str(v))
        console.print(table)
    finally:
        pool.close()


if __name__ == "__main__":
    app()