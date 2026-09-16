"""Phase two: walk → parse → clean → chunk → EMBED → INSERT.

Writes to the v2 schema:
    documents → chunks → chunk_embeddings_<active model table>
per document, inside ONE transaction per document (a crash never leaves
a document with half its chunks).

Idempotent & incremental:
  * Unchanged documents (same content_hash) are skipped unless --force.
  * Changed documents are re-chunked: old chunks deleted (embeddings
    cascade), new ones inserted.
  * Documents that vanished from the corpus are tombstoned at the end.
  * Every run is recorded in ingestion_runs; per-file failures go to
    ingestion_errors instead of crashing the run.

Requires: pip install "psycopg[binary]" pgvector sentence-transformers
Environment: DATABASE_URL (e.g. postgresql://rag:rag@localhost:5432/rag)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from datetime import datetime, timezone

import psycopg
import typer
from pgvector.psycopg import register_vector
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer

from src.exl_enterprise_rag.config.settings import get_departments, get_settings
from src.exl_enterprise_rag.ingest.chunker import (
    build_chunks,
    split_into_sections,
)
from src.exl_enterprise_rag.ingest.parser import parse_markdown
from src.exl_enterprise_rag.ingest.shortcodes import clean_shortcodes
from src.exl_enterprise_rag.ingest.walker import walk_corpus
from src.exl_enterprise_rag.scripts.ingest import _is_stub_index, resolve_title

app = typer.Typer(add_completion=False)
console = Console()

SOURCE = "gitlab-handbook"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _git_commit(repo_root) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _active_model(conn) -> tuple[str, int, str]:
    """Read the active embedding model from the registry. Refuse to run
    without exactly one active model — never guess where vectors go."""
    row = conn.execute(
        "SELECT model_name, dimensions, table_name "
        "FROM embedding_models WHERE is_active"
    ).fetchone()
    if row is None:
        raise SystemExit(
            "No active embedding model in embedding_models. "
            "Seed/activate one before ingesting."
        )
    return row[0], row[1], row[2]


_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def _safe_ident(name: str) -> str:
    """The embeddings table name comes from a DB registry row and is
    interpolated into SQL — validate it as a bare identifier first."""
    if not _IDENT_RE.match(name):
        raise SystemExit(f"Unsafe table name from registry: {name!r}")
    return name


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

@app.command()
def main(
    department: str = typer.Option("all", "--department", "-d"),
    batch_size: int = typer.Option(64, "--batch-size",
                                   help="Chunks per embedding batch."),
    min_tokens: int = typer.Option(100, "--min-tokens"),
    max_tokens: int = typer.Option(800, "--max-tokens"),
    force: bool = typer.Option(False, "--force",
                               help="Re-embed even unchanged documents."),
    dry_run: bool = typer.Option(False, "--dry-run",
                                 help="Report what would change; no writes."),
) -> None:
    settings = get_settings()
    departments = get_departments()
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("Set DATABASE_URL, e.g. postgresql://rag:rag@localhost:5432/rag")

    only = None if department == "all" else {department}
    files = walk_corpus(settings, departments, only_departments=only)
    console.print(f"[bold]Found {len(files)} files[/bold]")

    conn = psycopg.connect(dsn, autocommit=False)
    register_vector(conn)

    model_name, dims, emb_table = _active_model(conn)
    emb_table = _safe_ident(emb_table)
    console.print(f"Active embedder: [bold]{model_name}[/bold] "
                  f"({dims} dims → {emb_table})")

    console.print("Loading model (first run downloads weights)...")
    model = SentenceTransformer(model_name)
    probe = model.encode(["dimension probe"], normalize_embeddings=True)
    if probe.shape[1] != dims:
        raise SystemExit(
            f"Model emits {probe.shape[1]} dims but registry says {dims}. "
            f"Fix the registry/table before ingesting."
        )

    # Open the ingestion run
    run_id = None
    if not dry_run:
        run_id = conn.execute(
            "INSERT INTO ingestion_runs (git_commit) VALUES (%s) RETURNING id",
            (_git_commit(settings.handbook_root),),
        ).fetchone()[0]
        conn.commit()

    includes_dir = settings.handbook_root / "content" / "includes"
    stats = {
        "walked": 0, "added": 0, "updated": 0, "unchanged": 0,
        "skipped_stub": 0, "skipped_empty": 0, "errored": 0,
        "chunks_written": 0, "tombstoned": 0,
    }
    seen_paths: list[str] = []

    for sf in files:
        stats["walked"] += 1
        try:
            parsed = parse_markdown(sf.path)
            body = clean_shortcodes(parsed.body, includes_dir=includes_dir)

            if sf.path.name == "_index.md" and _is_stub_index(body):
                stats["skipped_stub"] += 1
                continue

            raw_sections = split_into_sections(body)
            title = resolve_title(parsed.title, raw_sections, sf.path)
            chunks = build_chunks(raw_sections, title=title,
                                  min_tokens=min_tokens, max_tokens=max_tokens)
            if not chunks:
                stats["skipped_empty"] += 1
                continue

            seen_paths.append(sf.source_path)

            # Change detection
            row = conn.execute(
                "SELECT id, content_hash FROM documents "
                "WHERE source = %s AND source_path = %s",
                (SOURCE, sf.source_path),
            ).fetchone()
            if row and row[1] == parsed.content_hash and not force:
                stats["unchanged"] += 1
                continue

            if dry_run:
                stats["updated" if row else "added"] += 1
                stats["chunks_written"] += len(chunks)
                continue

            # Embed (batched) BEFORE opening the write transaction, so
            # slow GPU work never holds row locks.
            vectors = []
            for i in range(0, len(chunks), batch_size):
                batch = [c.content for c in chunks[i:i + batch_size]]
                vectors.extend(model.encode(batch, normalize_embeddings=True))

            # One transaction per document
            with conn.transaction():
                doc_id = conn.execute(
                    """
                    INSERT INTO documents
                        (source, source_path, title, department,
                         content_hash, indexed_at, status)
                    VALUES (%s, %s, %s, %s, %s, now(), 'active')
                    ON CONFLICT (source, source_path) DO UPDATE SET
                        title = EXCLUDED.title,
                        department = EXCLUDED.department,
                        content_hash = EXCLUDED.content_hash,
                        indexed_at = now(),
                        status = 'active'
                    RETURNING id
                    """,
                    (SOURCE, sf.source_path, title,
                     sf.department, parsed.content_hash),
                ).fetchone()[0]

                # Re-chunk = wipe & rewrite (embeddings cascade on delete)
                conn.execute("DELETE FROM chunks WHERE document_id = %s",
                             (doc_id,))

                chunk_rows = [
                    (uuid.uuid4(), doc_id, idx, c.content, c.heading_path,
                     c.token_count, sf.department)
                    for idx, c in enumerate(chunks)
                ]
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO chunks
                            (id, document_id, chunk_index, content,
                             heading_path, token_count, department)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        chunk_rows,
                    )
                    cur.executemany(
                        f"""
                        INSERT INTO {emb_table} (chunk_id, department, embedding)
                        VALUES (%s, %s, %s)
                        """,
                        [(cr[0], sf.department, vec)
                         for cr, vec in zip(chunk_rows, vectors)],
                    )

            stats["updated" if row else "added"] += 1
            stats["chunks_written"] += len(chunks)

            done = stats["added"] + stats["updated"] + stats["unchanged"]
            if done % 100 == 0:
                console.print(f"  ...{done} documents processed")

        except Exception as e:
            stats["errored"] += 1
            conn.rollback()
            console.print(f"[red]error[/red] {sf.source_path}: {e}")
            if not dry_run and run_id is not None:
                conn.execute(
                    "INSERT INTO ingestion_errors (run_id, source_path, error) "
                    "VALUES (%s, %s, %s)",
                    (run_id, sf.source_path, str(e)[:2000]),
                )
                conn.commit()

    # Tombstone documents that disappeared from the corpus.
    # Only safe on a full run — a filtered run doesn't see other
    # departments and would tombstone them all.
    if not dry_run and only is None and seen_paths:
        with conn.transaction():
            cur = conn.execute(
                """
                UPDATE documents SET status = 'tombstoned'
                WHERE source = %s AND status = 'active'
                  AND NOT (source_path = ANY(%s))
                RETURNING id
                """,
                (SOURCE, seen_paths),
            )
            gone_ids = [r[0] for r in cur.fetchall()]
            if gone_ids:
                conn.execute(
                    "UPDATE chunks SET status = 'tombstoned' "
                    "WHERE document_id = ANY(%s)", (gone_ids,))
                conn.execute(
                    f"DELETE FROM {emb_table} WHERE chunk_id IN "
                    f"(SELECT id FROM chunks WHERE document_id = ANY(%s))",
                    (gone_ids,))
            stats["tombstoned"] = len(gone_ids)

    # Close the run
    if not dry_run and run_id is not None:
        conn.execute(
            "UPDATE ingestion_runs SET finished_at = now(), "
            "status = 'succeeded', stats = %s WHERE id = %s",
            (json.dumps(stats), run_id),
        )
        conn.commit()

    # Report
    table = Table(title=f"Phase Two — {'DRY RUN' if dry_run else 'Ingestion'} "
                        f"({datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta", justify="right")
    for k, v in stats.items():
        table.add_row(k, str(v))
    console.print(table)
    conn.close()


if __name__ == "__main__":
    app()