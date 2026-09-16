"""Data-access layer for the ingestion pipeline.

Every SQL statement the pipeline executes lives in this module — the
orchestrator never writes SQL. This is the seam that makes the pipeline
testable (repositories can be faked) and auditable (one file to review
when the schema changes).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Sequence

from psycopg import sql
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

SOURCE = "gitlab-handbook"

_IDENT_OK = set("abcdefghijklmnopqrstuvwxyz0123456789_")


def safe_identifier(name: str) -> str:
    if not name or name[0].isdigit() or not set(name) <= _IDENT_OK:
        raise ValueError(f"Unsafe SQL identifier from registry: {name!r}")
    return name


@dataclass(frozen=True)
class ActiveModel:
    model_name: str
    dimensions: int
    table_name: str


class EmbeddingModelRegistry:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    def active(self) -> ActiveModel:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT model_name, dimensions, table_name "
                "FROM embedding_models WHERE is_active"
            ).fetchone()
        if row is None:
            raise RuntimeError("No active embedding model in registry")
        return ActiveModel(row[0], row[1], safe_identifier(row[2]))


class IngestionRunRepository:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    def open(self, git_commit: str | None) -> uuid.UUID:
        with self.pool.connection() as conn:
            run_id = conn.execute(
                "INSERT INTO ingestion_runs (git_commit) VALUES (%s) "
                "RETURNING id",
                (git_commit,),
            ).fetchone()[0]
        return run_id

    def record_error(self, run_id: uuid.UUID, source_path: str,
                     error: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "INSERT INTO ingestion_errors (run_id, source_path, error) "
                "VALUES (%s, %s, %s)",
                (run_id, source_path, error[:2000]),
            )

    def close(self, run_id: uuid.UUID, status: str, stats: dict) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE ingestion_runs SET finished_at = now(), "
                "status = %s, stats = %s WHERE id = %s",
                (status, json.dumps(stats), run_id),
            )


class DocumentRepository:
    """Documents + chunks + embeddings, written atomically per document."""

    def __init__(self, pool: ConnectionPool, emb_table: str):
        self.pool = pool
        self.emb_table = sql.Identifier(safe_identifier(emb_table))

    def content_hash(self, source_path: str) -> str | None:
        """Hash of the currently indexed version, or None if unknown doc."""
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT content_hash FROM documents "
                "WHERE source = %s AND source_path = %s",
                (SOURCE, source_path),
            ).fetchone()
        return row[0] if row else None

    def write_document(
        self,
        *,
        source_path: str,
        title: str,
        department: str,
        content_hash: str,
        chunks: Sequence,            # objects with .content/.heading_path/.token_count
        vectors: Sequence,           # same length as chunks
    ) -> None:
        """Upsert document, replace its chunks and embeddings — ONE
        transaction. A crash never leaves a half-indexed document."""
        assert len(chunks) == len(vectors), "chunks/vectors length mismatch"
        with self.pool.connection() as conn:
            register_vector(conn)
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
                    (SOURCE, source_path, title, department, content_hash),
                ).fetchone()[0]

                conn.execute("DELETE FROM chunks WHERE document_id = %s",
                             (doc_id,))

                chunk_rows = [
                    (uuid.uuid4(), doc_id, idx, c.content, c.heading_path,
                     c.token_count, department)
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
                        sql.SQL(
                            "INSERT INTO {} (chunk_id, department, embedding) "
                            "VALUES (%s, %s, %s)"
                        ).format(self.emb_table),
                        [(row[0], department, vec)
                         for row, vec in zip(chunk_rows, vectors)],
                    )

    def tombstone_missing(self, seen_paths: list[str]) -> int:
        """Tombstone active documents whose files vanished from the corpus.
        Caller must only invoke this after a FULL (unfiltered) walk."""
        if not seen_paths:
            return 0
        with self.pool.connection() as conn:
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
                gone = [r[0] for r in cur.fetchall()]
                if gone:
                    conn.execute(
                        "UPDATE chunks SET status = 'tombstoned' "
                        "WHERE document_id = ANY(%s)", (gone,))
                    conn.execute(
                        sql.SQL(
                            "DELETE FROM {} WHERE chunk_id IN "
                            "(SELECT id FROM chunks WHERE document_id = ANY(%s))"
                        ).format(self.emb_table),
                        (gone,))
        return len(gone)